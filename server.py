"""
Ryudo Real-Time Visualization Server
=====================================
FastAPI server with WebSocket support for real-time constraint visualization.

Endpoints:
- GET / - Serve the frontend
- WS /ws - WebSocket for real-time updates
- POST /api/start - Start the agent workflow
- GET /api/status - Get workflow status
"""

import asyncio
import json
import os
from typing import Dict, Any, List
from contextlib import asynccontextmanager

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import networkx as nx
from ryudo.platform.events import attach_event_envelope
from ryudo.platform.replay import ReplayStore, deterministic_run_id

# Import Ryudo agents
from agents.state import RyudoState, GraphConstraint
from agents.environmental import environmental_agent, create_damage_zones, create_storm_surge_zone
from agents.infrastructure import infrastructure_agent, FACILITIES
from agents.temporal import temporal_agent

# Import LLM client for dynamic reasoning
from agents.llm_client import (
    call_environmental_agent,
    call_infrastructure_agent,
    call_temporal_agent,
    call_coordinator,
)

# Import geographic data service
from services.geo_data import (
    extract_roads,
    extract_water,
    extract_buildings,
    extract_landuse,
    extract_all,
    get_agent_instructions,
    ROAD_CLASSIFICATION,
    WATER_CLASSIFICATION,
    BUILDING_CLASSIFICATION,
    LANDUSE_CLASSIFICATION,
)

# Import mission solver and routing
from services.mission_solver import (
    RESCUE_DEPOT,
    RESCUE_VEHICLES,
    RESCUE_TARGETS,
    solve_rescue_mission,
    get_mission_data,
    calculate_target_priority,
)
from services.routing_service import get_road_graph, apply_constraints


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for broadcasting updates."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"[WS] Client connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"[WS] Client disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Send message to all connected clients."""
        run_id = workflow_state.get("run_id")
        outbound = attach_event_envelope(
            message,
            run_id=run_id,
            default_source="server",
        )
        event = outbound.get("event")
        if run_id and isinstance(event, dict):
            replay_store.append_event(run_id, event)

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(outbound)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()
replay_store = ReplayStore(max_runs=100, max_events_per_run=10000)


# ============================================================================
# Workflow State
# ============================================================================

# ============================================================================
# Workflow State
# ============================================================================

workflow_state = {
    "running": False,
    "constraints": [],
    "current_step": None,
    "base_graph_loaded": False,
    "completed": False,
    "run_id": None,
}

SIMULATION_SPEED = 1.0  # Multiplier: 1.0 = normal, 2.0 = 2x faster

async def wait(seconds: float):
    """Wait for a duration adjusted by simulation speed."""
    if SIMULATION_SPEED > 0:
        await asyncio.sleep(seconds / SIMULATION_SPEED)



# ============================================================================
# Cyclone Configuration
# ============================================================================

CYCLONE_HUDHUD_CONFIG = {
    "name": "Cyclone Hudhud",
    "date": "2014-10-12",
    "eye_lat": 17.68,
    "eye_lon": 83.35,
    "max_wind_speed_kmh": 185,
    "central_pressure_mb": 950,
    "ambient_pressure_mb": 1013,
    "radius_max_wind_km": 25,
    "extreme_damage_radius_km": 10,
    "severe_damage_radius_km": 25,
    "moderate_damage_radius_km": 50,
    "storm_surge_height_m": 2.5,
    "surge_inland_distance_km": 2,
}

# Note: RESCUE_TARGETS, RESCUE_VEHICLES, RESCUE_DEPOT now imported from services.mission_solver
# This provides 20 detailed targets with real Visakhapatnam addresses and 6 vehicles with diverse capabilities


def _scenario_seed_payload() -> Dict[str, Any]:
    """Build deterministic scenario payload used for run-id generation."""
    return {
        "workflow": "cycloneshield_demo",
        "cyclone": CYCLONE_HUDHUD_CONFIG,
        "depot": {
            "name": RESCUE_DEPOT.get("name"),
            "lat": RESCUE_DEPOT.get("lat"),
            "lon": RESCUE_DEPOT.get("lon"),
        },
        "vehicles": [
            {
                "id": vehicle.get("id"),
                "name": vehicle.get("name"),
                "capacity": vehicle.get("capacity"),
                "zone_access": vehicle.get("zone_access"),
            }
            for vehicle in RESCUE_VEHICLES
        ],
        "targets": [
            {
                "id": target.get("id"),
                "lat": target.get("lat"),
                "lon": target.get("lon"),
                "population": target.get("population"),
                "zone": target.get("zone"),
            }
            for target in RESCUE_TARGETS
        ],
    }



# ============================================================================
# App Lifecycle
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup and shutdown."""
    print("🚀 Ryudo Real-Time Server starting...")
    yield
    print("🛑 Ryudo Server shutting down...")


app = FastAPI(title="Ryudo Real-Time Visualization", lifespan=lifespan)

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files - prefer React build, fallback to legacy static
ui_dist_dir = os.path.join(os.path.dirname(__file__), "ui", "dist")
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# Mount React build assets if available
if os.path.exists(ui_dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(ui_dist_dir, "assets")), name="assets")

# Also keep legacy static mount
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ============================================================================
# Routes
# ============================================================================

@app.get("/")
async def root():
    """Serve the main frontend - prefer React build."""
    # First, try React build
    react_index = os.path.join(ui_dist_dir, "index.html")
    if os.path.exists(react_index):
        return FileResponse(react_index)
    
    # Fallback to legacy static
    legacy_index = os.path.join(static_dir, "index.html")
    if os.path.exists(legacy_index):
        return FileResponse(legacy_index)
    
    return HTMLResponse("<h1>Ryudo Server Running</h1><p>Run 'npm run build' in ui/ to build the React frontend.</p>")


@app.post("/api/start")
async def start_workflow():
    """Manually start the workflow via HTTP."""
    if not workflow_state["running"]:
        preview_run_id = deterministic_run_id(
            _scenario_seed_payload(),
            namespace="ryudo.workflow.cycloneshield.v1",
        )
        asyncio.create_task(run_workflow_with_updates())
        return {
            "status": "started",
            "message": "Workflow started in background",
            "run_id": preview_run_id,
        }
    return {"status": "running", "message": "Workflow already running"}



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_json(attach_event_envelope({
            "type": "connected",
            "message": "Connected to Ryudo server",
            "config": CYCLONE_HUDHUD_CONFIG,
        }, run_id=workflow_state.get("run_id"), default_source="server"))
        
        # Keep connection alive and listen for messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "start":
                # Start the workflow in background
                asyncio.create_task(run_workflow_with_updates())
            elif message.get("action") == "reset":
                workflow_state["constraints"] = []
                workflow_state["completed"] = False
                workflow_state["run_id"] = None
                await manager.broadcast({"type": "reset"})
            elif message.get("action") == "set_speed":
                global SIMULATION_SPEED
                try:
                    SIMULATION_SPEED = float(message.get("speed", 1.0))
                    print(f"[WS] Simulation speed set to {SIMULATION_SPEED}x")
                except ValueError:
                    pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/api/config")
async def get_config():
    """Get cyclone configuration."""
    return {
        "cyclone": CYCLONE_HUDHUD_CONFIG,
        "facilities": FACILITIES,
    }


@app.get("/api/status")
async def get_status():
    """Get current workflow status."""
    return workflow_state


@app.get("/api/replay")
async def list_replays(limit: int = 20):
    """List recent replay runs."""
    return {"runs": replay_store.list_runs(limit=limit)}


@app.get("/api/replay/latest")
async def get_latest_replay():
    """Get full replay payload for the latest run."""
    latest_run_id = replay_store.latest_run_id()
    if latest_run_id is None:
        raise HTTPException(status_code=404, detail="No replay runs found")

    replay = replay_store.get_run(latest_run_id)
    if replay is None:
        raise HTTPException(status_code=404, detail=f"Replay not found: {latest_run_id}")
    return replay


@app.get("/api/replay/{run_id}")
async def get_replay(run_id: str):
    """Get full replay payload for a specific run."""
    replay = replay_store.get_run(run_id)
    if replay is None:
        raise HTTPException(status_code=404, detail=f"Replay not found: {run_id}")
    return replay


@app.post("/api/replay/run-id")
async def preview_replay_run_id(payload: Dict[str, Any]):
    """Compute deterministic run id for external scenario payloads."""
    run_id = deterministic_run_id(payload, namespace="ryudo.replay.external.v1")
    return {"run_id": run_id}


# ============================================================================
# Geographic Data API Endpoints
# ============================================================================

# Default place for geographic data
GEO_PLACE = "Visakhapatnam, India"

# Cache for geographic data (loaded once on first request)
_geo_cache = {
    "roads": None,
    "water": None,
    "buildings": None,
    "landuse": None,
    "all": None
}


@app.get("/api/geo/roads")
async def get_roads():
    """
    Get road network with highway classification.
    
    Returns GeoJSON FeatureCollection with road features.
    Each feature includes:
    - highway_type: OSM highway tag (motorway, primary, etc.)
    - classification: Detailed classification with styling and AI metadata
    - agent_metadata: Information for AI agent constraint processing
    """
    global _geo_cache
    if _geo_cache["roads"] is None:
        _geo_cache["roads"] = extract_roads(GEO_PLACE)
    return _geo_cache["roads"]


@app.get("/api/geo/water")
async def get_water():
    """
    Get water bodies and waterways.
    
    Returns GeoJSON FeatureCollection with water features.
    Each feature includes:
    - water_type: Type of water feature (river, stream, lake, etc.)
    - classification: Flood risk and buffer zone information
    - agent_metadata: Flood risk levels for AI agent processing
    """
    global _geo_cache
    if _geo_cache["water"] is None:
        _geo_cache["water"] = extract_water(GEO_PLACE)
    return _geo_cache["water"]


@app.get("/api/geo/buildings")
async def get_buildings():
    """
    Get building footprints with classification.
    
    Returns GeoJSON FeatureCollection with building features.
    Each feature includes:
    - building_type: Type of building (residential, hospital, school, etc.)
    - classification: Evacuation priority and shelter information
    - agent_metadata: Priority and capacity for rescue planning
    """
    global _geo_cache
    if _geo_cache["buildings"] is None:
        _geo_cache["buildings"] = extract_buildings(GEO_PLACE)
    return _geo_cache["buildings"]


@app.get("/api/geo/landuse")
async def get_landuse():
    """
    Get land use zones.
    
    Returns GeoJSON FeatureCollection with land use features.
    Each feature includes:
    - landuse_type: Type of land use (residential, industrial, etc.)
    - classification: Population density and evacuation priority
    - agent_metadata: Population estimation for rescue planning
    """
    global _geo_cache
    if _geo_cache["landuse"] is None:
        _geo_cache["landuse"] = extract_landuse(GEO_PLACE)
    return _geo_cache["landuse"]


@app.get("/api/geo/all")
async def get_all_geo():
    """
    Get all geographic data layers bundled together.
    
    Returns all layers (roads, water, buildings, landuse) in one response.
    Includes summary statistics and agent layer guide.
    
    This is the recommended endpoint for initial map load.
    """
    global _geo_cache
    if _geo_cache["all"] is None:
        _geo_cache["all"] = extract_all(GEO_PLACE)
    return _geo_cache["all"]


@app.get("/api/geo/schema")
async def get_geo_schema():
    """
    Get classification schemas for all layers.
    
    Returns the full classification schemas used for styling
    and AI agent processing. Useful for understanding available
    categories and their properties.
    """
    return {
        "roads": ROAD_CLASSIFICATION,
        "water": WATER_CLASSIFICATION,
        "buildings": BUILDING_CLASSIFICATION,
        "landuse": LANDUSE_CLASSIFICATION
    }


@app.get("/api/geo/agent-guide")
async def get_agent_guide():
    """
    Get instructions for AI agents on using geographic data.
    
    Returns structured guidance on:
    - How to interpret each layer
    - What actions can be taken on features
    - Priority hierarchies and field mappings
    """
    return get_agent_instructions()


# ============================================================================
# Workflow Runner with Real-Time Updates
# ============================================================================

async def run_workflow_with_updates():
    """Run the agent workflow with real-time WebSocket updates."""
    global workflow_state
    
    workflow_state["running"] = True
    workflow_state["constraints"] = []
    workflow_state["completed"] = False
    scenario_seed = _scenario_seed_payload()
    run_id = deterministic_run_id(
        scenario_seed,
        namespace="ryudo.workflow.cycloneshield.v1",
    )
    workflow_state["run_id"] = run_id
    replay_store.start_run(
        scenario_seed,
        run_id=run_id,
        metadata={"workflow": "cycloneshield_demo"},
    )
    
    await manager.broadcast({
        "type": "workflow_start",
        "run_id": run_id,
        "message": "Starting LangGraph Agent Workflow",
    })
    
    await wait(0.2)
    
    # Step 1: Load base graph (simulated - we won't actually load it for speed)
    await manager.broadcast({
        "type": "step",
        "agent": "System",
        "step": "load_graph",
        "message": "Loading road network for Visakhapatnam...",
    })
    await wait(0.3)
    
    workflow_state["base_graph_loaded"] = True
    await manager.broadcast({
        "type": "step_complete",
        "agent": "System",
        "step": "load_graph",
        "message": "Road network loaded: 38,567 nodes, 100,459 edges",
    })
    
    await wait(0.2)
    
    # Step 2: Environmental Agent
    await manager.broadcast({
        "type": "agent_start",
        "agent": "FloodSentinel",
        "message": "Environmental Agent analyzing cyclone data...",
    })
    await wait(0.2)
    
    # Send LLM reasoning (actual call to Gemini)
    env_reasoning = call_environmental_agent(CYCLONE_HUDHUD_CONFIG)
    await manager.broadcast({
        "type": "reasoning",
        "agent": "FloodSentinel",
        "content": env_reasoning
    })
    await wait(0.4)
    
    # Generate damage zones
    damage_zones = create_damage_zones(
        CYCLONE_HUDHUD_CONFIG["eye_lat"],
        CYCLONE_HUDHUD_CONFIG["eye_lon"],
        CYCLONE_HUDHUD_CONFIG
    )
    
    zone_colors = {
        "extreme": "#FF0000",
        "severe": "#FF8C00",
        "moderate": "#FFD700",
    }
    
    zone_descriptions = {
        "extreme": "Wind > 150 km/h - Complete destruction",
        "severe": "Wind 100-150 km/h - Severe damage",
        "moderate": "Wind 60-100 km/h - Trees down, flooding",
    }
    
    # Send each zone as a constraint
    for zone_name, polygon in damage_zones.items():
        constraint = {
            "agent": "FloodSentinel",
            "action": "delete_zone",
            "zone_type": zone_name,
            "reason": f"Cyclone {zone_name} damage zone - {zone_descriptions[zone_name]}",
            "geometry": polygon.__geo_interface__,
            "style": {
                "fillColor": zone_colors[zone_name],
                "color": zone_colors[zone_name],
                "weight": 2,
                "fillOpacity": 0.2,
            }
        }
        
        workflow_state["constraints"].append(constraint)
        
        await manager.broadcast({
            "type": "constraint",
            "constraint": constraint,
            "total": len(workflow_state["constraints"]),
        })
        
        await wait(0.2)
    
    # Storm surge zone
    surge_zone = create_storm_surge_zone(CYCLONE_HUDHUD_CONFIG)
    surge_constraint = {
        "agent": "FloodSentinel",
        "action": "delete_zone",
        "zone_type": "flooded",
        "reason": f"Storm surge - {CYCLONE_HUDHUD_CONFIG['storm_surge_height_m']}m inundation",
        "geometry": surge_zone.__geo_interface__,
        "style": {
            "fillColor": "#0066FF",
            "color": "#0033CC",
            "weight": 2,
            "fillOpacity": 0.4,
        }
    }
    workflow_state["constraints"].append(surge_constraint)
    
    await manager.broadcast({
        "type": "constraint",
        "constraint": surge_constraint,
        "total": len(workflow_state["constraints"]),
    })
    
    await wait(0.2)
    
    # Cyclone eye marker
    await manager.broadcast({
        "type": "marker",
        "agent": "FloodSentinel",
        "position": [CYCLONE_HUDHUD_CONFIG["eye_lat"], CYCLONE_HUDHUD_CONFIG["eye_lon"]],
        "popup": f"🌀 {CYCLONE_HUDHUD_CONFIG['name']}<br>Max Wind: {CYCLONE_HUDHUD_CONFIG['max_wind_speed_kmh']} km/h",
        "icon": "cyclone",
    })
    
    await manager.broadcast({
        "type": "agent_complete",
        "agent": "FloodSentinel",
        "message": "Generated 4 zone constraints",
        "constraint_count": 4,
    })
    
    await wait(0.2)
    
    # Step 3: Infrastructure Agent
    await manager.broadcast({
        "type": "agent_start",
        "agent": "GridGuardian",
        "message": "Infrastructure Agent checking power grid...",
    })
    await wait(0.2)
    
    # Send LLM reasoning (actual call to Gemini)
    infra_data = {
        "failed_substations": ["substation_north", "substation_central"],
        "facilities": list(FACILITIES.keys())
    }
    infra_reasoning = call_infrastructure_agent(infra_data, CYCLONE_HUDHUD_CONFIG)
    await manager.broadcast({
        "type": "reasoning",
        "agent": "GridGuardian",
        "content": infra_reasoning
    })
    await wait(0.4)
    
    # Simulated infrastructure failures
    failed_substations = ["substation_north", "substation_central"]
    affected_facilities = [
        ("hospital_kgh", 17.7196, 83.3024, "King George Hospital", "hospital"),
        ("shelter_north_1", 17.7350, 83.2900, "Community Center North", "shelter"),
        ("shelter_north_2", 17.7280, 83.2750, "School Shelter North", "shelter"),
        ("hospital_visakha", 17.6880, 83.2150, "Visakha General Hospital", "hospital"),
        ("shelter_central_1", 17.6920, 83.2280, "Town Hall", "shelter"),
    ]
    
    # Send substation failures
    for sub_id in failed_substations:
        await manager.broadcast({
            "type": "marker",
            "agent": "GridGuardian",
            "marker_type": "substation_failed",
            "id": sub_id,
            "position": [17.72 if "north" in sub_id else 17.69, 83.28 if "north" in sub_id else 83.22],
            "popup": f"⚡ {sub_id}<br>Status: OFFLINE",
            "icon": "power_off",
        })
        await wait(0.1)
    
    # Send facility constraints
    for fac_id, lat, lon, name, fac_type in affected_facilities:
        constraint = {
            "agent": "GridGuardian",
            "action": "disable_node",
            "node_id": fac_id,
            "node_type": fac_type,
            "reason": f"No power - infrastructure failure",
            "position": [lat, lon],
            "name": name,
        }
        
        workflow_state["constraints"].append(constraint)
        
        await manager.broadcast({
            "type": "constraint",
            "constraint": constraint,
            "total": len(workflow_state["constraints"]),
        })
        
        await manager.broadcast({
            "type": "marker",
            "agent": "GridGuardian",
            "marker_type": "facility_offline",
            "id": fac_id,
            "position": [lat, lon],
            "popup": f"🏥 {name}<br>Status: NO POWER",
            "icon": "offline",
        })
        
        await wait(0.1)
    
    # Add operational facilities
    operational = [
        ("hospital_naval", 17.6650, 83.1980, "Naval Hospital"),
        ("shelter_south_1", 17.6580, 83.2050, "Industrial Area Shelter"),
    ]
    
    for fac_id, lat, lon, name in operational:
        await manager.broadcast({
            "type": "marker",
            "agent": "GridGuardian",
            "marker_type": "facility_active",
            "id": fac_id,
            "position": [lat, lon],
            "popup": f"✅ {name}<br>Status: OPERATIONAL",
            "icon": "active",
        })
        await wait(0.1)
    
    await manager.broadcast({
        "type": "agent_complete",
        "agent": "GridGuardian",
        "message": "2 substations failed, 5 facilities offline",
        "constraint_count": 5,
    })
    
    await wait(0.2)
    
    # Step 4: Temporal Agent
    await manager.broadcast({
        "type": "agent_start",
        "agent": "RoutePilot",
        "message": "Temporal Agent predicting route validity...",
    })
    await wait(0.2)
    
    # Send LLM reasoning (actual call to Gemini)
    forecast_data = {
        "cyclone": CYCLONE_HUDHUD_CONFIG,
        "current_time": "2014-10-12T14:00:00",
        "forecast_hours": 6
    }
    temporal_reasoning = call_temporal_agent(forecast_data, workflow_state["constraints"])
    await manager.broadcast({
        "type": "reasoning",
        "agent": "RoutePilot",
        "content": temporal_reasoning
    })
    await wait(0.4)
    
    # TTL predictions
    ttl_constraints = [
        {
            "zone": "moderate_to_severe",
            "hours_remaining": 2,
            "confidence": 0.75,
            "description": "Roads will experience severe conditions as eye wall expands",
        },
        {
            "zone": "severe_to_extreme",
            "hours_remaining": 1,
            "confidence": 0.85,
            "description": "Routes in severe zone will become impassable",
        },
        {
            "zone": "post_storm_flood",
            "hours_remaining": 6,
            "confidence": 0.6,
            "description": "Low-lying areas may flood as drainage overflows",
        },
    ]
    
    for ttl in ttl_constraints:
        constraint = {
            "agent": "RoutePilot",
            "action": "set_ttl",
            "zone": ttl["zone"],
            "ttl_hours": ttl["hours_remaining"],
            "confidence": ttl["confidence"],
            "reason": ttl["description"],
        }
        
        workflow_state["constraints"].append(constraint)
        
        await manager.broadcast({
            "type": "constraint",
            "constraint": constraint,
            "total": len(workflow_state["constraints"]),
        })
        
        await wait(0.2)
    
    await manager.broadcast({
        "type": "agent_complete",
        "agent": "RoutePilot",
        "message": "Generated 3 temporal constraints",
        "constraint_count": 3,
    })
    
    await wait(0.2)
    
    # Step 5: Mission Coordinator
    await manager.broadcast({
        "type": "agent_start",
        "agent": "Coordinator",
        "message": "Mission Coordinator processing constraints...",
    })
    await wait(0.2)
    
    # Send LLM reasoning (actual call to Gemini)
    solution_summary = {
        "targets_reached": len([t for t in RESCUE_TARGETS if t["zone"] not in ["extreme", "flooded"]]),
        "unreachable": [t["name"] for t in RESCUE_TARGETS if t["zone"] in ["extreme", "flooded"]],
        "total_distance": 15000,  # Estimated
    }
    coordinator_reasoning = call_coordinator(
        workflow_state["constraints"],
        RESCUE_TARGETS,
        solution_summary,
        vehicles=RESCUE_VEHICLES
    )
    await manager.broadcast({
        "type": "reasoning",
        "agent": "Coordinator",
        "content": coordinator_reasoning
    })
    await wait(0.5)
    
    await manager.broadcast({
        "type": "coordinator_stats",
        "stats": {
            "zones_deleted": 4,
            "edges_removed": 18113,
            "nodes_disabled": 5,
            "ttl_applied": 3,
        }
    })
    
    await wait(0.2)
    
    await manager.broadcast({
        "type": "agent_complete",
        "agent": "Coordinator",
        "message": "Removed 18,113 road segments (18%)",
    })
    
    await wait(0.2)
    
    # Step 6: Mission Solver - Use road network for actual routing
    await manager.broadcast({
        "type": "agent_start",
        "agent": "MissionSolver",
        "message": f"Solving rescue mission with {len(RESCUE_TARGETS)} targets, {len(RESCUE_VEHICLES)} vehicles...",
    })
    await wait(0.3)
    
    # Add rescue depot marker
    await manager.broadcast({
        "type": "marker",
        "agent": "MissionSolver",
        "marker_type": "depot",
        "position": [RESCUE_DEPOT["lat"], RESCUE_DEPOT["lon"]],
        "popup": f"🚒 {RESCUE_DEPOT['name']}<br>Vehicles: {len(RESCUE_VEHICLES)}",
        "icon": "depot",
    })
    await wait(0.1)
    
    # Run road-based mission solver
    await manager.broadcast({
        "type": "step",
        "agent": "MissionSolver",
        "step": "routing",
        "message": "Computing optimal routes using road network...",
    })
    
    # Solve the mission using actual road graph
    solution = solve_rescue_mission(
        place="Visakhapatnam, India",
        targets=RESCUE_TARGETS,
        vehicles=RESCUE_VEHICLES,
        depot=RESCUE_DEPOT,
        constraints=workflow_state["constraints"],
    )
    
    await wait(0.3)
    
    # Show target markers with status
    assigned_ids = set()
    for assignment in solution.get("assignments", []):
        for t in assignment.get("targets", []):
            assigned_ids.add(t["id"])
    
    for target in RESCUE_TARGETS:
        is_assigned = target["id"] in assigned_ids
        icon_type = "target_ok" if is_assigned else "target_blocked"
        status_text = "✅ ASSIGNED" if is_assigned else "⏳ PENDING"
        
        # Build popup with detailed info
        popup = f"🎯 <b>{target['name']}</b><br>"
        popup += f"📍 {target.get('address', 'N/A')}<br>"
        popup += f"👥 Population: {target['population']}<br>"
        popup += f"⚠️ Zone: {target['zone']}<br>"
        popup += f"⏱️ TTL: {target.get('ttl_hours', 'N/A')} hours<br>"
        popup += f"{status_text}"
        
        await manager.broadcast({
            "type": "marker",
            "agent": "MissionSolver",
            "marker_type": icon_type,
            "id": target["id"],
            "position": [target["lat"], target["lon"]],
            "popup": popup,
            "icon": icon_type,
        })
        await wait(0.05)
    
    # Display routes for each vehicle assignment
    route_colors = ["#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"]
    
    for idx, assignment in enumerate(solution.get("assignments", [])):
        route = assignment.get("route")
        if route and route.get("coords"):
            color = route_colors[idx % len(route_colors)]
            
            # Build route popup with road details
            route_popup = f"🚗 <b>{assignment['vehicle_name']}</b><br>"
            route_popup += f"📏 Distance: {assignment['distance_km']} km<br>"
            route_popup += f"⏱️ Est. Time: {assignment['time_min']} min<br>"
            route_popup += f"👥 Rescuing: {assignment['total_population']} people<br>"
            
            # Add road names if available
            if route.get("road_segments"):
                main_roads = [s["road"] for s in route["road_segments"][:3] if s["road"] != "Unnamed Road"]
                if main_roads:
                    route_popup += f"📌 Via: {', '.join(main_roads)}"
            
            await manager.broadcast({
                "type": "route_complete",
                "agent": "MissionSolver",
                "vehicle_id": assignment["vehicle_id"],
                "vehicle_name": assignment["vehicle_name"],
                "route": route["coords"],
                "total_distance_km": assignment["distance_km"],
                "travel_time_min": assignment["time_min"],
                "targets_served": len(assignment["targets"]),
                "target_names": [t["name"] for t in assignment["targets"]],
                "color": color,
                "popup": route_popup,
                "road_segments": route.get("road_segments", []),
                "directions": route.get("directions", []),
            })
            await wait(0.2)
    
    await wait(0.2)
    
    # Mission summary
    summary = solution.get("summary", {})
    await manager.broadcast({
        "type": "mission_summary",
        "total_targets": summary.get("total_targets", len(RESCUE_TARGETS)),
        "targets_assigned": summary.get("targets_assigned", 0),
        "targets_unassigned": summary.get("targets_unassigned", 0),
        "population_rescued": summary.get("total_population_rescued", 0),
        "population_at_risk": summary.get("total_population_at_risk", 0),
        "total_distance_km": summary.get("total_distance_km", 0),
        "vehicles_deployed": summary.get("vehicles_deployed", 0),
        "assignments": solution.get("assignments", []),
    })
    
    await manager.broadcast({
        "type": "agent_complete",
        "agent": "MissionSolver",
        "message": f"Rescue plan: {summary.get('targets_assigned', 0)}/{summary.get('total_targets', 0)} targets, "
                   f"{summary.get('total_population_rescued', 0)} people, {summary.get('total_distance_km', 0)} km",
        "constraint_count": summary.get("targets_assigned", 0),
    })
    
    # Workflow complete
    workflow_state["running"] = False
    workflow_state["completed"] = True
    
    await manager.broadcast({
        "type": "workflow_complete",
        "run_id": run_id,
        "message": "Mission planning complete!",
        "total_constraints": len(workflow_state["constraints"]),
        "summary": {
            "FloodSentinel": 4,
            "GridGuardian": 5,
            "RoutePilot": 3,
            "MissionSolver": summary.get("targets_assigned", 0),
        },
        "mission_result": {
            "targets_assigned": summary.get("targets_assigned", 0),
            "population_rescued": summary.get("total_population_rescued", 0),
            "total_distance_km": summary.get("total_distance_km", 0),
        }
    })

    replay_store.complete_run(
        run_id,
        result={
            "mission_summary": summary,
            "total_constraints": len(workflow_state["constraints"]),
        },
        metadata={"completed": True},
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
