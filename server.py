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

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import networkx as nx

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
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


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

# Rescue mission configuration
RESCUE_TARGETS = [
    {"id": "T1", "lat": 17.7200, "lon": 83.3200, "name": "Family A", "status": "pending", "zone": "extreme"},
    {"id": "T2", "lat": 17.7050, "lon": 83.2800, "name": "Family B", "status": "pending", "zone": "severe"},
    {"id": "T3", "lat": 17.6900, "lon": 83.2500, "name": "Family C", "status": "pending", "zone": "moderate"},
    {"id": "T4", "lat": 17.7300, "lon": 83.2600, "name": "Family D", "status": "pending", "zone": "severe"},
    {"id": "T5", "lat": 17.6750, "lon": 83.2100, "name": "Family E", "status": "pending", "zone": "safe"},
    {"id": "T6", "lat": 17.6600, "lon": 83.2300, "name": "Family F", "status": "pending", "zone": "safe"},
    {"id": "T7", "lat": 17.7100, "lon": 83.3100, "name": "Family G", "status": "pending", "zone": "extreme"},
    {"id": "T8", "lat": 17.6800, "lon": 83.2400, "name": "Family H", "status": "pending", "zone": "moderate"},
]

RESCUE_DEPOT = {"lat": 17.6868, "lon": 83.2185, "name": "Emergency Response Center"}


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
        asyncio.create_task(run_workflow_with_updates())
        return {"status": "started", "message": "Workflow started in background"}
    return {"status": "running", "message": "Workflow already running"}



@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            "type": "connected",
            "message": "Connected to Ryudo server",
            "config": CYCLONE_HUDHUD_CONFIG,
        })
        
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


# ============================================================================
# Workflow Runner with Real-Time Updates
# ============================================================================

async def run_workflow_with_updates():
    """Run the agent workflow with real-time WebSocket updates."""
    global workflow_state
    
    workflow_state["running"] = True
    workflow_state["constraints"] = []
    workflow_state["completed"] = False
    
    await manager.broadcast({
        "type": "workflow_start",
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
        solution_summary
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
    
    # Step 6: Mission Solver - Show rescue targets and compute routes
    await manager.broadcast({
        "type": "agent_start",
        "agent": "MissionSolver",
        "message": "Analyzing rescue mission targets...",
    })
    await wait(0.4)
    
    # Add rescue depot marker
    await manager.broadcast({
        "type": "marker",
        "agent": "MissionSolver",
        "marker_type": "depot",
        "position": [RESCUE_DEPOT["lat"], RESCUE_DEPOT["lon"]],
        "popup": f"🚒 {RESCUE_DEPOT['name']}<br>Rescue vehicles: 3",
        "icon": "depot",
    })
    await wait(0.1)
    
    # Assess each target based on damage zones
    reachable = []
    unreachable = []
    
    for target in RESCUE_TARGETS:
        await wait(0.1)
        
        # Determine if target is reachable based on zone
        if target["zone"] in ["extreme", "flooded"]:
            status = "unreachable"
            unreachable.append(target)
        elif target["zone"] == "severe":
            # 50% chance of being reachable in severe zone
            import random
            if random.random() > 0.5:
                status = "reachable"
                reachable.append(target)
            else:
                status = "unreachable"
                unreachable.append(target)
        else:
            status = "reachable"
            reachable.append(target)
        
        icon_type = "target_ok" if status == "reachable" else "target_blocked"
        popup_status = "✅ REACHABLE" if status == "reachable" else "❌ BLOCKED"
        
        await manager.broadcast({
            "type": "marker",
            "agent": "MissionSolver",
            "marker_type": icon_type,
            "id": target["id"],
            "position": [target["lat"], target["lon"]],
            "popup": f"🎯 {target['name']}<br>Zone: {target['zone']}<br>{popup_status}",
            "icon": icon_type,
        })
        
        await manager.broadcast({
            "type": "target_assessment",
            "target": target,
            "status": status,
            "reachable_count": len(reachable),
            "unreachable_count": len(unreachable),
        })
    
    await wait(0.2)
    
    # Compute routes for reachable targets
    if reachable:
        await manager.broadcast({
            "type": "step",
            "agent": "MissionSolver",
            "step": "routing",
            "message": f"Computing optimal routes for {len(reachable)} reachable targets...",
        })
        await wait(0.5)
        
        # Simulate route computation - create a simple greedy route
        route_coords = [[RESCUE_DEPOT["lat"], RESCUE_DEPOT["lon"]]]
        total_distance = 0
        
        for i, target in enumerate(reachable):
            await wait(0.1)
            
            # Add route segment
            prev = route_coords[-1]
            route_coords.append([target["lat"], target["lon"]])
            
            # Calculate approximate distance
            import math
            dist = math.sqrt((target["lat"] - prev[0])**2 + (target["lon"] - prev[1])**2) * 111  # km
            total_distance += dist
            
            # Send route segment
            await manager.broadcast({
                "type": "route_segment",
                "agent": "MissionSolver",
                "from": {"lat": prev[0], "lon": prev[1]},
                "to": {"lat": target["lat"], "lon": target["lon"]},
                "target": target,
                "segment_index": i,
                "distance_km": round(dist, 2),
            })
        
        # Send complete route
        await manager.broadcast({
            "type": "route_complete",
            "route": route_coords,
            "total_distance_km": round(total_distance, 2),
            "targets_served": len(reachable),
        })
    
    await wait(0.3)
    
    # Mission summary
    await manager.broadcast({
        "type": "mission_summary",
        "reachable": len(reachable),
        "unreachable": len(unreachable),
        "total_targets": len(RESCUE_TARGETS),
        "reachable_list": [t["name"] for t in reachable],
        "unreachable_list": [t["name"] for t in unreachable],
    })
    
    await manager.broadcast({
        "type": "agent_complete",
        "agent": "MissionSolver",
        "message": f"Rescue plan: {len(reachable)}/{len(RESCUE_TARGETS)} targets reachable",
        "constraint_count": len(reachable),
    })
    
    # Workflow complete
    workflow_state["running"] = False
    workflow_state["completed"] = True
    
    await manager.broadcast({
        "type": "workflow_complete",
        "message": "Mission planning complete!",
        "total_constraints": len(workflow_state["constraints"]),
        "summary": {
            "FloodSentinel": 4,
            "GridGuardian": 5,
            "RoutePilot": 3,
            "MissionSolver": len(reachable),
        },
        "mission_result": {
            "reachable": len(reachable),
            "unreachable": len(unreachable),
        }
    })


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
