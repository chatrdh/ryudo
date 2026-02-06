# Ryudo: Technical Overview

> **A General-Purpose Engine for Dynamic Spatial Optimization**  
> Treating the physical world as a programmable, non-stationary graph where AI Agents define the "Rules of Engagement" in real-time.

---

## Executive Summary

Ryudo is smart mapping engine that solves **dynamic movement optimization problems** at scale. Unlike traditional mapping platforms (Google Maps, Uber) that treat the world as static, Ryudo treats the world as a **living system** where AI agents continuously modify the graph based on real-time conditions.

**Core Innovation**: A three-layer architecture where:
1. **State Agents** observe reality and constrain the graph (what is physically possible)
2. **Living Graph** maintains both static topology and dynamic properties
3. **Mission Coordinator** solves optimization on the constrained graph (what is optimal)

**Current Implementation**: `CycloneShield` — a disaster response system demonstrating the engine with Cyclone Hudhud (2014) hitting Visakhapatnam, India.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     External Signals                             │
│  (Satellite Data, IoT Sensors, Weather APIs)                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│               Module A: State Agents (LLM-Powered)               │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐    │
│  │  FloodSentinel  │ │  GridGuardian   │ │   RoutePilot    │    │
│  │ (Environmental) │ │ (Infrastructure)│ │   (Temporal)    │    │
│  └────────┬────────┘ └────────┬────────┘ └────────┬────────┘    │
└───────────┼───────────────────┼───────────────────┼─────────────┘
            │ Zone Deletion     │ Edge Severing     │ TTL
            ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Living Graph                                │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │    Base Layer       │───▶│       Overlay Layer             │ │
│  │  (OSM Road Network) │    │  (Dynamic Weights/Constraints)  │ │
│  │  38,567 nodes       │    │  Risk scores, availability,     │ │
│  │  100,459 edges      │    │  TTL, traversal costs           │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
┌──────────────────────────────────▼──────────────────────────────┐
│              Module B: Mission Coordinator                       │
│           (CVRPTW Solver + OSMnx Routing Engine)                 │
│                                                                  │
│  Input:  {targets, vehicles, depot, constraints}                 │
│  Output: {routes, assignments, schedules}                        │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
                            Action Plan Output
                     (Rescue manifest, deployment schedule)
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Backend** | FastAPI + Uvicorn | Async HTTP/WebSocket server |
| **Graph Engine** | NetworkX + OSMnx | Road network loading and manipulation |
| **Geospatial** | Shapely + OSMnx | Polygon operations, GeoJSON |
| **LLM Reasoning** | Google Gemini (gemini-2.5-flash) | Agent reasoning and analysis |
| **Routing** | OSMnx shortest_path | Real road network routing |
| **Frontend** | React + Vite + Leaflet | Real-time map visualization |
| **Real-time** | WebSocket | Live constraint/route streaming |

---

## Core Components

### 1. State Agents (Module A)

State Agents are LLM-powered observers that **do not solve problems**. Their only job is to observe reality and emit constraints that modify the graph.

#### FloodSentinel (Environmental Agent)
**File**: `agents/environmental.py`  
**LLM Function**: `call_environmental_agent()` in `agents/llm_client.py`

- **Input**: Cyclone data (eye position, wind speed, pressure, surge height)
- **Action**: `Zone_Deletion` — marks polygonal regions as impassable
- **Physics Model**: Holland Wind Profile Model for damage zone calculation
- **Output Constraints**:
  - Extreme damage zone (wind > 150 km/h)
  - Severe damage zone (wind 100-150 km/h)
  - Moderate damage zone (wind 60-100 km/h)
  - Storm surge inundation zone

```python
# Example constraint emitted by FloodSentinel
{
    "agent": "FloodSentinel",
    "action": "delete_zone",
    "target": {"polygon": <Shapely Polygon>, "zone_type": "extreme"},
    "reason": "Wind > 150 km/h - Complete destruction"
}
```

#### GridGuardian (Infrastructure Agent)
**File**: `agents/infrastructure.py`  
**LLM Function**: `call_infrastructure_agent()`

- **Input**: Substation status, facility locations, dependency graph
- **Action**: `Edge_Severing` + `Dependency_Chaining`
- **Models**: Infrastructure dependency cascade (substation → hospital chain)
- **Output Constraints**:
  - Disabled nodes (powerless hospitals/shelters)
  - Severed edges (roads to offline facilities)

```python
# Dependency graph structure
SUBSTATIONS = {
    "substation_north": {
        "dependents": ["hospital_kgh", "shelter_north_1", "shelter_north_2"]
    }
}
```

#### RoutePilot (Temporal Agent)
**File**: `agents/temporal.py`  
**LLM Function**: `call_temporal_agent()`

- **Input**: Current constraints, forecast data, time horizon
- **Action**: `Time_To_Live` — assigns validity windows to routes
- **Output Constraints**:
  - TTL on moderate zones (2 hours before escalation)
  - Post-storm flood predictions (6 hours)

---

### 2. Living Graph

The Living Graph has two layers:

**Base Layer** (Static):
- Loaded via OSMnx from OpenStreetMap
- 38,567 nodes, 100,459 edges for Visakhapatnam
- Contains road geometry, types, and base weights

**Overlay Layer** (Dynamic):
- Populated by State Agent constraints
- Properties: availability (boolean), risk weights, TTL timestamps
- Applied via `apply_constraints()` in `services/routing_service.py`

---

### 3. Geographic Data Service
**File**: `services/geo_data.py`

Extracts and classifies OSM data for agent consumption:

| Layer | Features | Agent Use |
|-------|----------|-----------|
| **Roads** | Highway classification, capacity, flood passability | Routing, evacuation priority |
| **Water** | Rivers, coastline, flood risk buffers | Flood zone definition |
| **Buildings** | Hospitals, schools, shelters with capacity | Rescue target identification |
| **Land Use** | Residential density, industrial zones | Population estimation |

Each feature includes `agent_metadata` for LLM consumption:
```python
ROAD_CLASSIFICATION = {
    "motorway": {
        "level": 1,
        "capacity": "very_high",
        "passable_during_flood": True,
        "evacuation_priority": "primary"
    }
}
```

---

### 4. Mission Solver
**File**: `services/mission_solver.py`

Solves Capacitated Vehicle Routing Problem with Time Windows (CVRPTW):

**Input**:
- `RESCUE_DEPOT`: NDRF Emergency Response Center
- `RESCUE_VEHICLES`: 6 vehicles with varying capacities (8-25 people), medical equipment, water fording capability
- `RESCUE_TARGETS`: 20 detailed targets with population, medical needs, mobility issues, TTL

**Solver Logic**:
1. Load constrained road graph
2. Calculate priority scores: `priority = base + (10 * medical) + (5 * mobility / population)`
3. Match vehicles to targets by capability (medical, water fording)
4. Compute actual road routes via OSMnx `shortest_path`
5. Output assignment manifest with ETAs

**Target Example**:
```python
{
    "id": "T03",
    "name": "Coastal Dialysis Patient",
    "lat": 17.7420, "lon": 83.3650,
    "zone": "extreme",
    "population": 3,
    "medical_needs": ["dialysis", "oxygen_dependent"],
    "ttl_hours": 0.5,
    "priority_score": 98
}
```

---

### 5. LLM Client
**File**: `agents/llm_client.py`

**Primary Model**: `gemini-2.5-flash` (Google AI Studio)  
**Fallback**: OpenRouter API with Claude/GPT-4o-mini

Each agent has a specialized prompt with:
- System role definition
- Domain-specific instructions
- Input data schema
- Expected output format

**Logging**: All LLM reasoning is logged to `output/llm_reasoning.md` with timestamps, prompts, and responses.

---

### 6. Real-Time Server
**File**: `server.py`

FastAPI server with WebSocket broadcasting:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve React frontend |
| `/ws` | WebSocket | Real-time constraint/route streaming |
| `/api/start` | POST | Trigger workflow execution |
| `/api/geo/all` | GET | Get all geographic layers |
| `/api/config` | GET | Get cyclone configuration |

**WebSocket Message Types**:
- `workflow_start` — Begin simulation
- `agent_start/complete` — Agent lifecycle
- `reasoning` — LLM analysis text
- `constraint` — New graph constraint
- `route` — Computed rescue route
- `marker` — Map marker (cyclone eye, depot, facilities)

---

## LangGraph Workflow
**File**: `agents/graph.py`

The workflow is structured as a **parallel state machine**:

```
        START
          │
          ▼
      load_graph
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
   Env  Infra  Temp     ← Run in parallel
    │     │     │
    └─────┴─────┘
          │
          ▼
    coordinator
          │
          ▼
         END
```

State is typed via `RyudoState` (TypedDict):
```python
class RyudoState(TypedDict):
    base_graph: nx.MultiDiGraph
    constraints: List[GraphConstraint]
    environmental_data: Dict
    infrastructure_data: Dict
    temporal_data: Dict
    mission: Dict
    modified_graph: nx.MultiDiGraph
    solution: Dict
    visualization_layers: List[Dict]
```

---

## Frontend Architecture
**Directory**: `ui/`

React + Vite application with:
- **Leaflet** for interactive maps
- **WebSocket hook** for real-time updates
- **Components**:
  - `MapContainer` — Main map with layers
  - `ReasoningPanel` — Agent LLM outputs
  - `Sidebar` — Workflow control
  - `TimelineStep` — Progress visualization

---

## Data Flow Example: Cyclone Hudhud

```
1. User clicks "Start Simulation"
   └─▶ WebSocket sends {"action": "start"}

2. Server loads OSM graph for Visakhapatnam
   └─▶ 38,567 nodes, 100,459 edges

3. FloodSentinel analyzes cyclone data
   └─▶ LLM reasons about wind profile
   └─▶ Emits 4 zone constraints (extreme/severe/moderate/surge)
   └─▶ 18,113 road edges marked impassable

4. GridGuardian checks infrastructure
   └─▶ 2 substations in extreme zone → offline
   └─▶ 5 dependent facilities lose power
   └─▶ Emits node disable constraints

5. RoutePilot predicts temporal windows
   └─▶ Moderate zone → severe in 2 hours
   └─▶ Emits TTL constraints

6. Mission Coordinator solves CVRPTW
   └─▶ Matches 6 vehicles to 20 targets
   └─▶ Computes road-based routes
   └─▶ Outputs rescue manifest with ETAs

7. Frontend renders in real-time
   └─▶ Damage zones, routes, markers animate on map
   └─▶ Agent reasoning displayed in side panel
```

---

## Future Extensibility

The architecture is **domain-agnostic**. New "skins" require:

1. **New State Agents** that emit standard constraints:
   - `delete_zone` (polygon)
   - `disable_node` (node_id)
   - `reweight_edge` (edge_id, new_weight)
   - `set_ttl` (entity, expiry)

2. **New Mission Definition**:
   - Targets with lat/lon and priority
   - Vehicles with capacity and capabilities
   - Depot location

| Skin | Context | Agents | Goal |
|------|---------|--------|------|
| **CycloneShield** | Cyclone hitting coast | FloodSentinel, GridGuardian | Rescue 50 families |
| **Agri-Mind** | Harvest + approaching rain | CropRipenessAgent, MachineryHealthAgent | Harvest 500 acres |
| **UrbanFleet** | City logistics | TrafficFlow, DeliveryDemand | Optimize 100 deliveries |

---

## Running the System

**Backend**:
```bash
source .venv/bin/activate
python server.py
```

**Frontend** (hot reload):
```bash
cd ui
npm run dev
```

Access at `http://localhost:5173` (dev) or `http://localhost:8000` (production build)

---

## Key Files Reference

| File | Lines | Purpose |
|------|-------|---------|
| `server.py` | 928 | FastAPI server, WebSocket, workflow orchestration |
| `agents/llm_client.py` | 689 | Gemini/OpenRouter client, agent prompts |
| `agents/graph.py` | 164 | LangGraph workflow definition |
| `agents/environmental.py` | 192 | FloodSentinel agent |
| `agents/infrastructure.py` | 252 | GridGuardian agent |
| `agents/temporal.py` | 167 | RoutePilot agent |
| `services/mission_solver.py` | 822 | CVRPTW solver, target/vehicle data |
| `services/geo_data.py` | 1053 | OSM data extraction, classification |
| `services/routing_service.py` | — | Graph constraint application |

---

## Why This Matters

Traditional routing systems fail in dynamic environments because they:
- Cannot reason about cascading failures
- Treat all roads as equally available
- Ignore temporal validity windows
- Lack domain knowledge integration

Ryudo solves this by:
1. **Separating observation from optimization** — Agents constrain, solver optimizes
2. **Using LLMs for domain reasoning** — Physics models + natural language analysis
3. **Maintaining a living graph** — Real-time constraint application
4. **Streaming results** — WebSocket enables interactive visualization

This architecture scales to any domain where movement must be optimized under dynamic, uncertain conditions.
