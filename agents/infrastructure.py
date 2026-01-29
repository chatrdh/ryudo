"""
Infrastructure Agent (Grid Guardian)
=====================================
Monitors infrastructure dependencies and creates edge/node constraints
when critical systems fail.

This agent models cascading failures - when a substation goes down,
all dependent facilities (hospitals, shelters) become unavailable.
"""

from typing import Dict, Any, List
import networkx as nx

from agents.state import RyudoState, GraphConstraint


# Sample infrastructure dependency graph
# In production, this would come from a real database
INFRASTRUCTURE_DEPENDENCIES = {
    "substation_north": {
        "type": "power",
        "lat": 17.72,
        "lon": 83.28,
        "dependents": ["hospital_kgh", "shelter_north_1", "shelter_north_2"],
    },
    "substation_central": {
        "type": "power",
        "lat": 17.69,
        "lon": 83.22,
        "dependents": ["hospital_visakha", "shelter_central_1"],
    },
    "substation_south": {
        "type": "power",
        "lat": 17.65,
        "lon": 83.20,
        "dependents": ["hospital_naval", "shelter_south_1"],
    },
}

FACILITIES = {
    "hospital_kgh": {
        "name": "King George Hospital",
        "type": "hospital",
        "lat": 17.7196,
        "lon": 83.3024,
        "capacity": 500,
    },
    "hospital_visakha": {
        "name": "Visakha General Hospital",
        "type": "hospital",
        "lat": 17.6880,
        "lon": 83.2150,
        "capacity": 300,
    },
    "hospital_naval": {
        "name": "Naval Hospital",
        "type": "hospital",
        "lat": 17.6650,
        "lon": 83.1980,
        "capacity": 200,
    },
    "shelter_north_1": {
        "name": "Community Center North",
        "type": "shelter",
        "lat": 17.7350,
        "lon": 83.2900,
        "capacity": 150,
    },
    "shelter_north_2": {
        "name": "School Shelter North",
        "type": "shelter",
        "lat": 17.7280,
        "lon": 83.2750,
        "capacity": 100,
    },
    "shelter_central_1": {
        "name": "Town Hall",
        "type": "shelter",
        "lat": 17.6920,
        "lon": 83.2280,
        "capacity": 200,
    },
    "shelter_south_1": {
        "name": "Industrial Area Shelter",
        "type": "shelter",
        "lat": 17.6580,
        "lon": 83.2050,
        "capacity": 180,
    },
}


def get_dependent_facilities(substation_id: str) -> List[Dict[str, Any]]:
    """Get all facilities dependent on a substation."""
    substation = INFRASTRUCTURE_DEPENDENCIES.get(substation_id)
    if not substation:
        return []
    
    dependents = []
    for facility_id in substation.get("dependents", []):
        if facility_id in FACILITIES:
            facility = FACILITIES[facility_id].copy()
            facility["id"] = facility_id
            dependents.append(facility)
    
    return dependents


def check_wind_damage_to_infrastructure(eye_lat: float, eye_lon: float, 
                                        extreme_radius_km: float) -> List[str]:
    """
    Determine which substations are within the extreme damage zone.
    Returns list of failed substation IDs.
    """
    import math
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))
    
    failed = []
    for sub_id, sub_data in INFRASTRUCTURE_DEPENDENCIES.items():
        dist = haversine(eye_lat, eye_lon, sub_data["lat"], sub_data["lon"])
        if dist <= extreme_radius_km * 1.5:  # Power lines fail in severe zone too
            failed.append(sub_id)
    
    return failed


def infrastructure_agent(state: RyudoState) -> Dict[str, Any]:
    """
    Infrastructure Agent: Maps system dependencies and cascading failures.
    
    Generic Actions:
    - Edge_Severing: Remove connections to failed infrastructure
    - Dependency_Chaining: Cascade failures through the dependency graph
    
    This is the "Grid Guardian" in the CycloneShield implementation.
    """
    constraints: List[GraphConstraint] = []
    viz_layers: List[Dict[str, Any]] = []
    
    infra_data = state.get("infrastructure_data", {})
    env_data = state.get("environmental_data", {})
    
    # Determine failed substations
    failed_substations = infra_data.get("failed_substations", [])
    
    # If cyclone data exists, calculate wind damage to infrastructure
    if env_data.get("type") == "cyclone":
        config = env_data.get("config", {})
        eye_lat = env_data.get("eye_lat", config.get("eye_lat"))
        eye_lon = env_data.get("eye_lon", config.get("eye_lon"))
        extreme_radius = config.get("extreme_damage_radius_km", 10)
        
        if eye_lat and eye_lon:
            wind_damaged = check_wind_damage_to_infrastructure(
                eye_lat, eye_lon, extreme_radius
            )
            failed_substations = list(set(failed_substations + wind_damaged))
    
    # Call LLM for infrastructure analysis (if API key available)
    from agents.llm_client import call_infrastructure_agent
    llm_analysis = call_infrastructure_agent(
        {"failed_substations": failed_substations, "facilities": list(FACILITIES.keys())},
        env_data.get("config", {})
    )
    print(f"[GridGuardian] LLM Analysis:\n{llm_analysis[:200]}..." if len(llm_analysis) > 200 else f"[GridGuardian] LLM Analysis: {llm_analysis}")
    
    # Process cascading failures
    affected_facilities = []
    
    for substation_id in failed_substations:
        substation = INFRASTRUCTURE_DEPENDENCIES.get(substation_id)
        if not substation:
            continue
        
        # Add substation failure visualization
        viz_layers.append({
            "type": "marker",
            "agent": "GridGuardian",
            "name": f"Failed Substation: {substation_id}",
            "position": (substation["lat"], substation["lon"]),
            "popup": f"⚡ {substation_id}<br>Status: OFFLINE<br>Type: Power Substation",
            "icon": {"color": "black", "icon": "bolt", "prefix": "fa"}
        })
        
        # Get dependent facilities
        dependents = get_dependent_facilities(substation_id)
        
        for facility in dependents:
            # Create constraint to disable this node
            constraints.append({
                "agent": "GridGuardian",
                "action": "disable_node",
                "target": {
                    "node_id": facility["id"],
                    "lat": facility["lat"],
                    "lon": facility["lon"],
                    "type": facility["type"],
                },
                "reason": f"No power - depends on failed {substation_id}",
                "metadata": {
                    "facility_name": facility["name"],
                    "facility_type": facility["type"],
                    "capacity_lost": facility["capacity"],
                    "upstream_failure": substation_id,
                }
            })
            
            affected_facilities.append(facility)
            
            # Add visualization for affected facility
            icon_color = "gray" if facility["type"] == "hospital" else "lightgray"
            viz_layers.append({
                "type": "marker",
                "agent": "GridGuardian",
                "name": f"Offline: {facility['name']}",
                "position": (facility["lat"], facility["lon"]),
                "popup": f"🏥 {facility['name']}<br>"
                         f"Status: NO POWER<br>"
                         f"Capacity: {facility['capacity']}<br>"
                         f"Depends on: {substation_id}",
                "icon": {"color": icon_color, "icon": "hospital" if facility["type"] == "hospital" else "home", "prefix": "fa"}
            })
    
    # Also add operational facilities as green markers
    all_affected_ids = [f["id"] for f in affected_facilities]
    for fac_id, facility in FACILITIES.items():
        if fac_id not in all_affected_ids:
            viz_layers.append({
                "type": "marker",
                "agent": "GridGuardian",
                "name": f"Active: {facility['name']}",
                "position": (facility["lat"], facility["lon"]),
                "popup": f"✅ {facility['name']}<br>"
                         f"Status: OPERATIONAL<br>"
                         f"Capacity: {facility['capacity']}",
                "icon": {"color": "green", "icon": "hospital" if facility["type"] == "hospital" else "home", "prefix": "fa"}
            })
    
    print(f"[GridGuardian] {len(failed_substations)} substations failed, "
          f"{len(affected_facilities)} facilities offline")
    
    return {
        "constraints": constraints,
        "visualization_layers": viz_layers,
    }
