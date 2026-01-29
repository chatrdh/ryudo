"""
Ryudo LangGraph Agents Demo
============================
Demonstrates the multi-agent system running on Cyclone Hudhud scenario.

This script:
1. Loads the road network for Visakhapatnam
2. Configures Cyclone Hudhud parameters
3. Runs the LangGraph workflow with all 3 agents
4. Visualizes the results on an interactive map

Usage:
    python demo_langgraph_agents.py
"""

import os
import sys

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.graph import run_ryudo_workflow
from visualization.map_renderer import render_agent_map


# ============================================================================
# CYCLONE HUDHUD CONFIGURATION (Same as Phase 2)
# ============================================================================

CYCLONE_HUDHUD_CONFIG = {
    "name": "Cyclone Hudhud",
    "date": "2014-10-12",
    
    # Landfall position (adjusted for visualization)
    "eye_lat": 17.68,
    "eye_lon": 83.35,
    
    # Meteorological parameters
    "max_wind_speed_kmh": 185,
    "central_pressure_mb": 950,
    "ambient_pressure_mb": 1013,
    "radius_max_wind_km": 25,
    
    # Impact zones (km from eye)
    "extreme_damage_radius_km": 10,
    "severe_damage_radius_km": 25,
    "moderate_damage_radius_km": 50,
    
    # Storm surge
    "storm_surge_height_m": 2.5,
    "surge_inland_distance_km": 2,
}

# Sample rescue targets (families needing evacuation)
RESCUE_TARGETS = [
    {"lat": 17.7200, "lon": 83.3200, "name": "Family A - Near Beach"},
    {"lat": 17.7050, "lon": 83.2800, "name": "Family B - Low Ground"},
    {"lat": 17.6900, "lon": 83.2500, "name": "Family C - Near River"},
    {"lat": 17.7300, "lon": 83.2600, "name": "Family D - Coastal Area"},
    {"lat": 17.6750, "lon": 83.2100, "name": "Family E - Industrial Zone"},
    {"lat": 17.6600, "lon": 83.2300, "name": "Family F - Port Area"},
    {"lat": 17.7100, "lon": 83.3100, "name": "Family G - Flood Zone"},
    {"lat": 17.6800, "lon": 83.2400, "name": "Family H - Central City"},
]

# Rescue depot location (fire station / emergency center)
RESCUE_DEPOT = {
    "lat": 17.6868,
    "lon": 83.2185,
    "name": "Emergency Response Center"
}


def main():
    """Run the LangGraph agent demo."""
    print("=" * 70)
    print("🧠 RYUDO: LangGraph Multi-Agent Disaster Response System")
    print("=" * 70)
    print()
    
    print("📋 Scenario: Cyclone Hudhud hitting Visakhapatnam")
    print(f"   🌀 Max Wind: {CYCLONE_HUDHUD_CONFIG['max_wind_speed_kmh']} km/h")
    print(f"   📍 Eye Position: {CYCLONE_HUDHUD_CONFIG['eye_lat']}°N, {CYCLONE_HUDHUD_CONFIG['eye_lon']}°E")
    print(f"   🎯 Rescue Targets: {len(RESCUE_TARGETS)} families")
    print()
    
    # Configure agent inputs
    environmental_data = {
        "type": "cyclone",
        "eye_lat": CYCLONE_HUDHUD_CONFIG["eye_lat"],
        "eye_lon": CYCLONE_HUDHUD_CONFIG["eye_lon"],
        "config": CYCLONE_HUDHUD_CONFIG,
    }
    
    infrastructure_data = {
        # Let the agent determine failures based on cyclone position
        "failed_substations": [],
    }
    
    temporal_data = {
        # Agent will generate predictions based on cyclone
        "flood_progression": [],
    }
    
    mission = {
        "type": "rescue",
        "place": "Visakhapatnam, India",
        "targets": RESCUE_TARGETS,
        "depot": RESCUE_DEPOT,
    }
    
    # Run the workflow
    print("🚀 Starting LangGraph Workflow...")
    print("-" * 70)
    
    result = run_ryudo_workflow(
        environmental_data=environmental_data,
        infrastructure_data=infrastructure_data,
        temporal_data=temporal_data,
        mission=mission,
    )
    
    print("-" * 70)
    
    # Print results summary
    print()
    print("📊 WORKFLOW RESULTS:")
    print()
    
    constraints = result.get("constraints", [])
    print(f"   Total Constraints Generated: {len(constraints)}")
    
    # Group by agent
    by_agent = {}
    for c in constraints:
        agent = c.get("agent", "Unknown")
        by_agent[agent] = by_agent.get(agent, 0) + 1
    
    for agent, count in by_agent.items():
        print(f"   └─ {agent}: {count} constraints")
    
    print()
    
    # Coordinator stats
    stats = result.get("coordinator_stats", {})
    if stats:
        print(f"   Graph Modifications:")
        print(f"   ├─ Zones deleted: {stats.get('zones_deleted', 0)}")
        print(f"   ├─ Edges removed: {stats.get('edges_removed', 0)}")
        print(f"   └─ Nodes disabled: {stats.get('nodes_disabled', 0)}")
    
    print()
    
    # Solution
    solution = result.get("solution")
    if solution:
        print(f"   🚒 Rescue Solution:")
        print(f"   ├─ Targets Reachable: {solution.get('targets_reached', 0)}")
        print(f"   ├─ Targets Unreachable: {len(solution.get('unreachable', []))}")
        print(f"   └─ Total Distance: {solution.get('total_distance', 0):.0f} meters")
        
        if solution.get("unreachable"):
            print()
            print("   ⚠️ Unreachable Targets:")
            for target in solution["unreachable"]:
                print(f"      └─ {target.get('name', 'Unknown')} at ({target['lat']}, {target['lon']})")
    
    print()
    
    # Render map
    print("🗺️ Generating Interactive Map...")
    output_path = "output/agent_workflow_demo.html"
    
    render_agent_map(
        result,
        output_path=output_path,
        title="Ryudo LangGraph Agents - Cyclone Hudhud",
        show_roads=True,
    )
    
    print()
    print("=" * 70)
    print(f"✅ Demo Complete! Open {output_path} in a browser.")
    print("=" * 70)
    
    return result


if __name__ == "__main__":
    result = main()
