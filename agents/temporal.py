"""
Temporal Agent (Route Lifespan)
================================
Predicts the time-based validity of routes and assets.
Adds TTL (Time-To-Live) constraints to edges.

This agent forecasts when roads will become impassable due to
rising water levels, predicted weather changes, or traffic buildup.
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
import math

from agents.state import RyudoState, GraphConstraint


def predict_flood_progression(eye_lat: float, eye_lon: float, 
                              config: Dict[str, Any],
                              base_graph) -> List[Dict[str, Any]]:
    """
    Predict how flood zones will expand over time.
    
    Returns a list of predictions with affected road segments and TTL.
    """
    predictions = []
    
    # Simple model: moderate zone edges become severe in 2 hours
    # Severe zone edges become extreme in 1 hour
    moderate_radius = config.get("moderate_damage_radius_km", 50)
    severe_radius = config.get("severe_damage_radius_km", 25)
    
    # The "expansion front" - roads currently in moderate zone
    # but will be in severe zone as cyclone progresses
    predictions.append({
        "zone": "moderate_to_severe",
        "hours_remaining": 2,
        "affected_radius_start_km": severe_radius,
        "affected_radius_end_km": severe_radius + 10,
        "confidence": 0.75,
        "description": "Roads will experience severe conditions as eye wall expands"
    })
    
    predictions.append({
        "zone": "severe_to_extreme",
        "hours_remaining": 1,
        "affected_radius_start_km": config.get("extreme_damage_radius_km", 10),
        "affected_radius_end_km": severe_radius,
        "confidence": 0.85,
        "description": "Routes in severe zone will become impassable"
    })
    
    # Post-storm flooding prediction
    predictions.append({
        "zone": "post_storm_flood",
        "hours_remaining": 6,
        "affected_zones": ["low_elevation"],
        "confidence": 0.6,
        "description": "Low-lying areas may flood as drainage systems overflow"
    })
    
    return predictions


def temporal_agent(state: RyudoState) -> Dict[str, Any]:
    """
    Temporal Agent: Predicts the lifespan of routes and assets.
    
    Generic Action: Time_To_Live (TTL)
    - Forecasts when edges will become invalid
    - Adds time constraints to routing decisions
    
    This is the "Route Pilot" / "Drainage Pilot" in Ryudo.
    """
    constraints: List[GraphConstraint] = []
    viz_layers: List[Dict[str, Any]] = []
    
    temporal_data = state.get("temporal_data", {})
    env_data = state.get("environmental_data", {})
    
    # Get flood progression forecasts
    flood_forecast = temporal_data.get("flood_progression", [])
    
    # If cyclone data exists, generate predictions
    if env_data.get("type") == "cyclone" and not flood_forecast:
        config = env_data.get("config", {})
        eye_lat = env_data.get("eye_lat", config.get("eye_lat"))
        eye_lon = env_data.get("eye_lon", config.get("eye_lon"))
        
        if eye_lat and eye_lon:
            flood_forecast = predict_flood_progression(
                eye_lat, eye_lon, config, state.get("base_graph")
            )
    
    # Call LLM for temporal analysis (if API key available)
    from agents.llm_client import call_temporal_agent
    llm_analysis = call_temporal_agent(
        {"flood_forecast": flood_forecast, "cyclone": env_data.get("config", {})},
        state.get("constraints", [])
    )
    print(f"[RoutePilot] LLM Analysis:\n{llm_analysis[:200]}..." if len(llm_analysis) > 200 else f"[RoutePilot] LLM Analysis: {llm_analysis}")
    
    # Process each forecast
    for prediction in flood_forecast:
        ttl_hours = prediction.get("hours_remaining", 1)
        confidence = prediction.get("confidence", 0.5)
        zone = prediction.get("zone", "unknown")
        
        # Create TTL constraint
        constraints.append({
            "agent": "RoutePilot",
            "action": "set_ttl",
            "target": {
                "zone": zone,
                "radius_range": (
                    prediction.get("affected_radius_start_km", 0),
                    prediction.get("affected_radius_end_km", 100)
                ),
            },
            "reason": prediction.get("description", f"Route valid for {ttl_hours}h"),
            "metadata": {
                "ttl_hours": ttl_hours,
                "confidence": confidence,
                "predicted_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(hours=ttl_hours)).isoformat(),
            }
        })
        
        # Add visualization for time-limited zones
        if confidence > 0.6:  # Only show high-confidence predictions
            viz_layers.append({
                "type": "info",
                "agent": "RoutePilot",
                "name": f"TTL Warning: {zone}",
                "content": {
                    "zone": zone,
                    "hours_remaining": ttl_hours,
                    "confidence": f"{confidence * 100:.0f}%",
                    "description": prediction.get("description", ""),
                }
            })
    
    # Check for traffic-based TTLs (rush hour, evacuation congestion)
    traffic_data = temporal_data.get("traffic", {})
    if traffic_data.get("evacuation_active"):
        # Major arteries will be congested during evacuation
        constraints.append({
            "agent": "RoutePilot",
            "action": "set_ttl",
            "target": {
                "road_type": ["primary", "secondary"],
                "direction": "outbound",
            },
            "reason": "Evacuation traffic - expect 4x normal travel time",
            "metadata": {
                "ttl_hours": 8,
                "congestion_multiplier": 4.0,
                "confidence": 0.9,
            }
        })
    
    print(f"[RoutePilot] Generated {len(constraints)} temporal constraints")
    
    return {
        "constraints": constraints,
        "visualization_layers": viz_layers,
    }
