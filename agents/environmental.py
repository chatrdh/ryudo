"""
Environmental Agent (Flood Sentinel)
=====================================
Observes weather/satellite data and creates zone deletion constraints
for flooded or damaged areas.

This is a "State Agent" that modifies the Living Graph by marking
regions as impassable based on environmental conditions.
"""

from typing import Dict, Any, List
from shapely.geometry import Point, Polygon
import math

from agents.state import RyudoState, GraphConstraint


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371  # Earth's radius in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def create_damage_zones(eye_lat: float, eye_lon: float, config: Dict[str, Any]) -> Dict[str, Polygon]:
    """
    Create circular damage zones around cyclone eye.
    
    Returns a dict of zone_name -> shapely Polygon
    """
    center = Point(eye_lon, eye_lat)
    km_to_deg = 1 / 111.0  # Approximate conversion
    
    zones = {
        "extreme": center.buffer(config.get("extreme_damage_radius_km", 10) * km_to_deg),
        "severe": center.buffer(config.get("severe_damage_radius_km", 25) * km_to_deg),
        "moderate": center.buffer(config.get("moderate_damage_radius_km", 50) * km_to_deg),
    }
    
    return zones


def create_storm_surge_zone(config: Dict[str, Any]) -> Polygon:
    """
    Create storm surge flood zone along the coast.
    Uses simplified coastal geometry for Visakhapatnam.
    """
    from shapely.geometry import LineString
    
    # Approximate coastline near Visakhapatnam
    coastal_points = [
        (83.1, 17.4),
        (83.2, 17.5),
        (83.3, 17.6),
        (83.4, 17.7),
        (83.5, 17.8),
    ]
    
    coastline = LineString(coastal_points)
    surge_distance = config.get("surge_inland_distance_km", 2) / 111.0
    return coastline.buffer(surge_distance)


def environmental_agent(state: RyudoState) -> Dict[str, Any]:
    """
    Environmental Agent: Maps external forces to spatial zones.
    
    Generic Action: Zone_Deletion
    - Observes weather data (cyclone, floods, etc.)
    - Uses LLM for reasoning about environmental hazards
    - Creates damage zone polygons
    - Emits constraints to delete roads within affected zones
    
    This is the "Flood Sentinel" in the CycloneShield implementation.
    """
    from agents.llm_client import call_environmental_agent
    
    constraints: List[GraphConstraint] = []
    viz_layers: List[Dict[str, Any]] = []
    
    env_data = state.get("environmental_data", {})
    
    # Handle cyclone scenario
    if env_data.get("type") == "cyclone":
        config = env_data.get("config", {})
        eye_lat = env_data.get("eye_lat", config.get("eye_lat"))
        eye_lon = env_data.get("eye_lon", config.get("eye_lon"))
        
        if eye_lat and eye_lon:
            # Call LLM for reasoning (if API key available)
            llm_analysis = call_environmental_agent(config)
            print(f"[FloodSentinel] LLM Analysis:\n{llm_analysis[:200]}..." if len(llm_analysis) > 200 else f"[FloodSentinel] LLM Analysis: {llm_analysis}")
            
            # Create damage zones using Holland model principles
            damage_zones = create_damage_zones(eye_lat, eye_lon, config)
            
            # Zone colors for visualization
            zone_colors = {
                "extreme": "#FF0000",
                "severe": "#FF8C00",
                "moderate": "#FFD700",
            }
            
            zone_descriptions = {
                "extreme": f"Wind > 150 km/h - Complete destruction",
                "severe": f"Wind 100-150 km/h - Severe damage, debris",
                "moderate": f"Wind 60-100 km/h - Trees down, flooding",
            }
            
            for zone_name, polygon in damage_zones.items():
                # Emit zone deletion constraint
                constraints.append({
                    "agent": "FloodSentinel",
                    "action": "delete_zone",
                    "target": {
                        "polygon": polygon,
                        "zone_type": zone_name,
                    },
                    "reason": f"Cyclone {zone_name} damage zone - {zone_descriptions[zone_name]}",
                    "metadata": {
                        "wind_speed_range": zone_descriptions[zone_name],
                        "cyclone_name": config.get("name", "Unknown"),
                        "eye_position": (eye_lat, eye_lon),
                    }
                })
                
                # Add visualization layer
                viz_layers.append({
                    "type": "polygon",
                    "agent": "FloodSentinel",
                    "name": f"{zone_name.title()} Damage Zone",
                    "geometry": polygon.__geo_interface__,
                    "style": {
                        "fillColor": zone_colors[zone_name],
                        "color": zone_colors[zone_name],
                        "weight": 2,
                        "fillOpacity": 0.15,
                    }
                })
            
            # Add storm surge zone
            surge_zone = create_storm_surge_zone(config)
            constraints.append({
                "agent": "FloodSentinel",
                "action": "delete_zone",
                "target": {
                    "polygon": surge_zone,
                    "zone_type": "flooded",
                },
                "reason": f"Storm surge flooding - {config.get('storm_surge_height_m', 2)}m inundation",
                "metadata": {
                    "surge_height_m": config.get("storm_surge_height_m", 2),
                    "inland_distance_km": config.get("surge_inland_distance_km", 2),
                }
            })
            
            viz_layers.append({
                "type": "polygon",
                "agent": "FloodSentinel",
                "name": "Storm Surge Zone",
                "geometry": surge_zone.__geo_interface__,
                "style": {
                    "fillColor": "#0066FF",
                    "color": "#0033CC",
                    "weight": 2,
                    "fillOpacity": 0.4,
                }
            })
            
            # Add cyclone eye marker
            viz_layers.append({
                "type": "marker",
                "agent": "FloodSentinel",
                "name": "Cyclone Eye",
                "position": (eye_lat, eye_lon),
                "popup": f"🌀 {config.get('name', 'Cyclone')}<br>"
                         f"Max Wind: {config.get('max_wind_speed_kmh', 'N/A')} km/h<br>"
                         f"Pressure: {config.get('central_pressure_mb', 'N/A')} mb",
                "icon": {"color": "red", "icon": "cloud", "prefix": "fa"}
            })
    
    print(f"[FloodSentinel] Generated {len(constraints)} zone constraints")
    
    return {
        "constraints": constraints,
        "visualization_layers": viz_layers,
    }
