"""
Geographic Data Extraction Service
====================================
Extracts and provides GIS data from OpenStreetMap via OSMnx.

This module provides structured geographic data with clear metadata
for use by both the frontend map visualization and AI agents.

Data Layers:
- Roads: Classified by highway type (motorway, primary, secondary, etc.)
- Water: Water bodies (lakes, ponds) and waterways (rivers, streams)
- Buildings: Building footprints with type classification
- Land Use: Land use zones (residential, commercial, industrial, etc.)

All data is returned as GeoJSON with rich metadata for AI agent processing.
"""

import os
import json
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import osmnx as ox
import geopandas as gpd
from shapely.geometry import mapping

# Cache directory for geographic data
CACHE_DIR = Path(__file__).parent.parent / "cache" / "geo"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Road Classification Schema
# =============================================================================
# This classification is used by both the frontend for styling and AI agents
# for understanding road hierarchy and capacity.

ROAD_CLASSIFICATION = {
    "motorway": {
        "level": 1,
        "name": "Motorway",
        "description": "High-speed limited-access highway",
        "color": "#e892a2",
        "weight": 5,
        "speed_limit_kmh": 100,
        "capacity": "very_high",
        "passable_during_flood": False,
        "evacuation_priority": "primary"
    },
    "motorway_link": {
        "level": 1,
        "name": "Motorway Link",
        "description": "Motorway on/off ramp",
        "color": "#e892a2",
        "weight": 4,
        "speed_limit_kmh": 60,
        "capacity": "high",
        "passable_during_flood": False,
        "evacuation_priority": "primary"
    },
    "trunk": {
        "level": 2,
        "name": "Trunk Road",
        "description": "Major arterial road",
        "color": "#f9b29c",
        "weight": 4,
        "speed_limit_kmh": 80,
        "capacity": "high",
        "passable_during_flood": False,
        "evacuation_priority": "primary"
    },
    "trunk_link": {
        "level": 2,
        "name": "Trunk Link",
        "description": "Trunk road connector",
        "color": "#f9b29c",
        "weight": 3,
        "speed_limit_kmh": 50,
        "capacity": "medium",
        "passable_during_flood": False,
        "evacuation_priority": "secondary"
    },
    "primary": {
        "level": 3,
        "name": "Primary Road",
        "description": "Major city road",
        "color": "#fcd6a4",
        "weight": 3,
        "speed_limit_kmh": 60,
        "capacity": "medium",
        "passable_during_flood": False,
        "evacuation_priority": "secondary"
    },
    "primary_link": {
        "level": 3,
        "name": "Primary Link",
        "description": "Primary road connector",
        "color": "#fcd6a4",
        "weight": 2,
        "speed_limit_kmh": 40,
        "capacity": "medium",
        "passable_during_flood": False,
        "evacuation_priority": "secondary"
    },
    "secondary": {
        "level": 4,
        "name": "Secondary Road",
        "description": "Secondary city road",
        "color": "#f7fabf",
        "weight": 3,
        "speed_limit_kmh": 50,
        "capacity": "medium",
        "passable_during_flood": False,
        "evacuation_priority": "tertiary"
    },
    "secondary_link": {
        "level": 4,
        "name": "Secondary Link",
        "description": "Secondary road connector",
        "color": "#f7fabf",
        "weight": 2,
        "speed_limit_kmh": 40,
        "capacity": "low",
        "passable_during_flood": False,
        "evacuation_priority": "tertiary"
    },
    "tertiary": {
        "level": 5,
        "name": "Tertiary Road",
        "description": "Local connector road",
        "color": "#ffffff",
        "weight": 2,
        "speed_limit_kmh": 40,
        "capacity": "low",
        "passable_during_flood": False,
        "evacuation_priority": "tertiary"
    },
    "tertiary_link": {
        "level": 5,
        "name": "Tertiary Link",
        "description": "Tertiary road connector",
        "color": "#ffffff",
        "weight": 1,
        "speed_limit_kmh": 30,
        "capacity": "low",
        "passable_during_flood": False,
        "evacuation_priority": "low"
    },
    "residential": {
        "level": 6,
        "name": "Residential Road",
        "description": "Road in residential area",
        "color": "#ffffff",
        "weight": 1,
        "speed_limit_kmh": 30,
        "capacity": "low",
        "passable_during_flood": False,
        "evacuation_priority": "low"
    },
    "living_street": {
        "level": 7,
        "name": "Living Street",
        "description": "Pedestrian-priority street",
        "color": "#ededed",
        "weight": 1,
        "speed_limit_kmh": 20,
        "capacity": "very_low",
        "passable_during_flood": False,
        "evacuation_priority": "low"
    },
    "unclassified": {
        "level": 6,
        "name": "Unclassified Road",
        "description": "Minor road",
        "color": "#ffffff",
        "weight": 1,
        "speed_limit_kmh": 30,
        "capacity": "low",
        "passable_during_flood": False,
        "evacuation_priority": "low"
    },
    "service": {
        "level": 7,
        "name": "Service Road",
        "description": "Access road for services",
        "color": "#ededed",
        "weight": 1,
        "speed_limit_kmh": 20,
        "capacity": "very_low",
        "passable_during_flood": False,
        "evacuation_priority": "none"
    }
}


# =============================================================================
# Water Classification Schema
# =============================================================================

WATER_CLASSIFICATION = {
    "river": {
        "type": "waterway",
        "name": "River",
        "description": "Major flowing water body",
        "color": "#aad3df",
        "stroke_color": "#6699cc",
        "weight": 3,
        "flood_risk": "high",
        "buffer_zone_m": 100
    },
    "stream": {
        "type": "waterway",
        "name": "Stream",
        "description": "Smaller flowing water",
        "color": "#aad3df",
        "stroke_color": "#6699cc",
        "weight": 2,
        "flood_risk": "medium",
        "buffer_zone_m": 50
    },
    "canal": {
        "type": "waterway",
        "name": "Canal",
        "description": "Artificial waterway",
        "color": "#aad3df",
        "stroke_color": "#6699cc",
        "weight": 2,
        "flood_risk": "medium",
        "buffer_zone_m": 30
    },
    "drain": {
        "type": "waterway",
        "name": "Drain",
        "description": "Drainage channel",
        "color": "#aad3df",
        "stroke_color": "#99b3cc",
        "weight": 1,
        "flood_risk": "medium",
        "buffer_zone_m": 20
    },
    "water": {
        "type": "water_body",
        "name": "Water Body",
        "description": "Lake, pond, or reservoir",
        "color": "#aad3df",
        "stroke_color": "#6699cc",
        "fill_opacity": 0.6,
        "flood_risk": "high",
        "buffer_zone_m": 50
    },
    "wetland": {
        "type": "water_body",
        "name": "Wetland",
        "description": "Marsh or swamp area",
        "color": "#c8d7ab",
        "stroke_color": "#8fbc8f",
        "fill_opacity": 0.5,
        "flood_risk": "very_high",
        "buffer_zone_m": 100
    },
    "coastline": {
        "type": "boundary",
        "name": "Coastline",
        "description": "Sea/ocean boundary",
        "color": "#6699cc",
        "stroke_color": "#4477aa",
        "weight": 2,
        "flood_risk": "very_high",
        "buffer_zone_m": 200
    }
}


# =============================================================================
# Building Classification Schema
# =============================================================================

BUILDING_CLASSIFICATION = {
    "residential": {
        "name": "Residential Building",
        "description": "Houses, apartments, homes",
        "color": "#d9b38c",
        "priority_evacuation": "high",
        "capacity_estimate": "medium"
    },
    "commercial": {
        "name": "Commercial Building",
        "description": "Shops, offices, businesses",
        "color": "#c9a0dc",
        "priority_evacuation": "medium",
        "capacity_estimate": "high"
    },
    "industrial": {
        "name": "Industrial Building",
        "description": "Factories, warehouses",
        "color": "#a0a0a0",
        "priority_evacuation": "low",
        "capacity_estimate": "low"
    },
    "hospital": {
        "name": "Hospital",
        "description": "Medical facility",
        "color": "#ff6b6b",
        "priority_evacuation": "critical",
        "capacity_estimate": "high",
        "is_shelter": False,
        "is_critical_infrastructure": True
    },
    "school": {
        "name": "School",
        "description": "Educational facility",
        "color": "#98d8c8",
        "priority_evacuation": "high",
        "capacity_estimate": "high",
        "is_shelter": True,
        "is_critical_infrastructure": False
    },
    "public": {
        "name": "Public Building",
        "description": "Government, community buildings",
        "color": "#f7dc6f",
        "priority_evacuation": "medium",
        "capacity_estimate": "medium",
        "is_shelter": True,
        "is_critical_infrastructure": False
    },
    "religious": {
        "name": "Religious Building",
        "description": "Temple, church, mosque",
        "color": "#aed6f1",
        "priority_evacuation": "medium",
        "capacity_estimate": "medium",
        "is_shelter": True,
        "is_critical_infrastructure": False
    },
    "yes": {
        "name": "Building",
        "description": "Unspecified building type",
        "color": "#cccccc",
        "priority_evacuation": "medium",
        "capacity_estimate": "unknown"
    }
}


# =============================================================================
# Land Use Classification Schema
# =============================================================================

LANDUSE_CLASSIFICATION = {
    "residential": {
        "name": "Residential Area",
        "description": "Housing and living areas",
        "color": "#ffc0cb",
        "fill_opacity": 0.3,
        "population_density": "high",
        "evacuation_priority": "high"
    },
    "commercial": {
        "name": "Commercial Area",
        "description": "Business and shopping districts",
        "color": "#ff69b4",
        "fill_opacity": 0.3,
        "population_density": "high",
        "evacuation_priority": "high"
    },
    "industrial": {
        "name": "Industrial Area",
        "description": "Factories and industrial zones",
        "color": "#808080",
        "fill_opacity": 0.3,
        "population_density": "low",
        "evacuation_priority": "medium",
        "hazard_potential": "high"
    },
    "farmland": {
        "name": "Farmland",
        "description": "Agricultural land",
        "color": "#eef3c1",
        "fill_opacity": 0.3,
        "population_density": "very_low",
        "evacuation_priority": "low"
    },
    "forest": {
        "name": "Forest",
        "description": "Wooded area",
        "color": "#228b22",
        "fill_opacity": 0.3,
        "population_density": "none",
        "evacuation_priority": "none"
    },
    "grass": {
        "name": "Grassland",
        "description": "Open grass areas, parks",
        "color": "#98fb98",
        "fill_opacity": 0.3,
        "population_density": "none",
        "evacuation_priority": "none",
        "is_evacuation_zone": True
    },
    "recreation_ground": {
        "name": "Recreation Ground",
        "description": "Sports fields, playgrounds",
        "color": "#90ee90",
        "fill_opacity": 0.3,
        "population_density": "variable",
        "evacuation_priority": "low",
        "is_evacuation_zone": True
    },
    "cemetery": {
        "name": "Cemetery",
        "description": "Burial ground",
        "color": "#acdca0",
        "fill_opacity": 0.2,
        "population_density": "none",
        "evacuation_priority": "none"
    },
    "quarry": {
        "name": "Quarry",
        "description": "Mining or quarry site",
        "color": "#c0c0c0",
        "fill_opacity": 0.3,
        "population_density": "very_low",
        "evacuation_priority": "low",
        "hazard_potential": "medium"
    }
}


# =============================================================================
# Data Extraction Functions
# =============================================================================

def _get_cache_path(place: str, data_type: str) -> Path:
    """Generate cache file path based on place and data type."""
    place_hash = hashlib.md5(place.encode()).hexdigest()[:8]
    return CACHE_DIR / f"{place_hash}_{data_type}.geojson"


def _load_from_cache(cache_path: Path) -> Optional[Dict]:
    """Load GeoJSON from cache if exists."""
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None


def _sanitize_value(val):
    """Sanitize a single value for JSON serialization."""
    import math
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, dict):
        return {k: _sanitize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_sanitize_value(v) for v in val]
    # Handle pandas NA/NaN
    try:
        import pandas as pd
        if pd.isna(val):
            return None
    except (ImportError, TypeError):
        pass
    return val


def _sanitize_properties(props: Dict) -> Dict:
    """Sanitize all properties in a feature for JSON serialization."""
    return {k: _sanitize_value(v) for k, v in props.items()}


def _save_to_cache(cache_path: Path, data: Dict):
    """Save GeoJSON to cache."""
    with open(cache_path, 'w') as f:
        json.dump(data, f)


def extract_roads(place: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Extract road network with classification metadata.
    
    Parameters
    ----------
    place : str
        Place name (e.g., "Visakhapatnam, India")
    use_cache : bool
        Whether to use cached data if available
        
    Returns
    -------
    dict
        GeoJSON FeatureCollection with road features and metadata
        
    Metadata per feature:
    - highway_type: OSM highway classification
    - classification: Detailed classification from ROAD_CLASSIFICATION
    - name: Road name if available
    - oneway: Whether road is one-way
    - lanes: Number of lanes
    """
    cache_path = _get_cache_path(place, "roads")
    
    if use_cache:
        cached = _load_from_cache(cache_path)
        if cached:
            print(f"[GeoData] Roads loaded from cache: {len(cached['features'])} features")
            return cached
    
    print(f"[GeoData] Extracting roads for {place}...")
    
    # Get road network graph from OSMnx
    G = ox.graph_from_place(place, network_type="drive")
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    
    features = []
    for idx, row in edges_gdf.iterrows():
        if row.geometry is None:
            continue
            
        # Get highway type (might be a list)
        highway = row.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]
        
        # Get classification info
        classification = ROAD_CLASSIFICATION.get(highway, ROAD_CLASSIFICATION["unclassified"])
        
        # Build feature properties
        properties = {
            "id": f"road_{idx[0]}_{idx[1]}",
            "highway_type": highway,
            "classification": classification,
            "name": row.get("name", None),
            "oneway": row.get("oneway", False),
            "lanes": row.get("lanes", None),
            "length_m": row.get("length", 0),
            "maxspeed": row.get("maxspeed", classification.get("speed_limit_kmh")),
            # AI Agent metadata
            "agent_metadata": {
                "layer_type": "road",
                "can_be_blocked": True,
                "flood_vulnerable": classification.get("passable_during_flood", False) == False,
                "evacuation_priority": classification.get("evacuation_priority", "low"),
                "capacity": classification.get("capacity", "low")
            }
        }
        
        feature = {
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": _sanitize_properties(properties)
        }
        features.append(feature)
    
    result = {
        "type": "FeatureCollection",
        "metadata": {
            "layer_name": "roads",
            "layer_description": "Road network with highway classification",
            "place": place,
            "total_features": len(features),
            "classification_schema": ROAD_CLASSIFICATION,
            "agent_instructions": {
                "description": "Road network layer for routing and constraint application",
                "usage": "Use highway_type and classification to determine road importance",
                "constraint_actions": ["block_road", "slow_traffic", "set_weight"],
                "priority_hierarchy": ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]
            }
        },
        "features": features
    }
    
    _save_to_cache(cache_path, result)
    print(f"[GeoData] Extracted {len(features)} road segments")
    
    return result


def extract_water(place: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Extract water bodies and waterways.
    
    Returns
    -------
    dict
        GeoJSON FeatureCollection with water features
        
    Metadata per feature:
    - water_type: Type of water feature
    - classification: Detailed classification from WATER_CLASSIFICATION
    - name: Water body name if available
    """
    cache_path = _get_cache_path(place, "water")
    
    if use_cache:
        cached = _load_from_cache(cache_path)
        if cached:
            print(f"[GeoData] Water loaded from cache: {len(cached['features'])} features")
            return cached
    
    print(f"[GeoData] Extracting water features for {place}...")
    
    features = []
    
    # Extract water bodies (lakes, ponds, reservoirs)
    try:
        water_bodies = ox.features_from_place(place, tags={"natural": "water"})
        for idx, row in water_bodies.iterrows():
            if row.geometry is None:
                continue
            
            properties = {
                "id": f"water_body_{idx}",
                "water_type": "water",
                "classification": WATER_CLASSIFICATION["water"],
                "name": row.get("name", None),
                "agent_metadata": {
                    "layer_type": "water_body",
                    "flood_risk": "high",
                    "buffer_zone_m": 50,
                    "impassable": True
                }
            }
            
            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": _sanitize_properties(properties)
            }
            features.append(feature)
    except Exception as e:
        print(f"[GeoData] Could not fetch water bodies: {e}")
    
    # Extract waterways (rivers, streams, canals)
    try:
        waterways = ox.features_from_place(place, tags={"waterway": True})
        for idx, row in waterways.iterrows():
            if row.geometry is None:
                continue
            
            waterway_type = row.get("waterway", "stream")
            if isinstance(waterway_type, list):
                waterway_type = waterway_type[0]
            
            classification = WATER_CLASSIFICATION.get(waterway_type, WATER_CLASSIFICATION["stream"])
            
            properties = {
                "id": f"waterway_{idx}",
                "water_type": waterway_type,
                "classification": classification,
                "name": row.get("name", None),
                "agent_metadata": {
                    "layer_type": "waterway",
                    "flood_risk": classification.get("flood_risk", "medium"),
                    "buffer_zone_m": classification.get("buffer_zone_m", 30),
                    "impassable": True
                }
            }
            
            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": _sanitize_properties(properties)
            }
            features.append(feature)
    except Exception as e:
        print(f"[GeoData] Could not fetch waterways: {e}")
    
    # Extract wetlands
    try:
        wetlands = ox.features_from_place(place, tags={"natural": "wetland"})
        for idx, row in wetlands.iterrows():
            if row.geometry is None:
                continue
            
            properties = {
                "id": f"wetland_{idx}",
                "water_type": "wetland",
                "classification": WATER_CLASSIFICATION["wetland"],
                "name": row.get("name", None),
                "agent_metadata": {
                    "layer_type": "wetland",
                    "flood_risk": "very_high",
                    "buffer_zone_m": 100,
                    "impassable": True
                }
            }
            
            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": _sanitize_properties(properties)
            }
            features.append(feature)
    except Exception as e:
        print(f"[GeoData] Could not fetch wetlands: {e}")
    
    result = {
        "type": "FeatureCollection",
        "metadata": {
            "layer_name": "water",
            "layer_description": "Water bodies, waterways, and wetlands",
            "place": place,
            "total_features": len(features),
            "classification_schema": WATER_CLASSIFICATION,
            "agent_instructions": {
                "description": "Water features for flood risk assessment",
                "usage": "Use flood_risk and buffer_zone_m for flood zone calculation",
                "constraint_actions": ["expand_flood_zone", "mark_impassable"],
                "risk_hierarchy": ["very_high", "high", "medium", "low"]
            }
        },
        "features": features
    }
    
    _save_to_cache(cache_path, result)
    print(f"[GeoData] Extracted {len(features)} water features")
    
    return result


def extract_buildings(place: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Extract building footprints with classification.
    
    Returns
    -------
    dict
        GeoJSON FeatureCollection with building features
    """
    cache_path = _get_cache_path(place, "buildings")
    
    if use_cache:
        cached = _load_from_cache(cache_path)
        if cached:
            print(f"[GeoData] Buildings loaded from cache: {len(cached['features'])} features")
            return cached
    
    print(f"[GeoData] Extracting buildings for {place}...")
    
    features = []
    
    try:
        buildings = ox.features_from_place(place, tags={"building": True})
        
        for idx, row in buildings.iterrows():
            if row.geometry is None:
                continue
            
            building_type = row.get("building", "yes")
            if isinstance(building_type, list):
                building_type = building_type[0]
            
            classification = BUILDING_CLASSIFICATION.get(
                building_type, 
                BUILDING_CLASSIFICATION["yes"]
            )
            
            # Check for specific amenities that override building type
            amenity = row.get("amenity", None)
            if amenity == "hospital":
                classification = BUILDING_CLASSIFICATION["hospital"]
                building_type = "hospital"
            elif amenity == "school":
                classification = BUILDING_CLASSIFICATION["school"]
                building_type = "school"
            
            properties = {
                "id": f"building_{idx}",
                "building_type": building_type,
                "classification": classification,
                "name": row.get("name", None),
                "levels": row.get("building:levels", None),
                "amenity": amenity,
                "agent_metadata": {
                    "layer_type": "building",
                    "evacuation_priority": classification.get("priority_evacuation", "medium"),
                    "is_shelter": classification.get("is_shelter", False),
                    "is_critical_infrastructure": classification.get("is_critical_infrastructure", False),
                    "capacity_estimate": classification.get("capacity_estimate", "unknown")
                }
            }
            
            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": _sanitize_properties(properties)
            }
            features.append(feature)
            
    except Exception as e:
        print(f"[GeoData] Could not fetch buildings: {e}")
    
    result = {
        "type": "FeatureCollection",
        "metadata": {
            "layer_name": "buildings",
            "layer_description": "Building footprints with type classification",
            "place": place,
            "total_features": len(features),
            "classification_schema": BUILDING_CLASSIFICATION,
            "agent_instructions": {
                "description": "Building features for rescue prioritization",
                "usage": "Use evacuation_priority and is_shelter for mission planning",
                "constraint_actions": ["mark_shelter", "mark_critical", "set_priority"],
                "priority_hierarchy": ["critical", "high", "medium", "low"]
            }
        },
        "features": features
    }
    
    _save_to_cache(cache_path, result)
    print(f"[GeoData] Extracted {len(features)} building footprints")
    
    return result


def extract_landuse(place: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Extract land use zones.
    
    Returns
    -------
    dict
        GeoJSON FeatureCollection with land use features
    """
    cache_path = _get_cache_path(place, "landuse")
    
    if use_cache:
        cached = _load_from_cache(cache_path)
        if cached:
            print(f"[GeoData] Land use loaded from cache: {len(cached['features'])} features")
            return cached
    
    print(f"[GeoData] Extracting land use for {place}...")
    
    features = []
    
    try:
        landuse = ox.features_from_place(place, tags={"landuse": True})
        
        for idx, row in landuse.iterrows():
            if row.geometry is None:
                continue
            
            landuse_type = row.get("landuse", "unknown")
            if isinstance(landuse_type, list):
                landuse_type = landuse_type[0]
            
            classification = LANDUSE_CLASSIFICATION.get(
                landuse_type,
                {
                    "name": landuse_type.capitalize(),
                    "description": f"{landuse_type} land use",
                    "color": "#dddddd",
                    "fill_opacity": 0.2,
                    "population_density": "unknown",
                    "evacuation_priority": "medium"
                }
            )
            
            properties = {
                "id": f"landuse_{idx}",
                "landuse_type": landuse_type,
                "classification": classification,
                "name": row.get("name", None),
                "agent_metadata": {
                    "layer_type": "landuse",
                    "population_density": classification.get("population_density", "unknown"),
                    "evacuation_priority": classification.get("evacuation_priority", "medium"),
                    "is_evacuation_zone": classification.get("is_evacuation_zone", False),
                    "hazard_potential": classification.get("hazard_potential", "low")
                }
            }
            
            feature = {
                "type": "Feature",
                "geometry": mapping(row.geometry),
                "properties": _sanitize_properties(properties)
            }
            features.append(feature)
            
    except Exception as e:
        print(f"[GeoData] Could not fetch land use: {e}")
    
    result = {
        "type": "FeatureCollection",
        "metadata": {
            "layer_name": "landuse",
            "layer_description": "Land use zones for population and hazard assessment",
            "place": place,
            "total_features": len(features),
            "classification_schema": LANDUSE_CLASSIFICATION,
            "agent_instructions": {
                "description": "Land use zones for population estimation",
                "usage": "Use population_density and evacuation_priority for rescue planning",
                "constraint_actions": ["mark_evacuation_zone", "estimate_population"],
                "density_hierarchy": ["very_high", "high", "medium", "low", "very_low", "none"]
            }
        },
        "features": features
    }
    
    _save_to_cache(cache_path, result)
    print(f"[GeoData] Extracted {len(features)} land use zones")
    
    return result


def extract_all(place: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Extract all geographic data layers at once.
    
    Returns
    -------
    dict
        Dictionary containing all layers and combined metadata
    """
    print(f"[GeoData] Extracting all geographic data for {place}...")
    
    roads = extract_roads(place, use_cache)
    water = extract_water(place, use_cache)
    buildings = extract_buildings(place, use_cache)
    landuse = extract_landuse(place, use_cache)
    
    result = {
        "place": place,
        "layers": {
            "roads": roads,
            "water": water,
            "buildings": buildings,
            "landuse": landuse
        },
        "summary": {
            "total_roads": len(roads["features"]),
            "total_water_features": len(water["features"]),
            "total_buildings": len(buildings["features"]),
            "total_landuse_zones": len(landuse["features"])
        },
        "agent_layer_guide": {
            "roads": {
                "purpose": "Road network for routing and constraint application",
                "key_properties": ["highway_type", "evacuation_priority", "capacity"],
                "actions": ["block_road", "slow_traffic", "set_weight"]
            },
            "water": {
                "purpose": "Water features for flood risk zones",
                "key_properties": ["water_type", "flood_risk", "buffer_zone_m"],
                "actions": ["expand_flood_zone", "mark_impassable"]
            },
            "buildings": {
                "purpose": "Buildings for rescue prioritization",
                "key_properties": ["building_type", "evacuation_priority", "is_shelter"],
                "actions": ["mark_shelter", "set_priority", "disable_node"]
            },
            "landuse": {
                "purpose": "Land use for population estimation",
                "key_properties": ["landuse_type", "population_density", "is_evacuation_zone"],
                "actions": ["estimate_population", "mark_evacuation_zone"]
            }
        }
    }
    
    print(f"[GeoData] All data extracted: {result['summary']}")
    
    return result


# =============================================================================
# Utility Functions for AI Agents
# =============================================================================

def get_layer_schema(layer_name: str) -> Dict[str, Any]:
    """
    Get the classification schema for a layer.
    
    Useful for AI agents to understand available classifications.
    """
    schemas = {
        "roads": ROAD_CLASSIFICATION,
        "water": WATER_CLASSIFICATION,
        "buildings": BUILDING_CLASSIFICATION,
        "landuse": LANDUSE_CLASSIFICATION
    }
    return schemas.get(layer_name, {})


def get_agent_instructions() -> Dict[str, Any]:
    """
    Get instructions for AI agents on how to use geographic data.
    
    Returns a structured guide for agent consumption.
    """
    return {
        "overview": "Geographic data layers for disaster response and rescue planning",
        "layers": {
            "roads": {
                "description": "Road network with highway classification",
                "primary_use": "Routing, constraint application, evacuation paths",
                "key_actions": [
                    {"action": "block_road", "when": "Road is flooded or damaged"},
                    {"action": "slow_traffic", "when": "Road is partially affected"},
                    {"action": "set_weight", "when": "Adjusting route costs"}
                ],
                "priority_field": "evacuation_priority",
                "capacity_field": "capacity"
            },
            "water": {
                "description": "Water bodies and waterways",
                "primary_use": "Flood risk assessment, buffer zone calculation",
                "key_actions": [
                    {"action": "expand_flood_zone", "when": "Storm surge or heavy rain"},
                    {"action": "mark_impassable", "when": "Water level exceeds threshold"}
                ],
                "risk_field": "flood_risk",
                "buffer_field": "buffer_zone_m"
            },
            "buildings": {
                "description": "Building footprints with type and capacity",
                "primary_use": "Rescue prioritization, shelter identification",
                "key_actions": [
                    {"action": "mark_shelter", "when": "Building can serve as evacuation shelter"},
                    {"action": "set_priority", "when": "Adjusting rescue order"},
                    {"action": "disable_node", "when": "Building has no power or is damaged"}
                ],
                "priority_field": "evacuation_priority",
                "shelter_field": "is_shelter"
            },
            "landuse": {
                "description": "Land use zones for population estimation",
                "primary_use": "Population density estimation, evacuation zone marking",
                "key_actions": [
                    {"action": "estimate_population", "when": "Planning rescue capacity"},
                    {"action": "mark_evacuation_zone", "when": "Identifying safe areas"}
                ],
                "density_field": "population_density",
                "evacuation_field": "is_evacuation_zone"
            }
        },
        "workflow": [
            "1. Load all layers using extract_all(place)",
            "2. Apply environmental constraints (floods, damage zones)",
            "3. Update road weights based on conditions",
            "4. Identify shelters and critical infrastructure",
            "5. Calculate evacuation routes using modified graph"
        ]
    }
