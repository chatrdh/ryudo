"""
Road Network Routing Service
=============================
Provides road network-based routing using OSMnx and NetworkX.

This service enables:
- Loading and caching road network graphs
- Finding shortest paths between coordinates
- Applying damage zone constraints
- Returning routes with street names and distances
"""

import os
import json
import hashlib
import math
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import osmnx as ox
import networkx as nx
from shapely.geometry import Point, Polygon, shape
from shapely.ops import unary_union

# Configure OSMnx
ox.settings.use_cache = True
ox.settings.log_console = False

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "cache" / "routing"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Graph Management
# =============================================================================

_graph_cache: Dict[str, nx.MultiDiGraph] = {}


def _get_graph_cache_path(place: str) -> Path:
    """Get cache file path for a graph."""
    place_hash = hashlib.md5(place.encode()).hexdigest()[:12]
    return CACHE_DIR / f"{place_hash}_graph.graphml"


def get_road_graph(place: str, network_type: str = "drive") -> nx.MultiDiGraph:
    """
    Load or create a road network graph for routing.
    
    Parameters
    ----------
    place : str
        Place name (e.g., "Visakhapatnam, India")
    network_type : str
        Type of network ('drive', 'walk', 'bike', 'all')
        
    Returns
    -------
    nx.MultiDiGraph
        NetworkX graph with road network
    """
    cache_key = f"{place}_{network_type}"
    
    # Check memory cache
    if cache_key in _graph_cache:
        print(f"[Routing] Graph loaded from memory cache")
        return _graph_cache[cache_key]
    
    # Check file cache
    cache_path = _get_graph_cache_path(f"{place}_{network_type}")
    if cache_path.exists():
        print(f"[Routing] Loading graph from file cache...")
        G = ox.load_graphml(cache_path)
        _graph_cache[cache_key] = G
        print(f"[Routing] Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")
        return G
    
    # Download from OSM
    print(f"[Routing] Downloading road network for {place}...")
    G = ox.graph_from_place(place, network_type=network_type)
    
    # Add edge speeds and travel times
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    # Save to cache
    ox.save_graphml(G, cache_path)
    _graph_cache[cache_key] = G
    
    print(f"[Routing] Graph created: {len(G.nodes)} nodes, {len(G.edges)} edges")
    return G


def apply_constraints(G: nx.MultiDiGraph, constraints: List[Dict], apply_zones: bool = True) -> nx.MultiDiGraph:
    """
    Apply constraints to the graph by removing blocked edges.
    
    Parameters
    ----------
    G : nx.MultiDiGraph
        Original road network graph
    constraints : List[Dict]
        List of constraints with 'geometry' field for zone constraints
    apply_zones : bool
        If False, skip constraint application (useful for testing)
        
    Returns
    -------
    nx.MultiDiGraph
        Modified graph with blocked edges removed
    """
    if not apply_zones:
        print(f"[Routing] Skipping constraint application")
        return G
    
    # Work on a copy
    G_constrained = G.copy()
    
    # Collect only extreme/flooded zone geometries (not severe/moderate)
    zone_polygons = []
    for constraint in constraints:
        if constraint.get("action") == "delete_zone" and "geometry" in constraint:
            zone_type = constraint.get("zone_type", "")
            # Only block roads in extreme damage or flooded zones
            if zone_type in ["extreme", "flooded"]:
                try:
                    geom = shape(constraint["geometry"])
                    if geom.is_valid:
                        zone_polygons.append(geom)
                except Exception as e:
                    print(f"[Routing] Could not parse constraint geometry: {e}")
    
    if not zone_polygons:
        print(f"[Routing] No blocking zones to apply")
        return G_constrained
    
    # Merge all zones into one
    blocked_zone = unary_union(zone_polygons)
    print(f"[Routing] Checking {len(zone_polygons)} damage zones for blocked roads...")
    
    # Find edges that are INSIDE blocked zones (both endpoints)
    edges_to_remove = []
    
    for u, v, key in G.edges(keys=True):
        try:
            # OSMnx stores coordinates as x=lon, y=lat
            u_lon, u_lat = G.nodes[u]['x'], G.nodes[u]['y']
            v_lon, v_lat = G.nodes[v]['x'], G.nodes[v]['y']
            
            u_point = Point(u_lon, u_lat)
            v_point = Point(v_lon, v_lat)
            
            # Only remove if BOTH endpoints are in the blocked zone
            if blocked_zone.contains(u_point) and blocked_zone.contains(v_point):
                edges_to_remove.append((u, v, key))
        except KeyError:
            continue
    
    # Remove blocked edges
    if edges_to_remove:
        G_constrained.remove_edges_from([(u, v, k) for u, v, k in edges_to_remove])
    
    print(f"[Routing] Removed {len(edges_to_remove)} edges in damage zones (kept {len(G_constrained.edges())} edges)")
    return G_constrained


# =============================================================================
# Routing Functions
# =============================================================================

def find_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """Find the nearest graph node to a coordinate."""
    return ox.nearest_nodes(G, lon, lat)


def find_route(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    weight: str = "travel_time"
) -> Optional[Dict[str, Any]]:
    """
    Find the shortest route between two coordinates.
    
    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph
    start_lat, start_lon : float
        Starting coordinates
    end_lat, end_lon : float
        Ending coordinates
    weight : str
        Edge weight to optimize ('travel_time' or 'length')
        
    Returns
    -------
    dict or None
        Route information including:
        - path: List of node IDs
        - coords: List of [lat, lon] coordinates
        - distance_m: Total distance in meters
        - travel_time_s: Estimated travel time in seconds
        - road_segments: List of road names with distances
        - directions: Step-by-step directions
    """
    try:
        # Find nearest nodes
        start_node = find_nearest_node(G, start_lat, start_lon)
        end_node = find_nearest_node(G, end_lat, end_lon)
        
        # Find shortest path
        path = nx.shortest_path(G, start_node, end_node, weight=weight)
        
        if len(path) < 2:
            return None
        
        # Extract route details
        coords = []
        road_segments = []
        directions = []
        total_distance = 0
        total_time = 0
        
        current_road = None
        current_road_distance = 0
        
        for i in range(len(path)):
            node = path[i]
            node_data = G.nodes[node]
            coords.append([node_data['y'], node_data['x']])  # [lat, lon]
            
            if i < len(path) - 1:
                # Get edge data
                edge_data = G.get_edge_data(path[i], path[i+1])
                if edge_data:
                    # Get first edge if multiple
                    edge = list(edge_data.values())[0]
                    
                    distance = edge.get('length', 0)
                    travel_time = edge.get('travel_time', distance / 10)  # Default 10 m/s
                    road_name = edge.get('name', 'Unnamed Road')
                    
                    # Handle list of names
                    if isinstance(road_name, list):
                        road_name = road_name[0] if road_name else 'Unnamed Road'
                    
                    total_distance += distance
                    total_time += travel_time
                    
                    # Track road segments
                    if road_name == current_road:
                        current_road_distance += distance
                    else:
                        if current_road:
                            road_segments.append({
                                "road": current_road,
                                "distance_m": round(current_road_distance, 1)
                            })
                            directions.append(
                                f"Continue on {current_road} for {round(current_road_distance)}m"
                            )
                        current_road = road_name
                        current_road_distance = distance
        
        # Add final road segment
        if current_road:
            road_segments.append({
                "road": current_road,
                "distance_m": round(current_road_distance, 1)
            })
            directions.append(f"Arrive via {current_road}")
        
        return {
            "success": True,
            "path": path,
            "coords": coords,
            "distance_m": round(total_distance, 1),
            "distance_km": round(total_distance / 1000, 2),
            "travel_time_s": round(total_time, 1),
            "travel_time_min": round(total_time / 60, 1),
            "road_segments": road_segments,
            "directions": directions,
            "start_node": start_node,
            "end_node": end_node,
        }
        
    except nx.NetworkXNoPath:
        return {
            "success": False,
            "error": "No path found",
            "coords": [[start_lat, start_lon], [end_lat, end_lon]],
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "coords": [[start_lat, start_lon], [end_lat, end_lon]],
        }


def find_multi_stop_route(
    G: nx.MultiDiGraph,
    stops: List[Tuple[float, float]],
    weight: str = "travel_time"
) -> Dict[str, Any]:
    """
    Find a route through multiple stops in order.
    
    Parameters
    ----------
    G : nx.MultiDiGraph
        Road network graph
    stops : List[Tuple[float, float]]
        List of (lat, lon) coordinates in order
    weight : str
        Edge weight to optimize
        
    Returns
    -------
    dict
        Combined route information
    """
    if len(stops) < 2:
        return {"success": False, "error": "Need at least 2 stops"}
    
    all_coords = []
    all_segments = []
    all_directions = []
    total_distance = 0
    total_time = 0
    leg_details = []
    
    for i in range(len(stops) - 1):
        start = stops[i]
        end = stops[i + 1]
        
        route = find_route(G, start[0], start[1], end[0], end[1], weight)
        
        if route and route.get("success"):
            # Skip first coord if not first leg (avoid duplicates)
            if i > 0 and all_coords:
                route_coords = route["coords"][1:]
            else:
                route_coords = route["coords"]
            
            all_coords.extend(route_coords)
            all_segments.extend(route["road_segments"])
            all_directions.append(f"--- Leg {i+1}: Stop {i+1} → Stop {i+2} ---")
            all_directions.extend(route["directions"])
            total_distance += route["distance_m"]
            total_time += route["travel_time_s"]
            
            leg_details.append({
                "leg": i + 1,
                "from": start,
                "to": end,
                "distance_km": route["distance_km"],
                "travel_time_min": route["travel_time_min"],
                "road_count": len(route["road_segments"]),
            })
        else:
            # Mark unreachable leg
            leg_details.append({
                "leg": i + 1,
                "from": start,
                "to": end,
                "reachable": False,
                "error": route.get("error", "Unknown error") if route else "No route",
            })
    
    return {
        "success": True,
        "coords": all_coords,
        "distance_m": round(total_distance, 1),
        "distance_km": round(total_distance / 1000, 2),
        "travel_time_s": round(total_time, 1),
        "travel_time_min": round(total_time / 60, 1),
        "road_segments": all_segments,
        "directions": all_directions,
        "legs": leg_details,
        "stop_count": len(stops),
    }


# =============================================================================
# Utility Functions
# =============================================================================

def get_route_summary(route: Dict[str, Any]) -> str:
    """Generate a human-readable route summary."""
    if not route or not route.get("success"):
        return "Route not available"
    
    lines = [
        f"📍 Distance: {route['distance_km']} km",
        f"⏱️ Est. Time: {route['travel_time_min']} min",
        f"🛣️ Roads: {len(route['road_segments'])} segments",
    ]
    
    # Add main roads
    if route.get("road_segments"):
        main_roads = [s["road"] for s in route["road_segments"][:3]]
        main_roads = [r for r in main_roads if r != "Unnamed Road"]
        if main_roads:
            lines.append(f"📌 Via: {', '.join(main_roads)}")
    
    return "\n".join(lines)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate straight-line distance between two coordinates in meters."""
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


if __name__ == "__main__":
    # Test the routing service
    print("Testing routing service...")
    
    place = "Visakhapatnam, India"
    G = get_road_graph(place)
    
    # Test route from depot to a target
    depot = (17.6868, 83.2185)
    target = (17.7200, 83.3200)
    
    route = find_route(G, depot[0], depot[1], target[0], target[1])
    
    if route and route.get("success"):
        print(f"\nRoute found!")
        print(f"  Distance: {route['distance_km']} km")
        print(f"  Travel time: {route['travel_time_min']} min")
        print(f"  Road segments: {len(route['road_segments'])}")
        print(f"\nDirections:")
        for d in route['directions'][:5]:
            print(f"  - {d}")
    else:
        print(f"No route found: {route}")
