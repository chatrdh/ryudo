"""
Mission Coordinator
====================
Applies all agent constraints to the graph and solves the optimization problem.

This is the "solver" that takes the modified Living Graph and computes
optimal routes based on the user's mission.
"""

from typing import Dict, Any, List, Optional, Tuple
import networkx as nx
import math

from agents.state import RyudoState, GraphConstraint


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def remove_edges_in_polygon(G: nx.MultiDiGraph, polygon, 
                             zone_type: str = "unknown") -> Tuple[nx.MultiDiGraph, List]:
    """Remove all edges that intersect with the given polygon."""
    import osmnx as ox
    from shapely.geometry import LineString
    
    G_modified = G.copy()
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    
    removed_edges = []
    
    for idx, row in edges_gdf.iterrows():
        geom = row.geometry
        
        # Check if edge intersects with damage zone
        if geom.intersects(polygon):
            u, v, k = idx
            if G_modified.has_edge(u, v, k):
                G_modified.remove_edge(u, v, k)
                removed_edges.append(idx)
    
    return G_modified, removed_edges


def apply_weight_multiplier(G: nx.MultiDiGraph, polygon, 
                            multiplier: float, zone_type: str) -> nx.MultiDiGraph:
    """Apply weight multiplier to edges within a polygon."""
    import osmnx as ox
    
    G_modified = G.copy()
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    
    for idx, row in edges_gdf.iterrows():
        if row.geometry.intersects(polygon):
            u, v, k = idx
            if G_modified.has_edge(u, v, k):
                original_length = G_modified[u][v][k].get("length", 1)
                G_modified[u][v][k]["length"] = original_length * multiplier
                G_modified[u][v][k]["damage"] = zone_type
    
    return G_modified


def find_nearest_node(G: nx.MultiDiGraph, lat: float, lon: float) -> Optional[int]:
    """Find the nearest node in the graph to the given coordinates."""
    import osmnx as ox
    try:
        return ox.distance.nearest_nodes(G, lon, lat)
    except:
        return None


def solve_rescue_routing(G: nx.MultiDiGraph, 
                         targets: List[Dict[str, Any]], 
                         depot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Solve a simple rescue routing problem.
    
    For now, uses greedy nearest neighbor. In production, would use
    OR-Tools or similar for proper VRP solving.
    """
    if not targets:
        return {"routes": [], "total_distance": 0, "unreachable": []}
    
    # Find depot node
    depot_node = find_nearest_node(G, depot["lat"], depot["lon"])
    if depot_node is None:
        return {"error": "Depot location not reachable", "routes": [], "unreachable": targets}
    
    routes = []
    unreachable = []
    total_distance = 0
    
    # Find target nodes
    target_nodes = []
    for target in targets:
        node = find_nearest_node(G, target["lat"], target["lon"])
        if node is not None:
            target_nodes.append({"target": target, "node": node})
        else:
            unreachable.append(target)
    
    # Simple greedy routing
    visited = set()
    current_node = depot_node
    route = [{"type": "depot", "node": depot_node, "location": depot}]
    
    while len(visited) < len(target_nodes):
        best_next = None
        best_distance = float("inf")
        
        for i, tn in enumerate(target_nodes):
            if i in visited:
                continue
            
            try:
                distance = nx.shortest_path_length(G, current_node, tn["node"], weight="length")
                if distance < best_distance:
                    best_distance = distance
                    best_next = i
            except nx.NetworkXNoPath:
                unreachable.append(tn["target"])
                visited.add(i)
        
        if best_next is not None:
            visited.add(best_next)
            tn = target_nodes[best_next]
            
            # Get the path
            try:
                path = nx.shortest_path(G, current_node, tn["node"], weight="length")
                route.append({
                    "type": "target",
                    "node": tn["node"],
                    "location": tn["target"],
                    "path": path,
                    "distance": best_distance,
                })
                total_distance += best_distance
                current_node = tn["node"]
            except:
                unreachable.append(tn["target"])
        else:
            break
    
    routes.append(route)
    
    return {
        "routes": routes,
        "total_distance": total_distance,
        "targets_reached": len(visited) - len([t for t in targets if t in unreachable]),
        "unreachable": unreachable,
    }


def mission_coordinator(state: RyudoState) -> Dict[str, Any]:
    """
    Mission Coordinator: Applies all constraints and solves the mission.
    
    This node:
    1. Takes all constraints from the 3 State Agents
    2. Applies them to the base graph (zone deletion, edge severing, TTL)
    3. Solves the routing problem based on mission type
    4. Returns the solution with visualization data
    """
    G = state["base_graph"].copy()
    
    constraints = state.get("constraints", [])
    mission = state.get("mission", {})
    
    # Track statistics
    stats = {
        "zones_deleted": 0,
        "edges_removed": 0,
        "nodes_disabled": 0,
        "ttl_applied": 0,
    }
    
    disabled_nodes = set()
    removed_edges = []
    
    print(f"[Coordinator] Processing {len(constraints)} constraints...")
    
    # Apply constraints in order of severity
    # 1. First apply zone deletions (Environmental Agent)
    for constraint in constraints:
        if constraint["action"] == "delete_zone":
            polygon = constraint["target"].get("polygon")
            zone_type = constraint["target"].get("zone_type", "unknown")
            
            if polygon:
                if zone_type in ["flooded", "extreme"]:
                    # Complete removal
                    G, edges = remove_edges_in_polygon(G, polygon, zone_type)
                    removed_edges.extend(edges)
                    stats["edges_removed"] += len(edges)
                elif zone_type == "severe":
                    # 10x weight multiplier
                    G = apply_weight_multiplier(G, polygon, 10.0, zone_type)
                elif zone_type == "moderate":
                    # 3x weight multiplier
                    G = apply_weight_multiplier(G, polygon, 3.0, zone_type)
                
                stats["zones_deleted"] += 1
    
    # 2. Apply node disabling (Infrastructure Agent)
    for constraint in constraints:
        if constraint["action"] == "disable_node":
            node_id = constraint["target"].get("node_id")
            if node_id:
                disabled_nodes.add(node_id)
                stats["nodes_disabled"] += 1
    
    # 3. Apply TTL constraints (Temporal Agent) - just store metadata for now
    for constraint in constraints:
        if constraint["action"] == "set_ttl":
            stats["ttl_applied"] += 1
            # In production, would add TTL metadata to edges
    
    print(f"[Coordinator] Stats: {stats}")
    print(f"[Coordinator] Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges "
          f"(removed {len(removed_edges)} edges)")
    
    # Solve the mission
    solution = None
    
    if mission.get("type") == "rescue":
        targets = mission.get("targets", [])
        depot = mission.get("depot", {"lat": 17.6868, "lon": 83.2185})
        
        print(f"[Coordinator] Solving rescue mission: {len(targets)} targets")
        solution = solve_rescue_routing(G, targets, depot)
        
        if solution:
            print(f"[Coordinator] Solution: {solution.get('targets_reached', 0)} reachable, "
                  f"{len(solution.get('unreachable', []))} unreachable")
            
            # Call LLM for mission synthesis (if API key available)
            from agents.llm_client import call_coordinator
            llm_briefing = call_coordinator(constraints, targets, solution)
            print(f"[Coordinator] LLM Mission Briefing:\n{llm_briefing[:300]}..." if len(llm_briefing) > 300 else f"[Coordinator] LLM Briefing: {llm_briefing}")
            solution["llm_briefing"] = llm_briefing
    
    return {
        "modified_graph": G,
        "solution": solution,
        "coordinator_stats": stats,
        "disabled_facilities": list(disabled_nodes),
    }
