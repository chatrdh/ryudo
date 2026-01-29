"""
Map Renderer
=============
Renders agent decisions and solutions on interactive Folium maps.

Features:
- Color-coded damage zones from each agent
- Agent reasoning in popups
- Solution route visualization
- Layer controls for toggling agent contributions
"""

from typing import Dict, Any, List, Optional
import folium
from folium import FeatureGroup, Marker, PolyLine, GeoJson, Element
from folium.plugins import HeatMap
import networkx as nx


# Default center for Visakhapatnam
DEFAULT_CENTER = (17.6868, 83.2185)


def create_base_map(center: tuple = DEFAULT_CENTER, zoom: int = 11) -> folium.Map:
    """Create a base Folium map with multiple tile layers."""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="CartoDB positron"
    )
    
    # Add satellite layer option
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        control=True
    ).add_to(m)
    
    # Add OpenStreetMap layer
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True
    ).add_to(m)
    
    return m


def add_visualization_layers(m: folium.Map, viz_layers: List[Dict[str, Any]]) -> folium.Map:
    """
    Add visualization layers from agents to the map.
    
    Each layer is a dict with:
    - type: "polygon", "marker", "polyline", "info"
    - agent: which agent created this
    - name: layer name
    - geometry/position: spatial data
    - style/icon: visual styling
    """
    # Group layers by agent
    agent_groups = {}
    
    for layer in viz_layers:
        agent = layer.get("agent", "Unknown")
        if agent not in agent_groups:
            agent_groups[agent] = FeatureGroup(name=agent)
        
        layer_type = layer.get("type")
        
        if layer_type == "polygon":
            GeoJson(
                layer["geometry"],
                name=layer.get("name", "Polygon"),
                style_function=lambda x, s=layer.get("style", {}): s
            ).add_to(agent_groups[agent])
            
        elif layer_type == "marker":
            icon_config = layer.get("icon", {})
            Marker(
                layer["position"],
                popup=layer.get("popup", ""),
                icon=folium.Icon(
                    color=icon_config.get("color", "blue"),
                    icon=icon_config.get("icon", "info-sign"),
                    prefix=icon_config.get("prefix", "glyphicon")
                )
            ).add_to(agent_groups[agent])
            
        elif layer_type == "polyline":
            PolyLine(
                layer["coordinates"],
                color=layer.get("color", "blue"),
                weight=layer.get("weight", 3),
                opacity=layer.get("opacity", 0.8)
            ).add_to(agent_groups[agent])
    
    # Add all agent groups to map
    for group in agent_groups.values():
        group.add_to(m)
    
    return m


def add_road_network(m: folium.Map, G: nx.MultiDiGraph, 
                     color: str = "#888888", name: str = "Road Network") -> folium.Map:
    """Add road network to the map."""
    import osmnx as ox
    
    roads = FeatureGroup(name=name)
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    
    for _, row in edges_gdf.iterrows():
        if row.geometry.geom_type == "LineString":
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            
            # Check for damage attribute
            damage = row.get("damage") if hasattr(row, "get") else None
            
            if damage == "severe":
                edge_color = "#FF8C00"
                weight = 3
            elif damage == "moderate":
                edge_color = "#FFD700"
                weight = 2
            else:
                edge_color = color
                weight = 1
            
            PolyLine(
                coords,
                color=edge_color,
                weight=weight,
                opacity=0.6
            ).add_to(roads)
    
    roads.add_to(m)
    return m


def add_solution_routes(m: folium.Map, solution: Dict[str, Any], G: nx.MultiDiGraph) -> folium.Map:
    """Add solution routes to the map."""
    import osmnx as ox
    
    if not solution or not solution.get("routes"):
        return m
    
    routes_layer = FeatureGroup(name="Rescue Routes")
    
    for route_idx, route in enumerate(solution["routes"]):
        route_color = ["#00FF00", "#00CCFF", "#FF00FF", "#FFFF00"][route_idx % 4]
        
        for stop_idx, stop in enumerate(route):
            if stop["type"] == "depot":
                # Depot marker
                Marker(
                    (stop["location"]["lat"], stop["location"]["lon"]),
                    popup="🚒 Rescue Depot<br>Start Point",
                    icon=folium.Icon(color="green", icon="home", prefix="fa")
                ).add_to(routes_layer)
                
            elif stop["type"] == "target":
                # Target marker
                Marker(
                    (stop["location"]["lat"], stop["location"]["lon"]),
                    popup=f"🎯 Target {stop_idx}<br>"
                          f"Distance: {stop.get('distance', 0):.0f}m",
                    icon=folium.Icon(color="red", icon="flag", prefix="fa")
                ).add_to(routes_layer)
                
                # Draw path if available
                if "path" in stop and len(stop["path"]) > 1:
                    # Get node coordinates
                    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
                    path_coords = []
                    
                    for node_id in stop["path"]:
                        if node_id in nodes_gdf.index:
                            node = nodes_gdf.loc[node_id]
                            path_coords.append((node.geometry.y, node.geometry.x))
                    
                    if path_coords:
                        PolyLine(
                            path_coords,
                            color=route_color,
                            weight=4,
                            opacity=0.9,
                            dash_array="10"
                        ).add_to(routes_layer)
    
    # Add unreachable targets
    for target in solution.get("unreachable", []):
        Marker(
            (target["lat"], target["lon"]),
            popup="⚠️ UNREACHABLE<br>No safe route",
            icon=folium.Icon(color="gray", icon="exclamation-triangle", prefix="fa")
        ).add_to(routes_layer)
    
    routes_layer.add_to(m)
    return m


def add_legend(m: folium.Map, agents_active: List[str]) -> folium.Map:
    """Add a legend explaining the map layers."""
    legend_html = '''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000; 
                background: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-size: 12px;
                max-width: 300px;">
        <h4 style="margin: 0 0 10px 0;">🤖 Ryudo Agent System</h4>
        <hr style="margin: 5px 0;">
        <p style="margin: 3px 0;"><b>FloodSentinel</b> (Environmental)</p>
        <p style="margin: 3px 0; padding-left: 10px;"><span style="color: #FF0000;">●</span> Extreme - Impassable</p>
        <p style="margin: 3px 0; padding-left: 10px;"><span style="color: #FF8C00;">●</span> Severe - 10x slower</p>
        <p style="margin: 3px 0; padding-left: 10px;"><span style="color: #FFD700;">●</span> Moderate - 3x slower</p>
        <p style="margin: 3px 0; padding-left: 10px;"><span style="color: #0066FF;">●</span> Storm Surge</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 3px 0;"><b>GridGuardian</b> (Infrastructure)</p>
        <p style="margin: 3px 0; padding-left: 10px;">⚡ Failed Substations</p>
        <p style="margin: 3px 0; padding-left: 10px;">🏥 Facilities (green=active, gray=offline)</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 3px 0;"><b>RoutePilot</b> (Temporal)</p>
        <p style="margin: 3px 0; padding-left: 10px;">⏱️ TTL warnings on routes</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 3px 0;"><b>Rescue Routes</b></p>
        <p style="margin: 3px 0; padding-left: 10px;"><span style="color: #00FF00;">━━</span> Computed paths</p>
    </div>
    '''
    m.get_root().html.add_child(Element(legend_html))
    return m


def add_title(m: folium.Map, title: str, subtitle: str = "") -> folium.Map:
    """Add a title overlay to the map."""
    title_html = f'''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 1000; 
                background: white; padding: 10px 15px; border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
        <h4 style="margin: 0;">🧠 {title}</h4>
        <p style="margin: 5px 0 0 0; font-size: 12px; color: #666;">{subtitle}</p>
    </div>
    '''
    m.get_root().html.add_child(Element(title_html))
    return m


def render_agent_map(
    workflow_result: Dict[str, Any],
    output_path: str = "output/agent_workflow_map.html",
    title: str = "Ryudo LangGraph Agents",
    show_roads: bool = True,
) -> folium.Map:
    """
    Render the complete agent workflow result as an interactive map.
    
    Parameters
    ----------
    workflow_result : dict
        The result from run_ryudo_workflow()
    output_path : str
        Where to save the HTML map
    title : str
        Map title
    show_roads : bool
        Whether to show the road network
        
    Returns
    -------
    folium.Map
        The rendered map object
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create base map
    m = create_base_map()
    
    # Add visualization layers from agents
    viz_layers = workflow_result.get("visualization_layers", [])
    if viz_layers:
        m = add_visualization_layers(m, viz_layers)
    
    # Add modified road network
    modified_graph = workflow_result.get("modified_graph")
    if show_roads and modified_graph:
        m = add_road_network(m, modified_graph, name="Modified Road Network")
    
    # Add solution routes
    solution = workflow_result.get("solution")
    if solution and modified_graph:
        m = add_solution_routes(m, solution, modified_graph)
    
    # Add legend and title
    m = add_legend(m, ["FloodSentinel", "GridGuardian", "RoutePilot"])
    
    stats = workflow_result.get("coordinator_stats", {})
    subtitle = f"Zones: {stats.get('zones_deleted', 0)} | Facilities offline: {stats.get('nodes_disabled', 0)}"
    m = add_title(m, title, subtitle)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save
    m.save(output_path)
    print(f"✅ Map saved to: {output_path}")
    
    return m


def create_solution_layers(solution: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create visualization layers from a routing solution."""
    layers = []
    
    if not solution:
        return layers
    
    for route in solution.get("routes", []):
        for stop in route:
            if stop["type"] == "depot":
                layers.append({
                    "type": "marker",
                    "agent": "Coordinator",
                    "name": "Depot",
                    "position": (stop["location"]["lat"], stop["location"]["lon"]),
                    "popup": "🚒 Rescue Depot",
                    "icon": {"color": "green", "icon": "home", "prefix": "fa"}
                })
            elif stop["type"] == "target":
                layers.append({
                    "type": "marker",
                    "agent": "Coordinator",
                    "name": f"Target",
                    "position": (stop["location"]["lat"], stop["location"]["lon"]),
                    "popup": f"🎯 Rescue Target",
                    "icon": {"color": "red", "icon": "flag", "prefix": "fa"}
                })
    
    return layers
