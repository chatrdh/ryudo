"""
Ryudo Phase 2: Realistic Cyclone Simulation
============================================
Simulates Cyclone Hudhud (2014) impact on Visakhapatnam road network.

Uses the Holland Wind Profile Model to calculate:
- Wind speed decay from cyclone eye
- Storm surge / flood zones
- Road network damage assessment

Cyclone Hudhud Data (IMD, Oct 12, 2014):
- Landfall: 17.4°N, 83.8°E near Visakhapatnam
- Max sustained wind: 185 km/h (3-min avg)
- Central pressure: 950 mbar
- Ambient pressure: 1013 mbar
"""

import osmnx as ox
import networkx as nx
import folium
from folium.plugins import HeatMap
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import geopandas as gpd
import math

# ============================================================================
# CYCLONE HUDHUD PARAMETERS (October 12, 2014)
# ============================================================================

CYCLONE_CONFIG = {
    "name": "Cyclone Hudhud",
    "date": "2014-10-12",
    
    # Landfall position - adjusted slightly west toward city for demo
    # Actual landfall was 17.4°N, 83.8°E (east of city)
    # Using 17.68°N, 83.35°E (closer to city center for better visualization)
    "eye_lat": 17.68,
    "eye_lon": 83.35,
    
    # Meteorological parameters
    "max_wind_speed_kmh": 185,      # km/h (IMD 3-min sustained)
    "central_pressure_mb": 950,      # mbar
    "ambient_pressure_mb": 1013,     # mbar
    "radius_max_wind_km": 25,        # Rmax - estimated
    
    # Impact zones (km from eye) - adjusted for city-scale visualization
    "extreme_damage_radius_km": 10,   # >150 km/h winds - eye wall
    "severe_damage_radius_km": 25,    # >100 km/h winds
    "moderate_damage_radius_km": 50,  # >60 km/h winds
    
    # Storm surge zone (coastal)
    "storm_surge_height_m": 2.5,
    "surge_inland_distance_km": 2,
}

# City center for map
VIZAG_CENTER = (17.6868, 83.2185)


# ============================================================================
# HOLLAND WIND PROFILE MODEL
# ============================================================================

def holland_wind_speed(r_km, Rmax_km, Vmax_kmh, Pc_mb, Pn_mb, lat=17.0):
    """
    Calculate wind speed at distance r from cyclone center using simplified Holland model.
    
    Uses the empirical decay formula based on Holland (1980).
    """
    if r_km <= 0:
        return Vmax_kmh
    
    if r_km <= Rmax_km:
        # Inside Rmax - linear increase to max
        return Vmax_kmh * (r_km / Rmax_km) ** 0.5
    
    # Outside Rmax - exponential decay
    # V(r) = Vmax * (Rmax/r)^x where x depends on B parameter
    B = 1.5  # Typical value for Bay of Bengal cyclones
    
    decay = (Rmax_km / r_km) ** B
    wind = Vmax_kmh * decay
    
    return max(0, wind)


def calculate_wind_field(center_lat, center_lon, config, grid_size=0.01, extent=2.0):
    """
    Generate a wind speed grid around the cyclone center.
    
    Returns:
    --------
    list of [lat, lon, wind_speed] for heatmap visualization
    """
    wind_data = []
    
    Rmax = config["radius_max_wind_km"]
    Vmax = config["max_wind_speed_kmh"]
    Pc = config["central_pressure_mb"]
    Pn = config["ambient_pressure_mb"]
    
    for lat in np.arange(center_lat - extent, center_lat + extent, grid_size):
        for lon in np.arange(center_lon - extent, center_lon + extent, grid_size):
            # Distance from eye (approximate using Haversine)
            r_km = haversine_distance(center_lat, center_lon, lat, lon)
            
            # Calculate wind speed
            wind = holland_wind_speed(r_km, Rmax, Vmax, Pc, Pn, lat)
            
            if wind > 30:  # Only include significant winds
                # Normalize for heatmap (0-1 scale)
                intensity = min(1.0, wind / Vmax)
                wind_data.append([lat, lon, intensity])
    
    return wind_data


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


# ============================================================================
# DAMAGE ZONE GENERATION
# ============================================================================

def create_damage_zones(center_lat, center_lon, config):
    """
    Create circular damage zones around cyclone eye.
    
    Returns:
    --------
    dict of zone_name -> shapely Polygon
    """
    center = Point(center_lon, center_lat)
    
    # Create buffer zones (in degrees, approximate)
    # 1 degree ≈ 111 km at equator
    km_to_deg = 1 / 111.0
    
    zones = {
        "extreme": center.buffer(config["extreme_damage_radius_km"] * km_to_deg),
        "severe": center.buffer(config["severe_damage_radius_km"] * km_to_deg),
        "moderate": center.buffer(config["moderate_damage_radius_km"] * km_to_deg),
    }
    
    return zones


def create_storm_surge_zone(center_lat, center_lon, config):
    """
    Create storm surge flood zone along the coast.
    
    For simplicity, we create a coastal buffer zone.
    In production, this would use DEM and coastal geometry.
    """
    # Approximate coastline near Visakhapatnam (simplified)
    # The coast runs roughly NE-SW
    coastal_points = [
        (83.1, 17.4),
        (83.2, 17.5),
        (83.3, 17.6),
        (83.4, 17.7),
        (83.5, 17.8),
    ]
    
    # Create coastal buffer
    from shapely.geometry import LineString
    coastline = LineString(coastal_points)
    
    surge_distance = config["surge_inland_distance_km"] / 111.0  # km to degrees
    surge_zone = coastline.buffer(surge_distance)
    
    return surge_zone


# ============================================================================
# ROAD NETWORK DAMAGE ASSESSMENT
# ============================================================================

def assess_road_damage(G, damage_zones, surge_zone, config=CYCLONE_CONFIG):
    """
    Assess damage to road network based on distance from cyclone eye.
    
    Uses distance-based classification for proper graduated damage levels.
    """
    # Get edges as GeoDataFrame
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    
    eye_lat = config["eye_lat"]
    eye_lon = config["eye_lon"]
    
    stats = {
        "total_edges": len(edges_gdf),
        "extreme_damage": 0,
        "severe_damage": 0,
        "moderate_damage": 0,
        "flooded": 0,
        "unaffected": 0,
    }
    
    # Damage thresholds (km from eye)
    extreme_radius = config["extreme_damage_radius_km"]
    severe_radius = config["severe_damage_radius_km"]
    moderate_radius = config["moderate_damage_radius_km"]
    
    # Assign damage levels to edges
    edge_damage = {}
    
    for idx, row in edges_gdf.iterrows():
        geom = row.geometry
        
        # Get centroid of edge for distance calculation
        centroid = geom.centroid
        edge_lat = centroid.y
        edge_lon = centroid.x
        
        # Calculate distance from cyclone eye
        dist_km = haversine_distance(eye_lat, eye_lon, edge_lat, edge_lon)
        
        # Check storm surge first (coastal flooding)
        if geom.intersects(surge_zone):
            damage_level = "flooded"
            stats["flooded"] += 1
        # Then check distance-based damage zones (innermost to outermost)
        elif dist_km <= extreme_radius:
            damage_level = "extreme"
            stats["extreme_damage"] += 1
        elif dist_km <= severe_radius:
            damage_level = "severe"
            stats["severe_damage"] += 1
        elif dist_km <= moderate_radius:
            damage_level = "moderate"
            stats["moderate_damage"] += 1
        else:
            damage_level = "unaffected"
            stats["unaffected"] += 1
        
        edge_damage[idx] = damage_level
    
    return stats, edge_damage


def create_damaged_graph(G, edge_damage):
    """
    Create a modified graph with damaged edges removed or weighted.
    
    Damage effects:
    - flooded: Edge removed (impassable)
    - extreme: Edge removed (debris, trees)
    - severe: Edge weight x10 (slow, dangerous)
    - moderate: Edge weight x3 (caution)
    """
    G_damaged = G.copy()
    
    edges_to_remove = []
    
    for (u, v, k), damage_level in edge_damage.items():
        if damage_level in ["flooded", "extreme"]:
            edges_to_remove.append((u, v, k))
        elif damage_level == "severe":
            if G_damaged.has_edge(u, v, k):
                G_damaged[u][v][k]["length"] *= 10
                G_damaged[u][v][k]["damage"] = "severe"
        elif damage_level == "moderate":
            if G_damaged.has_edge(u, v, k):
                G_damaged[u][v][k]["length"] *= 3
                G_damaged[u][v][k]["damage"] = "moderate"
    
    # Remove impassable edges
    for u, v, k in edges_to_remove:
        if G_damaged.has_edge(u, v, k):
            G_damaged.remove_edge(u, v, k)
    
    return G_damaged


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_cyclone_map(G, config, damage_zones, surge_zone, edge_damage, wind_data):
    """
    Create an interactive map showing cyclone impact.
    """
    eye_lat = config["eye_lat"]
    eye_lon = config["eye_lon"]
    
    # Create base map
    m = folium.Map(
        location=VIZAG_CENTER,
        zoom_start=10,
        tiles="CartoDB positron"
    )
    
    # Add satellite layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        control=True
    ).add_to(m)
    
    # Add wind speed heatmap
    if wind_data:
        HeatMap(
            wind_data,
            name="Wind Speed Field",
            min_opacity=0.3,
            radius=15,
            blur=20,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}
        ).add_to(m)
    
    # Add damage zones
    zone_colors = {
        "extreme": "#FF0000",
        "severe": "#FF8C00", 
        "moderate": "#FFD700",
    }
    
    for zone_name, zone_poly in damage_zones.items():
        folium.GeoJson(
            zone_poly.__geo_interface__,
            name=f"{zone_name.title()} Damage Zone",
            style_function=lambda x, c=zone_colors[zone_name]: {
                'fillColor': c,
                'color': c,
                'weight': 2,
                'fillOpacity': 0.15
            }
        ).add_to(m)
    
    # Add storm surge zone
    folium.GeoJson(
        surge_zone.__geo_interface__,
        name="Storm Surge Zone",
        style_function=lambda x: {
            'fillColor': '#0066FF',
            'color': '#0033CC',
            'weight': 2,
            'fillOpacity': 0.4
        }
    ).add_to(m)
    
    # Add cyclone eye marker
    folium.Marker(
        [eye_lat, eye_lon],
        popup=f"<b>🌀 {config['name']}</b><br>"
               f"Landfall: {config['date']}<br>"
               f"Max Wind: {config['max_wind_speed_kmh']} km/h<br>"
               f"Pressure: {config['central_pressure_mb']} mb",
        icon=folium.Icon(color="red", icon="cloud", prefix="fa"),
    ).add_to(m)
    
    # Add road network with damage coloring
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    
    damage_colors = {
        "flooded": "#0066FF",
        "extreme": "#FF0000",
        "severe": "#FF8C00",
        "moderate": "#FFD700",
        "unaffected": "#00AA00",
    }
    
    # Create feature groups for each damage level
    road_layers = {level: folium.FeatureGroup(name=f"Roads - {level.title()}") 
                   for level in damage_colors.keys()}
    
    for idx, row in edges_gdf.iterrows():
        if idx in edge_damage:
            damage_level = edge_damage[idx]
            color = damage_colors.get(damage_level, "#888888")
            
            if row.geometry.geom_type == "LineString":
                coords = [(lat, lon) for lon, lat in row.geometry.coords]
                folium.PolyLine(
                    coords,
                    color=color,
                    weight=2 if damage_level == "unaffected" else 3,
                    opacity=0.7
                ).add_to(road_layers[damage_level])
    
    for layer in road_layers.values():
        layer.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title/legend
    legend_html = f'''
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000; 
                background: white; padding: 15px; border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.3); font-size: 12px;">
        <h4 style="margin: 0 0 10px 0;">🌀 {config['name']} Impact</h4>
        <p style="margin: 3px 0;"><span style="color: #FF0000;">●</span> Extreme Damage (impassable)</p>
        <p style="margin: 3px 0;"><span style="color: #FF8C00;">●</span> Severe Damage (10x slower)</p>
        <p style="margin: 3px 0;"><span style="color: #FFD700;">●</span> Moderate Damage (3x slower)</p>
        <p style="margin: 3px 0;"><span style="color: #0066FF;">●</span> Flooded (impassable)</p>
        <p style="margin: 3px 0;"><span style="color: #00AA00;">●</span> Unaffected</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    import os
    os.makedirs("output", exist_ok=True)
    
    print("=" * 60)
    print("RYUDO Phase 2: Cyclone Hudhud Simulation")
    print("=" * 60)
    
    config = CYCLONE_CONFIG
    
    print(f"\n🌀 Cyclone: {config['name']} ({config['date']})")
    print(f"   Eye Position: {config['eye_lat']}°N, {config['eye_lon']}°E")
    print(f"   Max Wind: {config['max_wind_speed_kmh']} km/h")
    print(f"   Central Pressure: {config['central_pressure_mb']} mb")
    
    # Step 1: Load road network from Phase 1 or download fresh
    print("\n[1/5] Loading road network...")
    try:
        G = ox.graph_from_place("Visakhapatnam, India", network_type="drive")
        print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    except Exception as e:
        print(f"   Error loading network: {e}")
        return
    
    # Step 2: Generate wind field data
    print("\n[2/5] Calculating wind field (Holland model)...")
    wind_data = calculate_wind_field(
        config["eye_lat"], 
        config["eye_lon"], 
        config,
        grid_size=0.02,
        extent=1.5
    )
    print(f"   Wind data points: {len(wind_data)}")
    
    # Step 3: Create damage zones
    print("\n[3/5] Creating damage zones...")
    damage_zones = create_damage_zones(config["eye_lat"], config["eye_lon"], config)
    surge_zone = create_storm_surge_zone(config["eye_lat"], config["eye_lon"], config)
    print(f"   Extreme zone: {config['extreme_damage_radius_km']} km radius")
    print(f"   Severe zone: {config['severe_damage_radius_km']} km radius")
    print(f"   Moderate zone: {config['moderate_damage_radius_km']} km radius")
    
    # Step 4: Assess road damage
    print("\n[4/5] Assessing road network damage...")
    stats, edge_damage = assess_road_damage(G, damage_zones, surge_zone)
    
    print(f"\n   📊 DAMAGE ASSESSMENT:")
    print(f"   ├─ Total road segments: {stats['total_edges']}")
    print(f"   ├─ Flooded (impassable): {stats['flooded']} ({100*stats['flooded']/stats['total_edges']:.1f}%)")
    print(f"   ├─ Extreme damage: {stats['extreme_damage']} ({100*stats['extreme_damage']/stats['total_edges']:.1f}%)")
    print(f"   ├─ Severe damage: {stats['severe_damage']} ({100*stats['severe_damage']/stats['total_edges']:.1f}%)")
    print(f"   ├─ Moderate damage: {stats['moderate_damage']} ({100*stats['moderate_damage']/stats['total_edges']:.1f}%)")
    print(f"   └─ Unaffected: {stats['unaffected']} ({100*stats['unaffected']/stats['total_edges']:.1f}%)")
    
    # Step 5: Create visualization
    print("\n[5/5] Generating cyclone impact map...")
    m = create_cyclone_map(G, config, damage_zones, surge_zone, edge_damage, wind_data)
    
    output_file = "output/phase2_cyclone_simulation.html"
    m.save(output_file)
    print(f"\n✅ Cyclone simulation map saved to: {output_file}")
    
    # Create damaged graph for routing
    G_damaged = create_damaged_graph(G, edge_damage)
    print(f"\n📍 Damaged Network Stats:")
    print(f"   Original edges: {G.number_of_edges()}")
    print(f"   Remaining edges: {G_damaged.number_of_edges()}")
    print(f"   Edges removed: {G.number_of_edges() - G_damaged.number_of_edges()}")
    
    return {
        "graph": G,
        "damaged_graph": G_damaged,
        "stats": stats,
        "output": output_file
    }


if __name__ == "__main__":
    result = main()
