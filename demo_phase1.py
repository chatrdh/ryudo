"""
Ryudo Phase 1: Base Map Visualization
=====================================
Demonstrates OSMnx road network extraction with Folium + Bhuvan WMS overlay.
"""

import osmnx as ox
import folium
from folium.raster_layers import WmsTileLayer

# Configuration
PLACE = "Visakhapatnam, India"  # Cyclone-prone coastal city
CENTER_LAT = 17.6868
CENTER_LON = 83.2185
OUTPUT_FILE = "output/phase1_map.html"


def create_base_map():
    """Create a Folium map with multiple tile layers."""
    m = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=12,
        tiles=None  # We'll add custom tiles
    )
    
    # Layer 1: OpenStreetMap base
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        control=True
    ).add_to(m)
    
    # Layer 2: Satellite view (for reference)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite View",
        control=True
    ).add_to(m)
    
    # Layer 3: Bhuvan LULC (Land Use Land Cover) - ISRO
    try:
        WmsTileLayer(
            url="https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms",
            layers="lulc50k_1112",
            fmt="image/png",
            transparent=True,
            name="Bhuvan LULC",
            attr="ISRO Bhuvan",
            control=True
        ).add_to(m)
    except Exception as e:
        print(f"Note: Bhuvan WMS may require network access: {e}")
    
    return m


def extract_road_network():
    """Extract road network from OpenStreetMap using OSMnx."""
    print(f"Downloading road network for {PLACE}...")
    
    # Get the road network graph
    G = ox.graph_from_place(PLACE, network_type="drive")
    
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    
    return G


def extract_water_features():
    """Extract water bodies and waterways."""
    print(f"Downloading water features for {PLACE}...")
    
    try:
        water = ox.features_from_place(
            PLACE, 
            tags={"natural": "water"}
        )
        waterways = ox.features_from_place(
            PLACE, 
            tags={"waterway": True}
        )
        print(f"  Water bodies: {len(water)}")
        print(f"  Waterways: {len(waterways)}")
        return water, waterways
    except Exception as e:
        print(f"  Could not fetch water features: {e}")
        return None, None


def add_network_to_map(m, G, color="blue", weight=1, opacity=0.6):
    """Add road network edges to Folium map."""
    # Convert graph to GeoDataFrame
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    
    # Create a feature group for roads
    road_layer = folium.FeatureGroup(name="Road Network")
    
    for _, row in edges_gdf.iterrows():
        # Get coordinates
        if row.geometry.geom_type == "LineString":
            coords = [(lat, lon) for lon, lat in row.geometry.coords]
            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=opacity
            ).add_to(road_layer)
    
    road_layer.add_to(m)
    return m


def add_water_to_map(m, water_gdf, waterways_gdf):
    """Add water features to the map."""
    water_layer = folium.FeatureGroup(name="Water Bodies")
    
    if water_gdf is not None:
        for _, row in water_gdf.iterrows():
            if row.geometry is not None:
                try:
                    folium.GeoJson(
                        row.geometry.__geo_interface__,
                        style_function=lambda x: {
                            'fillColor': '#3388ff',
                            'color': '#0066cc',
                            'weight': 1,
                            'fillOpacity': 0.5
                        }
                    ).add_to(water_layer)
                except:
                    pass
    
    if waterways_gdf is not None:
        for _, row in waterways_gdf.iterrows():
            if row.geometry is not None and row.geometry.geom_type == "LineString":
                try:
                    coords = [(lat, lon) for lon, lat in row.geometry.coords]
                    folium.PolyLine(
                        coords,
                        color="#0066cc",
                        weight=2,
                        opacity=0.7
                    ).add_to(water_layer)
                except:
                    pass
    
    water_layer.add_to(m)
    return m


def main():
    """Main execution."""
    import os
    os.makedirs("output", exist_ok=True)
    
    print("=" * 50)
    print("RYUDO Phase 1: Base Map Visualization")
    print("=" * 50)
    
    # Step 1: Create base map with tile layers
    print("\n[1/4] Creating base map...")
    m = create_base_map()
    
    # Step 2: Extract road network
    print("\n[2/4] Extracting road network...")
    G = extract_road_network()
    
    # Step 3: Extract water features
    print("\n[3/4] Extracting water features...")
    water, waterways = extract_water_features()
    
    # Step 4: Add layers to map
    print("\n[4/4] Building visualization...")
    m = add_network_to_map(m, G, color="#FF6B6B", weight=2, opacity=0.7)
    if water is not None or waterways is not None:
        m = add_water_to_map(m, water, waterways)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50px; z-index: 1000; 
                background: white; padding: 10px; border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);">
        <h4 style="margin: 0;">🌀 Ryudo - Phase 1: Base Map</h4>
        <p style="margin: 5px 0 0 0; font-size: 12px;">Visakhapatnam Road Network</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Save
    m.save(OUTPUT_FILE)
    print(f"\n✅ Map saved to: {OUTPUT_FILE}")
    print("Open this file in a browser to view the interactive map!")
    
    # Return stats
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "output": OUTPUT_FILE
    }


if __name__ == "__main__":
    stats = main()
    print(f"\nNetwork Stats: {stats['nodes']} nodes, {stats['edges']} edges")
