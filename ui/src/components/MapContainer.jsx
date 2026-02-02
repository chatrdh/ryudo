import { useEffect, useRef, useMemo, useState, useCallback } from 'react';
import { MapContainer as LeafletMap, TileLayer, ZoomControl, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-ant-path';
import './MapContainer.css';
import { useGeoData, getRoadStyle, getWaterStyle, getBuildingStyle, getLandUseStyle } from '../hooks/useGeoData';

// Debounce utility for batching updates
function useDebounce(value, delay) {
    const [debouncedValue, setDebouncedValue] = useState(value);

    useEffect(() => {
        const timer = setTimeout(() => setDebouncedValue(value), delay);
        return () => clearTimeout(timer);
    }, [value, delay]);

    return debouncedValue;
}

// Fix default marker icon paths for Leaflet in bundlers
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
    iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Visakhapatnam center coordinates
const DEFAULT_CENTER = [17.6868, 83.2185];
const DEFAULT_ZOOM = 12;

// Available basemaps
const BASEMAPS = {
    'dark': {
        name: 'Dark',
        url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attribution: '© CartoDB'
    },
    'light': {
        name: 'Light',
        url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
        attribution: '© CartoDB'
    },
    'osm': {
        name: 'OpenStreetMap',
        url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attribution: '© OpenStreetMap'
    },
    'satellite': {
        name: 'Satellite',
        url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attribution: '© Esri'
    },
    'terrain': {
        name: 'Terrain',
        url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attribution: '© OpenTopoMap'
    }
};

// Marker icon configurations
const MARKER_CONFIGS = {
    cyclone: { emoji: '🌀', bg: 'transparent', color: '#000' },
    power_off: { emoji: '⚡', bg: '#ef4444', color: '#fff' },
    offline: { emoji: '🏥', bg: '#52525b', color: '#a1a1aa' },
    active: { emoji: '🏥', bg: '#10b981', color: '#fff' },
    depot: { emoji: '🚒', bg: '#3b82f6', color: '#fff' },
    target_ok: { emoji: '✓', bg: '#22c55e', color: '#fff' },
    target_blocked: { emoji: '✗', bg: '#ef4444', color: '#fff' },
    default: { emoji: '📍', bg: '#fff', color: '#000' },
};

function createMarkerIcon(iconType) {
    const config = MARKER_CONFIGS[iconType] || MARKER_CONFIGS.default;
    return L.divIcon({
        className: 'custom-div-icon',
        html: `<div class="custom-icon-marker" style="width: 32px; height: 32px; background: ${config.bg}; color: ${config.color};">${config.emoji}</div>`,
        iconSize: [32, 32],
        iconAnchor: [16, 16],
    });
}

// Basemap switch component
function BasemapSwitcher({ current, onChange }) {
    return (
        <div className="basemap-switcher">
            <div className="layer-control-header">🗺️ Basemap</div>
            {Object.entries(BASEMAPS).map(([key, { name }]) => (
                <label key={key} className="layer-control-item">
                    <input
                        type="radio"
                        name="basemap"
                        checked={current === key}
                        onChange={() => onChange(key)}
                    />
                    <span>{name}</span>
                </label>
            ))}
        </div>
    );
}

// Geographic layers control component
function GeoLayerControl({ layers, onToggle }) {
    return (
        <div className="geo-layer-control">
            <div className="layer-control-header">📍 Geographic Layers</div>
            {Object.entries(layers).map(([name, { visible, label, count }]) => (
                <label key={name} className="layer-control-item">
                    <input
                        type="checkbox"
                        checked={visible}
                        onChange={() => onToggle(name)}
                    />
                    <span>{label}</span>
                    {count !== undefined && <span className="layer-count">({count})</span>}
                </label>
            ))}
        </div>
    );
}

// Agent layers control component
function AgentLayerControl({ layers, onToggle }) {
    return (
        <div className="agent-layer-control">
            <div className="layer-control-header">🤖 Agent Layers</div>
            {Object.entries(layers).map(([name, { visible, label }]) => (
                <label key={name} className="layer-control-item">
                    <input
                        type="checkbox"
                        checked={visible}
                        onChange={() => onToggle(name)}
                    />
                    <span>{label}</span>
                </label>
            ))}
        </div>
    );
}

// Layer control panel combining all controls
function LayerControlPanel({
    basemap,
    onBasemapChange,
    geoLayers,
    onGeoToggle,
    agentLayers,
    onAgentToggle,
    loading
}) {
    return (
        <div className="layer-control-panel">
            {loading && (
                <div className="loading-indicator">
                    <span className="loading-spinner">⏳</span> Loading map data...
                </div>
            )}
            <BasemapSwitcher current={basemap} onChange={onBasemapChange} />
            <GeoLayerControl layers={geoLayers} onToggle={onGeoToggle} />
            <AgentLayerControl layers={agentLayers} onToggle={onAgentToggle} />
        </div>
    );
}

// Geographic data layers renderer
function GeoDataLayers({ geoData, visibility }) {
    const map = useMap();

    if (!geoData.loaded) return null;

    return (
        <>
            {/* Land Use Layer (bottom) */}
            {visibility.landuse?.visible && geoData.landuse && (
                <GeoJSON
                    key="landuse"
                    data={geoData.landuse}
                    style={(feature) => getLandUseStyle(feature)}
                    onEachFeature={(feature, layer) => {
                        const props = feature.properties;
                        layer.bindPopup(`
                            <b>${props.classification?.name || 'Land Use'}</b><br/>
                            Type: ${props.landuse_type}<br/>
                            ${props.name ? `Name: ${props.name}<br/>` : ''}
                            Population: ${props.agent_metadata?.population_density || 'unknown'}
                        `);
                    }}
                />
            )}

            {/* Water Layer */}
            {visibility.water?.visible && geoData.water && (
                <GeoJSON
                    key="water"
                    data={geoData.water}
                    style={(feature) => getWaterStyle(feature)}
                    onEachFeature={(feature, layer) => {
                        const props = feature.properties;
                        layer.bindPopup(`
                            <b>${props.classification?.name || 'Water'}</b><br/>
                            Type: ${props.water_type}<br/>
                            ${props.name ? `Name: ${props.name}<br/>` : ''}
                            Flood Risk: ${props.agent_metadata?.flood_risk || 'unknown'}
                        `);
                    }}
                />
            )}

            {/* Buildings Layer */}
            {visibility.buildings?.visible && geoData.buildings && (
                <GeoJSON
                    key="buildings"
                    data={geoData.buildings}
                    style={(feature) => getBuildingStyle(feature)}
                    onEachFeature={(feature, layer) => {
                        const props = feature.properties;
                        layer.bindPopup(`
                            <b>${props.classification?.name || 'Building'}</b><br/>
                            ${props.name ? `Name: ${props.name}<br/>` : ''}
                            Type: ${props.building_type}<br/>
                            Priority: ${props.agent_metadata?.evacuation_priority || 'unknown'}
                        `);
                    }}
                />
            )}

            {/* Roads Layer (top of geo layers) */}
            {visibility.roads?.visible && geoData.roads && (
                <GeoJSON
                    key="roads"
                    data={geoData.roads}
                    style={(feature) => getRoadStyle(feature)}
                    onEachFeature={(feature, layer) => {
                        const props = feature.properties;
                        layer.bindPopup(`
                            <b>${props.name || 'Road'}</b><br/>
                            Type: ${props.classification?.name || props.highway_type}<br/>
                            Speed: ${props.maxspeed || 'unknown'} km/h<br/>
                            Evacuation: ${props.agent_metadata?.evacuation_priority || 'unknown'}
                        `);
                    }}
                />
            )}
        </>
    );
}

function MapContent({ constraints, markers, routes, onMapReady, geoLayers, agentLayers, geoData }) {
    const map = useMap();
    const layersRef = useRef({
        zones: L.layerGroup(),
        markers: L.layerGroup(),
        routes: L.layerGroup(),
    });
    const antPathsRef = useRef([]);
    const allMarkersRef = useRef([]);

    // Initialize layers on mount
    useEffect(() => {
        Object.values(layersRef.current).forEach(layer => layer.addTo(map));
        onMapReady?.(map);

        return () => {
            Object.values(layersRef.current).forEach(layer => {
                layer.clearLayers();
                map.removeLayer(layer);
            });
            antPathsRef.current.forEach(path => map.removeLayer(path));
        };
    }, [map, onMapReady]);

    // Toggle layer visibility
    useEffect(() => {
        Object.entries(agentLayers).forEach(([name, { visible }]) => {
            const layer = layersRef.current[name];
            if (layer) {
                if (visible && !map.hasLayer(layer)) {
                    layer.addTo(map);
                } else if (!visible && map.hasLayer(layer)) {
                    map.removeLayer(layer);
                }
            }
        });
        // Handle routes layer separately (antPaths)
        if (agentLayers.routes) {
            antPathsRef.current.forEach(path => {
                if (agentLayers.routes.visible && !map.hasLayer(path)) {
                    path.addTo(map);
                } else if (!agentLayers.routes.visible && map.hasLayer(path)) {
                    map.removeLayer(path);
                }
            });
        }
    }, [agentLayers, map]);

    // Handle constraints (flood zones, etc.)
    useEffect(() => {
        layersRef.current.zones.clearLayers();

        constraints.forEach(c => {
            if (c.action === 'delete_zone' && c.geometry) {
                const defaultStyle = {
                    color: c.style?.color || '#ef4444',
                    fillColor: c.style?.fillColor || '#ef4444',
                    weight: 2,
                    fillOpacity: 0.12,
                    opacity: 0.7,
                    dashArray: '8, 4',
                };

                L.geoJSON(c.geometry, {
                    style: defaultStyle,
                    onEachFeature: (feature, layer) => {
                        layer.bindPopup(`<b>${c.reason || 'Restricted Zone'}</b>`);
                    },
                }).addTo(layersRef.current.zones);
            }
        });
    }, [constraints]);

    // Handle markers and fit bounds
    useEffect(() => {
        layersRef.current.markers.clearLayers();
        allMarkersRef.current = [];

        markers.forEach(m => {
            const icon = createMarkerIcon(m.icon);
            L.marker(m.position, { icon })
                .bindPopup(m.popup || '')
                .addTo(layersRef.current.markers);

            allMarkersRef.current.push(m.position);
        });

        if (allMarkersRef.current.length > 2) {
            try {
                const bounds = L.latLngBounds(allMarkersRef.current);
                map.fitBounds(bounds, {
                    padding: [60, 60],
                    maxZoom: 14,
                    animate: true,
                    duration: 0.5
                });
            } catch (e) {
                // Ignore bounds errors
            }
        }
    }, [markers, map]);

    // Handle routes with AntPath animation - supports multiple vehicle routes
    useEffect(() => {
        antPathsRef.current.forEach(path => {
            if (map.hasLayer(path)) map.removeLayer(path);
        });
        antPathsRef.current = [];

        routes.forEach((route, idx) => {
            if (route.coords?.length > 0 || route.route?.length > 0) {
                const coords = route.coords || route.route;
                const color = route.color || '#6366f1';

                const antPath = L.polyline.antPath(coords, {
                    delay: 400 + (idx * 100), // Stagger animation
                    dashArray: [10, 20],
                    weight: 5,
                    color: color,
                    pulseColor: '#ffffff',
                    paused: false,
                    reverse: false,
                    hardwareAccelerated: true,
                });

                // Add popup with route info if available
                if (route.popup) {
                    antPath.bindPopup(route.popup);
                } else if (route.vehicle_name) {
                    let popup = `<b>🚗 ${route.vehicle_name}</b><br>`;
                    popup += `📏 Distance: ${route.total_distance_km || 0} km<br>`;
                    popup += `⏱️ Time: ${route.travel_time_min || 0} min<br>`;
                    if (route.target_names?.length) {
                        popup += `🎯 Targets: ${route.target_names.join(', ')}`;
                    }
                    antPath.bindPopup(popup);
                }

                antPath.addTo(map);
                antPathsRef.current.push(antPath);
            }
        });

        // Fit bounds to all routes if we have multiple
        if (routes.length > 0 && routes.some(r => (r.coords?.length || r.route?.length) > 1)) {
            const allCoords = routes.flatMap(r => r.coords || r.route || []);
            if (allCoords.length > 1) {
                try {
                    map.fitBounds(allCoords, { padding: [50, 50], maxZoom: 14 });
                } catch (e) {
                    // Ignore bounds errors
                }
            }
        }
    }, [routes, map]);

    return (
        <GeoDataLayers geoData={geoData} visibility={geoLayers} />
    );
}

export function MapView({ constraints = [], markers = [], routes = [], onMapReady }) {
    // Fetch geographic data
    const geoData = useGeoData();

    // Debounce rapid updates to prevent map stuttering
    const debouncedConstraints = useDebounce(constraints, 100);
    const debouncedMarkers = useDebounce(markers, 100);

    // Basemap state
    const [basemap, setBasemap] = useState('dark');

    // Geographic layers state with feature counts (disabled by default for performance)
    const [geoLayers, setGeoLayers] = useState({
        roads: { visible: false, label: '🛣️ Roads', count: undefined },
        water: { visible: false, label: '💧 Water', count: undefined },
        buildings: { visible: false, label: '🏢 Buildings', count: undefined },
        landuse: { visible: false, label: '🌳 Land Use', count: undefined },
    });

    // Update counts when data loads
    useEffect(() => {
        if (geoData.summary) {
            setGeoLayers(prev => ({
                roads: { ...prev.roads, count: geoData.summary.total_roads },
                water: { ...prev.water, count: geoData.summary.total_water_features },
                buildings: { ...prev.buildings, count: geoData.summary.total_buildings },
                landuse: { ...prev.landuse, count: geoData.summary.total_landuse_zones },
            }));
        }
    }, [geoData.summary]);

    // Agent layers state (existing functionality)
    const [agentLayers, setAgentLayers] = useState({
        zones: { visible: true, label: '🔴 Damage Zones' },
        markers: { visible: true, label: '📍 Markers' },
        routes: { visible: true, label: '🚗 Routes' },
    });

    const toggleGeoLayer = useCallback((name) => {
        setGeoLayers(prev => ({
            ...prev,
            [name]: { ...prev[name], visible: !prev[name].visible }
        }));
    }, []);

    const toggleAgentLayer = useCallback((name) => {
        setAgentLayers(prev => ({
            ...prev,
            [name]: { ...prev[name], visible: !prev[name].visible }
        }));
    }, []);

    const mapOptions = useMemo(() => ({
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        zoomControl: false,
        attributionControl: false,
        scrollWheelZoom: true,
        doubleClickZoom: true,
        dragging: true,
        preferCanvas: true,
    }), []);

    const currentBasemap = BASEMAPS[basemap];

    return (
        <div className="map-wrapper">
            <LayerControlPanel
                basemap={basemap}
                onBasemapChange={setBasemap}
                geoLayers={geoLayers}
                onGeoToggle={toggleGeoLayer}
                agentLayers={agentLayers}
                onAgentToggle={toggleAgentLayer}
                loading={geoData.loading}
            />
            <LeafletMap {...mapOptions} className="map-container">
                <TileLayer
                    key={basemap}
                    url={currentBasemap.url}
                    maxZoom={19}
                />
                <ZoomControl position="bottomright" />
                <MapContent
                    constraints={debouncedConstraints}
                    markers={debouncedMarkers}
                    routes={routes}
                    onMapReady={onMapReady}
                    geoLayers={geoLayers}
                    agentLayers={agentLayers}
                    geoData={geoData}
                />
            </LeafletMap>
        </div>
    );
}
