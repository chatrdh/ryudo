import { useEffect, useRef, useMemo, useState } from 'react';
import { MapContainer as LeafletMap, TileLayer, ZoomControl, LayersControl, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import 'leaflet-ant-path';
import './MapContainer.css';

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

// Layer control component
function LayerControlPanel({ layers, onToggle }) {
    return (
        <div className="layer-control-panel">
            <div className="layer-control-header">Layers</div>
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

function MapContent({ constraints, markers, routes, onMapReady, layerVisibility }) {
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
        Object.entries(layerVisibility).forEach(([name, { visible }]) => {
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
        if (layerVisibility.routes) {
            antPathsRef.current.forEach(path => {
                if (layerVisibility.routes.visible && !map.hasLayer(path)) {
                    path.addTo(map);
                } else if (!layerVisibility.routes.visible && map.hasLayer(path)) {
                    map.removeLayer(path);
                }
            });
        }
    }, [layerVisibility, map]);

    // Handle constraints (flood zones, etc.) - with reduced opacity
    useEffect(() => {
        layersRef.current.zones.clearLayers();

        constraints.forEach(c => {
            if (c.action === 'delete_zone' && c.geometry) {
                // Use server style but ensure low opacity and dashed stroke for clarity
                const defaultStyle = {
                    color: c.style?.color || '#ef4444',
                    fillColor: c.style?.fillColor || '#ef4444',
                    weight: 2,
                    fillOpacity: 0.12,  // Very low fill opacity
                    opacity: 0.7,       // Border opacity
                    dashArray: '8, 4',  // Dashed border for clarity
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

        // Fit bounds to all markers if we have multiple
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

    // Handle routes with AntPath animation
    useEffect(() => {
        // Clear previous ant paths
        antPathsRef.current.forEach(path => {
            if (map.hasLayer(path)) map.removeLayer(path);
        });
        antPathsRef.current = [];

        routes.forEach(route => {
            if (route.coords?.length > 0) {
                const antPath = L.polyline.antPath(route.coords, {
                    delay: 400,
                    dashArray: [10, 20],
                    weight: 5,
                    color: '#6366f1',
                    pulseColor: '#ffffff',
                    paused: false,
                    reverse: false,
                    hardwareAccelerated: true,
                });

                antPath.addTo(map);
                antPathsRef.current.push(antPath);

                // Fit map to route bounds
                if (route.coords.length > 1) {
                    map.fitBounds(route.coords, { padding: [50, 50], maxZoom: 14 });
                }
            }
        });
    }, [routes, map]);

    return null;
}

export function MapView({ constraints = [], markers = [], routes = [], onMapReady }) {
    const [layerVisibility, setLayerVisibility] = useState({
        zones: { visible: true, label: '🔴 Damage Zones' },
        markers: { visible: true, label: '📍 Markers' },
        routes: { visible: true, label: '🛣️ Routes' },
    });

    const toggleLayer = (name) => {
        setLayerVisibility(prev => ({
            ...prev,
            [name]: { ...prev[name], visible: !prev[name].visible }
        }));
    };

    const mapOptions = useMemo(() => ({
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        zoomControl: false,
        attributionControl: false,
        scrollWheelZoom: true,
        doubleClickZoom: true,
        dragging: true,
    }), []);

    return (
        <div className="map-wrapper">
            <LayerControlPanel layers={layerVisibility} onToggle={toggleLayer} />
            <LeafletMap {...mapOptions} className="map-container">
                <TileLayer
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                    maxZoom={19}
                />
                <ZoomControl position="bottomright" />
                <MapContent
                    constraints={constraints}
                    markers={markers}
                    routes={routes}
                    onMapReady={onMapReady}
                    layerVisibility={layerVisibility}
                />
            </LeafletMap>
        </div>
    );
}


