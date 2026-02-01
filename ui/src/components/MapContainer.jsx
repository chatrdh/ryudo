import { useEffect, useRef, useMemo } from 'react';
import { MapContainer as LeafletMap, TileLayer, ZoomControl, useMap } from 'react-leaflet';
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

function MapContent({ constraints, markers, routes, onMapReady }) {
    const map = useMap();
    const layersRef = useRef({
        flood: L.layerGroup(),
        grid: L.layerGroup(),
        temporal: L.layerGroup(),
        mission: L.layerGroup(),
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

    // Handle constraints (flood zones, etc.)
    useEffect(() => {
        constraints.forEach(c => {
            if (c.action === 'delete_zone' && c.geometry) {
                L.geoJSON(c.geometry, {
                    style: c.style || {
                        color: '#ef4444',
                        weight: 2,
                        fillOpacity: 0.3,
                    },
                    onEachFeature: (feature, layer) => {
                        layer.bindPopup(`<b>${c.reason || 'Restricted Zone'}</b>`);
                    },
                }).addTo(layersRef.current.flood);
            }
        });
    }, [constraints]);

    // Handle markers and fit bounds
    useEffect(() => {
        markers.forEach(m => {
            const icon = createMarkerIcon(m.icon);
            const layerName = m.agent?.toLowerCase().includes('flood') ? 'flood'
                : m.agent?.toLowerCase().includes('grid') ? 'grid'
                    : m.agent?.toLowerCase().includes('mission') ? 'mission'
                        : 'routes';

            L.marker(m.position, { icon })
                .bindPopup(m.popup || '')
                .addTo(layersRef.current[layerName]);

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
        routes.forEach(route => {
            if (route.coords?.length > 0) {
                // Clear previous ant paths
                antPathsRef.current.forEach(path => map.removeLayer(path));
                antPathsRef.current = [];

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
    const mapOptions = useMemo(() => ({
        center: DEFAULT_CENTER,
        zoom: DEFAULT_ZOOM,
        zoomControl: false,  // We'll add our own positioned control
        attributionControl: false,
        scrollWheelZoom: true,
        doubleClickZoom: true,
        dragging: true,
    }), []);

    return (
        <div className="map-wrapper">
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
                />
            </LeafletMap>
        </div>
    );
}

