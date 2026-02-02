import { useState, useEffect, useCallback } from 'react';

/**
 * Custom hook for fetching and managing geographic data.
 * 
 * Fetches all geographic layers (roads, water, buildings, landuse) on mount
 * and provides loading state and error handling.
 * 
 * Data includes rich metadata for AI agent processing:
 * - Each feature has agent_metadata with processing instructions
 * - Classification schemas define styling and priorities
 */
export function useGeoData() {
    const [data, setData] = useState({
        roads: null,
        water: null,
        buildings: null,
        landuse: null,
        summary: null,
        loaded: false
    });
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const fetchAllData = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            console.log('[GeoData] Fetching all geographic data...');
            const response = await fetch('/api/geo/all');

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            setData({
                roads: result.layers.roads,
                water: result.layers.water,
                buildings: result.layers.buildings,
                landuse: result.layers.landuse,
                summary: result.summary,
                agentGuide: result.agent_layer_guide,
                loaded: true
            });

            console.log('[GeoData] Data loaded:', result.summary);
        } catch (err) {
            console.error('[GeoData] Error fetching data:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchAllData();
    }, [fetchAllData]);

    // Reload function for manual refresh
    const reload = useCallback(() => {
        fetchAllData();
    }, [fetchAllData]);

    return {
        ...data,
        loading,
        error,
        reload
    };
}

/**
 * Get style for a road feature based on its highway type.
 */
export function getRoadStyle(feature) {
    const classification = feature.properties?.classification;
    if (!classification) {
        return {
            color: '#ffffff',
            weight: 1,
            opacity: 0.6
        };
    }

    return {
        color: classification.color || '#ffffff',
        weight: classification.weight || 1,
        opacity: 0.8
    };
}

/**
 * Get style for a water feature.
 */
export function getWaterStyle(feature) {
    const classification = feature.properties?.classification;
    const isLine = feature.geometry?.type === 'LineString' ||
        feature.geometry?.type === 'MultiLineString';

    if (isLine) {
        return {
            color: classification?.stroke_color || '#6699cc',
            weight: classification?.weight || 2,
            opacity: 0.8
        };
    }

    return {
        fillColor: classification?.color || '#aad3df',
        color: classification?.stroke_color || '#6699cc',
        weight: 1,
        fillOpacity: classification?.fill_opacity || 0.5,
        opacity: 0.8
    };
}

/**
 * Get style for a building feature.
 */
export function getBuildingStyle(feature) {
    const classification = feature.properties?.classification;

    return {
        fillColor: classification?.color || '#cccccc',
        color: '#666666',
        weight: 0.5,
        fillOpacity: 0.6,
        opacity: 0.8
    };
}

/**
 * Get style for a land use feature.
 */
export function getLandUseStyle(feature) {
    const classification = feature.properties?.classification;

    return {
        fillColor: classification?.color || '#dddddd',
        color: classification?.color || '#dddddd',
        weight: 1,
        fillOpacity: classification?.fill_opacity || 0.25,
        opacity: 0.5
    };
}

export default useGeoData;
