"""
Multi-Modal Routing
===================

Asset-specific graph view generation with constraint inversion.

Key Concept: Different assets see the same constraints differently.
- Truck: Flood zone = obstacle (high weight)
- Boat: Flood zone = roadway (low weight)
- Drone: Ignores ground obstacles

The ViewBuilder transforms graph constraints based on asset capabilities.
"""

from datetime import datetime, timezone
from typing import Any, Optional

import networkx as nx
from shapely.geometry import Point, shape

from ryudo.core.schema import ConstraintType, GraphConstraint
from ryudo.mission.profile import Asset, AssetCapability


# =============================================================================
# Capability Weight Profiles
# =============================================================================


# Weight multipliers by constraint type and capability
# Format: {zone_type: {capability: multiplier}}
WEIGHT_PROFILES: dict[str, dict[AssetCapability, float]] = {
    # Flood zones
    "flooded": {
        AssetCapability.ROAD: 1000.0,       # Avoid completely
        AssetCapability.WATER: 0.5,         # Prefer (navigable)
        AssetCapability.AIR: 1.0,           # Ignore
        AssetCapability.FLOOD_CAPABLE: 2.0, # Passable with penalty
        AssetCapability.OFFROAD: 100.0,     # Still very bad
    },
    "surge": {
        AssetCapability.ROAD: float("inf"), # Impassable
        AssetCapability.WATER: 0.3,         # Very good for boats
        AssetCapability.AIR: 1.0,           # Ignore
        AssetCapability.FLOOD_CAPABLE: 50.0,# Risky but possible
    },
    "extreme": {
        AssetCapability.ROAD: float("inf"), # Impassable
        AssetCapability.WATER: 1.0,         # Normal for boats
        AssetCapability.AIR: 1.0,           # Ignore
    },
    "severe": {
        AssetCapability.ROAD: 10.0,         # High cost
        AssetCapability.WATER: 0.8,         # Slightly prefer
        AssetCapability.AIR: 1.0,           # Ignore
        AssetCapability.FLOOD_CAPABLE: 3.0, # Manageable
    },
    "moderate": {
        AssetCapability.ROAD: 3.0,          # Moderate cost
        AssetCapability.WATER: 1.5,         # Slight penalty
        AssetCapability.AIR: 1.0,           # Ignore
        AssetCapability.FLOOD_CAPABLE: 1.5, # Slight penalty
    },
    # Road damage
    "road_damaged": {
        AssetCapability.ROAD: 50.0,         # Very high cost
        AssetCapability.WATER: 100.0,       # Can't use roads
        AssetCapability.AIR: 1.0,           # Ignore
        AssetCapability.OFFROAD: 5.0,       # Manageable
    },
    # High wind
    "high_wind": {
        AssetCapability.ROAD: 1.5,          # Slight penalty
        AssetCapability.WATER: 3.0,         # More dangerous
        AssetCapability.AIR: 100.0,         # Extremely dangerous
    },
}

# Default weight for unknown capability/zone combinations
DEFAULT_WEIGHT = 10.0


# =============================================================================
# ViewBuilder
# =============================================================================


class ViewBuilder:
    """
    Builds asset-specific graph views by transforming constraints.
    
    The core insight: the same physical constraints have different
    routing implications for different asset types.
    
    Example
    -------
    ```python
    builder = ViewBuilder(living_graph)
    
    truck = Asset(id="T1", capabilities=(AssetCapability.ROAD,))
    boat = Asset(id="B1", capabilities=(AssetCapability.WATER,))
    
    # Same graph, different views
    truck_view = builder.build_view(truck, now, constraints)
    boat_view = builder.build_view(boat, now, constraints)
    
    # Truck avoids flood, boat goes through
    truck_path = nx.shortest_path(truck_view, A, B)  # Around flood
    boat_path = nx.shortest_path(boat_view, A, B)   # Through flood
    ```
    """
    
    def __init__(self, living_graph: Optional["LivingGraph"] = None):
        """
        Initialize ViewBuilder.
        
        Parameters
        ----------
        living_graph : LivingGraph, optional
            Reference to the LivingGraph engine for base views.
            If not provided, must pass graph directly to build_view.
        """
        self._living_graph = living_graph
        self._weight_profiles = WEIGHT_PROFILES
    
    def build_view(
        self,
        asset: Asset,
        query_time: datetime,
        constraints: list[GraphConstraint],
        base_graph: Optional[nx.MultiDiGraph] = None,
    ) -> nx.MultiDiGraph:
        """
        Build a graph view customized for the asset's capabilities.
        
        Parameters
        ----------
        asset : Asset
            The asset requesting the view
        query_time : datetime
            Query time for filtering active constraints
        constraints : list[GraphConstraint]
            Current active constraints
        base_graph : nx.MultiDiGraph, optional
            Base graph to start from (uses LivingGraph if not provided)
        
        Returns
        -------
        nx.MultiDiGraph
            Asset-specific graph view with transformed weights
        """
        # Get base graph
        if base_graph is not None:
            G = base_graph.copy()
        elif self._living_graph is not None:
            G = self._living_graph.get_view(query_time=query_time)
        else:
            raise ValueError("Must provide base_graph or living_graph")
        
        # Build edge spatial index for efficiency
        edge_points = self._build_edge_index(G)
        
        # Apply capability-based transformations
        for constraint in constraints:
            if not self._is_active(constraint, query_time):
                continue
            
            if constraint.type == ConstraintType.ZONE_MASK:
                self._apply_zone_transform(G, constraint, asset, edge_points)
            elif constraint.type == ConstraintType.EDGE_WEIGHT:
                self._apply_edge_transform(G, constraint, asset)
            elif constraint.type == ConstraintType.NODE_STATUS:
                self._apply_node_transform(G, constraint, asset)
        
        return G
    
    def _is_active(self, constraint: GraphConstraint, query_time: datetime) -> bool:
        """Check if constraint is active at query time."""
        if constraint.validity is None:
            return True
        
        start = constraint.validity.start_time
        end = constraint.validity.end_time
        
        if start and query_time < start:
            return False
        if end and query_time > end:
            return False
        
        return True
    
    def _build_edge_index(
        self, 
        graph: nx.MultiDiGraph
    ) -> dict[tuple, Point]:
        """Build mapping of edges to their centroid points."""
        edge_points = {}
        
        for u, v, k in graph.edges(keys=True):
            u_data = graph.nodes.get(u, {})
            v_data = graph.nodes.get(v, {})
            
            u_lat = u_data.get("y", u_data.get("lat", 0))
            u_lon = u_data.get("x", u_data.get("lon", 0))
            v_lat = v_data.get("y", v_data.get("lat", 0))
            v_lon = v_data.get("x", v_data.get("lon", 0))
            
            centroid = Point((u_lon + v_lon) / 2, (u_lat + v_lat) / 2)
            edge_points[(u, v, k)] = centroid
        
        return edge_points
    
    def _apply_zone_transform(
        self,
        graph: nx.MultiDiGraph,
        constraint: GraphConstraint,
        asset: Asset,
        edge_points: dict[tuple, Point],
    ) -> None:
        """
        Apply zone constraint with capability-based weight transformation.
        
        Key Logic:
        - Get the zone type from constraint metadata
        - Look up weight multiplier based on asset's PRIMARY capability
        - Apply transformed weight to edges within zone
        """
        # Get zone polygon
        polygon_data = constraint.target.get("polygon")
        if not polygon_data:
            return
        
        try:
            polygon = shape(polygon_data)
        except Exception:
            return
        
        # Determine zone type
        zone_type = constraint.metadata.get("zone_type", "flooded")
        if zone_type not in self._weight_profiles:
            zone_type = "moderate"  # Default fallback
        
        # Get weight multiplier for this asset's primary capability
        multiplier = self._get_weight_multiplier(asset, zone_type)
        
        # Apply to edges within zone
        for (u, v, k), point in edge_points.items():
            if polygon.contains(point):
                edge_data = graph[u][v][k]
                current_weight = edge_data.get("weight", 1.0)
                
                if multiplier == float("inf"):
                    # Remove edge (impassable)
                    graph.remove_edge(u, v, key=k)
                else:
                    edge_data["weight"] = current_weight * multiplier
                    edge_data["zone_modified"] = True
                    edge_data["zone_type"] = zone_type
    
    def _apply_edge_transform(
        self,
        graph: nx.MultiDiGraph,
        constraint: GraphConstraint,
        asset: Asset,
    ) -> None:
        """Apply edge weight constraint (no capability inversion for direct edges)."""
        edge_spec = constraint.target.get("edge")
        if not edge_spec:
            return
        
        u, v = edge_spec[0], edge_spec[1]
        k = edge_spec[2] if len(edge_spec) > 2 else 0
        
        if graph.has_edge(u, v, key=k):
            factor = constraint.effect.get("weight_factor", 1.0)
            graph[u][v][k]["weight"] *= factor
    
    def _apply_node_transform(
        self,
        graph: nx.MultiDiGraph,
        constraint: GraphConstraint,
        asset: Asset,
    ) -> None:
        """Apply node status constraint."""
        node_id = constraint.target.get("node_id")
        action = constraint.effect.get("action", "disable")
        
        if node_id and node_id in graph.nodes:
            if action == "disable":
                # Remove all edges to/from node
                edges_to_remove = list(graph.in_edges(node_id, keys=True))
                edges_to_remove += list(graph.out_edges(node_id, keys=True))
                
                for edge in edges_to_remove:
                    if len(edge) == 3:
                        graph.remove_edge(edge[0], edge[1], key=edge[2])
                    else:
                        graph.remove_edge(edge[0], edge[1])
    
    def _get_weight_multiplier(self, asset: Asset, zone_type: str) -> float:
        """
        Get weight multiplier for asset in zone.
        
        Priority order for capabilities:
        1. WATER (for water-based zones)
        2. AIR (for all zones)
        3. FLOOD_CAPABLE
        4. OFFROAD
        5. ROAD (default)
        """
        profile = self._weight_profiles.get(zone_type, {})
        
        # Check capabilities in priority order
        priority_order = [
            AssetCapability.WATER,
            AssetCapability.AIR,
            AssetCapability.FLOOD_CAPABLE,
            AssetCapability.OFFROAD,
            AssetCapability.ROAD,
        ]
        
        for cap in priority_order:
            if asset.has_capability(cap) and cap in profile:
                return profile[cap]
        
        # Fallback to ROAD profile or default
        return profile.get(AssetCapability.ROAD, DEFAULT_WEIGHT)
    
    def get_capability_summary(self, asset: Asset) -> dict[str, float]:
        """Get summary of how asset sees each zone type."""
        summary = {}
        for zone_type in self._weight_profiles:
            summary[zone_type] = self._get_weight_multiplier(asset, zone_type)
        return summary


# =============================================================================
# Constraint Inverter (Utility)
# =============================================================================


class ConstraintInverter:
    """
    Utility for inverting constraints for specific asset types.
    
    Used when an asset's capabilities fundamentally change
    the meaning of a constraint.
    """
    
    @staticmethod
    def invert_flood_constraint(
        constraint: GraphConstraint,
        original_factor: float = 1000.0,
        inverted_factor: float = 0.5,
    ) -> GraphConstraint:
        """
        Invert a flood constraint for water-capable assets.
        
        Original: weight_factor = 1000 (avoid)
        Inverted: weight_factor = 0.5 (prefer)
        """
        if constraint.type != ConstraintType.ZONE_MASK:
            return constraint
        
        current_factor = constraint.effect.get("weight_factor", 1.0)
        
        # If it's a high-penalty constraint, invert it
        if current_factor >= original_factor:
            new_effect = {**constraint.effect, "weight_factor": inverted_factor}
        else:
            new_effect = constraint.effect
        
        return GraphConstraint(
            id=constraint.id,
            type=constraint.type,
            target=constraint.target,
            effect=new_effect,
            validity=constraint.validity,
            source_agent_id=constraint.source_agent_id,
            metadata={
                **constraint.metadata,
                "inverted": True,
                "original_factor": current_factor,
            },
            priority=constraint.priority,
        )
    
    @staticmethod
    def create_land_constraint_for_boat(
        graph: nx.MultiDiGraph,
        flood_polygon: Any,
        weight_factor: float = 5.0,
    ) -> list[tuple]:
        """
        Create constraints for land areas (obstacles for boats).
        
        For boats, non-flooded road areas are harder to traverse
        (they need to dock and drive, or can't access at all).
        
        Returns list of edges outside the flood zone that should
        have increased weight for water assets.
        """
        try:
            polygon = shape(flood_polygon)
        except Exception:
            return []
        
        land_edges = []
        
        for u, v, k, data in graph.edges(keys=True, data=True):
            u_data = graph.nodes.get(u, {})
            v_data = graph.nodes.get(v, {})
            
            centroid = Point(
                (u_data.get("x", 0) + v_data.get("x", 0)) / 2,
                (u_data.get("y", 0) + v_data.get("y", 0)) / 2,
            )
            
            if not polygon.contains(centroid):
                land_edges.append((u, v, k))
        
        return land_edges
