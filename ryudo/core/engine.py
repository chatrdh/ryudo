"""
Living Graph Engine
===================

The core engine that manages an immutable base graph with dynamic
constraint overlays applied at query time.

Key Design Principles:
1. Base graph is IMMUTABLE - never modified directly
2. Constraints are stored separately and applied via get_view()
3. Spatial queries use STRtree for O(log n) performance
4. Time-aware: constraints have validity windows
"""

from datetime import datetime, timezone
from dataclasses import dataclass, field
import json
import math
from typing import Any, Optional
from uuid import UUID

import networkx as nx
from shapely import STRtree
from shapely.geometry import LineString, Point, shape

from ryudo.core.schema import ConstraintType, GraphConstraint, TimeWindow
from ryudo.core.mapper import TagMapper


CONSTRAINT_PRECEDENCE: dict[ConstraintType, int] = {
    ConstraintType.NODE_STATUS: 400,
    ConstraintType.ZONE_MASK: 300,
    ConstraintType.EDGE_WEIGHT: 200,
    ConstraintType.VIRTUAL_EDGE: 100,
}
"""
Deterministic constraint precedence (higher applies first).

Why this order:
- NODE_STATUS first: hard availability constraints take top priority.
- ZONE_MASK second: spatial overlays set baseline risk and zone labels.
- EDGE_WEIGHT third: fine-grained weighting layers on top.
- VIRTUAL_EDGE last: synthetic links are added after topology mutations.
"""


@dataclass
class ConstraintApplicationRecord:
    """Per-constraint application status for view materialization."""

    constraint_id: UUID
    constraint_type: ConstraintType
    source_agent_id: str
    status: str
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a JSON-serializable dictionary."""
        return {
            "constraint_id": str(self.constraint_id),
            "constraint_type": self.constraint_type.value,
            "source_agent_id": self.source_agent_id,
            "status": self.status,
            "reason": self.reason,
            "details": self.details,
        }


class LivingGraph:
    """
    A graph engine that maintains an immutable base map with dynamic constraints.
    
    The base graph (typically OSM road network) is loaded once and never modified.
    Constraints are registered and applied at query time via `get_view()`, which
    returns a modified copy of the graph.
    
    Example
    -------
    >>> graph = LivingGraph()
    >>> graph.load_base_map(bbox=(17.65, 83.15, 17.75, 83.35))
    >>> 
    >>> # Add a zone constraint
    >>> constraint = GraphConstraint(
    ...     type=ConstraintType.ZONE_MASK,
    ...     target={"polygon": zone_geojson},
    ...     effect={"weight_factor": 100.0},
    ...     validity=TimeWindow(start_time=now, end_time=now+timedelta(hours=6)),
    ...     source_agent_id="test"
    ... )
    >>> graph.add_constraint(constraint)
    >>> 
    >>> # Get a view with constraints applied
    >>> view = graph.get_view(query_time=now)
    """
    
    def __init__(self, tag_mapper: Optional[TagMapper] = None):
        """
        Initialize the Living Graph.
        
        Parameters
        ----------
        tag_mapper : TagMapper, optional
            Mapper for converting OSM tags to weights.
            If None, uses default configuration.
        """
        self._base_graph: Optional[nx.MultiDiGraph] = None
        self._constraints: dict[UUID, GraphConstraint] = {}
        self._tag_mapper = tag_mapper or TagMapper()
        
        # Spatial index for edges (built during load)
        self._edge_index: Optional[STRtree] = None
        self._edge_list: list[tuple[int, int, int]] = []  # (u, v, key) indexed same as geometries
        self._edge_geometries: list[LineString] = []
    
    @property
    def base_graph(self) -> Optional[nx.MultiDiGraph]:
        """The immutable base graph (read-only access)."""
        return self._base_graph
    
    @property
    def constraint_count(self) -> int:
        """Number of registered constraints."""
        return len(self._constraints)

    @property
    def precedence_matrix(self) -> dict[ConstraintType, int]:
        """Read-only copy of type precedence values."""
        return dict(CONSTRAINT_PRECEDENCE)
    
    def load_base_map(
        self, 
        bbox: Optional[tuple[float, float, float, float]] = None,
        place: Optional[str] = None,
        network_type: str = "drive"
    ) -> None:
        """
        Load the base map from OpenStreetMap.
        
        Parameters
        ----------
        bbox : tuple, optional
            Bounding box as (south, west, north, east) in degrees
        place : str, optional
            Place name (e.g., "Visakhapatnam, India")
        network_type : str
            OSMnx network type: 'drive', 'walk', 'bike', 'all'
        
        Raises
        ------
        ValueError
            If neither bbox nor place is provided
        """
        import osmnx as ox
        
        if bbox is None and place is None:
            raise ValueError("Must provide either bbox or place")
        
        # Load the graph
        if place:
            print(f"[LivingGraph] Loading road network for {place}...")
            self._base_graph = ox.graph_from_place(place, network_type=network_type)
        else:
            south, west, north, east = bbox
            print(f"[LivingGraph] Loading road network for bbox {bbox}...")
            self._base_graph = ox.graph_from_bbox(
                north=north, south=south, east=east, west=west,
                network_type=network_type
            )
        
        print(f"[LivingGraph] Loaded {self._base_graph.number_of_nodes()} nodes, "
              f"{self._base_graph.number_of_edges()} edges")
        
        # Build spatial index for edges
        self._build_edge_index()
    
    def load_from_graph(self, graph: nx.MultiDiGraph) -> None:
        """
        Load from an existing NetworkX graph (for testing).
        
        Parameters
        ----------
        graph : nx.MultiDiGraph
            Pre-loaded graph with 'x' and 'y' attributes on nodes
        """
        self._base_graph = graph
        self._build_edge_index()
        print(f"[LivingGraph] Loaded graph with {graph.number_of_nodes()} nodes, "
              f"{graph.number_of_edges()} edges")
    
    def _build_edge_index(self) -> None:
        """Build spatial R-tree index for edges."""
        if self._base_graph is None:
            return
        
        self._edge_geometries = []
        self._edge_list = []
        
        for u, v, key in self._base_graph.edges(keys=True):
            try:
                u_x, u_y = self._base_graph.nodes[u]['x'], self._base_graph.nodes[u]['y']
                v_x, v_y = self._base_graph.nodes[v]['x'], self._base_graph.nodes[v]['y']
                
                geom = LineString([(u_x, u_y), (v_x, v_y)])
                self._edge_geometries.append(geom)
                self._edge_list.append((u, v, key))
            except KeyError:
                continue
        
        if self._edge_geometries:
            self._edge_index = STRtree(self._edge_geometries)
            print(f"[LivingGraph] Built spatial index for {len(self._edge_geometries)} edges")
    
    def add_constraint(self, constraint: GraphConstraint) -> UUID:
        """
        Register a new constraint.
        
        Parameters
        ----------
        constraint : GraphConstraint
            The constraint to add
        
        Returns
        -------
        UUID
            The constraint's unique identifier
        """
        self._constraints[constraint.id] = constraint
        return constraint.id
    
    def remove_constraint(self, constraint_id: UUID) -> bool:
        """
        Remove a constraint by ID.
        
        Parameters
        ----------
        constraint_id : UUID
            ID of the constraint to remove
        
        Returns
        -------
        bool
            True if constraint was found and removed
        """
        if constraint_id in self._constraints:
            del self._constraints[constraint_id]
            return True
        return False
    
    def get_constraint(self, constraint_id: UUID) -> Optional[GraphConstraint]:
        """Get a constraint by ID."""
        return self._constraints.get(constraint_id)
    
    def list_constraints(
        self, 
        active_at: Optional[datetime] = None,
        source_agent_id: Optional[str] = None
    ) -> list[GraphConstraint]:
        """
        List constraints, optionally filtered.
        
        Parameters
        ----------
        active_at : datetime, optional
            Only return constraints active at this time
        source_agent_id : str, optional
            Only return constraints from this agent
        
        Returns
        -------
        list[GraphConstraint]
            Matching constraints
        """
        result = list(self._constraints.values())
        
        if active_at is not None:
            result = [c for c in result if c.is_active(active_at)]
        
        if source_agent_id is not None:
            result = [c for c in result if c.source_agent_id == source_agent_id]
        
        return result
    
    def clear_expired_constraints(self, reference_time: Optional[datetime] = None) -> int:
        """
        Remove all constraints that have expired.
        
        Parameters
        ----------
        reference_time : datetime, optional
            Time to check against. Defaults to now (UTC).
        
        Returns
        -------
        int
            Number of constraints removed
        """
        if reference_time is None:
            reference_time = datetime.now(timezone.utc)
        
        expired_ids = [
            cid for cid, c in self._constraints.items()
            if c.validity.end_time < reference_time
        ]
        
        for cid in expired_ids:
            del self._constraints[cid]
        
        return len(expired_ids)
    
    def get_view(
        self,
        query_time: Optional[datetime] = None,
        filter_tags: Optional[dict[str, Any]] = None,
        weight_attr: str = "weight"
    ) -> nx.MultiDiGraph:
        """
        Get a view of the graph with active constraints applied.
        
        This is the core method that applies all constraints to produce
        a routing-ready graph. The base graph is never modified.
        
        Parameters
        ----------
        query_time : datetime, optional
            Time for filtering constraint validity. Defaults to now (UTC).
        filter_tags : dict, optional
            Only include constraints with matching metadata tags
        weight_attr : str
            Name of the weight attribute to modify (default: "weight")
        
        Returns
        -------
        nx.MultiDiGraph
            Copy of base graph with constraints applied
        """
        view, _ = self.get_view_with_report(
            query_time=query_time,
            filter_tags=filter_tags,
            weight_attr=weight_attr,
        )
        return view

    def get_view_with_report(
        self,
        query_time: Optional[datetime] = None,
        filter_tags: Optional[dict[str, Any]] = None,
        weight_attr: str = "weight"
    ) -> tuple[nx.MultiDiGraph, list[ConstraintApplicationRecord]]:
        """
        Get a constrained graph view plus per-constraint apply/skip report.

        Returns
        -------
        tuple[nx.MultiDiGraph, list[ConstraintApplicationRecord]]
            Graph view and application records in processing order.
        """
        if self._base_graph is None:
            raise ValueError("No base graph loaded. Call load_base_map() first.")
        
        if query_time is None:
            query_time = datetime.now(timezone.utc)
        
        # Start with a copy
        G = self._base_graph.copy()
        
        # Filter to active constraints
        active_constraints = [
            c for c in self._constraints.values()
            if c.is_active(query_time)
        ]
        
        # Apply filter_tags if provided
        if filter_tags:
            active_constraints = [
                c for c in active_constraints
                if all(c.metadata.get(k) == v for k, v in filter_tags.items())
            ]
        
        report: list[ConstraintApplicationRecord] = []

        if not active_constraints:
            return G, report

        ordered_constraints = sorted(
            active_constraints,
            key=self._constraint_sort_key,
        )
        edge_selector_cache: dict[str, list[tuple[int, int, int]]] = {}

        for c in ordered_constraints:
            if c.type == ConstraintType.NODE_STATUS:
                self._apply_node_status_constraint(G, c, report)
            elif c.type == ConstraintType.ZONE_MASK:
                self._apply_zone_masks(G, [c], weight_attr, report)
            elif c.type == ConstraintType.EDGE_WEIGHT:
                self._apply_edge_weight_constraint(
                    G,
                    c,
                    weight_attr,
                    report,
                    selector_cache=edge_selector_cache,
                )
            elif c.type == ConstraintType.VIRTUAL_EDGE:
                self._apply_virtual_edge_constraint(G, c, weight_attr, report)
        
        return G, report

    def _constraint_sort_key(self, constraint: GraphConstraint) -> tuple[int, int]:
        """
        Sort key for deterministic conflict precedence.

        Higher precedence first, then higher explicit priority first.
        Stable sort preserves registration order when ties remain.
        """
        precedence = CONSTRAINT_PRECEDENCE.get(constraint.type, 0)
        return (-precedence, -constraint.priority)

    def _apply_node_status_constraint(
        self,
        G: nx.MultiDiGraph,
        constraint: GraphConstraint,
        report: list[ConstraintApplicationRecord],
    ) -> None:
        """Apply a NODE_STATUS constraint."""
        if G.number_of_nodes() == 0:
            report.append(self._record(
                constraint,
                status="skipped",
                reason="graph_has_no_nodes",
            ))
            return

        action = constraint.effect.get("action", "disable")
        if action != "disable":
            report.append(self._record(
                constraint,
                status="skipped",
                reason="unsupported_node_action",
                details={"action": action},
            ))
            return

        nodes_to_remove = self._resolve_node_ids(G, constraint.target)
        if not nodes_to_remove:
            report.append(self._record(
                constraint,
                status="skipped",
                reason="no_nodes_resolved",
            ))
            return

        removed_count = sum(1 for n in nodes_to_remove if n in G)
        if removed_count == 0:
            report.append(self._record(
                constraint,
                status="skipped",
                reason="nodes_not_present_in_view",
                details={"nodes_resolved": len(nodes_to_remove)},
            ))
            return

        G.remove_nodes_from(nodes_to_remove)
        report.append(self._record(
            constraint,
            status="applied",
            reason="nodes_disabled",
            details={"nodes_removed": removed_count},
        ))

    def _apply_edge_weight_constraint(
        self,
        G: nx.MultiDiGraph,
        constraint: GraphConstraint,
        weight_attr: str,
        report: list[ConstraintApplicationRecord],
        selector_cache: Optional[dict[str, list[tuple[int, int, int]]]] = None,
    ) -> None:
        """Apply an EDGE_WEIGHT constraint."""
        if G.number_of_edges() == 0:
            report.append(self._record(
                constraint,
                status="skipped",
                reason="graph_has_no_edges",
            ))
            return

        factor = constraint.effect.get("weight_factor", 1.0)
        cache_key = self._edge_selector_cache_key(constraint.target)
        if selector_cache is not None and cache_key in selector_cache:
            target_edges = selector_cache[cache_key]
        else:
            target_edges = self._resolve_edge_keys(G, constraint.target)
            if selector_cache is not None:
                selector_cache[cache_key] = target_edges

        if not target_edges:
            report.append(self._record(
                constraint,
                status="skipped",
                reason="no_edges_resolved",
            ))
            return

        applied_count = 0
        for u, v, key in target_edges:
            if G.has_edge(u, v, key):
                current = G[u][v][key].get(weight_attr, 1.0)
                G[u][v][key][weight_attr] = current * factor
                applied_count += 1

        if applied_count == 0:
            report.append(self._record(
                constraint,
                status="skipped",
                reason="edges_not_present_in_view",
                details={"edges_resolved": len(target_edges)},
            ))
            return

        report.append(self._record(
            constraint,
            status="applied",
            reason="edge_weights_updated",
            details={"edges_modified": applied_count, "weight_factor": factor},
        ))

    def _apply_virtual_edge_constraint(
        self,
        G: nx.MultiDiGraph,
        constraint: GraphConstraint,
        weight_attr: str,
        report: list[ConstraintApplicationRecord],
    ) -> None:
        """Apply a VIRTUAL_EDGE constraint."""
        if G.number_of_nodes() == 0:
            report.append(self._record(
                constraint,
                status="skipped",
                reason="graph_has_no_nodes",
            ))
            return

        from_node = constraint.target.get("from_node")
        to_node = constraint.target.get("to_node")

        if from_node is None or to_node is None:
            report.append(self._record(
                constraint,
                status="skipped",
                reason="missing_virtual_edge_nodes",
            ))
            return

        if from_node not in G or to_node not in G:
            report.append(self._record(
                constraint,
                status="skipped",
                reason="virtual_edge_nodes_not_in_view",
                details={"from_node": from_node, "to_node": to_node},
            ))
            return

        weight = constraint.effect.get("weight", 1.0)
        bidirectional = constraint.effect.get("bidirectional", False)

        G.add_edge(from_node, to_node, **{weight_attr: weight, "virtual": True})
        if bidirectional:
            G.add_edge(to_node, from_node, **{weight_attr: weight, "virtual": True})

        report.append(self._record(
            constraint,
            status="applied",
            reason="virtual_edge_added",
            details={"bidirectional": bidirectional, "weight": weight},
        ))

    def _resolve_node_ids(
        self,
        G: nx.MultiDiGraph,
        target: dict[str, Any]
    ) -> set[int]:
        """
        Resolve node selector target into concrete node IDs.

        Supported selectors:
        - {"node_id": int}
        - {"node_ids": [int, ...]}
        - {"lat": float, "lon": float} -> nearest node
        - {"lat": float, "lon": float, "radius_m": float} -> all nodes in radius
        """
        if G.number_of_nodes() == 0:
            return set()

        node_ids: set[int] = set()

        node_id = target.get("node_id")
        if node_id is not None:
            node_ids.add(node_id)

        for candidate in target.get("node_ids", []):
            node_ids.add(candidate)

        lat = self._as_float(target.get("lat"))
        lon = self._as_float(target.get("lon"))

        if lat is not None and lon is not None:
            radius_m = self._as_float(target.get("radius_m"))
            if radius_m is None:
                radius_km = self._as_float(target.get("radius_km"))
                radius_m = radius_km * 1000.0 if radius_km is not None else None

            nearest_node = None
            nearest_distance = float("inf")

            for candidate, attrs in G.nodes(data=True):
                node_lat = self._as_float(attrs.get("y", attrs.get("lat")))
                node_lon = self._as_float(attrs.get("x", attrs.get("lon")))
                if node_lat is None or node_lon is None:
                    continue

                distance_m = self._haversine_m(lat, lon, node_lat, node_lon)

                if distance_m < nearest_distance:
                    nearest_distance = distance_m
                    nearest_node = candidate

                if radius_m is not None and distance_m <= radius_m:
                    node_ids.add(candidate)

            if radius_m is None and nearest_node is not None:
                node_ids.add(nearest_node)

        return {n for n in node_ids if n in G}

    def _edge_selector_cache_key(self, target: dict[str, Any]) -> str:
        """Create a stable cache key for selector targets."""
        normalized = self._normalize_for_cache(target)
        return json.dumps(normalized, sort_keys=True, separators=(",", ":"), default=str)

    def _normalize_for_cache(self, value: Any) -> Any:
        """Normalize nested structures into deterministic JSON-safe form."""
        if isinstance(value, dict):
            return {
                str(key): self._normalize_for_cache(val)
                for key, val in sorted(value.items(), key=lambda item: str(item[0]))
            }

        if isinstance(value, (list, tuple)):
            return [self._normalize_for_cache(item) for item in value]

        if isinstance(value, (set, frozenset)):
            return sorted(self._normalize_for_cache(item) for item in value)

        return value

    def _resolve_edge_keys(
        self,
        G: nx.MultiDiGraph,
        target: dict[str, Any]
    ) -> list[tuple[int, int, int]]:
        """
        Resolve edge selector target into concrete (u, v, key) tuples.

        Supported selectors:
        - {"edge": (u, v)} or {"edge": (u, v, key)}
        - {"edges": [(u, v), ...]}
        - {"road_types": ["primary", "secondary"]}
        - {"tag_filter": {"highway": ["primary"]}}
        - {"zone": "moderate"} or {"zones": ["moderate", ...]}
        """
        resolved: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()

        def _append(edge_key: tuple[int, int, int]) -> None:
            if edge_key not in seen:
                seen.add(edge_key)
                resolved.append(edge_key)

        direct_edge = target.get("edge")
        if direct_edge is not None:
            for edge_key in self._resolve_edge_spec(G, direct_edge):
                _append(edge_key)

        for edge_spec in target.get("edges", []):
            for edge_key in self._resolve_edge_spec(G, edge_spec):
                _append(edge_key)

        road_types = self._normalize_to_set(target.get("road_types"))
        tag_filter = target.get("tag_filter", {})
        zones = self._normalize_to_set(target.get("zone"))
        zones.update(self._normalize_to_set(target.get("zones")))

        if road_types or tag_filter or zones:
            normalized_filter = dict(tag_filter) if isinstance(tag_filter, dict) else {}
            if road_types:
                normalized_filter["highway"] = road_types

            for u, v, key, data in G.edges(keys=True, data=True):
                if normalized_filter and not self._matches_filters(data, normalized_filter):
                    continue
                if zones and not self._matches_zone(data, zones):
                    continue
                _append((u, v, key))

        return resolved

    def _resolve_edge_spec(
        self,
        G: nx.MultiDiGraph,
        edge_spec: Any
    ) -> list[tuple[int, int, int]]:
        """Resolve a single edge selector into concrete edge keys."""
        if not isinstance(edge_spec, (list, tuple)):
            return []

        if len(edge_spec) == 2:
            u, v = edge_spec
            if not G.has_edge(u, v):
                return []
            return [(u, v, key) for key in G[u][v].keys()]

        if len(edge_spec) >= 3:
            u, v, key = edge_spec[0], edge_spec[1], edge_spec[2]
            if G.has_edge(u, v, key):
                return [(u, v, key)]

        return []

    def _matches_filters(self, data: dict[str, Any], filters: dict[str, Any]) -> bool:
        """Return True when all attribute filters match edge data."""
        for key, expected in filters.items():
            expected_values = self._normalize_to_set(expected)
            if not expected_values:
                continue

            actual_values = self._normalize_to_set(data.get(key))
            if not actual_values.intersection(expected_values):
                return False
        return True

    def _matches_zone(self, data: dict[str, Any], zones: set[str]) -> bool:
        """Return True when edge data carries one of the target zone labels."""
        if not zones:
            return True

        actual = self._normalize_to_set(data.get("zone_labels"))
        actual.update(self._normalize_to_set(data.get("zone_type")))
        actual.update(self._normalize_to_set(data.get("prediction_zone")))

        return bool(actual.intersection(zones))

    def _normalize_to_set(self, value: Any) -> set[str]:
        """Normalize scalar or list-like values to a lowercase string set."""
        if value is None:
            return set()

        if isinstance(value, (list, tuple, set, frozenset)):
            return {str(v).lower() for v in value if v is not None}

        return {str(value).lower()}

    def _as_float(self, value: Any) -> Optional[float]:
        """Best-effort numeric conversion."""
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _haversine_m(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in meters."""
        r = 6371000.0

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        return r * 2 * math.asin(math.sqrt(a))
    
    def _apply_zone_masks(
        self, 
        G: nx.MultiDiGraph, 
        constraints: list[GraphConstraint],
        weight_attr: str,
        report: Optional[list[ConstraintApplicationRecord]] = None,
    ) -> None:
        """
        Apply zone mask constraints using spatial index.
        
        This modifies G in-place for efficiency.
        """
        if not constraints or self._edge_index is None:
            if report is not None:
                for c in constraints:
                    report.append(self._record(
                        c,
                        status="skipped",
                        reason="edge_index_unavailable",
                    ))
            return

        if G.number_of_edges() == 0:
            if report is not None:
                for c in constraints:
                    report.append(self._record(
                        c,
                        status="skipped",
                        reason="graph_has_no_edges",
                    ))
            return
        
        for c in constraints:
            polygon_data = c.target.get("polygon")
            if polygon_data is None:
                if report is not None:
                    report.append(self._record(
                        c,
                        status="skipped",
                        reason="missing_polygon_target",
                    ))
                continue
            
            try:
                polygon = shape(polygon_data)
            except Exception:
                if report is not None:
                    report.append(self._record(
                        c,
                        status="skipped",
                        reason="invalid_polygon_geometry",
                    ))
                continue
            
            if not polygon.is_valid:
                if report is not None:
                    report.append(self._record(
                        c,
                        status="skipped",
                        reason="invalid_polygon_topology",
                    ))
                continue
            
            weight_factor = c.effect.get("weight_factor", 1.0)
            node_action = c.effect.get("node_action")
            zone_label = (
                c.metadata.get("zone_type")
                or c.metadata.get("prediction_zone")
                or c.metadata.get("zone")
            )
            
            # Find edges that intersect the polygon using spatial index
            affected_indices = self._edge_index.query(polygon)
            
            edges_to_modify = []
            nodes_to_disable = set()
            
            for idx in affected_indices:
                edge_geom = self._edge_geometries[idx]
                
                # Verify actual intersection (STRtree uses bounding boxes)
                if polygon.intersects(edge_geom):
                    u, v, key = self._edge_list[idx]
                    if u not in G or v not in G:
                        continue
                    edges_to_modify.append((u, v, key))
                    
                    # Collect nodes if node_action is set
                    if node_action == "disable":
                        # Only disable nodes fully inside the polygon
                        u_point = Point(
                            G.nodes[u].get("x", G.nodes[u].get("lon", 0.0)),
                            G.nodes[u].get("y", G.nodes[u].get("lat", 0.0)),
                        )
                        v_point = Point(
                            G.nodes[v].get("x", G.nodes[v].get("lon", 0.0)),
                            G.nodes[v].get("y", G.nodes[v].get("lat", 0.0)),
                        )
                        if polygon.contains(u_point):
                            nodes_to_disable.add(u)
                        if polygon.contains(v_point):
                            nodes_to_disable.add(v)
            
            # Apply weight factor to edges
            for u, v, key in edges_to_modify:
                if G.has_edge(u, v, key):
                    edge_data = G[u][v][key]
                    current = edge_data.get(weight_attr, 1.0)
                    edge_data[weight_attr] = current * weight_factor

                    if zone_label:
                        labels = self._normalize_to_set(edge_data.get("zone_labels"))
                        labels.add(str(zone_label).lower())
                        edge_data["zone_labels"] = sorted(labels)
                        edge_data["zone_type"] = str(zone_label).lower()
            
            # Disable nodes if requested
            if nodes_to_disable:
                G.remove_nodes_from(nodes_to_disable)

            if report is not None:
                if edges_to_modify:
                    report.append(self._record(
                        c,
                        status="applied",
                        reason="zone_mask_applied",
                        details={
                            "edges_modified": len(edges_to_modify),
                            "nodes_removed": len(nodes_to_disable),
                            "weight_factor": weight_factor,
                        },
                    ))
                else:
                    report.append(self._record(
                        c,
                        status="skipped",
                        reason="zone_mask_no_intersections",
                    ))

    def _record(
        self,
        constraint: GraphConstraint,
        status: str,
        reason: str,
        details: Optional[dict[str, Any]] = None,
    ) -> ConstraintApplicationRecord:
        """Create a constraint application record."""
        return ConstraintApplicationRecord(
            constraint_id=constraint.id,
            constraint_type=constraint.type,
            source_agent_id=constraint.source_agent_id,
            status=status,
            reason=reason,
            details=details or {},
        )
    
    def __repr__(self) -> str:
        nodes = self._base_graph.number_of_nodes() if self._base_graph else 0
        edges = self._base_graph.number_of_edges() if self._base_graph else 0
        return f"LivingGraph(nodes={nodes}, edges={edges}, constraints={len(self._constraints)})"
