"""
Mission Solver Architecture
===========================

Pluggable solver abstraction for different mission optimization problems.

Solvers:
- RescueSolver: CVRPTW for rescue operations (visit all targets)
- PatrolSolver: Edge coverage maximization
- SupplySolver: Delivery optimization
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

import networkx as nx

from ryudo.mission.profile import (
    Asset,
    MissionProfile,
    ObjectiveType,
    Target,
    AssetCapability,
)


# =============================================================================
# Solution Data Structures
# =============================================================================


@dataclass
class RouteSegment:
    """A segment of a route between two points."""
    
    from_id: str
    to_id: str
    distance_km: float
    travel_time_min: float
    path_nodes: list[int] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "from": self.from_id,
            "to": self.to_id,
            "distance_km": round(self.distance_km, 2),
            "travel_time_min": round(self.travel_time_min, 1),
            "path_nodes": self.path_nodes,
        }


@dataclass
class AssetAssignment:
    """Assignment of an asset to a set of targets."""
    
    asset_id: str
    asset_name: str
    targets: list[Target]
    route_segments: list[RouteSegment]
    total_population: int
    total_distance_km: float
    total_time_min: float
    score: float = 0.0
    
    @property
    def target_ids(self) -> list[str]:
        return [t.id for t in self.targets]
    
    def to_dict(self) -> dict:
        return {
            "asset_id": self.asset_id,
            "asset_name": self.asset_name,
            "targets": [
                {"id": t.id, "name": t.name, "population": t.population}
                for t in self.targets
            ],
            "route": [seg.to_dict() for seg in self.route_segments],
            "total_population": self.total_population,
            "distance_km": round(self.total_distance_km, 2),
            "time_min": round(self.total_time_min, 1),
            "score": round(self.score, 2),
        }


@dataclass
class SolutionResult:
    """
    Complete mission solution.
    
    Attributes
    ----------
    success : bool
        Whether a valid solution was found
    assignments : list[AssetAssignment]
        Asset-to-target assignments with routes
    unassigned_targets : list[Target]
        Targets that couldn't be assigned
    statistics : dict
        Summary statistics
    metadata : dict
        Additional solver-specific data
    """
    
    success: bool = True
    assignments: list[AssetAssignment] = field(default_factory=list)
    unassigned_targets: list[Target] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    solve_time_ms: float = 0.0
    
    @property
    def total_assigned_population(self) -> int:
        return sum(a.total_population for a in self.assignments)
    
    @property
    def total_distance_km(self) -> float:
        return sum(a.total_distance_km for a in self.assignments)
    
    @property
    def assets_used(self) -> int:
        return len(self.assignments)
    
    @property
    def targets_reached(self) -> int:
        return sum(len(a.targets) for a in self.assignments)
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "assignments": [a.to_dict() for a in self.assignments],
            "unassigned_targets": [
                {"id": t.id, "name": t.name, "population": t.population}
                for t in self.unassigned_targets
            ],
            "summary": {
                "targets_reached": self.targets_reached,
                "targets_unreachable": len(self.unassigned_targets),
                "total_population_rescued": self.total_assigned_population,
                "total_distance_km": round(self.total_distance_km, 1),
                "assets_deployed": self.assets_used,
                "solve_time_ms": round(self.solve_time_ms, 2),
                **self.statistics,
            },
            "metadata": self.metadata,
        }


# =============================================================================
# Base Solver
# =============================================================================


class BaseSolver(ABC):
    """
    Abstract base class for mission solvers.
    
    Subclasses implement different optimization strategies:
    - RescueSolver: CVRPTW (visit all high-priority targets)
    - PatrolSolver: Edge coverage maximization
    - SupplySolver: Delivery with capacity constraints
    
    Interface
    ---------
    solve(graph, mission, asset_graphs) -> SolutionResult
    
    Parameters
    ----------
    graph : nx.MultiDiGraph
        Base graph (for reference/fallback)
    mission : MissionProfile
        Mission specification
    asset_graphs : dict[str, nx.MultiDiGraph]
        Per-asset graph views (after capability-based filtering)
    """
    
    solver_id: str = "base"
    name: str = "Base Solver"
    api_version: str = "1.0"
    implementation_version: str = "0.1.0"
    supported_objectives: list[ObjectiveType] = []
    
    @abstractmethod
    def solve(
        self,
        graph: nx.MultiDiGraph,
        mission: MissionProfile,
        asset_graphs: dict[str, nx.MultiDiGraph],
    ) -> SolutionResult:
        """
        Solve the mission optimization problem.
        
        Must be implemented by subclasses.
        """
        ...
    
    def supports_objective(self, objective: ObjectiveType) -> bool:
        """Check if solver supports an objective."""
        return objective in self.supported_objectives

    def descriptor(self, source: str = "unknown") -> dict[str, Any]:
        """Return metadata describing this solver registration."""
        return {
            "solver_id": self.solver_id,
            "name": self.name,
            "source": source,
            "api_version": self.api_version,
            "implementation_version": self.implementation_version,
            "supported_objectives": [objective.value for objective in self.supported_objectives],
        }
    
    def score_solution(
        self,
        solution: SolutionResult,
        mission: MissionProfile
    ) -> float:
        """
        Calculate weighted objective score.
        
        Higher is better.
        """
        if not solution.success:
            return 0.0
        
        weights = mission.weights
        
        # Coverage: fraction of population reached
        total_pop = sum(t.population for t in mission.targets) or 1
        coverage_score = solution.total_assigned_population / total_pop
        
        # Time: inverse of total time (normalized)
        max_time = mission.max_duration_hours * 60 if mission.max_duration_hours else 360
        total_time = sum(a.total_time_min for a in solution.assignments)
        time_score = max(0, 1 - (total_time / max_time))
        
        # Risk: fraction of critical targets reached
        critical_ids = {t.id for t in mission.critical_targets}
        reached_critical = sum(
            1 for a in solution.assignments
            for t in a.targets if t.id in critical_ids
        )
        risk_score = reached_critical / len(critical_ids) if critical_ids else 1.0
        
        return (
            weights.get("coverage", 0.3) * coverage_score +
            weights.get("time", 0.4) * time_score +
            weights.get("risk", 0.3) * risk_score
        )


# =============================================================================
# Solver Registry
# =============================================================================


class SolverRegistry:
    """
    Registry for solver discovery and selection.
    
    Usage
    -----
    ```python
    # Register solvers
    SolverRegistry.register(RescueSolver())
    SolverRegistry.register(PatrolSolver())
    
    # Get solver for objective
    solver = SolverRegistry.get_for_objective(ObjectiveType.MIN_TIME)
    solution = solver.solve(graph, mission, asset_graphs)
    ```
    """
    
    REGISTRY_API_VERSION = "1.0"

    _solvers: dict[str, BaseSolver] = {}
    _sources: dict[str, str] = {}

    @classmethod
    def register(cls, solver: BaseSolver, *, source: str = "builtin") -> None:
        """Register a solver."""
        if not cls.is_api_compatible(solver.api_version):
            raise ValueError(
                f"Incompatible solver API version {solver.api_version!r} for "
                f"registry API {cls.REGISTRY_API_VERSION!r}"
            )
        cls._solvers[solver.solver_id] = solver
        cls._sources[solver.solver_id] = source
        print(f"[SolverRegistry] Registered: {solver.solver_id} ({solver.name}) [{source}]")

    @classmethod
    def register_factory(
        cls,
        factory: Callable[[], BaseSolver],
        *,
        source: str = "external",
    ) -> BaseSolver:
        """
        Register a solver using a factory hook.

        This is the primary external plugin hook: callers provide a no-arg
        factory that constructs a solver instance at registration time.
        """
        solver = factory()
        if not isinstance(solver, BaseSolver):
            raise TypeError("factory must return a BaseSolver instance")

        cls.register(solver, source=source)
        return solver
    
    @classmethod
    def unregister(cls, solver_id: str) -> bool:
        """Unregister a solver."""
        if solver_id in cls._solvers:
            del cls._solvers[solver_id]
            cls._sources.pop(solver_id, None)
            return True
        return False
    
    @classmethod
    def get(cls, solver_id: str) -> Optional[BaseSolver]:
        """Get solver by ID."""
        return cls._solvers.get(solver_id)
    
    @classmethod
    def get_for_objective(cls, objective: ObjectiveType) -> BaseSolver:
        """
        Get a solver that supports the given objective.
        
        Raises ValueError if no suitable solver found.
        """
        for solver in cls._solvers.values():
            if solver.supports_objective(objective):
                return solver
        
        raise ValueError(
            f"No solver found for objective: {objective}. "
            f"Available: {list(cls._solvers.keys())}"
        )
    
    @classmethod
    def list_solvers(cls) -> list[BaseSolver]:
        """List all registered solvers."""
        return list(cls._solvers.values())

    @classmethod
    def list_solver_descriptors(cls) -> list[dict[str, Any]]:
        """List solver metadata with source and versioning details."""
        descriptors: list[dict[str, Any]] = []
        for solver in cls._solvers.values():
            source = cls._sources.get(solver.solver_id, "unknown")
            descriptors.append(solver.descriptor(source=source))
        return descriptors

    @classmethod
    def clear(cls) -> None:
        """Clear all registered solvers."""
        cls._solvers.clear()
        cls._sources.clear()

    @classmethod
    def is_api_compatible(cls, solver_api_version: str) -> bool:
        """
        Check solver API compatibility against registry expectations.

        Current policy: major versions must match.
        """
        return cls._major_version(solver_api_version) == cls._major_version(
            cls.REGISTRY_API_VERSION
        )

    @staticmethod
    def _major_version(version: str) -> int:
        """Extract semantic major version from a version string."""
        if not version:
            return 0

        major_part = str(version).split(".", maxsplit=1)[0]
        try:
            return int(major_part)
        except ValueError:
            return 0


# =============================================================================
# Rescue Solver (CVRPTW)
# =============================================================================


class RescueSolver(BaseSolver):
    """
    Capacitated Vehicle Routing with Time Windows (CVRPTW) solver.
    
    Optimizes for:
    - Visiting all high-priority targets
    - Respecting vehicle capacity
    - Meeting TTL constraints
    - Matching vehicle capabilities to target requirements
    
    Algorithm: Greedy priority-based assignment with route optimization
    """
    
    solver_id = "rescue_cvrptw"
    name = "Rescue CVRPTW Solver"
    api_version = "1.0"
    implementation_version = "1.0.0"
    supported_objectives = [
        ObjectiveType.MIN_TIME,
        ObjectiveType.MIN_RISK,
        ObjectiveType.BALANCED,
    ]
    
    def __init__(self, max_stops_per_trip: int = 4):
        self.max_stops_per_trip = max_stops_per_trip
    
    def solve(
        self,
        graph: nx.MultiDiGraph,
        mission: MissionProfile,
        asset_graphs: dict[str, nx.MultiDiGraph],
    ) -> SolutionResult:
        """Solve rescue mission with priority-based assignment."""
        import time
        start_time = time.time()
        
        assignments: list[AssetAssignment] = []
        assigned_ids: set[str] = set()
        
        # Sort targets by priority (highest first)
        sorted_targets = sorted(
            mission.targets,
            key=lambda t: (t.priority, -(t.ttl_hours or 999)),
            reverse=True
        )
        
        # Sort assets: medical first, then by capacity
        sorted_assets = sorted(
            mission.assets,
            key=lambda a: (
                AssetCapability.MEDICAL in a.capabilities,
                a.capacity
            ),
            reverse=True
        )
        
        depot_lat = mission.depot.get("lat", 0)
        depot_lon = mission.depot.get("lon", 0)
        
        for asset in sorted_assets:
            # Get asset-specific graph
            G = asset_graphs.get(asset.id, graph)
            
            # Find targets this asset can serve
            remaining = [t for t in sorted_targets if t.id not in assigned_ids]
            if not remaining:
                break
            
            matched = self._match_targets_to_asset(
                asset, remaining, G, depot_lat, depot_lon
            )
            
            if matched:
                # Build route
                route_segments = self._build_route(
                    asset, matched, G, depot_lat, depot_lon
                )
                
                for t in matched:
                    assigned_ids.add(t.id)
                
                assignment = AssetAssignment(
                    asset_id=asset.id,
                    asset_name=asset.name,
                    targets=matched,
                    route_segments=route_segments,
                    total_population=sum(t.population for t in matched),
                    total_distance_km=sum(s.distance_km for s in route_segments),
                    total_time_min=sum(s.travel_time_min for s in route_segments),
                )
                assignments.append(assignment)
        
        unassigned = [t for t in mission.targets if t.id not in assigned_ids]
        
        solve_time = (time.time() - start_time) * 1000
        
        result = SolutionResult(
            success=len(assignments) > 0,
            assignments=assignments,
            unassigned_targets=unassigned,
            statistics={
                "algorithm": "greedy_priority",
                "max_stops_per_trip": self.max_stops_per_trip,
            },
            solve_time_ms=solve_time,
        )
        
        # Score the solution
        for a in result.assignments:
            a.score = self._score_assignment(a, mission)
        
        return result
    
    def _match_targets_to_asset(
        self,
        asset: Asset,
        targets: list[Target],
        graph: nx.MultiDiGraph,
        depot_lat: float,
        depot_lon: float,
    ) -> list[Target]:
        """
        Match targets to asset based on capability, reachability, and TTL.

        Targets are selected sequentially from the current route position so
        TTL checks use estimated arrival time, not just depot reachability.
        """
        matched: list[Target] = []
        remaining_capacity = asset.capacity

        depot_node = self._find_nearest_node(graph, depot_lat, depot_lon)
        if depot_node is None:
            return matched

        current_node = depot_node
        elapsed_time_min = 0.0
        remaining_targets = list(targets)

        while (
            remaining_capacity > 0
            and len(matched) < self.max_stops_per_trip
            and remaining_targets
        ):
            best_candidate = None
            best_score = None

            for target in remaining_targets:
                if target.population > remaining_capacity:
                    continue

                if target.required_capabilities:
                    has_all = all(
                        asset.has_capability(cap)
                        for cap in target.required_capabilities
                    )
                    if not has_all:
                        continue

                target_node = self._find_nearest_node(graph, target.lat, target.lon)
                if target_node is None:
                    continue

                try:
                    _, segment_distance_km = self._compute_path_and_distance(
                        graph, current_node, target_node
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

                segment_time_min = self._distance_to_travel_time_min(
                    segment_distance_km, asset.speed_kmh
                )
                arrival_time_min = elapsed_time_min + segment_time_min

                if not self._meets_ttl_window(target, arrival_time_min):
                    continue

                # Higher priority first, then earlier arrival, then shorter hop.
                score = (target.priority, -arrival_time_min, -segment_distance_km)
                if best_score is None or score > best_score:
                    best_score = score
                    best_candidate = (target, target_node, segment_time_min)

            if best_candidate is None:
                break

            target, target_node, segment_time_min = best_candidate
            matched.append(target)
            remaining_targets.remove(target)
            remaining_capacity -= target.population
            elapsed_time_min += segment_time_min
            current_node = target_node
        
        return matched
    
    def _build_route(
        self,
        asset: Asset,
        targets: list[Target],
        graph: nx.MultiDiGraph,
        depot_lat: float,
        depot_lon: float,
    ) -> list[RouteSegment]:
        """Build route segments for assigned targets."""
        segments: list[RouteSegment] = []
        
        depot_node = self._find_nearest_node(graph, depot_lat, depot_lon)
        if depot_node is None:
            return segments
        
        current = depot_node
        current_id = "DEPOT"
        
        for target in targets:
            target_node = self._find_nearest_node(graph, target.lat, target.lon)
            if target_node is None:
                continue
            
            try:
                path, distance = self._compute_path_and_distance(
                    graph, current, target_node
                )
                travel_time = self._distance_to_travel_time_min(
                    distance, asset.speed_kmh
                )
                
                segments.append(RouteSegment(
                    from_id=current_id,
                    to_id=target.id,
                    distance_km=distance,
                    travel_time_min=travel_time,
                    path_nodes=path,
                ))
                
                current = target_node
                current_id = target.id
            except nx.NetworkXNoPath:
                continue
        
        # Return to depot
        if current != depot_node:
            try:
                path, distance = self._compute_path_and_distance(
                    graph, current, depot_node
                )
                travel_time = self._distance_to_travel_time_min(
                    distance, asset.speed_kmh
                )
                
                segments.append(RouteSegment(
                    from_id=current_id,
                    to_id="DEPOT",
                    distance_km=distance,
                    travel_time_min=travel_time,
                    path_nodes=path,
                ))
            except nx.NetworkXNoPath:
                pass
        
        return segments

    def _compute_path_and_distance(
        self,
        graph: nx.MultiDiGraph,
        source: int,
        target: int,
    ) -> tuple[list[int], float]:
        """
        Compute weighted path and physical distance (km) along that path.

        For MultiDiGraph parallel edges, distance uses the edge with the
        minimum routing weight on each hop to stay consistent with the chosen
        weighted path.
        """
        path = nx.shortest_path(graph, source, target, weight="weight")
        distance_m = 0.0

        for u, v in zip(path, path[1:]):
            edge_bundle = graph.get_edge_data(u, v, default={})
            if not edge_bundle:
                raise nx.NetworkXNoPath(f"No edge between {u} and {v}")

            best_edge = min(
                edge_bundle.values(),
                key=lambda data: data.get("weight", 1.0),
            )
            distance_m += float(best_edge.get("length", 0.0))

        return path, distance_m / 1000.0

    def _distance_to_travel_time_min(self, distance_km: float, speed_kmh: float) -> float:
        """Convert segment distance to travel time in minutes."""
        if speed_kmh <= 0:
            return float("inf")
        return (distance_km / speed_kmh) * 60.0

    def _meets_ttl_window(self, target: Target, arrival_time_min: float) -> bool:
        """Check whether target can be reached before its TTL (if provided)."""
        if target.ttl_hours is None:
            return True
        return arrival_time_min <= target.ttl_hours * 60.0
    
    def _find_nearest_node(
        self,
        graph: nx.MultiDiGraph,
        lat: float,
        lon: float
    ) -> Optional[int]:
        """Find nearest graph node to coordinates."""
        min_dist = float("inf")
        nearest = None
        
        for node, data in graph.nodes(data=True):
            node_lat = data.get("y", data.get("lat", 0))
            node_lon = data.get("x", data.get("lon", 0))
            
            # Simple Euclidean approximation
            dist = ((lat - node_lat) ** 2 + (lon - node_lon) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _score_assignment(
        self,
        assignment: AssetAssignment,
        mission: MissionProfile
    ) -> float:
        """Score an individual assignment."""
        # Priority score
        priority_sum = sum(t.priority for t in assignment.targets)
        
        # Efficiency (population per km)
        if assignment.total_distance_km > 0:
            efficiency = assignment.total_population / assignment.total_distance_km
        else:
            efficiency = 0
        
        return priority_sum * 0.7 + efficiency * 0.3


# =============================================================================
# Patrol Solver (Edge Coverage)
# =============================================================================


class PatrolSolver(BaseSolver):
    """
    Edge coverage maximization solver.
    
    Optimizes for covering the maximum road network within constraints.
    Uses Chinese Postman / Arc Routing approaches.
    
    Suitable for:
    - Patrol missions
    - Damage assessment
    - Supply distribution along routes
    """
    
    solver_id = "patrol_coverage"
    name = "Patrol Coverage Solver"
    api_version = "1.0"
    implementation_version = "1.0.0"
    supported_objectives = [ObjectiveType.MAX_COVERAGE]
    
    def solve(
        self,
        graph: nx.MultiDiGraph,
        mission: MissionProfile,
        asset_graphs: dict[str, nx.MultiDiGraph],
    ) -> SolutionResult:
        """Solve patrol mission for maximum edge coverage."""
        import time
        start_time = time.time()
        
        assignments: list[AssetAssignment] = []
        covered_edges: set[tuple] = set()
        
        depot_lat = mission.depot.get("lat", 0)
        depot_lon = mission.depot.get("lon", 0)
        
        for asset in mission.assets:
            G = asset_graphs.get(asset.id, graph)
            
            # Generate patrol route (simplified: DFS from depot)
            route_edges = self._generate_patrol_route(
                G, depot_lat, depot_lon, 
                asset.fuel_range_km, 
                covered_edges
            )
            
            if route_edges:
                # Mark edges as covered
                for edge in route_edges:
                    covered_edges.add(edge)
                
                # Calculate stats
                total_dist = sum(
                    G[u][v].get(0, {}).get("length", 0) / 1000 
                    for u, v in route_edges
                )
                
                assignment = AssetAssignment(
                    asset_id=asset.id,
                    asset_name=asset.name,
                    targets=[],  # No point targets for patrol
                    route_segments=[],  # Would convert edges to segments
                    total_population=0,
                    total_distance_km=total_dist,
                    total_time_min=(total_dist / asset.speed_kmh) * 60,
                    score=len(route_edges),
                )
                assignments.append(assignment)
        
        solve_time = (time.time() - start_time) * 1000
        
        total_edges = graph.number_of_edges()
        coverage_pct = (len(covered_edges) / total_edges * 100) if total_edges > 0 else 0
        
        return SolutionResult(
            success=len(assignments) > 0,
            assignments=assignments,
            unassigned_targets=[],
            statistics={
                "edges_covered": len(covered_edges),
                "total_edges": total_edges,
                "coverage_percent": round(coverage_pct, 1),
            },
            solve_time_ms=solve_time,
        )
    
    def _generate_patrol_route(
        self,
        graph: nx.MultiDiGraph,
        depot_lat: float,
        depot_lon: float,
        max_distance_km: float,
        already_covered: set[tuple],
    ) -> list[tuple]:
        """Generate patrol route using greedy edge selection."""
        route: list[tuple] = []
        total_distance = 0.0
        
        # Find depot node
        depot = self._find_nearest_node(graph, depot_lat, depot_lon)
        if depot is None:
            return route
        
        current = depot
        visited_nodes = {current}
        
        while total_distance < max_distance_km:
            # Find uncovered neighbor edge
            next_edge = None
            next_node = None
            min_length = float("inf")
            
            for _, neighbor, data in graph.out_edges(current, data=True):
                edge = (current, neighbor)
                if edge in already_covered or edge in route:
                    continue
                
                length = data.get("length", 1000) / 1000
                if length < min_length:
                    min_length = length
                    next_edge = edge
                    next_node = neighbor
            
            if next_edge is None:
                break
            
            if total_distance + min_length > max_distance_km:
                break
            
            route.append(next_edge)
            total_distance += min_length
            current = next_node
            visited_nodes.add(current)
        
        return route
    
    def _find_nearest_node(self, graph, lat, lon):
        """Find nearest node (same as RescueSolver)."""
        min_dist = float("inf")
        nearest = None
        for node, data in graph.nodes(data=True):
            node_lat = data.get("y", data.get("lat", 0))
            node_lon = data.get("x", data.get("lon", 0))
            dist = ((lat - node_lat) ** 2 + (lon - node_lon) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = node
        return nearest


# =============================================================================
# Auto-register default solvers
# =============================================================================


def register_default_solvers():
    """Register built-in solvers."""
    SolverRegistry.register(RescueSolver())
    SolverRegistry.register(PatrolSolver())


# Register on import
register_default_solvers()
