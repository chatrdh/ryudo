"""
Mission Coordinator
===================

Top-level orchestrator for mission solving.

The MissionCoordinator:
1. Takes a MissionProfile (objectives, assets, targets)
2. Builds asset-specific graph views
3. Selects appropriate solver
4. Runs optimization
5. Returns unified solution
"""

from datetime import datetime, timezone
from typing import Any, Optional

import networkx as nx

from ryudo.core.schema import GraphConstraint
from ryudo.mission.multimodal import ViewBuilder
from ryudo.mission.profile import Asset, MissionProfile
from ryudo.mission.solver import BaseSolver, SolverRegistry, SolutionResult
from ryudo.platform.replay import deterministic_run_id


class MissionCoordinator:
    """
    High-level mission orchestration.
    
    Coordinates between:
    - LivingGraph (constraint storage)
    - ViewBuilder (multi-modal views)
    - Solvers (optimization)
    
    Example
    -------
    ```python
    coordinator = MissionCoordinator(living_graph)
    
    profile = MissionProfile(
        objective=ObjectiveType.MIN_TIME,
        assets=[truck, boat],
        targets=targets,
    )
    
    solution = coordinator.solve_mission(profile, constraints)
    
    for assignment in solution.assignments:
        print(f"{assignment.asset_name}: {assignment.target_ids}")
    ```
    """
    
    def __init__(
        self,
        living_graph: Optional["LivingGraph"] = None,
        solver_override: Optional[BaseSolver] = None,
    ):
        """
        Initialize MissionCoordinator.
        
        Parameters
        ----------
        living_graph : LivingGraph, optional
            Reference to the living graph engine
        solver_override : BaseSolver, optional
            Force use of a specific solver (ignores objective matching)
        """
        self._living_graph = living_graph
        self._view_builder = ViewBuilder(living_graph)
        self._solver_override = solver_override
    
    def solve_mission(
        self,
        profile: MissionProfile,
        constraints: list[GraphConstraint],
        query_time: Optional[datetime] = None,
        base_graph: Optional[nx.MultiDiGraph] = None,
        run_id: Optional[str] = None,
    ) -> SolutionResult:
        """
        Solve a complete mission.
        
        Steps:
        1. Build asset-specific graph views (multi-modal routing)
        2. Select appropriate solver for objective
        3. Run optimization
        4. Return unified solution
        
        Parameters
        ----------
        profile : MissionProfile
            Complete mission specification
        constraints : list[GraphConstraint]
            Current active constraints from agents
        query_time : datetime, optional
            Query time (defaults to now)
        base_graph : nx.MultiDiGraph, optional
            Base graph to use (defaults to living graph)
        
        Returns
        -------
        SolutionResult
            Complete solution with assignments and routes
        """
        if query_time is None:
            query_time = datetime.now(timezone.utc)
        
        print(f"[Coordinator] Solving mission: {profile.name}")
        print(f"[Coordinator] Objective: {profile.objective.value}")
        print(f"[Coordinator] Assets: {len(profile.assets)}, Targets: {len(profile.targets)}")
        
        # Step 1: Build asset-specific views
        asset_views = self._build_asset_views(
            profile.assets, constraints, query_time, base_graph
        )
        
        # Step 2: Select solver
        solver = self._select_solver(profile)
        print(f"[Coordinator] Using solver: {solver.name}")
        
        # Step 3: Build deterministic run identity
        resolved_run_id = run_id or deterministic_run_id(
            {
                "mission": profile.to_dict(),
                "constraints": [self._constraint_payload(c) for c in constraints],
            },
            namespace="ryudo.mission.solve.v1",
        )

        # Step 4: Get base graph for reference
        if base_graph is not None:
            graph = base_graph
        elif self._living_graph is not None:
            graph = self._living_graph.get_view(query_time=query_time)
        else:
            # Use first asset view as fallback
            graph = next(iter(asset_views.values()), nx.MultiDiGraph())
        
        # Step 5: Run solver
        solution = solver.solve(graph, profile, asset_views)
        
        # Step 6: Enrich solution
        solution.metadata["mission_id"] = profile.id
        solution.metadata["mission_name"] = profile.name
        solution.metadata["query_time"] = query_time.isoformat()
        solution.metadata["run_id"] = resolved_run_id
        solution.metadata["solver_id"] = solver.solver_id
        solution.metadata["solver_name"] = solver.name
        solution.metadata["solver_api_version"] = solver.api_version
        solution.metadata["solver_implementation_version"] = solver.implementation_version
        
        # Calculate overall score
        overall_score = solver.score_solution(solution, profile)
        solution.statistics["overall_score"] = round(overall_score, 3)
        
        print(f"[Coordinator] Solution: {solution.targets_reached} targets, "
              f"{solution.total_distance_km:.1f} km, score={overall_score:.3f}")
        
        return solution

    def _constraint_payload(self, constraint: GraphConstraint) -> dict[str, Any]:
        """Create stable, deterministic payload shape for run hashing."""
        payload = constraint.model_dump(mode="json")
        payload.pop("id", None)
        return payload
    
    def _build_asset_views(
        self,
        assets: list[Asset],
        constraints: list[GraphConstraint],
        query_time: datetime,
        base_graph: Optional[nx.MultiDiGraph] = None,
    ) -> dict[str, nx.MultiDiGraph]:
        """Build graph view for each asset."""
        views = {}
        
        for asset in assets:
            print(f"[Coordinator] Building view for {asset.name} "
                  f"(capabilities: {[c.value for c in asset.capabilities]})")
            
            view = self._view_builder.build_view(
                asset=asset,
                query_time=query_time,
                constraints=constraints,
                base_graph=base_graph,
            )
            
            views[asset.id] = view
            print(f"[Coordinator] View for {asset.id}: "
                  f"{view.number_of_nodes()} nodes, {view.number_of_edges()} edges")
        
        return views
    
    def _select_solver(self, profile: MissionProfile) -> BaseSolver:
        """Select appropriate solver for mission objective."""
        if self._solver_override:
            return self._solver_override
        
        try:
            return SolverRegistry.get_for_objective(profile.objective)
        except ValueError as e:
            print(f"[Coordinator] Warning: {e}")
            # Fallback to first available solver
            solvers = SolverRegistry.list_solvers()
            if solvers:
                return solvers[0]
            raise ValueError("No solvers registered")
    
    def compare_asset_views(
        self,
        assets: list[Asset],
        constraints: list[GraphConstraint],
        query_time: Optional[datetime] = None,
        base_graph: Optional[nx.MultiDiGraph] = None,
    ) -> dict[str, dict[str, Any]]:
        """
        Compare how different assets see the same constraints.
        
        Useful for debugging and visualization.
        
        Returns dict of asset_id -> {
            "nodes": int,
            "edges": int,
            "capability_weights": {...},
            "removed_edges": int,
        }
        """
        if query_time is None:
            query_time = datetime.now(timezone.utc)
        
        # Get base counts
        if base_graph is not None:
            base_nodes = base_graph.number_of_nodes()
            base_edges = base_graph.number_of_edges()
        elif self._living_graph is not None:
            base = self._living_graph.get_view(query_time=query_time)
            base_nodes = base.number_of_nodes()
            base_edges = base.number_of_edges()
        else:
            base_nodes = 0
            base_edges = 0
        
        comparison = {}
        
        for asset in assets:
            view = self._view_builder.build_view(
                asset=asset,
                query_time=query_time,
                constraints=constraints,
                base_graph=base_graph,
            )
            
            comparison[asset.id] = {
                "name": asset.name,
                "capabilities": [c.value for c in asset.capabilities],
                "nodes": view.number_of_nodes(),
                "edges": view.number_of_edges(),
                "edges_removed": base_edges - view.number_of_edges(),
                "capability_weights": self._view_builder.get_capability_summary(asset),
            }
        
        return comparison
    
    def estimate_mission_feasibility(
        self,
        profile: MissionProfile,
        constraints: list[GraphConstraint],
        query_time: Optional[datetime] = None,
        base_graph: Optional[nx.MultiDiGraph] = None,
    ) -> dict[str, Any]:
        """
        Estimate mission feasibility before full solve.
        
        Quick checks:
        - Total capacity vs total population
        - Critical target reachability
        - Asset-to-target capability matching
        """
        if query_time is None:
            query_time = datetime.now(timezone.utc)
        
        # Capacity check
        capacity_ratio = (
            profile.total_capacity / profile.total_population
            if profile.total_population > 0 else float("inf")
        )
        
        # Build asset views
        asset_views = self._build_asset_views(
            profile.assets, constraints, query_time, base_graph
        )
        
        # Check reachability of critical targets
        critical = profile.critical_targets
        reachable_critical = 0
        
        depot_lat = profile.depot.get("lat", 0)
        depot_lon = profile.depot.get("lon", 0)
        
        for target in critical:
            for asset_id, view in asset_views.items():
                # Simple reachability check
                depot_node = self._find_nearest_node(view, depot_lat, depot_lon)
                target_node = self._find_nearest_node(view, target.lat, target.lon)
                
                if depot_node is not None and target_node is not None:
                    if nx.has_path(view, depot_node, target_node):
                        reachable_critical += 1
                        break
        
        return {
            "feasible": capacity_ratio >= 0.8 and reachable_critical >= len(critical) * 0.8,
            "capacity_ratio": round(capacity_ratio, 2),
            "total_capacity": profile.total_capacity,
            "total_population": profile.total_population,
            "critical_targets": len(critical),
            "reachable_critical": reachable_critical,
            "estimated_trips": profile.estimate_minimum_trips(),
        }
    
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
            dist = ((lat - node_lat) ** 2 + (lon - node_lon) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
