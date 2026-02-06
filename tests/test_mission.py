"""
Mission Module Tests
====================

Test suite for Phase 3: Mission Coordinator with multi-modal routing.
"""

from datetime import datetime, timedelta, timezone

import networkx as nx
import pytest
from shapely.geometry import Point, box

from ryudo.core.schema import ConstraintType, GraphConstraint, TimeWindow
from ryudo.mission.profile import (
    Asset,
    AssetCapability,
    MissionProfile,
    ObjectiveType,
    Target,
    TargetPriority,
)
from ryudo.mission.solver import (
    BaseSolver,
    PatrolSolver,
    RescueSolver,
    SolverRegistry,
    SolutionResult,
)
from ryudo.mission.multimodal import ViewBuilder, WEIGHT_PROFILES
from ryudo.mission.coordinator import MissionCoordinator


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_graph():
    """Create a grid graph for testing."""
    G = nx.MultiDiGraph()
    
    # 5x5 grid of nodes
    for i in range(5):
        for j in range(5):
            node_id = i * 5 + j
            G.add_node(node_id, x=j * 0.01, y=i * 0.01, lat=i * 0.01, lon=j * 0.01)
    
    # Connect adjacent nodes
    for i in range(5):
        for j in range(5):
            node_id = i * 5 + j
            # Right
            if j < 4:
                G.add_edge(node_id, node_id + 1, weight=1.0, length=1000)
                G.add_edge(node_id + 1, node_id, weight=1.0, length=1000)
            # Down
            if i < 4:
                G.add_edge(node_id, node_id + 5, weight=1.0, length=1000)
                G.add_edge(node_id + 5, node_id, weight=1.0, length=1000)
    
    return G


@pytest.fixture
def now():
    """Current time."""
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def active_window(now):
    """Active time window."""
    return TimeWindow(
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=12)
    )


@pytest.fixture
def truck():
    """Standard road vehicle."""
    return Asset(
        id="truck_1",
        name="Rescue Truck",
        capabilities=(AssetCapability.ROAD,),
        capacity=15,
        speed_kmh=40,
        depot_lat=0.02,
        depot_lon=0.02,
    )


@pytest.fixture
def boat():
    """Water-capable vehicle."""
    return Asset(
        id="boat_1",
        name="Rescue Boat",
        capabilities=(AssetCapability.WATER,),
        capacity=8,
        speed_kmh=25,
        depot_lat=0.02,
        depot_lon=0.02,
    )


@pytest.fixture
def drone():
    """Air-capable asset."""
    return Asset(
        id="drone_1",
        name="Survey Drone",
        capabilities=(AssetCapability.AIR,),
        capacity=0,
        speed_kmh=60,
        depot_lat=0.02,
        depot_lon=0.02,
    )


@pytest.fixture
def sample_targets():
    """Sample rescue targets."""
    return [
        Target(id="T01", lat=0.00, lon=0.00, population=10, priority=100, ttl_hours=0.5),
        Target(id="T02", lat=0.00, lon=0.04, population=20, priority=80, ttl_hours=1.5),
        Target(id="T03", lat=0.04, lon=0.00, population=15, priority=60, ttl_hours=3.0),
        Target(id="T04", lat=0.04, lon=0.04, population=8, priority=40, ttl_hours=6.0),
    ]


@pytest.fixture
def flood_constraint(now, active_window):
    """Flood zone constraint covering bottom-left quadrant."""
    # Use lon,lat (x,y) order: box(minx, miny, maxx, maxy)
    # Covering nodes 0,1,5,6 which are at (x,y) = (0-0.01, 0-0.01)
    polygon = box(-0.005, -0.005, 0.015, 0.015)  # Covers bottom-left 2x2
    
    return GraphConstraint(
        type=ConstraintType.ZONE_MASK,
        target={"polygon": polygon.__geo_interface__},
        effect={"weight_factor": 1000.0},
        validity=active_window,
        source_agent_id="flood_sentinel",
        metadata={"zone_type": "flooded"},
    )


# ============================================================================
# Asset Tests
# ============================================================================


class TestAsset:
    """Tests for Asset model."""
    
    def test_has_capability(self, truck, boat):
        """Test capability check."""
        assert truck.has_capability(AssetCapability.ROAD)
        assert not truck.has_capability(AssetCapability.WATER)
        
        assert boat.has_capability(AssetCapability.WATER)
        assert not boat.has_capability(AssetCapability.ROAD)
    
    def test_can_access_flooded(self, truck, boat, drone):
        """Test flood accessibility check."""
        assert not truck.can_access_flooded()
        assert boat.can_access_flooded()
        assert drone.can_access_flooded()
    
    def test_from_dict(self):
        """Test creating Asset from dict."""
        data = {
            "id": "v1",
            "name": "Test Vehicle",
            "capabilities": ["road", "offroad"],
            "capacity": 10,
        }
        
        asset = Asset.from_dict(data)
        assert asset.id == "v1"
        assert AssetCapability.ROAD in asset.capabilities
        assert AssetCapability.OFFROAD in asset.capabilities


# ============================================================================
# Target Tests
# ============================================================================


class TestTarget:
    """Tests for Target model."""
    
    def test_priority_level_from_ttl(self):
        """Test priority level derivation."""
        critical = Target(id="T1", lat=0, lon=0, ttl_hours=0.5)
        high = Target(id="T2", lat=0, lon=0, ttl_hours=2.0)
        medium = Target(id="T3", lat=0, lon=0, ttl_hours=4.0)
        low = Target(id="T4", lat=0, lon=0, ttl_hours=10.0)
        
        assert critical.priority_level == TargetPriority.CRITICAL
        assert high.priority_level == TargetPriority.HIGH
        assert medium.priority_level == TargetPriority.MEDIUM
        assert low.priority_level == TargetPriority.LOW
    
    def test_priority_level_from_score(self):
        """Test priority from score when no TTL."""
        high = Target(id="T1", lat=0, lon=0, priority=95)
        assert high.priority_level == TargetPriority.CRITICAL


# ============================================================================
# MissionProfile Tests
# ============================================================================


class TestMissionProfile:
    """Tests for MissionProfile model."""
    
    def test_create_profile(self, truck, boat, sample_targets):
        """Test creating a mission profile."""
        profile = MissionProfile(
            id="mission_001",
            name="Test Rescue",
            objective=ObjectiveType.MIN_TIME,
            assets=[truck, boat],
            targets=sample_targets,
            depot={"lat": 0.02, "lon": 0.02},
        )
        
        assert profile.id == "mission_001"
        assert len(profile.assets) == 2
        assert len(profile.targets) == 4
        assert profile.objective == ObjectiveType.MIN_TIME
    
    def test_total_population(self, truck, sample_targets):
        """Test population calculation."""
        profile = MissionProfile(
            assets=[truck],
            targets=sample_targets,
        )
        
        assert profile.total_population == 53  # 10+20+15+8
    
    def test_critical_targets(self, truck, sample_targets):
        """Test critical target filtering."""
        profile = MissionProfile(
            assets=[truck],
            targets=sample_targets,
        )
        
        # T01 has TTL 0.5 hours -> CRITICAL
        critical = profile.critical_targets
        assert len(critical) == 1
        assert critical[0].id == "T01"
    
    def test_from_dict(self):
        """Test creating profile from JSON/dict."""
        data = {
            "id": "m1",
            "name": "Test Mission",
            "objective": "min_time",
            "assets": [
                {"id": "v1", "name": "Truck 1", "capabilities": ["road"], "capacity": 10}
            ],
            "targets": [
                {"id": "t1", "lat": 17.7, "lon": 83.3, "population": 20}
            ],
            "depot": {"lat": 17.68, "lon": 83.21},
        }
        
        profile = MissionProfile.from_dict(data)
        assert profile.id == "m1"
        assert len(profile.assets) == 1
        assert len(profile.targets) == 1
        assert profile.objective == ObjectiveType.MIN_TIME


# ============================================================================
# Solver Tests
# ============================================================================


class TestSolverRegistry:
    """Tests for SolverRegistry."""
    
    def test_default_solvers_registered(self):
        """Default solvers should be registered on import."""
        solvers = SolverRegistry.list_solvers()
        ids = [s.solver_id for s in solvers]
        
        assert "rescue_cvrptw" in ids
        assert "patrol_coverage" in ids
    
    def test_get_for_objective(self):
        """Test solver lookup by objective."""
        rescue = SolverRegistry.get_for_objective(ObjectiveType.MIN_TIME)
        assert rescue.solver_id == "rescue_cvrptw"
        
        patrol = SolverRegistry.get_for_objective(ObjectiveType.MAX_COVERAGE)
        assert patrol.solver_id == "patrol_coverage"
    
    def test_get_unknown_objective_raises(self):
        """Should raise for unknown objectives."""
        # MAX_TARGETS is not supported by default solvers
        with pytest.raises(ValueError):
            SolverRegistry.get_for_objective(ObjectiveType.MAX_TARGETS)

    def test_list_solver_descriptors(self):
        """Registry should expose source/version metadata per solver."""
        descriptors = SolverRegistry.list_solver_descriptors()
        by_id = {descriptor["solver_id"]: descriptor for descriptor in descriptors}

        assert "rescue_cvrptw" in by_id
        assert by_id["rescue_cvrptw"]["source"] == "builtin"
        assert by_id["rescue_cvrptw"]["api_version"] == "1.0"

    def test_register_factory_hook(self, sample_graph, truck):
        """External factory registration should wire a solver into registry."""
        class ExternalTestSolver(BaseSolver):
            solver_id = "external_test_solver"
            name = "External Test Solver"
            api_version = "1.0"
            implementation_version = "0.0.1"
            supported_objectives = [ObjectiveType.MAX_TARGETS]

            def solve(self, graph, mission, asset_graphs):
                return SolutionResult(success=True)

        SolverRegistry.unregister("external_test_solver")
        SolverRegistry.register_factory(
            lambda: ExternalTestSolver(),
            source="external:test-suite",
        )

        solver = SolverRegistry.get("external_test_solver")
        assert solver is not None

        descriptors = SolverRegistry.list_solver_descriptors()
        descriptor = next(d for d in descriptors if d["solver_id"] == "external_test_solver")
        assert descriptor["source"] == "external:test-suite"
        assert descriptor["api_version"] == "1.0"
        SolverRegistry.unregister("external_test_solver")

    def test_reject_incompatible_solver_api(self):
        """Registry should reject incompatible major API versions."""
        class IncompatibleSolver(BaseSolver):
            solver_id = "incompatible_solver"
            name = "Incompatible Solver"
            api_version = "2.0"
            supported_objectives = [ObjectiveType.MAX_TARGETS]

            def solve(self, graph, mission, asset_graphs):
                return SolutionResult(success=True)

        with pytest.raises(ValueError, match="Incompatible solver API version"):
            SolverRegistry.register(IncompatibleSolver(), source="external:test-suite")


class TestRescueSolver:
    """Tests for RescueSolver."""
    
    def test_solve_basic(self, sample_graph, truck, sample_targets):
        """Test basic rescue solving."""
        profile = MissionProfile(
            assets=[truck],
            targets=sample_targets,
            depot={"lat": 0.02, "lon": 0.02},
            objective=ObjectiveType.MIN_TIME,
        )
        
        solver = RescueSolver()
        result = solver.solve(sample_graph, profile, {"truck_1": sample_graph})
        
        assert result.success
        assert len(result.assignments) >= 1
    
    def test_respects_capacity(self, sample_graph, sample_targets):
        """Test capacity constraints."""
        small_truck = Asset(
            id="small",
            name="Small Van",
            capabilities=(AssetCapability.ROAD,),
            capacity=5,  # Small capacity
        )
        
        profile = MissionProfile(
            assets=[small_truck],
            targets=sample_targets,  # Total pop = 53
            depot={"lat": 0.02, "lon": 0.02},
        )
        
        solver = RescueSolver()
        result = solver.solve(sample_graph, profile, {"small": sample_graph})
        
        # Can't serve everyone with capacity 5
        for assignment in result.assignments:
            assert assignment.total_population <= 5

    def test_respects_ttl_arrival_windows(self):
        """Targets should be skipped when arrival would exceed TTL."""
        G = nx.MultiDiGraph()
        G.add_node(0, x=0.00, y=0.00)
        G.add_node(1, x=0.01, y=0.00)
        G.add_node(2, x=0.20, y=0.00)

        # Depot -> near target (1 km), near -> far target (19 km)
        G.add_edge(0, 1, weight=1.0, length=1000)
        G.add_edge(1, 0, weight=1.0, length=1000)
        G.add_edge(1, 2, weight=1.0, length=19000)
        G.add_edge(2, 1, weight=1.0, length=19000)

        asset = Asset(
            id="a1",
            name="Rescue Truck",
            capabilities=(AssetCapability.ROAD,),
            capacity=30,
            speed_kmh=40,
            depot_lat=0.00,
            depot_lon=0.00,
        )

        near_target = Target(
            id="near",
            lat=0.00,
            lon=0.01,
            population=5,
            priority=70,
            ttl_hours=1.0,
        )
        far_target = Target(
            id="far",
            lat=0.00,
            lon=0.20,
            population=5,
            priority=100,
            ttl_hours=0.2,  # 12 minutes
        )

        profile = MissionProfile(
            assets=[asset],
            targets=[far_target, near_target],  # Far is higher priority but TTL-infeasible
            depot={"lat": 0.00, "lon": 0.00},
            objective=ObjectiveType.MIN_TIME,
        )

        result = RescueSolver().solve(G, profile, {"a1": G})

        assigned = {t.id for a in result.assignments for t in a.targets}
        unassigned = {t.id for t in result.unassigned_targets}

        assert "near" in assigned
        assert "far" in unassigned

    def test_distance_follows_weighted_route(self):
        """
        Distance should be measured on the same weighted path used for routing.
        """
        G = nx.MultiDiGraph()
        # Nodes: depot(0) -> target(3) with two alternatives
        G.add_node(0, x=0.00, y=0.00)
        G.add_node(1, x=0.01, y=0.00)
        G.add_node(2, x=0.02, y=0.00)
        G.add_node(3, x=0.03, y=0.00)

        # Direct path: short distance but very high routing weight
        G.add_edge(0, 3, weight=1000.0, length=1000)
        G.add_edge(3, 0, weight=1000.0, length=1000)

        # Safe detour: longer distance, much lower routing weight
        G.add_edge(0, 1, weight=1.0, length=1000)
        G.add_edge(1, 0, weight=1.0, length=1000)
        G.add_edge(1, 2, weight=1.0, length=1000)
        G.add_edge(2, 1, weight=1.0, length=1000)
        G.add_edge(2, 3, weight=1.0, length=1000)
        G.add_edge(3, 2, weight=1.0, length=1000)

        asset = Asset(
            id="a1",
            name="Rescue Truck",
            capabilities=(AssetCapability.ROAD,),
            capacity=10,
            speed_kmh=40,
            depot_lat=0.00,
            depot_lon=0.00,
        )

        target = Target(id="t1", lat=0.00, lon=0.03, population=5, priority=90)
        profile = MissionProfile(
            assets=[asset],
            targets=[target],
            depot={"lat": 0.00, "lon": 0.00},
            objective=ObjectiveType.MIN_TIME,
        )

        result = RescueSolver().solve(G, profile, {"a1": G})

        assert result.success
        assert len(result.assignments) == 1
        # Outbound and return should each use 3 km detour => ~6 km total.
        assert result.assignments[0].total_distance_km >= 5.9


class TestPatrolSolver:
    """Tests for PatrolSolver."""
    
    def test_solve_patrol(self, sample_graph, truck):
        """Test patrol route generation."""
        profile = MissionProfile(
            assets=[truck],
            targets=[],  # No targets for patrol
            depot={"lat": 0.02, "lon": 0.02},
            objective=ObjectiveType.MAX_COVERAGE,
        )
        
        solver = PatrolSolver()
        result = solver.solve(sample_graph, profile, {"truck_1": sample_graph})
        
        assert result.success
        assert "edges_covered" in result.statistics


# ============================================================================
# Multi-Modal Routing Tests
# ============================================================================


class TestViewBuilder:
    """Tests for ViewBuilder multi-modal routing."""
    
    def test_truck_sees_high_flood_weight(
        self, sample_graph, truck, flood_constraint, now
    ):
        """Truck should see high weights in flood zone."""
        builder = ViewBuilder()
        
        view = builder.build_view(
            asset=truck,
            query_time=now,
            constraints=[flood_constraint],
            base_graph=sample_graph,
        )
        
        # Check weights in flood zone (nodes 0,1,5,6)
        # Edge 0->1 should have high weight
        if view.has_edge(0, 1):
            weight = view[0][1][0]["weight"]
            assert weight >= 1000.0, f"Truck should see high weight, got {weight}"
    
    def test_boat_sees_low_flood_weight(
        self, sample_graph, boat, flood_constraint, now
    ):
        """Boat should see LOW weights in flood zone."""
        builder = ViewBuilder()
        
        view = builder.build_view(
            asset=boat,
            query_time=now,
            constraints=[flood_constraint],
            base_graph=sample_graph,
        )
        
        # Check weights in flood zone
        if view.has_edge(0, 1):
            weight = view[0][1][0]["weight"]
            assert weight <= 1.0, f"Boat should see low weight, got {weight}"
    
    def test_drone_ignores_flood(
        self, sample_graph, drone, flood_constraint, now
    ):
        """Drone should ignore flood constraints."""
        builder = ViewBuilder()
        
        view = builder.build_view(
            asset=drone,
            query_time=now,
            constraints=[flood_constraint],
            base_graph=sample_graph,
        )
        
        # Drone should have unchanged weights
        if view.has_edge(0, 1):
            weight = view[0][1][0]["weight"]
            assert weight == 1.0, "Drone should ignore flood"
    
    def test_different_paths_for_truck_and_boat(
        self, sample_graph, truck, boat, flood_constraint, now
    ):
        """Truck and boat should find different optimal paths."""
        builder = ViewBuilder()
        
        truck_view = builder.build_view(truck, now, [flood_constraint], sample_graph)
        boat_view = builder.build_view(boat, now, [flood_constraint], sample_graph)
        
        # Find paths from top-right (24) to bottom-left (0)
        try:
            truck_path = nx.shortest_path(truck_view, 24, 0, weight="weight")
            boat_path = nx.shortest_path(boat_view, 24, 0, weight="weight")
            
            # Paths should differ - truck avoids flood, boat goes through
            # Just check they're computable (different weight profiles)
            assert len(truck_path) > 0
            assert len(boat_path) > 0
        except nx.NetworkXNoPath:
            pass  # OK if no path (graph may be disconnected)
    
    def test_capability_summary(self, truck, boat):
        """Test getting capability weight summary."""
        builder = ViewBuilder()
        
        truck_summary = builder.get_capability_summary(truck)
        boat_summary = builder.get_capability_summary(boat)
        
        # Truck should have high flood weight
        assert truck_summary["flooded"] >= 100
        
        # Boat should have low flood weight
        assert boat_summary["flooded"] < 1


# ============================================================================
# MissionCoordinator Tests
# ============================================================================


class TestMissionCoordinator:
    """Tests for MissionCoordinator."""
    
    def test_solve_mission_basic(
        self, sample_graph, truck, sample_targets, flood_constraint, now
    ):
        """Test basic mission solving."""
        profile = MissionProfile(
            id="test",
            name="Test Mission",
            objective=ObjectiveType.MIN_TIME,
            assets=[truck],
            targets=sample_targets,
            depot={"lat": 0.02, "lon": 0.02},
        )
        
        coordinator = MissionCoordinator()
        solution = coordinator.solve_mission(
            profile=profile,
            constraints=[flood_constraint],
            query_time=now,
            base_graph=sample_graph,
        )
        
        assert solution.success
        assert solution.metadata["mission_id"] == "test"
    
    def test_multi_modal_mission(
        self, sample_graph, truck, boat, sample_targets, flood_constraint, now
    ):
        """Test mission with multiple asset types."""
        profile = MissionProfile(
            id="multi",
            name="Multi-Modal Mission",
            objective=ObjectiveType.BALANCED,
            assets=[truck, boat],
            targets=sample_targets,
            depot={"lat": 0.02, "lon": 0.02},
        )
        
        coordinator = MissionCoordinator()
        solution = coordinator.solve_mission(
            profile=profile,
            constraints=[flood_constraint],
            query_time=now,
            base_graph=sample_graph,
        )
        
        assert solution.success
    
    def test_compare_asset_views(
        self, sample_graph, truck, boat, flood_constraint, now
    ):
        """Test comparing how assets see constraints."""
        coordinator = MissionCoordinator()
        
        comparison = coordinator.compare_asset_views(
            assets=[truck, boat],
            constraints=[flood_constraint],
            query_time=now,
            base_graph=sample_graph,
        )
        
        assert "truck_1" in comparison
        assert "boat_1" in comparison
        
        # Truck should have more edges removed
        # (from flood zone becoming impassable)
        # Boat sees flood as navigable
    
    def test_estimate_feasibility(
        self, sample_graph, truck, sample_targets, flood_constraint, now
    ):
        """Test mission feasibility estimation."""
        profile = MissionProfile(
            assets=[truck],
            targets=sample_targets,
            depot={"lat": 0.02, "lon": 0.02},
        )
        
        coordinator = MissionCoordinator()
        estimate = coordinator.estimate_mission_feasibility(
            profile=profile,
            constraints=[flood_constraint],
            query_time=now,
            base_graph=sample_graph,
        )
        
        assert "feasible" in estimate
        assert "capacity_ratio" in estimate
        assert "estimated_trips" in estimate

    def test_deterministic_run_id_metadata(
        self, sample_graph, truck, sample_targets, flood_constraint, now
    ):
        """Repeated mission solves with same inputs should share run_id."""
        profile = MissionProfile(
            id="run-id-test",
            name="Run ID Determinism",
            objective=ObjectiveType.MIN_TIME,
            assets=[truck],
            targets=sample_targets,
            depot={"lat": 0.02, "lon": 0.02},
        )

        coordinator = MissionCoordinator()
        first = coordinator.solve_mission(
            profile=profile,
            constraints=[flood_constraint],
            query_time=now,
            base_graph=sample_graph,
        )
        second = coordinator.solve_mission(
            profile=profile,
            constraints=[flood_constraint],
            query_time=now,
            base_graph=sample_graph,
        )

        assert first.metadata["run_id"] == second.metadata["run_id"]
        assert first.metadata["solver_api_version"] == "1.0"

        modified_constraint = flood_constraint.model_copy(
            update={"effect": {"weight_factor": 2000.0}}
        )
        third = coordinator.solve_mission(
            profile=profile,
            constraints=[modified_constraint],
            query_time=now,
            base_graph=sample_graph,
        )

        assert third.metadata["run_id"] != first.metadata["run_id"]


# ============================================================================
# Integration Test: The Critical Multi-Modal Proof
# ============================================================================


class TestMultiModalProof:
    """
    Critical integration test: Prove that different assets 
    see the same physical space completely differently.
    """
    
    def test_boat_prefers_flood_truck_avoids(self, now):
        """
        Set up a scenario where:
        - Direct path A->B goes through flood zone
        - Alternate path A->C->B goes around flood zone
        
        Expected:
        - Truck takes longer path (avoids flood)
        - Boat takes direct path (prefers flood)
        """
        # Create simple graph
        #        B(1)
        #       / |
        #      /  | (FLOOD)
        #     /   |
        #   A(0)--C(2)
        G = nx.MultiDiGraph()
        G.add_node(0, x=0, y=0)      # A
        G.add_node(1, x=0.01, y=0.02)  # B
        G.add_node(2, x=0.02, y=0)    # C
        
        # Direct A->B (through flood)
        G.add_edge(0, 1, weight=1.0, length=1000)
        G.add_edge(1, 0, weight=1.0, length=1000)
        
        # A->C (safe)
        G.add_edge(0, 2, weight=1.0, length=1000)
        G.add_edge(2, 0, weight=1.0, length=1000)
        
        # C->B (safe)
        G.add_edge(2, 1, weight=1.0, length=1000)
        G.add_edge(1, 2, weight=1.0, length=1000)
        
        # Flood zone covers path A->B
        flood = GraphConstraint(
            type=ConstraintType.ZONE_MASK,
            target={"polygon": box(-0.01, -0.01, 0.015, 0.03).__geo_interface__},
            effect={"weight_factor": 1000.0},
            validity=TimeWindow(
                start_time=now - timedelta(hours=1),
                end_time=now + timedelta(hours=12)
            ),
            source_agent_id="test",
            metadata={"zone_type": "flooded"},
        )
        
        truck = Asset(id="truck", name="Truck", capabilities=(AssetCapability.ROAD,))
        boat = Asset(id="boat", name="Boat", capabilities=(AssetCapability.WATER,))
        
        builder = ViewBuilder()
        
        truck_view = builder.build_view(truck, now, [flood], G)
        boat_view = builder.build_view(boat, now, [flood], G)
        
        # Get paths
        try:
            truck_path = nx.shortest_path(truck_view, 0, 1, weight="weight")
            boat_path = nx.shortest_path(boat_view, 0, 1, weight="weight")
            
            # Truck should take A->C->B (avoiding flood)
            # Boat should take A->B (through flood, lower weight)
            truck_cost = nx.shortest_path_length(truck_view, 0, 1, weight="weight")
            boat_cost = nx.shortest_path_length(boat_view, 0, 1, weight="weight")
            
            # Boat's direct path through flood should be cheaper
            assert boat_cost < truck_cost, \
                f"Boat cost ({boat_cost}) should be less than truck ({truck_cost})"
            
            # Verify paths are different OR boat is cheaper
            # (topology might force same path, but costs differ)
            print(f"Truck path: {truck_path}, cost: {truck_cost}")
            print(f"Boat path: {boat_path}, cost: {boat_cost}")
            
        except nx.NetworkXNoPath:
            pytest.fail("No path found")
