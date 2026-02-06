"""
Agent SDK Tests
===============

Test suite for the Agent SDK interface, orchestrator, and reference agents.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import networkx as nx
import pytest

from ryudo.core.schema import ConstraintType, GraphConstraint, TimeWindow
from ryudo.agents.sdk.interface import AgentResult, BaseAgent, WorldState
from ryudo.agents.sdk.orchestrator import AgentOrchestrator, ConflictStrategy
from ryudo.agents.sdk.reference import FloodSentinel, GridGuardian, RoutePilot


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_graph():
    """Create a simple test graph."""
    G = nx.MultiDiGraph()
    for i in range(1, 10):
        G.add_node(i, x=i * 0.01, y=i * 0.01)
    
    edges = [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9),
             (1,4), (4,7), (2,5), (5,8), (3,6), (6,9)]
    for u, v in edges:
        G.add_edge(u, v, weight=1.0, length=100)
    
    return G


@pytest.fixture
def now():
    """Current time."""
    return datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def world_state(sample_graph, now):
    """Create a WorldState fixture."""
    return WorldState(
        graph_view=sample_graph,
        query_time=now,
        signals={
            "cyclone": {
                "eye_lat": 17.72,
                "eye_lon": 83.30,
                "config": {
                    "name": "Test Cyclone",
                    "max_wind_speed_kmh": 180,
                    "extreme_damage_radius_km": 10,
                    "severe_damage_radius_km": 25,
                    "moderate_damage_radius_km": 50,
                }
            }
        },
        metadata={"mission": "test"},
    )


@pytest.fixture
def living_graph(sample_graph):
    """Create a LivingGraph with test data."""
    from ryudo.core.engine import LivingGraph
    lg = LivingGraph()
    lg.load_from_graph(sample_graph)
    return lg


# ============================================================================
# WorldState Tests
# ============================================================================


class TestWorldState:
    """Tests for WorldState dataclass."""
    
    def test_worldstate_is_frozen(self, sample_graph, now):
        """WorldState should be immutable."""
        state = WorldState(
            graph_view=sample_graph,
            query_time=now,
            signals={"test": 123},
        )
        
        with pytest.raises(AttributeError):
            state.query_time = now + timedelta(hours=1)
    
    def test_get_signal(self, world_state):
        """Test signal access."""
        assert world_state.get_signal("cyclone") is not None
        assert world_state.get_signal("nonexistent") is None
        assert world_state.get_signal("nonexistent", "default") == "default"
    
    def test_has_signal(self, world_state):
        """Test signal presence check."""
        assert world_state.has_signal("cyclone")
        assert not world_state.has_signal("nonexistent")


# ============================================================================
# AgentResult Tests
# ============================================================================


class TestAgentResult:
    """Tests for AgentResult dataclass."""
    
    def test_empty_result(self):
        """Empty result check."""
        result = AgentResult()
        assert result.is_empty()
    
    def test_result_with_constraints(self, now):
        """Result with constraints."""
        c = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 1},
            effect={"action": "disable"},
            validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
            source_agent_id="test"
        )
        result = AgentResult(constraints=[c])
        assert not result.is_empty()
    
    def test_merge_results(self, now):
        """Test merging two results."""
        c1 = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 1},
            effect={"action": "disable"},
            validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
            source_agent_id="agent1"
        )
        c2 = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 2},
            effect={"action": "disable"},
            validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
            source_agent_id="agent2"
        )
        
        r1 = AgentResult(constraints=[c1], confidence=0.9)
        r2 = AgentResult(constraints=[c2], confidence=0.8)
        
        merged = r1.merge(r2)
        assert len(merged.constraints) == 2
        assert merged.confidence == 0.8  # min


# ============================================================================
# BaseAgent Tests
# ============================================================================


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""
    
    def test_cannot_instantiate_base(self):
        """Cannot instantiate abstract class."""
        with pytest.raises(TypeError):
            BaseAgent()
    
    def test_concrete_agent(self, world_state, now):
        """Test a concrete agent implementation."""
        class TestAgent(BaseAgent):
            agent_id = "test_agent"
            priority = 5
            
            def observe(self, state: WorldState) -> AgentResult:
                c = GraphConstraint(
                    type=ConstraintType.NODE_STATUS,
                    target={"node_id": 1},
                    effect={"action": "disable"},
                    validity=TimeWindow(
                        start_time=state.query_time,
                        end_time=state.query_time + timedelta(hours=1)
                    ),
                    source_agent_id=self.agent_id
                )
                return AgentResult(constraints=[c])
        
        agent = TestAgent()
        result = agent.observe(world_state)
        
        assert len(result.constraints) == 1
        assert result.constraints[0].source_agent_id == "test_agent"
    
    def test_validate_result(self, now):
        """Test result validation."""
        class TestAgent(BaseAgent):
            agent_id = "test_agent"
            def observe(self, state): 
                return AgentResult()
        
        agent = TestAgent()
        
        # Valid result
        valid = AgentResult(constraints=[
            GraphConstraint(
                type=ConstraintType.NODE_STATUS,
                target={"node_id": 1},
                effect={"action": "disable"},
                validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
                source_agent_id="test_agent"
            )
        ])
        assert len(agent.validate_result(valid)) == 0
        
        # Invalid: wrong source_agent_id
        invalid = AgentResult(constraints=[
            GraphConstraint(
                type=ConstraintType.NODE_STATUS,
                target={"node_id": 1},
                effect={"action": "disable"},
                validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
                source_agent_id="wrong_agent"
            )
        ])
        errors = agent.validate_result(invalid)
        assert len(errors) == 1


# ============================================================================
# FloodSentinel Tests
# ============================================================================


class TestFloodSentinel:
    """Tests for FloodSentinel reference agent."""
    
    def test_creates_zone_constraints(self, world_state):
        """Should create zone constraints from cyclone data."""
        agent = FloodSentinel()
        result = agent.observe(world_state)
        
        assert len(result.constraints) >= 3  # At least 3 zones
        assert all(c.type == ConstraintType.ZONE_MASK for c in result.constraints)
        assert all(c.source_agent_id == "flood_sentinel" for c in result.constraints)
    
    def test_no_data_returns_empty(self, sample_graph, now):
        """Should return empty if no environmental data."""
        state = WorldState(
            graph_view=sample_graph,
            query_time=now,
            signals={},
        )
        
        agent = FloodSentinel()
        result = agent.observe(state)
        
        assert result.is_empty()
    
    def test_custom_zones(self, world_state):
        """Should use custom zone configuration."""
        from ryudo.agents.sdk.reference.flood_sentinel import ZoneConfig
        
        custom_zones = [
            ZoneConfig("critical", 5, 2000.0, "#000000", "Critical zone"),
        ]
        
        agent = FloodSentinel(zones=custom_zones)
        result = agent.observe(world_state)
        
        assert len(result.constraints) >= 1
        # Find the critical zone constraint
        critical = [c for c in result.constraints if c.metadata.get("zone_type") == "critical"]
        assert len(critical) == 1
        assert critical[0].effect["weight_factor"] == 2000.0


# ============================================================================
# GridGuardian Tests
# ============================================================================


class TestGridGuardian:
    """Tests for GridGuardian reference agent."""
    
    def test_processes_failed_substations(self, sample_graph, now):
        """Should process explicitly failed substations."""
        state = WorldState(
            graph_view=sample_graph,
            query_time=now,
            signals={
                "infrastructure": {
                    "failed_substations": ["substation_north"]
                }
            },
        )
        
        agent = GridGuardian()
        result = agent.observe(state)
        
        assert len(result.constraints) >= 1
        assert all(c.source_agent_id == "grid_guardian" for c in result.constraints)
    
    def test_assesses_environmental_impact(self, world_state):
        """Should detect substations in damage zone."""
        agent = GridGuardian()
        result = agent.observe(world_state)
        
        # Should detect impact based on cyclone position
        assert result.confidence > 0


# ============================================================================
# RoutePilot Tests
# ============================================================================


class TestRoutePilot:
    """Tests for RoutePilot reference agent."""
    
    def test_generates_temporal_predictions(self, world_state):
        """Should generate predictions from environmental data."""
        agent = RoutePilot()
        result = agent.observe(world_state)
        
        assert len(result.constraints) >= 1
        assert all(c.source_agent_id == "route_pilot" for c in result.constraints)
    
    def test_respects_min_confidence(self, world_state):
        """Should filter low-confidence predictions."""
        agent = RoutePilot(min_confidence=0.9)
        result = agent.observe(world_state)
        
        # All remaining constraints should have high confidence
        for c in result.constraints:
            assert c.metadata.get("confidence", 1.0) >= 0.9


# ============================================================================
# AgentOrchestrator Tests
# ============================================================================


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator."""
    
    def test_register_agent(self, living_graph):
        """Test agent registration."""
        orch = AgentOrchestrator(living_graph)
        
        orch.register_agent(FloodSentinel())
        assert orch.agent_count == 1
        
        orch.register_agent(GridGuardian())
        assert orch.agent_count == 2
    
    def test_duplicate_registration_fails(self, living_graph):
        """Cannot register same agent twice."""
        orch = AgentOrchestrator(living_graph)
        orch.register_agent(FloodSentinel())
        
        with pytest.raises(ValueError):
            orch.register_agent(FloodSentinel())
    
    def test_unregister_agent(self, living_graph):
        """Test agent unregistration."""
        orch = AgentOrchestrator(living_graph)
        orch.register_agent(FloodSentinel())
        
        assert orch.unregister_agent("flood_sentinel")
        assert orch.agent_count == 0
        
        assert not orch.unregister_agent("nonexistent")
    
    def test_list_agents_by_priority(self, living_graph):
        """Agents should be listed by priority (highest first)."""
        orch = AgentOrchestrator(living_graph)
        orch.register_agent(RoutePilot())      # priority=5
        orch.register_agent(FloodSentinel())   # priority=10
        orch.register_agent(GridGuardian())    # priority=8
        
        agents = orch.list_agents()
        priorities = [a.priority for a in agents]
        
        assert priorities == sorted(priorities, reverse=True)
    
    @pytest.mark.asyncio
    async def test_run_agents_parallel(self, living_graph, now):
        """Test parallel agent execution."""
        orch = AgentOrchestrator(living_graph)
        orch.register_agent(FloodSentinel())
        orch.register_agent(GridGuardian())
        orch.register_agent(RoutePilot())
        
        constraints = await orch.run_agents(
            signals={
                "cyclone": {
                    "eye_lat": 17.72,
                    "eye_lon": 83.30,
                    "config": {
                        "extreme_damage_radius_km": 10,
                        "severe_damage_radius_km": 25,
                        "moderate_damage_radius_km": 50,
                    }
                }
            },
            query_time=now,
        )
        
        assert len(constraints) > 0
        # Verify constraints from multiple agents
        sources = set(c.source_agent_id for c in constraints)
        assert len(sources) >= 2
    
    def test_run_agents_sync(self, living_graph, now):
        """Test synchronous wrapper."""
        orch = AgentOrchestrator(living_graph)
        orch.register_agent(FloodSentinel())
        
        constraints = orch.run_agents_sync(
            signals={
                "cyclone": {
                    "eye_lat": 17.72,
                    "eye_lon": 83.30,
                    "config": {}
                }
            },
            query_time=now,
        )
        
        assert len(constraints) > 0


# ============================================================================
# Conflict Resolution Tests
# ============================================================================


class TestConflictResolution:
    """Tests for constraint conflict resolution."""
    
    def test_highest_cost_strategy(self, living_graph, now):
        """HIGHEST_COST should use max weight factor."""
        orch = AgentOrchestrator(
            living_graph, 
            conflict_strategy=ConflictStrategy.HIGHEST_COST
        )
        
        class Agent1(BaseAgent):
            agent_id = "agent1"
            def observe(self, state):
                return AgentResult(constraints=[
                    GraphConstraint(
                        type=ConstraintType.EDGE_WEIGHT,
                        target={"edge": (1, 2, 0)},
                        effect={"weight_factor": 2.0},
                        validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
                        source_agent_id=self.agent_id
                    )
                ])
        
        class Agent2(BaseAgent):
            agent_id = "agent2"
            def observe(self, state):
                return AgentResult(constraints=[
                    GraphConstraint(
                        type=ConstraintType.EDGE_WEIGHT,
                        target={"edge": (1, 2, 0)},
                        effect={"weight_factor": 5.0},
                        validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
                        source_agent_id=self.agent_id
                    )
                ])
        
        orch.register_agent(Agent1())
        orch.register_agent(Agent2())
        
        constraints = orch.run_agents_sync(signals={}, query_time=now)
        
        # Should resolve to single constraint with factor 5.0
        edge_constraints = [c for c in constraints if c.type == ConstraintType.EDGE_WEIGHT]
        assert len(edge_constraints) == 1
        assert edge_constraints[0].effect["weight_factor"] == 5.0
    
    def test_merge_multiply_strategy(self, living_graph, now):
        """MERGE_MULTIPLY should multiply all factors."""
        orch = AgentOrchestrator(
            living_graph,
            conflict_strategy=ConflictStrategy.MERGE_MULTIPLY
        )
        
        class Agent1(BaseAgent):
            agent_id = "agent1"
            priority = 10
            def observe(self, state):
                return AgentResult(constraints=[
                    GraphConstraint(
                        type=ConstraintType.EDGE_WEIGHT,
                        target={"edge": (1, 2, 0)},
                        effect={"weight_factor": 2.0},
                        validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
                        source_agent_id=self.agent_id
                    )
                ])
        
        class Agent2(BaseAgent):
            agent_id = "agent2"
            priority = 5
            def observe(self, state):
                return AgentResult(constraints=[
                    GraphConstraint(
                        type=ConstraintType.EDGE_WEIGHT,
                        target={"edge": (1, 2, 0)},
                        effect={"weight_factor": 3.0},
                        validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
                        source_agent_id=self.agent_id
                    )
                ])
        
        orch.register_agent(Agent1())
        orch.register_agent(Agent2())
        
        constraints = orch.run_agents_sync(signals={}, query_time=now)
        
        edge_constraints = [c for c in constraints if c.type == ConstraintType.EDGE_WEIGHT]
        assert len(edge_constraints) == 1
        assert edge_constraints[0].effect["weight_factor"] == 6.0  # 2 * 3
    
    def test_node_disable_always_wins(self, living_graph, now):
        """Disable action should always win for safety."""
        orch = AgentOrchestrator(living_graph)
        
        class Agent1(BaseAgent):
            agent_id = "agent1"
            def observe(self, state):
                return AgentResult(constraints=[
                    GraphConstraint(
                        type=ConstraintType.NODE_STATUS,
                        target={"node_id": 5},
                        effect={"action": "enable"},
                        validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
                        source_agent_id=self.agent_id
                    )
                ])
        
        class Agent2(BaseAgent):
            agent_id = "agent2"
            def observe(self, state):
                return AgentResult(constraints=[
                    GraphConstraint(
                        type=ConstraintType.NODE_STATUS,
                        target={"node_id": 5},
                        effect={"action": "disable"},
                        validity=TimeWindow(start_time=now, end_time=now + timedelta(hours=1)),
                        source_agent_id=self.agent_id
                    )
                ])
        
        orch.register_agent(Agent1())
        orch.register_agent(Agent2())
        
        constraints = orch.run_agents_sync(signals={}, query_time=now)
        
        node_constraints = [c for c in constraints if c.type == ConstraintType.NODE_STATUS]
        assert len(node_constraints) == 1
        assert node_constraints[0].effect["action"] == "disable"
