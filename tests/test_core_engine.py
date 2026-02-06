"""
Tests for the Core Engine
=========================

Tests the LivingGraph engine, constraint schema, and tag mapper.
"""

import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import networkx as nx
from pydantic import ValidationError

from ryudo.core.schema import ConstraintType, TimeWindow, GraphConstraint
from ryudo.core.mapper import TagMapper, GraphAttributes
from ryudo.core.engine import LivingGraph


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_graph() -> nx.MultiDiGraph:
    """
    Create a simple test graph:
    
        1 ---- 2 ---- 3
        |      |      |
        4 ---- 5 ---- 6
        |      |      |
        7 ---- 8 ---- 9
    
    Node positions (x, y) = (col, row) for easy visualization.
    """
    G = nx.MultiDiGraph()
    
    # Add nodes with positions (x=lon, y=lat format like OSMnx)
    positions = {
        1: (0.0, 2.0), 2: (1.0, 2.0), 3: (2.0, 2.0),
        4: (0.0, 1.0), 5: (1.0, 1.0), 6: (2.0, 1.0),
        7: (0.0, 0.0), 8: (1.0, 0.0), 9: (2.0, 0.0),
    }
    
    for node_id, (x, y) in positions.items():
        G.add_node(node_id, x=x, y=y)
    
    # Add edges with weight = distance (all same for simplicity)
    edges = [
        (1, 2), (2, 3), (4, 5), (5, 6), (7, 8), (8, 9),  # Horizontal
        (1, 4), (4, 7), (2, 5), (5, 8), (3, 6), (6, 9),  # Vertical
    ]
    
    for u, v in edges:
        G.add_edge(u, v, weight=1.0, length=1.0)
        G.add_edge(v, u, weight=1.0, length=1.0)  # Bidirectional
    
    return G


@pytest.fixture
def living_graph(sample_graph) -> LivingGraph:
    """Create a LivingGraph with the sample graph loaded."""
    lg = LivingGraph()
    lg.load_from_graph(sample_graph)
    return lg


@pytest.fixture
def now() -> datetime:
    """Current time for testing."""
    return datetime.now(timezone.utc)


@pytest.fixture
def active_window(now) -> TimeWindow:
    """A time window that is currently active."""
    return TimeWindow(
        start_time=now - timedelta(hours=1),
        end_time=now + timedelta(hours=1)
    )


@pytest.fixture
def expired_window(now) -> TimeWindow:
    """A time window that has already expired."""
    return TimeWindow(
        start_time=now - timedelta(hours=2),
        end_time=now - timedelta(hours=1)
    )


@pytest.fixture
def future_window(now) -> TimeWindow:
    """A time window that hasn't started yet."""
    return TimeWindow(
        start_time=now + timedelta(hours=1),
        end_time=now + timedelta(hours=2)
    )


# =============================================================================
# Schema Tests
# =============================================================================

class TestTimeWindow:
    """Tests for TimeWindow model."""
    
    def test_is_active_within_window(self, now, active_window):
        assert active_window.is_active(now) is True
    
    def test_is_active_before_window(self, now, future_window):
        assert future_window.is_active(now) is False
    
    def test_is_active_after_window(self, now, expired_window):
        assert expired_window.is_active(now) is False
    
    def test_is_active_at_boundary(self, now):
        window = TimeWindow(start_time=now, end_time=now + timedelta(hours=1))
        assert window.is_active(now) is True

    def test_rejects_end_before_start(self, now):
        with pytest.raises(ValidationError):
            TimeWindow(
                start_time=now + timedelta(hours=1),
                end_time=now,
            )


class TestGraphConstraint:
    """Tests for GraphConstraint model."""
    
    def test_create_node_status_constraint(self, active_window):
        c = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 123},
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="test_agent"
        )
        assert c.type == ConstraintType.NODE_STATUS
        assert c.id is not None
    
    def test_create_zone_mask_constraint(self, active_window):
        c = GraphConstraint(
            type=ConstraintType.ZONE_MASK,
            target={"polygon": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}},
            effect={"weight_factor": 10.0},
            validity=active_window,
            source_agent_id="test_agent"
        )
        assert c.type == ConstraintType.ZONE_MASK
    
    def test_constraint_with_metadata(self, active_window):
        c = GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"edge": (1, 2)},
            effect={"weight_factor": 5.0},
            validity=active_window,
            source_agent_id="test_agent",
            metadata={"reason": "High congestion", "severity": 0.8}
        )
        assert c.metadata["severity"] == 0.8

    def test_rejects_node_status_without_selector(self, active_window):
        with pytest.raises(ValidationError):
            GraphConstraint(
                type=ConstraintType.NODE_STATUS,
                target={"facility_id": "f1"},
                effect={"action": "disable"},
                validity=active_window,
                source_agent_id="test_agent"
            )

    def test_rejects_node_status_without_action(self, active_window):
        with pytest.raises(ValidationError):
            GraphConstraint(
                type=ConstraintType.NODE_STATUS,
                target={"node_id": 1},
                effect={},
                validity=active_window,
                source_agent_id="test_agent"
            )

    def test_accepts_node_status_lat_lon_selector(self, active_window):
        c = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"lat": 17.72, "lon": 83.30},
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="test_agent"
        )
        assert c.type == ConstraintType.NODE_STATUS

    def test_rejects_edge_weight_without_selector(self, active_window):
        with pytest.raises(ValidationError):
            GraphConstraint(
                type=ConstraintType.EDGE_WEIGHT,
                target={},
                effect={"weight_factor": 5.0},
                validity=active_window,
                source_agent_id="test_agent"
            )

    def test_rejects_zone_mask_without_effect(self, active_window):
        with pytest.raises(ValidationError):
            GraphConstraint(
                type=ConstraintType.ZONE_MASK,
                target={"polygon": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}},
                effect={},
                validity=active_window,
                source_agent_id="test_agent"
            )

    def test_rejects_virtual_edge_without_nodes(self, active_window):
        with pytest.raises(ValidationError):
            GraphConstraint(
                type=ConstraintType.VIRTUAL_EDGE,
                target={"from_node": 1},
                effect={"weight": 1.0},
                validity=active_window,
                source_agent_id="test_agent"
            )


# =============================================================================
# Mapper Tests
# =============================================================================

class TestTagMapper:
    """Tests for TagMapper."""
    
    def test_default_config(self):
        mapper = TagMapper()
        attrs = mapper.normalize_attributes({"highway": "motorway"})
        assert attrs.base_weight == 0.8
        assert attrs.priority == 10
    
    def test_custom_config(self):
        config = {
            "highway": {
                "custom_road": {"base_weight": 2.5, "priority": 3}
            }
        }
        mapper = TagMapper(config)
        attrs = mapper.normalize_attributes({"highway": "custom_road"})
        assert attrs.base_weight == 2.5
        assert attrs.priority == 3
    
    def test_unknown_highway_type(self):
        mapper = TagMapper()
        attrs = mapper.normalize_attributes({"highway": "unknown_type"})
        # Should return defaults
        assert attrs.base_weight == 1.0
        assert attrs.priority == 5
    
    def test_maxspeed_override(self):
        mapper = TagMapper()
        attrs = mapper.normalize_attributes({"highway": "residential", "maxspeed": "40"})
        assert attrs.max_speed_kmh == 40
    
    def test_maxspeed_kmh_format(self):
        mapper = TagMapper()
        attrs = mapper.normalize_attributes({"highway": "residential", "maxspeed": "50 km/h"})
        assert attrs.max_speed_kmh == 50
    
    def test_oneway_detection(self):
        mapper = TagMapper()
        attrs = mapper.normalize_attributes({"highway": "primary", "oneway": "yes"})
        assert attrs.bidirectional is False
    
    def test_bidirectional_default(self):
        mapper = TagMapper()
        attrs = mapper.normalize_attributes({"highway": "primary"})
        assert attrs.bidirectional is True
    
    def test_get_edge_weight_travel_time(self):
        mapper = TagMapper()
        # 1000m at 36 km/h = 100 seconds
        weight = mapper.get_edge_weight({"highway": "residential"}, 1000, "travel_time")
        # residential max_speed is 30 km/h by default = 8.33 m/s
        expected = 1000 / (30 / 3.6)  # ~120s
        assert abs(weight - expected) < 0.1


# =============================================================================
# Engine Tests
# =============================================================================

class TestLivingGraphBasics:
    """Basic tests for LivingGraph."""
    
    def test_load_from_graph(self, sample_graph):
        lg = LivingGraph()
        lg.load_from_graph(sample_graph)
        assert lg.base_graph is not None
        assert lg.base_graph.number_of_nodes() == 9
    
    def test_constraint_lifecycle(self, living_graph, active_window):
        c = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 5},
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="test"
        )
        
        # Add
        cid = living_graph.add_constraint(c)
        assert living_graph.constraint_count == 1
        
        # Get
        retrieved = living_graph.get_constraint(cid)
        assert retrieved.id == c.id
        
        # Remove
        assert living_graph.remove_constraint(cid) is True
        assert living_graph.constraint_count == 0
    
    def test_list_active_constraints(self, living_graph, now, active_window, expired_window):
        c1 = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 1},
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="agent1"
        )
        c2 = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 2},
            effect={"action": "disable"},
            validity=expired_window,
            source_agent_id="agent2"
        )
        
        living_graph.add_constraint(c1)
        living_graph.add_constraint(c2)
        
        active = living_graph.list_constraints(active_at=now)
        assert len(active) == 1
        assert active[0].id == c1.id
    
    def test_clear_expired_constraints(self, living_graph, now, active_window, expired_window):
        c1 = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 1},
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="test"
        )
        c2 = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 2},
            effect={"action": "disable"},
            validity=expired_window,
            source_agent_id="test"
        )
        
        living_graph.add_constraint(c1)
        living_graph.add_constraint(c2)
        
        removed = living_graph.clear_expired_constraints(reference_time=now)
        assert removed == 1
        assert living_graph.constraint_count == 1


class TestGetView:
    """Tests for get_view() method."""
    
    def test_get_view_no_constraints(self, living_graph):
        view = living_graph.get_view()
        assert view.number_of_nodes() == 9
        assert view.number_of_edges() == living_graph.base_graph.number_of_edges()
    
    def test_node_status_disable(self, living_graph, now, active_window):
        c = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 5},
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)
        
        view = living_graph.get_view(query_time=now)
        assert 5 not in view.nodes()
        assert view.number_of_nodes() == 8
    
    def test_edge_weight_factor(self, living_graph, now, active_window):
        c = GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"edge": (1, 2)},
            effect={"weight_factor": 10.0},
            validity=active_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)
        
        view = living_graph.get_view(query_time=now)
        # Edge weight should be multiplied
        assert view[1][2][0]["weight"] == 10.0
    
    def test_virtual_edge(self, living_graph, now, active_window):
        # Add virtual edge from 1 to 9 (not directly connected)
        c = GraphConstraint(
            type=ConstraintType.VIRTUAL_EDGE,
            target={"from_node": 1, "to_node": 9},
            effect={"weight": 0.5, "bidirectional": True},
            validity=active_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)
        
        view = living_graph.get_view(query_time=now)
        assert view.has_edge(1, 9)
        assert view.has_edge(9, 1)
        assert view[1][9][0]["virtual"] is True
    
    def test_expired_constraint_not_applied(self, living_graph, now, expired_window):
        c = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 5},
            effect={"action": "disable"},
            validity=expired_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)
        
        view = living_graph.get_view(query_time=now)
        # Node should NOT be disabled because constraint expired
        assert 5 in view.nodes()

    def test_node_status_resolves_from_coordinates(self, living_graph, now, active_window):
        """NODE_STATUS should resolve nearest node when only lat/lon are provided."""
        c = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"lat": 1.0, "lon": 1.0},
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)

        view = living_graph.get_view(query_time=now)
        assert 5 not in view.nodes()

    def test_edge_weight_resolves_road_types(self, living_graph, now, active_window):
        """EDGE_WEIGHT should support selector-style road_types filters."""
        # Tag specific edges in the base graph.
        living_graph.base_graph[1][2][0]["highway"] = "primary"
        living_graph.base_graph[2][3][0]["highway"] = "residential"

        c = GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"road_types": ["primary"]},
            effect={"weight_factor": 5.0},
            validity=active_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)

        view = living_graph.get_view(query_time=now)
        assert view[1][2][0]["weight"] == 5.0
        assert view[2][3][0]["weight"] == 1.0

    def test_edge_weight_resolves_tag_filter(self, living_graph, now, active_window):
        """EDGE_WEIGHT should support generic tag_filter selectors."""
        living_graph.base_graph[4][5][0]["surface"] = "unpaved"
        living_graph.base_graph[5][6][0]["surface"] = "paved"

        c = GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"tag_filter": {"surface": "unpaved"}},
            effect={"weight_factor": 7.0},
            validity=active_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)

        view = living_graph.get_view(query_time=now)
        assert view[4][5][0]["weight"] == 7.0
        assert view[5][6][0]["weight"] == 1.0

    def test_get_view_with_report_tracks_applied_and_skipped(
        self, living_graph, now, active_window
    ):
        """Report should include per-constraint apply/skip outcomes."""
        disable = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 5},
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="test",
        )
        skipped = GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"road_types": ["primary"]},  # No matching tags in sample graph
            effect={"weight_factor": 2.0},
            validity=active_window,
            source_agent_id="test",
        )

        living_graph.add_constraint(disable)
        living_graph.add_constraint(skipped)

        view, report = living_graph.get_view_with_report(query_time=now)
        assert 5 not in view.nodes()

        by_id = {entry.constraint_id: entry for entry in report}
        assert by_id[disable.id].status == "applied"
        assert by_id[disable.id].reason == "nodes_disabled"
        assert by_id[skipped.id].status == "skipped"
        assert by_id[skipped.id].reason == "no_edges_resolved"
        serialized = by_id[disable.id].to_dict()
        assert serialized["constraint_id"] == str(disable.id)
        assert serialized["status"] == "applied"

    def test_precedence_matrix_is_explicit(self, living_graph):
        """Core precedence matrix should enforce deterministic type ordering."""
        matrix = living_graph.precedence_matrix
        assert matrix[ConstraintType.NODE_STATUS] > matrix[ConstraintType.ZONE_MASK]
        assert matrix[ConstraintType.ZONE_MASK] > matrix[ConstraintType.EDGE_WEIGHT]
        assert matrix[ConstraintType.EDGE_WEIGHT] > matrix[ConstraintType.VIRTUAL_EDGE]

    def test_mixed_collision_node_status_beats_edge_weight(
        self, living_graph, now, active_window
    ):
        """Node disable should run before edge weighting, regardless of add order."""
        edge_weight = GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"edge": (2, 5)},
            effect={"weight_factor": 20.0},
            validity=active_window,
            source_agent_id="edge_agent",
        )
        node_disable = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"lat": 1.0, "lon": 1.0},  # Resolves to node 5
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="node_agent",
        )

        # Add in reverse order to prove precedence, not insertion order.
        living_graph.add_constraint(edge_weight)
        living_graph.add_constraint(node_disable)

        view, report = living_graph.get_view_with_report(query_time=now)
        assert 5 not in view.nodes()
        assert not view.has_edge(2, 5)

        idx = {entry.constraint_id: i for i, entry in enumerate(report)}
        by_id = {entry.constraint_id: entry for entry in report}
        assert idx[node_disable.id] < idx[edge_weight.id]
        assert by_id[node_disable.id].status == "applied"
        assert by_id[edge_weight.id].status == "skipped"

    def test_mixed_collision_node_status_beats_virtual_edge(
        self, living_graph, now, active_window
    ):
        """Virtual edge should not be created when one endpoint is disabled first."""
        virtual = GraphConstraint(
            type=ConstraintType.VIRTUAL_EDGE,
            target={"from_node": 1, "to_node": 9},
            effect={"weight": 1.0, "bidirectional": False},
            validity=active_window,
            source_agent_id="virtual_agent",
        )
        node_disable = GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 9},
            effect={"action": "disable"},
            validity=active_window,
            source_agent_id="node_agent",
        )

        # Add in reverse order to prove precedence.
        living_graph.add_constraint(virtual)
        living_graph.add_constraint(node_disable)

        view, report = living_graph.get_view_with_report(query_time=now)
        assert 9 not in view.nodes()
        assert not view.has_edge(1, 9)

        idx = {entry.constraint_id: i for i, entry in enumerate(report)}
        by_id = {entry.constraint_id: entry for entry in report}
        assert idx[node_disable.id] < idx[virtual.id]
        assert by_id[node_disable.id].status == "applied"
        assert by_id[virtual.id].status == "skipped"
        assert by_id[virtual.id].reason == "virtual_edge_nodes_not_in_view"

    def test_mixed_collision_zone_mask_beats_zone_edge_weight(
        self, living_graph, now, active_window
    ):
        """Zone labels should exist before zone-based edge selectors execute."""
        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [-0.5, 0.5],
                [2.5, 0.5],
                [2.5, 1.5],
                [-0.5, 1.5],
                [-0.5, 0.5]
            ]]
        }
        zone_edge_weight = GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"zone": "priority_band"},
            effect={"weight_factor": 3.0},
            validity=active_window,
            source_agent_id="edge_agent",
        )
        zone_mask = GraphConstraint(
            type=ConstraintType.ZONE_MASK,
            target={"polygon": polygon},
            effect={"weight_factor": 2.0},
            validity=active_window,
            source_agent_id="zone_agent",
            metadata={"zone_type": "priority_band"},
        )

        # Add in reverse order to prove precedence.
        living_graph.add_constraint(zone_edge_weight)
        living_graph.add_constraint(zone_mask)

        view, report = living_graph.get_view_with_report(query_time=now)
        assert view[4][5][0]["weight"] == 6.0  # 1 * 2 * 3

        idx = {entry.constraint_id: i for i, entry in enumerate(report)}
        by_id = {entry.constraint_id: entry for entry in report}
        assert idx[zone_mask.id] < idx[zone_edge_weight.id]
        assert by_id[zone_mask.id].status == "applied"
        assert by_id[zone_edge_weight.id].status == "applied"


class TestZoneMask:
    """Tests for ZONE_MASK constraints with spatial queries."""
    
    def test_zone_mask_weight_factor(self, living_graph, now, active_window):
        # Create a polygon covering nodes 4, 5, 6 (middle row)
        # Polygon coordinates: (x, y) = (lon, lat)
        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [-0.5, 0.5],  # SW
                [2.5, 0.5],   # SE
                [2.5, 1.5],   # NE
                [-0.5, 1.5],  # NW
                [-0.5, 0.5]   # Close
            ]]
        }
        
        c = GraphConstraint(
            type=ConstraintType.ZONE_MASK,
            target={"polygon": polygon},
            effect={"weight_factor": 100.0},
            validity=active_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)
        
        view = living_graph.get_view(query_time=now)
        
        # Edge 4-5 should have high weight (inside zone)
        assert view[4][5][0]["weight"] == 100.0
        
        # Edge 1-2 should have original weight (outside zone)
        assert view[1][2][0]["weight"] == 1.0
    
    def test_zone_mask_path_deviation(self, living_graph, now, active_window):
        """
        THE CRITICAL TEST: Path should change to avoid the zone.
        
        Original shortest path 1 -> 9: 1 -> 4 -> 7 -> 8 -> 9 (4 hops)
        or 1 -> 2 -> 5 -> 8 -> 9 (4 hops via center)
        
        With center row (y=1) blocked by high weight:
        Path should go around: 1 -> 2 -> 3 -> 6 -> 9 (via top)
        or 1 -> 4 -> 7 -> 8 -> 9 (via bottom)
        """
        # Block the center (node 5 area)
        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [0.5, 0.5],   # SW
                [1.5, 0.5],   # SE
                [1.5, 1.5],   # NE
                [0.5, 1.5],   # NW
                [0.5, 0.5]    # Close
            ]]
        }
        
        # Get path BEFORE constraint
        view_before = living_graph.get_view(query_time=now)
        path_before = nx.shortest_path(view_before, 1, 9, weight="weight")
        
        # Add zone constraint with extreme weight
        c = GraphConstraint(
            type=ConstraintType.ZONE_MASK,
            target={"polygon": polygon},
            effect={"weight_factor": 1000.0},
            validity=active_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)
        
        # Get path AFTER constraint
        view_after = living_graph.get_view(query_time=now)
        path_after = nx.shortest_path(view_after, 1, 9, weight="weight")
        
        # Path should avoid node 5 (center of the zone)
        assert 5 not in path_after, f"Path {path_after} should avoid node 5"
    
    def test_zone_mask_node_disable(self, living_graph, now, active_window):
        """Test zone mask with node_action=disable."""
        # Polygon covering node 5
        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [0.5, 0.5],
                [1.5, 0.5],
                [1.5, 1.5],
                [0.5, 1.5],
                [0.5, 0.5]
            ]]
        }
        
        c = GraphConstraint(
            type=ConstraintType.ZONE_MASK,
            target={"polygon": polygon},
            effect={"node_action": "disable"},
            validity=active_window,
            source_agent_id="test"
        )
        living_graph.add_constraint(c)
        
        view = living_graph.get_view(query_time=now)
        
        # Node 5 should be removed
        assert 5 not in view.nodes()

    def test_zone_labels_enable_followup_edge_selectors(self, living_graph, now, active_window):
        """Zone metadata labels should be reusable by later EDGE_WEIGHT constraints."""
        polygon = {
            "type": "Polygon",
            "coordinates": [[
                [-0.5, 0.5],
                [2.5, 0.5],
                [2.5, 1.5],
                [-0.5, 1.5],
                [-0.5, 0.5]
            ]]
        }

        zone_constraint = GraphConstraint(
            type=ConstraintType.ZONE_MASK,
            target={"polygon": polygon},
            effect={"weight_factor": 2.0},
            validity=active_window,
            source_agent_id="test",
            metadata={"zone_type": "middle_band"},
        )
        weight_constraint = GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"zone": "middle_band"},
            effect={"weight_factor": 3.0},
            validity=active_window,
            source_agent_id="test",
        )

        living_graph.add_constraint(zone_constraint)
        living_graph.add_constraint(weight_constraint)

        view = living_graph.get_view(query_time=now)
        # In-zone edge gets both multipliers: 1 * 2 * 3
        assert view[4][5][0]["weight"] == 6.0
        # Out-of-zone edge unaffected by follow-up selector
        assert view[1][2][0]["weight"] == 1.0


# =============================================================================
# Domain Agnosticism Verification
# =============================================================================

class TestDomainAgnosticism:
    """Verify no domain-specific terms exist in the core module."""
    
    def test_no_domain_terms_in_schema(self):
        import inspect
        from ryudo.core import schema
        
        source = inspect.getsource(schema)
        domain_terms = ["flood", "cyclone", "disaster", "rescue", "hospital", "vehicle"]
        
        for term in domain_terms:
            assert term.lower() not in source.lower(), f"Found domain term '{term}' in schema.py"
    
    def test_no_domain_terms_in_engine(self):
        import inspect
        from ryudo.core import engine
        
        source = inspect.getsource(engine)
        domain_terms = ["flood", "cyclone", "disaster", "rescue", "hospital", "vehicle"]
        
        for term in domain_terms:
            assert term.lower() not in source.lower(), f"Found domain term '{term}' in engine.py"
    
    def test_no_domain_terms_in_mapper(self):
        import inspect
        from ryudo.core import mapper
        
        source = inspect.getsource(mapper)
        domain_terms = ["flood", "cyclone", "disaster", "rescue", "hospital", "vehicle"]
        
        for term in domain_terms:
            assert term.lower() not in source.lower(), f"Found domain term '{term}' in mapper.py"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
