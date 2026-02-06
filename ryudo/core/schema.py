"""
Constraint Schema
=================

Domain-agnostic data models for graph constraints.

These schemas represent abstract graph mutations without any reference
to specific application domains.

Constraint Types:
- NODE_STATUS: Enable/disable a specific node
- EDGE_WEIGHT: Multiply traversal cost by a factor
- ZONE_MASK: Apply effects to nodes/edges within a polygon
- VIRTUAL_EDGE: Create synthetic connections between nodes
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConstraintType(str, Enum):
    """Types of graph mutations that can be applied."""
    
    NODE_STATUS = "node_status"
    """Enable or disable a specific node."""
    
    EDGE_WEIGHT = "edge_weight"
    """Multiply an edge's traversal cost by a factor."""
    
    ZONE_MASK = "zone_mask"
    """Apply effects to all nodes/edges within a polygon."""
    
    VIRTUAL_EDGE = "virtual_edge"
    """Create a synthetic edge between two nodes."""


class TimeWindow(BaseModel):
    """Validity period for a constraint."""
    
    start_time: datetime
    """When the constraint becomes active."""
    
    end_time: datetime
    """When the constraint expires."""

    @model_validator(mode="after")
    def _validate_bounds(self) -> "TimeWindow":
        """Ensure time window bounds are valid."""
        if self.end_time < self.start_time:
            raise ValueError("end_time must be greater than or equal to start_time")
        return self
    
    def is_active(self, query_time: datetime) -> bool:
        """Check if this window contains the query time."""
        return self.start_time <= query_time <= self.end_time
    
    def __repr__(self) -> str:
        return f"TimeWindow({self.start_time.isoformat()} → {self.end_time.isoformat()})"


class GraphConstraint(BaseModel):
    """
    A constraint that modifies the graph.
    
    Constraints are the fundamental unit of change in the Living Graph.
    They describe WHAT to modify, HOW to modify it, and WHEN the 
    modification is valid.
    
    Examples
    --------
    Disable a node:
        GraphConstraint(
            type=ConstraintType.NODE_STATUS,
            target={"node_id": 12345},
            effect={"action": "disable"},
            validity=TimeWindow(...),
            source_agent_id="power_monitor"
        )
    
    Increase edge cost:
        GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"edge": (123, 456, 0)},
            effect={"weight_factor": 2.5},
            validity=TimeWindow(...),
            source_agent_id="congestion_sensor"
        )
    
    Zone-based weight increase:
        GraphConstraint(
            type=ConstraintType.ZONE_MASK,
            target={"polygon": {"type": "Polygon", "coordinates": [...]}},
            effect={"weight_factor": 10.0, "node_action": "disable"},
            validity=TimeWindow(...),
            source_agent_id="area_monitor"
        )
    
    Create virtual edge:
        GraphConstraint(
            type=ConstraintType.VIRTUAL_EDGE,
            target={"from_node": 123, "to_node": 789},
            effect={"weight": 100.0, "bidirectional": True},
            validity=TimeWindow(...),
            source_agent_id="bridge_builder"
        )
    """
    
    id: UUID = Field(default_factory=uuid4)
    """Unique identifier for this constraint."""
    
    type: ConstraintType
    """The type of graph mutation."""
    
    target: dict[str, Any]
    """
    What to modify. Structure depends on type:
    - NODE_STATUS: {"node_id": int}
    - EDGE_WEIGHT: {"edge": (u, v, key)} or {"edge": (u, v)}
    - ZONE_MASK: {"polygon": GeoJSON}
    - VIRTUAL_EDGE: {"from_node": int, "to_node": int}
    """
    
    effect: dict[str, Any]
    """
    How to modify. Structure depends on type:
    - NODE_STATUS: {"action": "disable" | "enable"}
    - EDGE_WEIGHT: {"weight_factor": float}
    - ZONE_MASK: {"weight_factor": float} and/or {"node_action": "disable"}
    - VIRTUAL_EDGE: {"weight": float, "bidirectional": bool}
    """
    
    validity: TimeWindow
    """When this constraint is active."""
    
    source_agent_id: str
    """Identifier of the agent that created this constraint."""
    
    metadata: dict[str, Any] = Field(default_factory=dict)
    """
    Arbitrary additional data. Used for:
    - Reasoning/explanation text
    - Numeric values that led to this constraint
    - Tags for filtering
    """
    
    priority: int = Field(default=0)
    """
    Priority for conflict resolution. Higher values take precedence.
    Used when multiple constraints affect the same target.
    """
    
    def is_active(self, query_time: datetime) -> bool:
        """Check if this constraint is active at the given time."""
        return self.validity.is_active(query_time)
    
    model_config = ConfigDict(frozen=False)

    @model_validator(mode="after")
    def _validate_contract(self) -> "GraphConstraint":
        """
        Validate target/effect structure for each constraint type.

        This keeps the schema domain-agnostic while enforcing typed semantics.
        """
        self._validate_common()

        if self.type == ConstraintType.NODE_STATUS:
            self._validate_node_status()
        elif self.type == ConstraintType.EDGE_WEIGHT:
            self._validate_edge_weight()
        elif self.type == ConstraintType.ZONE_MASK:
            self._validate_zone_mask()
        elif self.type == ConstraintType.VIRTUAL_EDGE:
            self._validate_virtual_edge()

        return self

    def _validate_common(self) -> None:
        """Validate common fields across all constraint types."""
        if not self.source_agent_id or not self.source_agent_id.strip():
            raise ValueError("source_agent_id must be a non-empty string")

    def _validate_node_status(self) -> None:
        """Validate NODE_STATUS target/effect contract."""
        action = self.effect.get("action")
        if action not in {"disable", "enable"}:
            raise ValueError("NODE_STATUS.effect.action must be 'disable' or 'enable'")

        if not self._has_any_target_key(("node_id", "node_ids", "lat", "lon")):
            raise ValueError(
                "NODE_STATUS.target must include one selector: "
                "node_id, node_ids, or lat/lon"
            )

        lat_present = "lat" in self.target and self.target.get("lat") is not None
        lon_present = "lon" in self.target and self.target.get("lon") is not None
        if lat_present != lon_present:
            raise ValueError("NODE_STATUS.target must include both lat and lon together")

    def _validate_edge_weight(self) -> None:
        """Validate EDGE_WEIGHT target/effect contract."""
        selector_keys = ("edge", "edges", "road_types", "tag_filter", "zone", "zones")
        if not self._has_any_target_key(selector_keys):
            raise ValueError(
                "EDGE_WEIGHT.target must include one selector: "
                "edge, edges, road_types, tag_filter, zone, or zones"
            )

        if "weight_factor" not in self.effect:
            raise ValueError("EDGE_WEIGHT.effect must include weight_factor")

        try:
            float(self.effect["weight_factor"])
        except (TypeError, ValueError) as exc:
            raise ValueError("EDGE_WEIGHT.effect.weight_factor must be numeric") from exc

    def _validate_zone_mask(self) -> None:
        """Validate ZONE_MASK target/effect contract."""
        polygon = self.target.get("polygon")
        if not isinstance(polygon, dict):
            raise ValueError("ZONE_MASK.target.polygon must be a GeoJSON-like dict")

        has_weight = "weight_factor" in self.effect
        has_node_action = "node_action" in self.effect
        if not has_weight and not has_node_action:
            raise ValueError(
                "ZONE_MASK.effect must include at least one of: weight_factor, node_action"
            )

        if has_weight:
            try:
                float(self.effect["weight_factor"])
            except (TypeError, ValueError) as exc:
                raise ValueError("ZONE_MASK.effect.weight_factor must be numeric") from exc

        if has_node_action and self.effect.get("node_action") not in {"disable", "enable"}:
            raise ValueError("ZONE_MASK.effect.node_action must be 'disable' or 'enable'")

    def _validate_virtual_edge(self) -> None:
        """Validate VIRTUAL_EDGE target/effect contract."""
        if "from_node" not in self.target or "to_node" not in self.target:
            raise ValueError("VIRTUAL_EDGE.target must include from_node and to_node")

        if "weight" in self.effect:
            try:
                float(self.effect["weight"])
            except (TypeError, ValueError) as exc:
                raise ValueError("VIRTUAL_EDGE.effect.weight must be numeric") from exc

        if "bidirectional" in self.effect and not isinstance(self.effect["bidirectional"], bool):
            raise ValueError("VIRTUAL_EDGE.effect.bidirectional must be a boolean")

    def _has_any_target_key(self, keys: tuple[str, ...]) -> bool:
        """Return True when any selector key is present and non-empty."""
        for key in keys:
            value = self.target.get(key)
            if value is None:
                continue

            if isinstance(value, str) and not value.strip():
                continue

            if isinstance(value, (list, tuple, dict, set, frozenset)) and len(value) == 0:
                continue

            return True

        return False
        
    def __repr__(self) -> str:
        return (
            f"GraphConstraint(id={str(self.id)[:8]}..., "
            f"type={self.type.value}, source={self.source_agent_id})"
        )
