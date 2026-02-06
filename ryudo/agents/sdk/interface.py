"""
Agent Interface
===============

Defines the strict input/output contract for agents.

Key Principle: Agents are pure functions that observe state and emit constraints.
They cannot modify the graph directly — only propose changes via constraints.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import networkx as nx

from ryudo.core.schema import GraphConstraint


@dataclass(frozen=True)
class WorldState:
    """
    Immutable snapshot of the world that agents observe.
    
    This is the input to every agent's `observe()` method. It provides
    a frozen view of the current graph state plus external signals.
    
    Attributes
    ----------
    graph_view : nx.MultiDiGraph
        A frozen copy of the current graph (with existing constraints applied)
    query_time : datetime
        The simulation/query time
    signals : dict
        External data (weather, sensors, IoT, etc.)
    existing_constraints : list[GraphConstraint]
        Constraints already emitted by other agents in this cycle
    metadata : dict
        Additional context (e.g., mission parameters)
    """
    
    graph_view: nx.MultiDiGraph
    query_time: datetime
    signals: dict[str, Any]
    existing_constraints: tuple[GraphConstraint, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def get_signal(self, key: str, default: Any = None) -> Any:
        """Safely get a signal value."""
        return self.signals.get(key, default)
    
    def has_signal(self, key: str) -> bool:
        """Check if a signal exists."""
        return key in self.signals


@dataclass
class AgentResult:
    """
    What agents return after observation.
    
    This is the output of every agent's `observe()` method.
    
    Attributes
    ----------
    constraints : list[GraphConstraint]
        New constraints to apply to the graph
    visualization : list[dict]
        Optional visualization layers for the frontend
    reasoning : str
        Optional explanation of the agent's decision process
    confidence : float
        0-1 confidence score for the constraints
    """
    
    constraints: list[GraphConstraint] = field(default_factory=list)
    visualization: list[dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 1.0
    
    def is_empty(self) -> bool:
        """Check if agent produced no constraints."""
        return len(self.constraints) == 0
    
    def merge(self, other: "AgentResult") -> "AgentResult":
        """Merge with another result."""
        return AgentResult(
            constraints=self.constraints + other.constraints,
            visualization=self.visualization + other.visualization,
            reasoning=f"{self.reasoning}\n{other.reasoning}".strip(),
            confidence=min(self.confidence, other.confidence),
        )


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Subclass this to create custom agents. Override the `observe()` method
    to implement your logic.
    
    Example
    -------
    >>> class MyAgent(BaseAgent):
    ...     agent_id = "my_agent"
    ...     priority = 5
    ...     
    ...     def observe(self, state: WorldState) -> AgentResult:
    ...         constraints = []
    ...         # ... analyze state.signals, emit constraints ...
    ...         return AgentResult(constraints=constraints)
    """
    
    # Must be set by subclasses
    agent_id: str = "base_agent"
    
    # Priority for conflict resolution (higher = wins conflicts)
    priority: int = 0
    
    # Optional description
    description: str = ""
    
    @abstractmethod
    def observe(self, state: WorldState) -> AgentResult:
        """
        Observe world state and emit constraints.
        
        This method MUST be:
        1. Pure - same input produces same output
        2. Stateless - don't store state between calls
        3. Side-effect free - don't modify external state
        
        Parameters
        ----------
        state : WorldState
            Frozen snapshot of the current world
        
        Returns
        -------
        AgentResult
            Constraints and optional visualization data
        """
        ...
    
    def validate_result(self, result: AgentResult) -> list[str]:
        """
        Validate the result before accepting.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        for i, c in enumerate(result.constraints):
            if c.source_agent_id != self.agent_id:
                errors.append(
                    f"Constraint {i}: source_agent_id '{c.source_agent_id}' "
                    f"doesn't match agent_id '{self.agent_id}'"
                )
        
        if not 0.0 <= result.confidence <= 1.0:
            errors.append(f"Confidence {result.confidence} not in [0, 1]")
        
        return errors
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, priority={self.priority})"
