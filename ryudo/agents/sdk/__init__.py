"""
Ryudo Agent SDK
===============

A framework for building agents that observe world state and emit constraints.

Public API:
- WorldState: Immutable snapshot that agents observe
- AgentResult: What agents return after observation
- BaseAgent: Abstract base class for agents
- AgentOrchestrator: Manages agent lifecycle and parallel execution
- ConflictStrategy: Enum for conflict resolution strategies
"""

from ryudo.agents.sdk.interface import (
    WorldState,
    AgentResult,
    BaseAgent,
)
from ryudo.agents.sdk.orchestrator import (
    AgentOrchestrator,
    ConflictStrategy,
)

__all__ = [
    "WorldState",
    "AgentResult",
    "BaseAgent",
    "AgentOrchestrator",
    "ConflictStrategy",
]
