"""
Agent Orchestrator
==================

Manages agent lifecycle, parallel execution, and conflict resolution.

The orchestrator:
1. Registers agents
2. Creates WorldState snapshots
3. Runs agents in parallel
4. Resolves conflicts between constraints
5. Returns merged constraints
"""

import asyncio
import inspect
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional
from uuid import UUID

import networkx as nx

from ryudo.core.engine import LivingGraph
from ryudo.core.schema import ConstraintType, GraphConstraint
from ryudo.agents.sdk.interface import AgentResult, BaseAgent, WorldState


class ConflictStrategy(Enum):
    """Strategies for resolving conflicting constraints."""
    
    HIGHEST_COST = "highest_cost"
    """Use the highest weight factor (safest for routing)."""
    
    LOWEST_COST = "lowest_cost"
    """Use the lowest weight factor (optimistic)."""
    
    PRIORITY_WINS = "priority_wins"
    """Higher priority agent wins."""
    
    MERGE_MULTIPLY = "merge_multiply"
    """Multiply all weight factors together."""
    
    FIRST_WINS = "first_wins"
    """First constraint registered wins."""


class AgentOrchestrator:
    """
    Orchestrates agent execution and constraint collection.
    
    Example
    -------
    >>> orchestrator = AgentOrchestrator(living_graph)
    >>> orchestrator.register_agent(FloodSentinel())
    >>> orchestrator.register_agent(GridGuardian())
    >>> 
    >>> constraints = await orchestrator.run_agents(
    ...     signals={"cyclone": cyclone_data},
    ...     query_time=datetime.now(timezone.utc)
    ... )
    >>> 
    >>> for c in constraints:
    ...     living_graph.add_constraint(c)
    """
    
    def __init__(
        self,
        living_graph: LivingGraph,
        conflict_strategy: ConflictStrategy = ConflictStrategy.HIGHEST_COST,
    ):
        """
        Initialize the orchestrator.
        
        Parameters
        ----------
        living_graph : LivingGraph
            The graph engine to get views from
        conflict_strategy : ConflictStrategy
            How to resolve conflicting constraints
        """
        self._living_graph = living_graph
        self._conflict_strategy = conflict_strategy
        self._agents: dict[str, BaseAgent] = {}
        self._execution_hooks: list[Callable] = []
    
    @property
    def agent_count(self) -> int:
        """Number of registered agents."""
        return len(self._agents)
    
    def register_agent(self, agent: BaseAgent) -> None:
        """
        Register an agent for execution.
        
        Parameters
        ----------
        agent : BaseAgent
            Agent instance to register
        
        Raises
        ------
        ValueError
            If agent with same ID already registered
        """
        if agent.agent_id in self._agents:
            raise ValueError(f"Agent '{agent.agent_id}' already registered")
        
        self._agents[agent.agent_id] = agent
        print(f"[Orchestrator] Registered agent: {agent}")
    
    def unregister_agent(self, agent_id: str) -> bool:
        """
        Remove an agent.
        
        Returns True if agent was found and removed.
        """
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> list[BaseAgent]:
        """List all registered agents, sorted by priority (highest first)."""
        return sorted(
            self._agents.values(),
            key=lambda a: a.priority,
            reverse=True
        )
    
    async def run_agents(
        self,
        signals: dict[str, Any],
        query_time: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[GraphConstraint]:
        """
        Run all agents in parallel and return merged constraints.
        
        Parameters
        ----------
        signals : dict
            External data for agents (weather, sensors, etc.)
        query_time : datetime, optional
            Simulation time. Defaults to now (UTC).
        metadata : dict, optional
            Additional context (mission parameters, etc.)
        
        Returns
        -------
        list[GraphConstraint]
            All constraints from all agents, with conflicts resolved
        """
        if query_time is None:
            query_time = datetime.now(timezone.utc)
        
        if metadata is None:
            metadata = {}
        
        if not self._agents:
            print("[Orchestrator] No agents registered")
            return []
        
        # Create world state snapshot
        graph_view = self._living_graph.get_view(query_time=query_time)
        
        world_state = WorldState(
            graph_view=graph_view,
            query_time=query_time,
            signals=signals,
            existing_constraints=tuple(),  # First pass has no existing constraints
            metadata=metadata,
        )
        
        # Run all agents in parallel
        print(f"[Orchestrator] Running {len(self._agents)} agents...")
        
        tasks = [
            self._run_single_agent(agent, world_state)
            for agent in self._agents.values()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all constraints and handle errors
        all_constraints: list[GraphConstraint] = []
        all_visualizations: list[dict] = []
        
        for agent_id, result in zip(self._agents.keys(), results):
            if isinstance(result, Exception):
                print(f"[Orchestrator] Agent '{agent_id}' failed: {result}")
                continue
            
            if result.constraints:
                all_constraints.extend(result.constraints)
                print(f"[Orchestrator] Agent '{agent_id}' emitted {len(result.constraints)} constraints")
            
            if result.visualization:
                all_visualizations.extend(result.visualization)
        
        # Resolve conflicts
        resolved = self._resolve_conflicts(all_constraints)
        
        print(f"[Orchestrator] Total: {len(all_constraints)} constraints, "
              f"{len(resolved)} after conflict resolution")
        
        return resolved
    
    def run_agents_sync(
        self,
        signals: dict[str, Any],
        query_time: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> list[GraphConstraint]:
        """Synchronous wrapper for run_agents()."""
        return asyncio.run(self.run_agents(signals, query_time, metadata))
    
    async def _run_single_agent(
        self,
        agent: BaseAgent,
        state: WorldState
    ) -> AgentResult:
        """
        Run a single agent and validate its result.

        Notes
        -----
        We execute sync agents in the current thread. Some geometry
        libraries used by reference agents may deadlock when called from
        executor threads in certain environments.
        """
        try:
            result_or_awaitable = agent.observe(state)
            result = (
                await result_or_awaitable
                if inspect.isawaitable(result_or_awaitable)
                else result_or_awaitable
            )
            
            # Validate result
            errors = agent.validate_result(result)
            if errors:
                print(f"[Orchestrator] Validation errors for '{agent.agent_id}': {errors}")
            
            return result
        
        except Exception as e:
            print(f"[Orchestrator] Agent '{agent.agent_id}' raised: {e}")
            raise
    
    def _resolve_conflicts(
        self,
        constraints: list[GraphConstraint]
    ) -> list[GraphConstraint]:
        """
        Resolve conflicting constraints.
        
        Conflicts occur when multiple constraints target the same entity
        (same node, same edge, overlapping zones).
        """
        if not constraints:
            return []
        
        # Group by target type and identity
        node_constraints: dict[int, list[GraphConstraint]] = defaultdict(list)
        edge_constraints: dict[tuple, list[GraphConstraint]] = defaultdict(list)
        zone_constraints: list[GraphConstraint] = []
        virtual_constraints: list[GraphConstraint] = []
        
        for c in constraints:
            if c.type == ConstraintType.NODE_STATUS:
                node_id = c.target.get("node_id")
                if node_id is not None:
                    node_constraints[node_id].append(c)
            
            elif c.type == ConstraintType.EDGE_WEIGHT:
                edge = c.target.get("edge")
                if edge is not None:
                    edge_key = tuple(edge) if isinstance(edge, list) else edge
                    edge_constraints[edge_key].append(c)
            
            elif c.type == ConstraintType.ZONE_MASK:
                # Zones are harder to compare - keep all for now
                zone_constraints.append(c)
            
            elif c.type == ConstraintType.VIRTUAL_EDGE:
                virtual_constraints.append(c)
        
        # Resolve node conflicts
        resolved_nodes = []
        for node_id, node_cs in node_constraints.items():
            resolved_nodes.append(self._resolve_node_conflict(node_cs))
        
        # Resolve edge conflicts
        resolved_edges = []
        for edge_key, edge_cs in edge_constraints.items():
            resolved_edges.append(self._resolve_edge_conflict(edge_cs))
        
        # Zone constraints: keep all (they're cumulative)
        # Could later add overlap detection and merging
        
        return resolved_nodes + resolved_edges + zone_constraints + virtual_constraints
    
    def _resolve_node_conflict(
        self,
        constraints: list[GraphConstraint]
    ) -> GraphConstraint:
        """Resolve conflicts for same node."""
        if len(constraints) == 1:
            return constraints[0]
        
        # For node status, "disable" always wins (safety)
        for c in constraints:
            if c.effect.get("action") == "disable":
                return c
        
        # Otherwise use priority
        return max(constraints, key=lambda c: self._get_agent_priority(c.source_agent_id))
    
    def _resolve_edge_conflict(
        self,
        constraints: list[GraphConstraint]
    ) -> GraphConstraint:
        """Resolve conflicts for same edge."""
        if len(constraints) == 1:
            return constraints[0]
        
        # Get all weight factors
        factors = [c.effect.get("weight_factor", 1.0) for c in constraints]
        
        if self._conflict_strategy == ConflictStrategy.HIGHEST_COST:
            winner_idx = factors.index(max(factors))
            return constraints[winner_idx]
        
        elif self._conflict_strategy == ConflictStrategy.LOWEST_COST:
            winner_idx = factors.index(min(factors))
            return constraints[winner_idx]
        
        elif self._conflict_strategy == ConflictStrategy.MERGE_MULTIPLY:
            # Merge into a single constraint with multiplied factor
            merged_factor = 1.0
            for f in factors:
                merged_factor *= f
            
            # Use highest priority agent as source
            best = max(constraints, key=lambda c: self._get_agent_priority(c.source_agent_id))
            
            # Create new merged constraint
            return GraphConstraint(
                type=best.type,
                target=best.target,
                effect={"weight_factor": merged_factor},
                validity=best.validity,
                source_agent_id="orchestrator",
                metadata={
                    "merged_from": [c.source_agent_id for c in constraints],
                    "original_factors": factors,
                },
            )
        
        elif self._conflict_strategy == ConflictStrategy.PRIORITY_WINS:
            return max(constraints, key=lambda c: self._get_agent_priority(c.source_agent_id))
        
        else:  # FIRST_WINS
            return constraints[0]
    
    def _get_agent_priority(self, agent_id: str) -> int:
        """Get priority for an agent by ID."""
        agent = self._agents.get(agent_id)
        return agent.priority if agent else 0
    
    def __repr__(self) -> str:
        return f"AgentOrchestrator(agents={len(self._agents)}, strategy={self._conflict_strategy.value})"
