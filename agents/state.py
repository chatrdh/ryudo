"""
Ryudo State Schema
==================
Defines the shared state that flows through the LangGraph workflow.
All agents read from and write to this state.
"""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
import networkx as nx
import operator


class GraphConstraint(TypedDict):
    """A constraint applied to the graph by an agent."""
    agent: str                    # Which agent created this constraint
    action: str                   # "delete_zone", "sever_edge", "set_ttl", "disable_node"
    target: Dict[str, Any]        # Polygon, edge_id, node_id, or edge_ids list
    reason: str                   # Human-readable explanation
    metadata: Dict[str, Any]      # Extra data (wind_speed, failure_type, ttl_hours, etc.)


class RyudoState(TypedDict):
    """
    Shared state passed through the LangGraph workflow.
    
    This state is the "Living Graph" - it starts with a base road network
    and accumulates constraints from all agents until the Mission Coordinator
    applies them and solves the routing problem.
    """
    # Base graph (NetworkX MultiDiGraph from OSMnx)
    base_graph: nx.MultiDiGraph
    
    # Accumulated constraints from all agents (uses operator.add for merging)
    constraints: Annotated[List[GraphConstraint], operator.add]
    
    # Agent-specific input data
    environmental_data: Dict[str, Any]   # Cyclone config, weather data, flood zones
    infrastructure_data: Dict[str, Any]  # Power grid status, sensor readings
    temporal_data: Dict[str, Any]        # Forecasts, predicted TTLs
    
    # Mission parameters (from user)
    mission: Dict[str, Any]  # e.g., {"type": "rescue", "targets": [...], "depot": {...}}
    
    # Final outputs (set by Mission Coordinator)
    modified_graph: Optional[nx.MultiDiGraph]
    solution: Optional[Dict[str, Any]]
    
    # Visualization data for map rendering (annotated for parallel agent writes)
    visualization_layers: Annotated[List[Dict[str, Any]], operator.add]
