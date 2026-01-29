"""
LangGraph Workflow Builder
===========================
Compiles the Ryudo agent workflow using LangGraph.

The workflow runs 3 State Agents in parallel, then merges their
constraints in the Mission Coordinator.
"""

from typing import Dict, Any
import networkx as nx

from langgraph.graph import StateGraph, START, END

from agents.state import RyudoState
from agents.environmental import environmental_agent
from agents.infrastructure import infrastructure_agent
from agents.temporal import temporal_agent
from agents.coordinator import mission_coordinator


def load_base_graph(state: RyudoState) -> Dict[str, Any]:
    """
    Load the base graph node.
    
    If base_graph is already provided in state, use it.
    Otherwise, download from OSMnx.
    """
    if state.get("base_graph") is not None:
        print("[LoadGraph] Using provided base graph")
        return {}
    
    import osmnx as ox
    
    place = state.get("mission", {}).get("place", "Visakhapatnam, India")
    print(f"[LoadGraph] Downloading road network for {place}...")
    
    try:
        G = ox.graph_from_place(place, network_type="drive")
        print(f"[LoadGraph] Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return {"base_graph": G}
    except Exception as e:
        print(f"[LoadGraph] Error: {e}")
        return {"base_graph": nx.MultiDiGraph()}


def merge_visualization_layers(state: RyudoState) -> Dict[str, Any]:
    """Merge all visualization layers from agents."""
    # This is handled automatically by the state annotation
    return {}


def build_ryudo_workflow():
    """
    Build the LangGraph workflow for Ryudo.
    
    Workflow structure:
    
        START
          │
          ▼
      load_graph
          │
          ├─────────┬─────────┐
          ▼         ▼         ▼
    environmental infrastructure temporal
          │         │         │
          └─────────┴─────────┘
                    │
                    ▼
             coordinator
                    │
                    ▼
                   END
    """
    workflow = StateGraph(RyudoState)
    
    # Add nodes
    workflow.add_node("load_graph", load_base_graph)
    workflow.add_node("environmental", environmental_agent)
    workflow.add_node("infrastructure", infrastructure_agent)
    workflow.add_node("temporal", temporal_agent)
    workflow.add_node("coordinator", mission_coordinator)
    
    # Define edges
    # Start -> Load Graph
    workflow.add_edge(START, "load_graph")
    
    # Load Graph -> All 3 agents in parallel
    workflow.add_edge("load_graph", "environmental")
    workflow.add_edge("load_graph", "infrastructure")
    workflow.add_edge("load_graph", "temporal")
    
    # All agents -> Coordinator
    workflow.add_edge("environmental", "coordinator")
    workflow.add_edge("infrastructure", "coordinator")
    workflow.add_edge("temporal", "coordinator")
    
    # Coordinator -> End
    workflow.add_edge("coordinator", END)
    
    # Compile the workflow
    app = workflow.compile()
    
    return app


def run_ryudo_workflow(
    base_graph: nx.MultiDiGraph = None,
    environmental_data: Dict[str, Any] = None,
    infrastructure_data: Dict[str, Any] = None,
    temporal_data: Dict[str, Any] = None,
    mission: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the full Ryudo workflow.
    
    Parameters
    ----------
    base_graph : NetworkX MultiDiGraph, optional
        Pre-loaded road network graph
    environmental_data : dict
        Weather/cyclone data for Environmental Agent
    infrastructure_data : dict
        Power grid/sensor data for Infrastructure Agent
    temporal_data : dict
        Forecast data for Temporal Agent
    mission : dict
        Mission definition (type, targets, depot, etc.)
    
    Returns
    -------
    dict
        Final state with modified_graph, solution, and visualization_layers
    """
    # Build workflow
    app = build_ryudo_workflow()
    
    # Prepare initial state
    initial_state = {
        "base_graph": base_graph,
        "constraints": [],
        "environmental_data": environmental_data or {},
        "infrastructure_data": infrastructure_data or {},
        "temporal_data": temporal_data or {},
        "mission": mission or {},
        "modified_graph": None,
        "solution": None,
        "visualization_layers": [],
    }
    
    # Run the workflow
    print("=" * 60)
    print("RYUDO: Running LangGraph Agent Workflow")
    print("=" * 60)
    
    result = app.invoke(initial_state)
    
    print("=" * 60)
    print("RYUDO: Workflow Complete")
    print("=" * 60)
    
    return result
