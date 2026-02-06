"""
Ryudo Mission Package
=====================

Pluggable mission solver architecture with multi-modal routing support.
"""

from ryudo.mission.profile import (
    MissionProfile,
    ObjectiveType,
    Asset,
    Target,
    AssetCapability,
)
from ryudo.mission.solver import (
    BaseSolver,
    SolverRegistry,
    SolutionResult,
)
from ryudo.mission.coordinator import MissionCoordinator

__all__ = [
    # Profile
    "MissionProfile",
    "ObjectiveType", 
    "Asset",
    "Target",
    "AssetCapability",
    # Solver
    "BaseSolver",
    "SolverRegistry",
    "SolutionResult",
    # Coordinator
    "MissionCoordinator",
]
