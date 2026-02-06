"""
Reference Agents
================

Example agent implementations demonstrating the SDK.

These are ports of the original CycloneShield agents:
- FloodSentinel: Environmental zone detection
- GridGuardian: Infrastructure dependency tracking
- RoutePilot: Temporal validity prediction
"""

from ryudo.agents.sdk.reference.flood_sentinel import FloodSentinel
from ryudo.agents.sdk.reference.grid_guardian import GridGuardian
from ryudo.agents.sdk.reference.route_pilot import RoutePilot

__all__ = [
    "FloodSentinel",
    "GridGuardian",
    "RoutePilot",
]
