"""
Ryudo Core: Domain-Agnostic Living Graph Engine
================================================

This package provides a graph engine that treats the physical world as a 
programmable, non-stationary graph where constraints define the "rules of 
engagement" at query time.

Public API:
- LivingGraph: The main engine class
- GraphConstraint: Typed constraint model
- ConstraintType: Enum of constraint types
- TimeWindow: Validity period for constraints
- TagMapper: OSM tag → graph weight converter
"""

from ryudo.core.schema import (
    ConstraintType,
    TimeWindow,
    GraphConstraint,
)
from ryudo.core.mapper import TagMapper, GraphAttributes
from ryudo.core.engine import (
    LivingGraph,
    ConstraintApplicationRecord,
    CONSTRAINT_PRECEDENCE,
)

__all__ = [
    "LivingGraph",
    "GraphConstraint",
    "ConstraintType",
    "TimeWindow",
    "TagMapper",
    "GraphAttributes",
    "ConstraintApplicationRecord",
    "CONSTRAINT_PRECEDENCE",
]
