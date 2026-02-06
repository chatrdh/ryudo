"""
Mission Profile Schema
======================

Configuration-driven mission definitions with objectives, assets, and targets.

This module defines the data models for mission specification:
- MissionProfile: Top-level mission configuration
- Asset: Vehicle/resource definitions with capabilities
- Target: Locations to visit/serve
- ObjectiveType: Optimization goals (min_time, min_risk, max_coverage)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Enums
# =============================================================================


class ObjectiveType(str, Enum):
    """Mission optimization objectives."""
    
    MIN_TIME = "min_time"
    """Minimize total mission completion time."""
    
    MIN_RISK = "min_risk"
    """Minimize exposure to hazardous areas."""
    
    MAX_COVERAGE = "max_coverage"
    """Maximize area/edge coverage (patrol mode)."""
    
    MAX_TARGETS = "max_targets"
    """Maximize number of targets served."""
    
    BALANCED = "balanced"
    """Multi-objective weighted optimization."""


class AssetCapability(str, Enum):
    """Asset capability flags for routing decisions."""
    
    ROAD = "road"
    """Standard road travel (default for vehicles)."""
    
    WATER = "water"
    """Can traverse flooded/water areas (boats, amphibious)."""
    
    AIR = "air"
    """Ignores ground obstacles (drones, helicopters)."""
    
    OFFROAD = "offroad"
    """Can traverse unpaved/damaged roads."""
    
    MEDICAL = "medical"
    """Has medical equipment for critical patients."""
    
    HIGH_CAPACITY = "high_capacity"
    """Can carry 20+ people."""
    
    FLOOD_CAPABLE = "flood_capable"
    """Can ford water up to 1m depth."""


class TargetPriority(str, Enum):
    """Target priority levels."""
    
    CRITICAL = "critical"
    """Immediate response required (TTL < 1 hour)."""
    
    HIGH = "high"
    """High priority (TTL 1-3 hours)."""
    
    MEDIUM = "medium"
    """Medium priority (TTL 3-6 hours)."""
    
    LOW = "low"
    """Low priority (TTL > 6 hours)."""


# =============================================================================
# Data Models
# =============================================================================


@dataclass(frozen=True)
class Asset:
    """
    A resource available for mission execution.
    
    Examples: rescue trucks, ambulances, boats, drones.
    
    Attributes
    ----------
    id : str
        Unique identifier
    name : str
        Human-readable name
    capabilities : tuple[AssetCapability, ...]
        What this asset can do (determines routing)
    capacity : int
        Maximum passengers/cargo units
    speed_kmh : float
        Average speed in km/h
    depot_lat : float
        Starting latitude
    depot_lon : float
        Starting longitude
    fuel_range_km : float
        Maximum travel distance without refueling
    metadata : dict
        Additional asset-specific data
    """
    
    id: str
    name: str
    capabilities: tuple[AssetCapability, ...] = (AssetCapability.ROAD,)
    capacity: int = 10
    speed_kmh: float = 40.0
    depot_lat: float = 0.0
    depot_lon: float = 0.0
    fuel_range_km: float = 200.0
    metadata: dict = field(default_factory=dict)
    
    def has_capability(self, cap: AssetCapability) -> bool:
        """Check if asset has a specific capability."""
        return cap in self.capabilities
    
    def can_access_flooded(self) -> bool:
        """Check if asset can traverse flooded areas."""
        return (
            AssetCapability.WATER in self.capabilities or
            AssetCapability.AIR in self.capabilities or
            AssetCapability.FLOOD_CAPABLE in self.capabilities
        )
    
    @classmethod
    def from_dict(cls, data: dict) -> "Asset":
        """Create Asset from dictionary."""
        caps = data.get("capabilities", ["road"])
        if isinstance(caps, list):
            caps = tuple(AssetCapability(c) for c in caps)
        
        return cls(
            id=data["id"],
            name=data.get("name", data["id"]),
            capabilities=caps,
            capacity=data.get("capacity", 10),
            speed_kmh=data.get("speed_kmh", 40.0),
            depot_lat=data.get("depot_lat", data.get("lat", 0.0)),
            depot_lon=data.get("depot_lon", data.get("lon", 0.0)),
            fuel_range_km=data.get("fuel_range_km", 200.0),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True)
class Target:
    """
    A location to be visited or served during a mission.
    
    Attributes
    ----------
    id : str
        Unique identifier
    lat : float
        Latitude
    lon : float
        Longitude
    population : int
        Number of people at this target
    priority : float
        Priority score (0-100, higher = more urgent)
    ttl_hours : float
        Time-to-live in hours (window before situation worsens)
    required_capabilities : tuple
        Capabilities needed to serve this target
    name : str
        Human-readable name
    metadata : dict
        Additional data (zone, building_type, etc.)
    """
    
    id: str
    lat: float
    lon: float
    population: int = 0
    priority: float = 50.0
    ttl_hours: Optional[float] = None
    required_capabilities: tuple[AssetCapability, ...] = ()
    name: str = ""
    metadata: dict = field(default_factory=dict)
    
    @property
    def priority_level(self) -> TargetPriority:
        """Get priority level from TTL."""
        if self.ttl_hours is None:
            if self.priority >= 90:
                return TargetPriority.CRITICAL
            elif self.priority >= 70:
                return TargetPriority.HIGH
            elif self.priority >= 40:
                return TargetPriority.MEDIUM
            return TargetPriority.LOW
        
        if self.ttl_hours < 1:
            return TargetPriority.CRITICAL
        elif self.ttl_hours < 3:
            return TargetPriority.HIGH
        elif self.ttl_hours < 6:
            return TargetPriority.MEDIUM
        return TargetPriority.LOW
    
    def requires_capability(self, cap: AssetCapability) -> bool:
        """Check if target requires a specific capability."""
        return cap in self.required_capabilities
    
    @classmethod
    def from_dict(cls, data: dict) -> "Target":
        """Create Target from dictionary."""
        req_caps = data.get("required_capabilities", [])
        if isinstance(req_caps, list):
            req_caps = tuple(AssetCapability(c) for c in req_caps)
        
        return cls(
            id=data["id"],
            lat=data["lat"],
            lon=data["lon"],
            population=data.get("population", 0),
            priority=data.get("priority", data.get("priority_score", 50.0)),
            ttl_hours=data.get("ttl_hours"),
            required_capabilities=req_caps,
            name=data.get("name", data["id"]),
            metadata={
                k: v for k, v in data.items()
                if k not in {"id", "lat", "lon", "population", "priority", 
                            "priority_score", "ttl_hours", "required_capabilities", "name"}
            },
        )


# =============================================================================
# Mission Profile
# =============================================================================


class MissionProfile(BaseModel):
    """
    Complete mission specification.
    
    A MissionProfile defines:
    - What to optimize (objective)
    - Available resources (assets)
    - Locations to serve (targets)
    - Constraints (time, distance, capacity)
    
    Example
    -------
    ```python
    profile = MissionProfile(
        id="rescue_001",
        name="Cyclone Hudhud Rescue",
        objective=ObjectiveType.MIN_TIME,
        assets=[
            Asset(id="truck1", capabilities=(AssetCapability.ROAD,), capacity=15),
            Asset(id="boat1", capabilities=(AssetCapability.WATER,), capacity=8),
        ],
        targets=[
            Target(id="T01", lat=17.78, lon=83.38, population=45, priority=100),
        ],
        depot={"lat": 17.6868, "lon": 83.2185},
    )
    ```
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Identity
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    name: str = "Unnamed Mission"
    description: str = ""
    
    # Optimization objective
    objective: ObjectiveType = ObjectiveType.BALANCED
    
    # Resources
    assets: list[Asset] = Field(default_factory=list)
    
    # Targets
    targets: list[Target] = Field(default_factory=list)
    
    # Depot location
    depot: dict[str, float] = Field(default_factory=lambda: {"lat": 0.0, "lon": 0.0})
    
    # Constraints
    max_duration_hours: Optional[float] = None
    max_distance_km: Optional[float] = None
    
    # Multi-objective weights
    weights: dict[str, float] = Field(default_factory=lambda: {
        "time": 0.4,
        "risk": 0.3,
        "coverage": 0.3,
    })
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def total_population(self) -> int:
        """Total population across all targets."""
        return sum(t.population for t in self.targets)
    
    @property
    def total_capacity(self) -> int:
        """Total capacity of all assets."""
        return sum(a.capacity for a in self.assets)
    
    @property
    def critical_targets(self) -> list[Target]:
        """Targets with CRITICAL priority."""
        return [t for t in self.targets if t.priority_level == TargetPriority.CRITICAL]
    
    @property
    def high_priority_targets(self) -> list[Target]:
        """Targets with HIGH or CRITICAL priority."""
        return [
            t for t in self.targets 
            if t.priority_level in (TargetPriority.CRITICAL, TargetPriority.HIGH)
        ]
    
    # -------------------------------------------------------------------------
    # Methods
    # -------------------------------------------------------------------------
    
    def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Get asset by ID."""
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        return None
    
    def get_target(self, target_id: str) -> Optional[Target]:
        """Get target by ID."""
        for target in self.targets:
            if target.id == target_id:
                return target
        return None
    
    def assets_with_capability(self, cap: AssetCapability) -> list[Asset]:
        """Get all assets with a specific capability."""
        return [a for a in self.assets if a.has_capability(cap)]
    
    def targets_requiring_capability(self, cap: AssetCapability) -> list[Target]:
        """Get targets requiring a specific capability."""
        return [t for t in self.targets if t.requires_capability(cap)]
    
    def estimate_minimum_trips(self) -> int:
        """Estimate minimum trips needed based on capacity."""
        if self.total_capacity == 0:
            return len(self.targets)
        return max(1, (self.total_population + self.total_capacity - 1) // self.total_capacity)
    
    @classmethod
    def from_dict(cls, data: dict) -> "MissionProfile":
        """Create MissionProfile from dictionary/JSON."""
        assets = [
            Asset.from_dict(a) if isinstance(a, dict) else a
            for a in data.get("assets", [])
        ]
        targets = [
            Target.from_dict(t) if isinstance(t, dict) else t
            for t in data.get("targets", [])
        ]
        
        objective = data.get("objective", "balanced")
        if isinstance(objective, str):
            objective = ObjectiveType(objective)
        
        return cls(
            id=data.get("id", str(uuid4())[:8]),
            name=data.get("name", "Unnamed Mission"),
            description=data.get("description", ""),
            objective=objective,
            assets=assets,
            targets=targets,
            depot=data.get("depot", {"lat": 0.0, "lon": 0.0}),
            max_duration_hours=data.get("max_duration_hours"),
            max_distance_km=data.get("max_distance_km"),
            weights=data.get("weights", {"time": 0.4, "risk": 0.3, "coverage": 0.3}),
            metadata=data.get("metadata", {}),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "objective": self.objective.value,
            "assets": [
                {
                    "id": a.id,
                    "name": a.name,
                    "capabilities": [c.value for c in a.capabilities],
                    "capacity": a.capacity,
                    "speed_kmh": a.speed_kmh,
                    "depot_lat": a.depot_lat,
                    "depot_lon": a.depot_lon,
                    "fuel_range_km": a.fuel_range_km,
                    "metadata": a.metadata,
                }
                for a in self.assets
            ],
            "targets": [
                {
                    "id": t.id,
                    "name": t.name,
                    "lat": t.lat,
                    "lon": t.lon,
                    "population": t.population,
                    "priority": t.priority,
                    "ttl_hours": t.ttl_hours,
                    "required_capabilities": [c.value for c in t.required_capabilities],
                    **t.metadata,
                }
                for t in self.targets
            ],
            "depot": self.depot,
            "max_duration_hours": self.max_duration_hours,
            "max_distance_km": self.max_distance_km,
            "weights": self.weights,
            "metadata": self.metadata,
        }
