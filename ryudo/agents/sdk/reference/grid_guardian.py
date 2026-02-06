"""
Grid Guardian Agent
===================

Monitors infrastructure dependencies and emits node/edge constraints
when critical systems fail.

This agent models cascading failures - when a substation goes offline,
all dependent facilities lose power and become unavailable.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Optional
import math

from ryudo.core.schema import ConstraintType, GraphConstraint, TimeWindow
from ryudo.agents.sdk.interface import AgentResult, BaseAgent, WorldState


@dataclass
class Substation:
    """Power substation definition."""
    id: str
    lat: float
    lon: float
    dependents: list[str] = field(default_factory=list)


@dataclass
class Facility:
    """Dependent facility definition."""
    id: str
    name: str
    facility_type: str  # "hospital", "shelter", etc.
    lat: float
    lon: float
    capacity: int = 0
    is_critical: bool = False


class GridGuardian(BaseAgent):
    """
    Infrastructure dependency monitoring agent.
    
    Observes infrastructure status and emits constraints when systems fail.
    Models cascading failures through dependency graphs.
    
    Signals Expected
    ----------------
    - "infrastructure": dict with:
        - failed_substations: list of substation IDs
        - sensor_readings: optional health data
    - "environmental": dict with center coords (for impact assessment)
    
    Example
    -------
    >>> agent = GridGuardian(substations=my_substations, facilities=my_facilities)
    >>> result = agent.observe(world_state)
    """
    
    agent_id = "grid_guardian"
    priority = 8  # High priority for infrastructure
    description = "Infrastructure dependency and cascading failure detection"
    
    def __init__(
        self,
        substations: Optional[list[Substation]] = None,
        facilities: Optional[dict[str, Facility]] = None,
        llm_client: Optional[Callable[[dict], str]] = None,
        validity_hours: float = 12.0,
    ):
        """
        Initialize the agent.
        
        Parameters
        ----------
        substations : list[Substation], optional
            Power substations with dependencies
        facilities : dict[str, Facility], optional
            Facilities that depend on power
        llm_client : Callable, optional
            LLM for reasoning
        validity_hours : float
            Constraint validity duration
        """
        self.substations = substations or self._default_substations()
        self.facilities = facilities or self._default_facilities()
        self.llm_client = llm_client
        self.validity_hours = validity_hours
    
    def observe(self, state: WorldState) -> AgentResult:
        """Observe infrastructure status and emit failure constraints."""
        constraints = []
        visualization = []
        reasoning = ""
        
        infra_data = state.get_signal("infrastructure", {})
        env_data = state.get_signal("environmental") or state.get_signal("cyclone")
        
        # Determine which substations have failed
        failed_substations = set(infra_data.get("failed_substations", []))
        
        # If no explicit failures but we have environmental data, assess impact
        if not failed_substations and env_data:
            failed_substations = self._assess_environmental_impact(env_data)
        
        if not failed_substations:
            return AgentResult(
                constraints=[],
                reasoning="All infrastructure nominal"
            )
        
        # Get LLM analysis
        if self.llm_client:
            try:
                reasoning = self.llm_client({
                    "failed_substations": list(failed_substations),
                    "total_substations": len(self.substations),
                })
            except Exception as e:
                reasoning = f"LLM unavailable: {e}"
        
        validity = TimeWindow(
            start_time=state.query_time,
            end_time=state.query_time + timedelta(hours=self.validity_hours)
        )
        
        # Process each failed substation
        affected_facilities = set()
        
        for sub_id in failed_substations:
            substation = self._get_substation(sub_id)
            if not substation:
                continue
            
            # Mark substation node as disabled
            # Note: In production, you'd map to actual graph node IDs
            visualization.append({
                "type": "marker",
                "agent": self.agent_id,
                "name": f"Failed: {sub_id}",
                "position": (substation.lat, substation.lon),
                "popup": f"⚡ Substation Offline: {sub_id}",
                "icon": {"color": "red", "icon": "bolt", "prefix": "fa"}
            })
            
            # Cascade to dependent facilities
            for facility_id in substation.dependents:
                affected_facilities.add(facility_id)
        
        # Create constraints for affected facilities
        for facility_id in affected_facilities:
            facility = self.facilities.get(facility_id)
            if not facility:
                continue
            
            # Mark facility as unavailable via metadata
            # The actual node disable would require graph node ID mapping
            constraints.append(GraphConstraint(
                type=ConstraintType.NODE_STATUS,
                target={"facility_id": facility_id, "lat": facility.lat, "lon": facility.lon},
                effect={"action": "disable", "reason": "power_loss"},
                validity=validity,
                source_agent_id=self.agent_id,
                priority=self.priority,
                metadata={
                    "facility_name": facility.name,
                    "facility_type": facility.facility_type,
                    "is_critical": facility.is_critical,
                    "capacity_lost": facility.capacity,
                }
            ))
            
            # Visualization
            icon_color = "darkred" if facility.is_critical else "orange"
            visualization.append({
                "type": "marker",
                "agent": self.agent_id,
                "name": f"No Power: {facility.name}",
                "position": (facility.lat, facility.lon),
                "popup": f"🔌 {facility.name} - Power Lost",
                "icon": {"color": icon_color, "icon": "hospital" if facility.facility_type == "hospital" else "home", "prefix": "fa"}
            })
        
        return AgentResult(
            constraints=constraints,
            visualization=visualization,
            reasoning=reasoning or f"Infrastructure failures affecting {len(affected_facilities)} facilities",
            confidence=0.85,
        )
    
    def _assess_environmental_impact(self, env_data: dict) -> set[str]:
        """Determine which substations are in the damage zone."""
        config = env_data.get("config", env_data)
        eye_lat = env_data.get("eye_lat") or config.get("eye_lat")
        eye_lon = env_data.get("eye_lon") or config.get("eye_lon")
        
        if eye_lat is None or eye_lon is None:
            return set()
        
        extreme_radius = config.get("extreme_damage_radius_km", 10)
        failed = set()
        
        for sub in self.substations:
            dist = self._haversine(eye_lat, eye_lon, sub.lat, sub.lon)
            if dist <= extreme_radius:
                failed.add(sub.id)
        
        return failed
    
    def _haversine(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km between two points."""
        R = 6371
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))
    
    def _get_substation(self, sub_id: str) -> Optional[Substation]:
        """Get substation by ID."""
        for sub in self.substations:
            if sub.id == sub_id:
                return sub
        return None
    
    def _default_substations(self) -> list[Substation]:
        """Default substations for Visakhapatnam."""
        return [
            Substation("substation_north", 17.75, 83.30, ["hospital_kgh", "shelter_north_1"]),
            Substation("substation_central", 17.70, 83.25, ["hospital_visakha", "shelter_central_1"]),
            Substation("substation_south", 17.65, 83.20, ["hospital_naval", "shelter_south_1"]),
        ]
    
    def _default_facilities(self) -> dict[str, Facility]:
        """Default facilities for Visakhapatnam."""
        return {
            "hospital_kgh": Facility("hospital_kgh", "King George Hospital", "hospital", 17.7196, 83.3024, 500, True),
            "hospital_visakha": Facility("hospital_visakha", "Visakha General Hospital", "hospital", 17.7050, 83.2500, 300, True),
            "hospital_naval": Facility("hospital_naval", "Naval Dockyard Hospital", "hospital", 17.6850, 83.2800, 150, True),
            "shelter_north_1": Facility("shelter_north_1", "North Shelter", "shelter", 17.7480, 83.2960, 200, False),
            "shelter_central_1": Facility("shelter_central_1", "Central Shelter", "shelter", 17.7100, 83.2280, 200, False),
            "shelter_south_1": Facility("shelter_south_1", "South Shelter", "shelter", 17.6580, 83.2050, 180, False),
        }
