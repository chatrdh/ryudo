"""
Flood Sentinel Agent
====================

Observes environmental data (weather, satellite imagery) and emits
zone-based constraints for hazardous areas.

This is the reference implementation for environmental monitoring agents.
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Optional
import math

from shapely.geometry import Point, Polygon, LineString

from ryudo.core.schema import ConstraintType, GraphConstraint, TimeWindow
from ryudo.agents.sdk.interface import AgentResult, BaseAgent, WorldState


@dataclass
class ZoneConfig:
    """Configuration for a damage zone."""
    name: str
    radius_km: float
    weight_factor: float
    color: str
    description: str


# Default zone configurations
DEFAULT_ZONES = [
    ZoneConfig("extreme", 10, 1000.0, "#FF0000", "Extreme conditions - impassable"),
    ZoneConfig("severe", 25, 100.0, "#FF8C00", "Severe conditions - high cost"),
    ZoneConfig("moderate", 50, 10.0, "#FFD700", "Moderate conditions - elevated cost"),
]


class FloodSentinel(BaseAgent):
    """
    Environmental monitoring agent.
    
    Observes weather/environmental signals and emits zone constraints
    for hazardous areas. Works with any area-based hazard (storms,
    floods, fires, etc.).
    
    Signals Expected
    ----------------
    - "environmental" or "cyclone": dict with:
        - eye_lat, eye_lon: Center coordinates
        - config: Zone radius configuration
    
    Example
    -------
    >>> agent = FloodSentinel()
    >>> result = agent.observe(world_state)
    >>> print(f"Emitted {len(result.constraints)} constraints")
    """
    
    agent_id = "flood_sentinel"
    priority = 10  # High priority for safety-critical constraints
    description = "Environmental hazard zone detection"
    
    def __init__(
        self,
        zones: Optional[list[ZoneConfig]] = None,
        llm_client: Optional[Callable[[dict], str]] = None,
        validity_hours: float = 24.0,
    ):
        """
        Initialize the agent.
        
        Parameters
        ----------
        zones : list[ZoneConfig], optional
            Custom zone configurations. Uses defaults if not provided.
        llm_client : Callable, optional
            Function that takes config dict and returns reasoning string
        validity_hours : float
            How long constraints are valid (default 24 hours)
        """
        self.zones = zones or DEFAULT_ZONES
        self.llm_client = llm_client
        self.validity_hours = validity_hours
    
    def observe(self, state: WorldState) -> AgentResult:
        """Observe environmental conditions and emit zone constraints."""
        constraints = []
        visualization = []
        reasoning = ""
        
        # Check for environmental data in signals
        env_data = state.get_signal("environmental") or state.get_signal("cyclone")
        
        if not env_data:
            return AgentResult(constraints=[], reasoning="No environmental data available")
        
        # Extract center point
        config = env_data.get("config", env_data)
        eye_lat = env_data.get("eye_lat") or config.get("eye_lat")
        eye_lon = env_data.get("eye_lon") or config.get("eye_lon")
        
        if eye_lat is None or eye_lon is None:
            return AgentResult(constraints=[], reasoning="Missing center coordinates")
        
        # Get LLM reasoning if available
        if self.llm_client:
            try:
                reasoning = self.llm_client(config)
            except Exception as e:
                reasoning = f"LLM analysis unavailable: {e}"
        
        # Create zone polygons and constraints
        validity = TimeWindow(
            start_time=state.query_time,
            end_time=state.query_time + timedelta(hours=self.validity_hours)
        )
        
        for zone in self.zones:
            # Create circular polygon for zone
            polygon = self._create_circle_polygon(eye_lat, eye_lon, zone.radius_km)
            
            # Override radius from config if available
            config_key = f"{zone.name}_damage_radius_km"
            if config_key in config:
                polygon = self._create_circle_polygon(
                    eye_lat, eye_lon, config[config_key]
                )
            
            # Create constraint
            constraints.append(GraphConstraint(
                type=ConstraintType.ZONE_MASK,
                target={"polygon": polygon.__geo_interface__},
                effect={"weight_factor": zone.weight_factor},
                validity=validity,
                source_agent_id=self.agent_id,
                priority=self.priority,
                metadata={
                    "zone_type": zone.name,
                    "center": (eye_lat, eye_lon),
                    "radius_km": zone.radius_km,
                    "description": zone.description,
                }
            ))
            
            # Add visualization
            visualization.append({
                "type": "polygon",
                "agent": self.agent_id,
                "name": f"{zone.name.title()} Zone",
                "geometry": polygon.__geo_interface__,
                "style": {
                    "fillColor": zone.color,
                    "color": zone.color,
                    "weight": 2,
                    "fillOpacity": 0.15,
                }
            })
        
        # Add surge zone if configured
        surge_constraint, surge_viz = self._create_surge_zone(config, validity)
        if surge_constraint:
            constraints.append(surge_constraint)
            visualization.append(surge_viz)
        
        # Add center marker
        visualization.append({
            "type": "marker",
            "agent": self.agent_id,
            "name": "Hazard Center",
            "position": (eye_lat, eye_lon),
            "popup": f"Center: ({eye_lat:.4f}, {eye_lon:.4f})",
            "icon": {"color": "red", "icon": "exclamation-triangle", "prefix": "fa"}
        })
        
        return AgentResult(
            constraints=constraints,
            visualization=visualization,
            reasoning=reasoning,
            confidence=0.9,
        )
    
    def _create_circle_polygon(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        num_points: int = 32
    ) -> Polygon:
        """Create a circular polygon approximation."""
        km_to_deg = 1 / 111.0  # Approximate conversion
        radius_deg = radius_km * km_to_deg
        
        points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_lon + radius_deg * math.cos(angle)
            y = center_lat + radius_deg * math.sin(angle)
            points.append((x, y))
        
        points.append(points[0])  # Close the polygon
        return Polygon(points)
    
    def _create_surge_zone(
        self,
        config: dict,
        validity: TimeWindow
    ) -> tuple[Optional[GraphConstraint], Optional[dict]]:
        """Create storm surge flood zone if configured."""
        surge_height = config.get("storm_surge_height_m")
        if not surge_height or surge_height <= 0:
            return None, None
        
        # Simplified coastal geometry (would be loaded from data in production)
        coastal_points = [
            (83.1, 17.4),
            (83.2, 17.5),
            (83.3, 17.6),
            (83.4, 17.7),
            (83.5, 17.8),
        ]
        
        coastline = LineString(coastal_points)
        surge_distance = config.get("surge_inland_distance_km", 2) / 111.0
        surge_polygon = coastline.buffer(surge_distance)
        
        constraint = GraphConstraint(
            type=ConstraintType.ZONE_MASK,
            target={"polygon": surge_polygon.__geo_interface__},
            effect={"weight_factor": 1000.0, "node_action": "disable"},
            validity=validity,
            source_agent_id=self.agent_id,
            priority=self.priority + 1,  # Higher priority than wind zones
            metadata={
                "zone_type": "surge",
                "surge_height_m": surge_height,
                "inland_distance_km": config.get("surge_inland_distance_km", 2),
            }
        )
        
        viz = {
            "type": "polygon",
            "agent": self.agent_id,
            "name": "Storm Surge Zone",
            "geometry": surge_polygon.__geo_interface__,
            "style": {
                "fillColor": "#0066FF",
                "color": "#0033CC",
                "weight": 2,
                "fillOpacity": 0.4,
            }
        }
        
        return constraint, viz
