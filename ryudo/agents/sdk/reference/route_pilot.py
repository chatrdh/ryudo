"""
Route Pilot Agent
=================

Predicts time-based validity of routes and emits TTL constraints.

This agent forecasts when roads will become impassable due to
evolving conditions (rising water, weather changes, traffic).
"""

from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable, Optional
import math

from shapely.geometry import Point

from ryudo.core.schema import ConstraintType, GraphConstraint, TimeWindow
from ryudo.agents.sdk.interface import AgentResult, BaseAgent, WorldState


@dataclass
class ZonePrediction:
    """Prediction for a zone's evolution."""
    zone: str
    hours_remaining: float
    weight_factor: float
    confidence: float
    description: str
    inner_radius_km: Optional[float] = None
    outer_radius_km: Optional[float] = None


class RoutePilot(BaseAgent):
    """
    Temporal prediction agent.
    
    Observes current conditions and forecasts how they will evolve,
    emitting time-limited constraints.
    
    Signals Expected
    ----------------
    - "temporal": dict with predictions
    - "environmental": for auto-generating predictions
    - "traffic": for congestion forecasts
    
    Example
    -------
    >>> agent = RoutePilot()
    >>> result = agent.observe(world_state)
    """
    
    agent_id = "route_pilot"
    priority = 5  # Medium priority
    description = "Temporal route validity prediction"
    
    def __init__(
        self,
        llm_client: Optional[Callable[[dict], str]] = None,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the agent.
        
        Parameters
        ----------
        llm_client : Callable, optional
            LLM for reasoning
        min_confidence : float
            Minimum confidence to emit constraint
        """
        self.llm_client = llm_client
        self.min_confidence = min_confidence
    
    def observe(self, state: WorldState) -> AgentResult:
        """Observe conditions and predict temporal constraints."""
        constraints = []
        visualization = []
        reasoning = ""
        
        temporal_data = state.get_signal("temporal", {})
        env_data = state.get_signal("environmental") or state.get_signal("cyclone")
        traffic_data = state.get_signal("traffic", {})
        
        # Get or generate predictions
        predictions = temporal_data.get("predictions", [])
        
        if not predictions and env_data:
            predictions = self._generate_predictions(env_data)
        
        if not predictions:
            return AgentResult(constraints=[], reasoning="No temporal data for predictions")
        
        # Get LLM analysis
        if self.llm_client:
            try:
                reasoning = self.llm_client({
                    "predictions": [p.__dict__ if hasattr(p, '__dict__') else p for p in predictions],
                    "existing_constraints": len(state.existing_constraints),
                })
            except Exception as e:
                reasoning = f"LLM unavailable: {e}"
        
        # Extract center point for zone predictions
        config = (env_data or {}).get("config", env_data or {})
        eye_lat = (env_data or {}).get("eye_lat") or config.get("eye_lat")
        eye_lon = (env_data or {}).get("eye_lon") or config.get("eye_lon")
        
        # Process each prediction
        for pred in predictions:
            if isinstance(pred, dict):
                pred = ZonePrediction(**pred)
            
            if pred.confidence < self.min_confidence:
                continue
            
            # Create time window for this prediction
            validity = TimeWindow(
                start_time=state.query_time,
                end_time=state.query_time + timedelta(hours=pred.hours_remaining)
            )
            
            # Create zone polygon if we have center and radii
            if eye_lat and eye_lon and pred.inner_radius_km and pred.outer_radius_km:
                polygon = self._create_ring_polygon(
                    eye_lat, eye_lon,
                    pred.inner_radius_km,
                    pred.outer_radius_km
                )
                
                constraints.append(GraphConstraint(
                    type=ConstraintType.ZONE_MASK,
                    target={"polygon": polygon.__geo_interface__},
                    effect={"weight_factor": pred.weight_factor},
                    validity=validity,
                    source_agent_id=self.agent_id,
                    priority=self.priority,
                    metadata={
                        "prediction_zone": pred.zone,
                        "hours_remaining": pred.hours_remaining,
                        "confidence": pred.confidence,
                        "description": pred.description,
                    }
                ))
            else:
                # Generic TTL constraint
                constraints.append(GraphConstraint(
                    type=ConstraintType.EDGE_WEIGHT,
                    target={"zone": pred.zone},
                    effect={"weight_factor": pred.weight_factor, "is_prediction": True},
                    validity=validity,
                    source_agent_id=self.agent_id,
                    priority=self.priority,
                    metadata={
                        "prediction_zone": pred.zone,
                        "hours_remaining": pred.hours_remaining,
                        "confidence": pred.confidence,
                        "description": pred.description,
                    }
                ))
            
            # Add info visualization
            if pred.confidence >= 0.6:
                visualization.append({
                    "type": "info",
                    "agent": self.agent_id,
                    "name": f"TTL: {pred.zone}",
                    "content": {
                        "zone": pred.zone,
                        "hours_remaining": pred.hours_remaining,
                        "confidence": f"{pred.confidence * 100:.0f}%",
                        "description": pred.description,
                    }
                })
        
        # Handle traffic predictions
        if traffic_data.get("evacuation_active"):
            evac_constraint = self._create_evacuation_constraint(
                state.query_time, traffic_data
            )
            if evac_constraint:
                constraints.append(evac_constraint)
        
        return AgentResult(
            constraints=constraints,
            visualization=visualization,
            reasoning=reasoning or f"Generated {len(constraints)} temporal predictions",
            confidence=min(p.confidence for p in predictions) if predictions else 1.0,
        )
    
    def _generate_predictions(self, env_data: dict) -> list[ZonePrediction]:
        """Generate predictions from environmental data."""
        config = env_data.get("config", env_data)
        
        extreme_radius = config.get("extreme_damage_radius_km", 10)
        severe_radius = config.get("severe_damage_radius_km", 25)
        moderate_radius = config.get("moderate_damage_radius_km", 50)
        
        return [
            ZonePrediction(
                zone="moderate_to_severe",
                hours_remaining=2.0,
                weight_factor=50.0,
                confidence=0.75,
                description="Moderate zone will escalate to severe conditions",
                inner_radius_km=severe_radius,
                outer_radius_km=severe_radius + 10,
            ),
            ZonePrediction(
                zone="severe_to_extreme",
                hours_remaining=1.0,
                weight_factor=500.0,
                confidence=0.85,
                description="Severe zone will become impassable",
                inner_radius_km=extreme_radius,
                outer_radius_km=severe_radius,
            ),
            ZonePrediction(
                zone="post_event_flooding",
                hours_remaining=6.0,
                weight_factor=100.0,
                confidence=0.6,
                description="Low-lying areas may flood as drainage overflows",
                inner_radius_km=None,
                outer_radius_km=None,
            ),
        ]
    
    def _create_ring_polygon(
        self,
        center_lat: float,
        center_lon: float,
        inner_radius_km: float,
        outer_radius_km: float,
        num_points: int = 32
    ):
        """Create a ring (donut) polygon."""
        from shapely.geometry import Polygon
        
        km_to_deg = 1 / 111.0
        
        # Outer ring
        outer_points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_lon + outer_radius_km * km_to_deg * math.cos(angle)
            y = center_lat + outer_radius_km * km_to_deg * math.sin(angle)
            outer_points.append((x, y))
        outer_points.append(outer_points[0])
        
        # Inner ring (hole)
        inner_points = []
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_lon + inner_radius_km * km_to_deg * math.cos(angle)
            y = center_lat + inner_radius_km * km_to_deg * math.sin(angle)
            inner_points.append((x, y))
        inner_points.append(inner_points[0])
        inner_points.reverse()  # Counterclockwise for hole
        
        return Polygon(outer_points, [inner_points])
    
    def _create_evacuation_constraint(
        self,
        query_time,
        traffic_data: dict
    ) -> Optional[GraphConstraint]:
        """Create constraint for evacuation traffic."""
        congestion_multiplier = traffic_data.get("congestion_multiplier", 4.0)
        duration_hours = traffic_data.get("duration_hours", 8.0)
        
        return GraphConstraint(
            type=ConstraintType.EDGE_WEIGHT,
            target={"road_types": ["primary", "secondary"], "direction": "outbound"},
            effect={"weight_factor": congestion_multiplier, "reason": "evacuation_traffic"},
            validity=TimeWindow(
                start_time=query_time,
                end_time=query_time + timedelta(hours=duration_hours)
            ),
            source_agent_id=self.agent_id,
            priority=self.priority,
            metadata={
                "traffic_type": "evacuation",
                "congestion_multiplier": congestion_multiplier,
            }
        )
