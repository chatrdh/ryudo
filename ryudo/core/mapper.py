"""
Tag Mapper
==========

Configuration-driven converter from OSM tags to abstract graph attributes.

This replaces hardcoded road classification dictionaries with a flexible,
JSON-configurable system that maps raw OSM tags to normalized weights 
and priorities.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
import json
from pathlib import Path


@dataclass
class GraphAttributes:
    """
    Normalized attributes for a graph element (node or edge).
    
    These are abstract properties used for routing and analysis,
    divorced from domain-specific semantics.
    """
    
    base_weight: float = 1.0
    """Base traversal cost multiplier. Lower = faster."""
    
    priority: int = 5
    """Priority level (1-10). Higher = more important routes."""
    
    capacity: str = "medium"
    """Throughput capacity: very_low, low, medium, high, very_high."""
    
    max_speed_kmh: int = 50
    """Maximum speed in km/h."""
    
    bidirectional: bool = True
    """Whether this edge can be traversed both ways."""
    
    tags: dict[str, Any] = field(default_factory=dict)
    """Additional tags preserved from the source data."""


# Default configuration for OSM highway tags
DEFAULT_HIGHWAY_CONFIG: dict[str, dict[str, Any]] = {
    "motorway": {
        "base_weight": 0.8,
        "priority": 10,
        "capacity": "very_high",
        "max_speed_kmh": 100,
    },
    "motorway_link": {
        "base_weight": 0.9,
        "priority": 9,
        "capacity": "high",
        "max_speed_kmh": 60,
    },
    "trunk": {
        "base_weight": 0.9,
        "priority": 9,
        "capacity": "very_high",
        "max_speed_kmh": 80,
    },
    "trunk_link": {
        "base_weight": 1.0,
        "priority": 8,
        "capacity": "high",
        "max_speed_kmh": 50,
    },
    "primary": {
        "base_weight": 1.0,
        "priority": 8,
        "capacity": "high",
        "max_speed_kmh": 60,
    },
    "primary_link": {
        "base_weight": 1.1,
        "priority": 7,
        "capacity": "medium",
        "max_speed_kmh": 40,
    },
    "secondary": {
        "base_weight": 1.1,
        "priority": 7,
        "capacity": "medium",
        "max_speed_kmh": 50,
    },
    "secondary_link": {
        "base_weight": 1.2,
        "priority": 6,
        "capacity": "medium",
        "max_speed_kmh": 40,
    },
    "tertiary": {
        "base_weight": 1.2,
        "priority": 6,
        "capacity": "medium",
        "max_speed_kmh": 40,
    },
    "tertiary_link": {
        "base_weight": 1.3,
        "priority": 5,
        "capacity": "low",
        "max_speed_kmh": 30,
    },
    "residential": {
        "base_weight": 1.5,
        "priority": 4,
        "capacity": "low",
        "max_speed_kmh": 30,
    },
    "living_street": {
        "base_weight": 2.0,
        "priority": 3,
        "capacity": "very_low",
        "max_speed_kmh": 20,
    },
    "unclassified": {
        "base_weight": 1.4,
        "priority": 4,
        "capacity": "low",
        "max_speed_kmh": 30,
    },
    "service": {
        "base_weight": 1.8,
        "priority": 2,
        "capacity": "very_low",
        "max_speed_kmh": 20,
    },
}


class TagMapper:
    """
    Maps raw OSM tags to normalized graph attributes.
    
    Example
    -------
    >>> config = {"highway": {"motorway": {"base_weight": 0.8, "priority": 10}}}
    >>> mapper = TagMapper(config)
    >>> attrs = mapper.normalize_attributes({"highway": "motorway"})
    >>> attrs.base_weight
    0.8
    """
    
    def __init__(self, config: Optional[dict[str, Any]] = None):
        """
        Initialize the mapper with configuration.
        
        Parameters
        ----------
        config : dict, optional
            Configuration mapping OSM tag keys to value mappings.
            If None, uses DEFAULT_HIGHWAY_CONFIG.
        """
        if config is None:
            self.config = {"highway": DEFAULT_HIGHWAY_CONFIG}
        else:
            self.config = config
    
    @classmethod
    def from_json_file(cls, path: Path | str) -> "TagMapper":
        """Load configuration from a JSON file."""
        with open(path) as f:
            config = json.load(f)
        return cls(config)
    
    def normalize_attributes(self, osm_tags: dict[str, Any]) -> GraphAttributes:
        """
        Convert OSM tags to normalized graph attributes.
        
        Parameters
        ----------
        osm_tags : dict
            Raw OSM tags (e.g., {"highway": "primary", "maxspeed": "60"})
        
        Returns
        -------
        GraphAttributes
            Normalized attributes for routing
        """
        attrs = GraphAttributes()
        
        # Check highway type first (most common for routing)
        highway_type = osm_tags.get("highway")
        if highway_type and "highway" in self.config:
            highway_config = self.config["highway"].get(highway_type, {})
            
            attrs.base_weight = highway_config.get("base_weight", attrs.base_weight)
            attrs.priority = highway_config.get("priority", attrs.priority)
            attrs.capacity = highway_config.get("capacity", attrs.capacity)
            attrs.max_speed_kmh = highway_config.get("max_speed_kmh", attrs.max_speed_kmh)
        
        # Override max speed if explicitly set
        if "maxspeed" in osm_tags:
            try:
                # Handle formats like "60", "60 km/h", "60 mph"
                speed_str = str(osm_tags["maxspeed"]).lower()
                speed_str = speed_str.replace("km/h", "").replace("kmh", "").strip()
                if "mph" in speed_str:
                    speed_str = speed_str.replace("mph", "").strip()
                    attrs.max_speed_kmh = int(float(speed_str) * 1.60934)
                else:
                    attrs.max_speed_kmh = int(float(speed_str))
            except (ValueError, TypeError):
                pass  # Keep default
        
        # Check one-way
        oneway = osm_tags.get("oneway", "no")
        attrs.bidirectional = oneway not in ("yes", "true", "1", "-1")
        
        # Preserve original tags
        attrs.tags = osm_tags.copy()
        
        return attrs
    
    def get_edge_weight(
        self, 
        osm_tags: dict[str, Any], 
        length_m: float,
        weight_type: str = "travel_time"
    ) -> float:
        """
        Calculate edge weight for routing.
        
        Parameters
        ----------
        osm_tags : dict
            Raw OSM tags
        length_m : float
            Edge length in meters
        weight_type : str
            One of "travel_time", "length", "weighted"
        
        Returns
        -------
        float
            Edge weight for routing algorithms
        """
        attrs = self.normalize_attributes(osm_tags)
        
        if weight_type == "length":
            return length_m
        
        if weight_type == "travel_time":
            # Time in seconds
            speed_ms = attrs.max_speed_kmh / 3.6  # Convert to m/s
            return length_m / speed_ms if speed_ms > 0 else float("inf")
        
        if weight_type == "weighted":
            # Length modified by base_weight
            return length_m * attrs.base_weight
        
        return length_m  # Fallback
