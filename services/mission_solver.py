"""
Mission Solver Service
=======================
Solves rescue mission routing with vehicle-target assignment.

Features:
- Priority-based target assignment
- Vehicle capacity matching
- Medical needs routing
- Road network-based routing
- TTL-aware scheduling
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import math

from services.routing_service import (
    get_road_graph,
    apply_constraints,
    find_route,
    find_multi_stop_route,
    haversine_distance,
)


# =============================================================================
# Expanded Rescue Mission Data
# =============================================================================

RESCUE_DEPOT = {
    "id": "DEPOT",
    "name": "NDRF Emergency Response Center",
    "address": "Dwaraka Nagar, Visakhapatnam",
    "lat": 17.6868,
    "lon": 83.2185,
    "type": "depot",
    "vehicles_available": 6,
}

RESCUE_VEHICLES = [
    {
        "id": "V1",
        "name": "Heavy Rescue Unit Alpha",
        "type": "heavy_rescue",
        "capacity": 15,
        "speed_kmh": 35,
        "fuel_range_km": 150,
        "medical_equipped": False,
        "water_fording_m": 0.8,
        "crew": 4,
        "status": "available",
    },
    {
        "id": "V2",
        "name": "Medical Ambulance 1",
        "type": "ambulance",
        "capacity": 4,
        "speed_kmh": 60,
        "fuel_range_km": 200,
        "medical_equipped": True,
        "water_fording_m": 0.3,
        "crew": 2,
        "status": "available",
        "equipment": ["stretcher", "oxygen", "defibrillator"],
    },
    {
        "id": "V3",
        "name": "Medical Ambulance 2",
        "type": "ambulance",
        "capacity": 4,
        "speed_kmh": 60,
        "fuel_range_km": 200,
        "medical_equipped": True,
        "water_fording_m": 0.3,
        "crew": 2,
        "status": "available",
        "equipment": ["stretcher", "oxygen", "dialysis_portable"],
    },
    {
        "id": "V4",
        "name": "Army Troop Carrier",
        "type": "troop_carrier",
        "capacity": 25,
        "speed_kmh": 30,
        "fuel_range_km": 120,
        "medical_equipped": False,
        "water_fording_m": 1.2,
        "crew": 3,
        "status": "available",
    },
    {
        "id": "V5",
        "name": "Light Rescue Van",
        "type": "light_rescue",
        "capacity": 8,
        "speed_kmh": 50,
        "fuel_range_km": 180,
        "medical_equipped": False,
        "water_fording_m": 0.4,
        "crew": 2,
        "status": "available",
    },
    {
        "id": "V6",
        "name": "High-Water Rescue Truck",
        "type": "flood_rescue",
        "capacity": 12,
        "speed_kmh": 25,
        "fuel_range_km": 100,
        "medical_equipped": True,
        "water_fording_m": 1.5,
        "crew": 4,
        "status": "available",
        "equipment": ["inflatable_boat", "life_jackets", "rescue_ropes"],
    },
]

RESCUE_TARGETS = [
    # =========================================================================
    # EXTREME ZONE - Coastal/Surge Areas (TTL < 1 hour)
    # =========================================================================
    {
        "id": "T01",
        "name": "Rushikonda Fishing Village",
        "address": "Beach Road, Rushikonda, Visakhapatnam 530045",
        "lat": 17.7823,
        "lon": 83.3842,
        "zone": "extreme",
        "population": 45,
        "type": "settlement",
        "building_type": "informal",
        "ttl_hours": 0.5,
        "medical_needs": ["elderly_care"],
        "mobility_issues": 5,
        "accessibility": "flooded",
        "contact": "+91-9876543210",
        "priority_score": 100,
        "notes": "15 fishing families, 5 elderly require assistance",
    },
    {
        "id": "T02",
        "name": "RK Beach Resort Staff",
        "address": "RK Beach Road, Visakhapatnam 530023",
        "lat": 17.7166,
        "lon": 83.3283,
        "zone": "extreme",
        "population": 28,
        "type": "commercial",
        "building_type": "concrete",
        "ttl_hours": 0.75,
        "medical_needs": [],
        "mobility_issues": 2,
        "accessibility": "road_damaged",
        "priority_score": 85,
        "notes": "Staff trapped in ground floor, water rising",
    },
    {
        "id": "T03",
        "name": "Coastal Dialysis Patient",
        "address": "Sagar Nagar, Beach Road, Visakhapatnam",
        "lat": 17.7420,
        "lon": 83.3650,
        "zone": "extreme",
        "population": 3,
        "type": "residential",
        "building_type": "apartment",
        "ttl_hours": 0.5,
        "medical_needs": ["dialysis", "critical_medical"],
        "mobility_issues": 1,
        "accessibility": "flooded",
        "priority_score": 98,
        "notes": "Patient missed dialysis, critical condition",
    },
    {
        "id": "T04",
        "name": "Port Area Workers",
        "address": "Visakhapatnam Port Trust Colony",
        "lat": 17.6842,
        "lon": 83.2867,
        "zone": "extreme",
        "population": 35,
        "type": "industrial",
        "building_type": "warehouse",
        "ttl_hours": 1.0,
        "medical_needs": [],
        "mobility_issues": 0,
        "accessibility": "road_damaged",
        "priority_score": 75,
        "notes": "Workers sheltering in warehouse, structure stable",
    },
    
    # =========================================================================
    # SEVERE ZONE - High Risk Areas (TTL 1-3 hours)
    # =========================================================================
    {
        "id": "T05",
        "name": "St. Joseph's Primary School",
        "address": "MVP Colony, Visakhapatnam 530017",
        "lat": 17.7350,
        "lon": 83.2850,
        "zone": "severe",
        "population": 120,
        "type": "school",
        "building_type": "concrete",
        "ttl_hours": 2.0,
        "medical_needs": [],
        "mobility_issues": 0,
        "accessibility": "road_clear",
        "priority_score": 90,
        "notes": "Children stranded after school flood, teachers present",
    },
    {
        "id": "T06",
        "name": "Kancharapalem Clinic Patients",
        "address": "Main Road, Kancharapalem, Visakhapatnam",
        "lat": 17.7150,
        "lon": 83.2950,
        "zone": "severe",
        "population": 18,
        "type": "medical",
        "building_type": "commercial",
        "ttl_hours": 1.5,
        "medical_needs": ["oxygen_dependent", "elderly_care"],
        "mobility_issues": 8,
        "accessibility": "road_clear",
        "priority_score": 92,
        "notes": "Small clinic, 3 on oxygen, generator failing",
    },
    {
        "id": "T07",
        "name": "Rajayyapeta Slum",
        "address": "Rajayyapeta, Visakhapatnam 530020",
        "lat": 17.7080,
        "lon": 83.3100,
        "zone": "severe",
        "population": 85,
        "type": "settlement",
        "building_type": "informal",
        "ttl_hours": 2.5,
        "medical_needs": ["pregnant_woman"],
        "mobility_issues": 12,
        "accessibility": "road_damaged",
        "priority_score": 88,
        "notes": "Low-lying area, 1 pregnant woman near due date",
    },
    {
        "id": "T08",
        "name": "Temple Compound Shelter",
        "address": "Sri Venkateswara Temple, Simhachalam Road",
        "lat": 17.7680,
        "lon": 83.2520,
        "zone": "severe",
        "population": 65,
        "type": "shelter",
        "building_type": "concrete",
        "ttl_hours": 3.0,
        "medical_needs": [],
        "mobility_issues": 8,
        "accessibility": "road_clear",
        "priority_score": 70,
        "notes": "Self-evacuated locals, food running low",
    },
    {
        "id": "T09",
        "name": "Gajuwaka Market Vendors",
        "address": "Gajuwaka Junction, Visakhapatnam 530026",
        "lat": 17.6980,
        "lon": 83.2150,
        "zone": "severe",
        "population": 42,
        "type": "commercial",
        "building_type": "mixed",
        "ttl_hours": 2.0,
        "medical_needs": [],
        "mobility_issues": 3,
        "accessibility": "road_clear",
        "priority_score": 65,
        "notes": "Vendors and customers trapped in market building",
    },
    {
        "id": "T10",
        "name": "Railway Colony Flooding",
        "address": "Railway New Colony, Visakhapatnam",
        "lat": 17.6920,
        "lon": 83.2880,
        "zone": "severe",
        "population": 55,
        "type": "residential",
        "building_type": "quarters",
        "ttl_hours": 2.5,
        "medical_needs": ["elderly_care"],
        "mobility_issues": 10,
        "accessibility": "road_damaged",
        "priority_score": 72,
        "notes": "Ground floor flooded, families on upper floors",
    },
    
    # =========================================================================
    # MODERATE ZONE - Medium Risk (TTL 3-6 hours)
    # =========================================================================
    {
        "id": "T11",
        "name": "Madhurawada Apartments",
        "address": "Madhurawada, IT Park Road, Visakhapatnam",
        "lat": 17.7920,
        "lon": 83.3750,
        "zone": "moderate",
        "population": 180,
        "type": "residential",
        "building_type": "highrise",
        "ttl_hours": 6.0,
        "medical_needs": ["elderly_care", "infant"],
        "mobility_issues": 15,
        "accessibility": "road_clear",
        "priority_score": 55,
        "notes": "Power out, medical cases need evacuation",
    },
    {
        "id": "T12",
        "name": "Industrial Valley Workers",
        "address": "Autonagar, VSEZ, Visakhapatnam",
        "lat": 17.6750,
        "lon": 83.2650,
        "zone": "moderate",
        "population": 75,
        "type": "industrial",
        "building_type": "factory",
        "ttl_hours": 4.0,
        "medical_needs": [],
        "mobility_issues": 2,
        "accessibility": "road_clear",
        "priority_score": 50,
        "notes": "Night shift workers, factory has supplies",
    },
    {
        "id": "T13",
        "name": "Pendurthi Old Age Home",
        "address": "Pendurthi, Visakhapatnam 531173",
        "lat": 17.7550,
        "lon": 83.2100,
        "zone": "moderate",
        "population": 35,
        "type": "shelter",
        "building_type": "institutional",
        "ttl_hours": 5.0,
        "medical_needs": ["elderly_care", "wheelchair"],
        "mobility_issues": 30,
        "accessibility": "road_clear",
        "priority_score": 78,
        "notes": "All residents elderly, 15 wheelchair-bound",
    },
    {
        "id": "T14",
        "name": "Comm Center River Bank",
        "address": "Gosala, Near Meghadri Gedda, Visakhapatnam",
        "lat": 17.7250,
        "lon": 83.2400,
        "zone": "moderate",
        "population": 95,
        "type": "shelter",
        "building_type": "community_hall",
        "ttl_hours": 4.0,
        "medical_needs": [],
        "mobility_issues": 8,
        "accessibility": "road_clear",
        "priority_score": 60,
        "notes": "Community shelter near river, water level rising",
    },
    {
        "id": "T15",
        "name": "Isukathota School Shelter",
        "address": "Isukathota, Visakhapatnam",
        "lat": 17.7380,
        "lon": 83.2680,
        "zone": "moderate",
        "population": 150,
        "type": "shelter",
        "building_type": "school",
        "ttl_hours": 6.0,
        "medical_needs": ["pregnant_woman"],
        "mobility_issues": 5,
        "accessibility": "road_clear",
        "priority_score": 58,
        "notes": "Official shelter, running low on water",
    },
    
    # =========================================================================
    # SAFE ZONE - Lower Priority (TTL > 6 hours)
    # =========================================================================
    {
        "id": "T16",
        "name": "Arilova Housing Complex",
        "address": "Arilova, Visakhapatnam 530040",
        "lat": 17.7650,
        "lon": 83.2280,
        "zone": "safe",
        "population": 220,
        "type": "residential",
        "building_type": "apartments",
        "ttl_hours": 12.0,
        "medical_needs": [],
        "mobility_issues": 10,
        "accessibility": "road_clear",
        "priority_score": 35,
        "notes": "Request for supplies, not urgent evacuation",
    },
    {
        "id": "T17",
        "name": "NAD Junction Stranded Bus",
        "address": "NAD Junction, Visakhapatnam",
        "lat": 17.7180,
        "lon": 83.2350,
        "zone": "safe",
        "population": 45,
        "type": "transport",
        "building_type": "vehicle",
        "ttl_hours": 8.0,
        "medical_needs": [],
        "mobility_issues": 2,
        "accessibility": "road_clear",
        "priority_score": 42,
        "notes": "Inter-city bus passengers, safe but stranded",
    },
    {
        "id": "T18",
        "name": "Marripalem VUDA Colony",
        "address": "Marripalem, Visakhapatnam",
        "lat": 17.7480,
        "lon": 83.2150,
        "zone": "safe",
        "population": 130,
        "type": "residential",
        "building_type": "houses",
        "ttl_hours": 10.0,
        "medical_needs": ["elderly_care"],
        "mobility_issues": 12,
        "accessibility": "road_clear",
        "priority_score": 38,
        "notes": "Uphill area, isolated but safe",
    },
    {
        "id": "T19",
        "name": "GVMC Community Hall",
        "address": "Siripuram, Visakhapatnam 530003",
        "lat": 17.7050,
        "lon": 83.3050,
        "zone": "safe",
        "population": 85,
        "type": "shelter",
        "building_type": "community_hall",
        "ttl_hours": 24.0,
        "medical_needs": [],
        "mobility_issues": 5,
        "accessibility": "road_clear",
        "priority_score": 25,
        "notes": "Official shelter, well-stocked",
    },
    {
        "id": "T20",
        "name": "Steel Plant Township",
        "address": "Ukkunagaram, Visakhapatnam 530032",
        "lat": 17.6350,
        "lon": 83.1650,
        "zone": "safe",
        "population": 300,
        "type": "residential",
        "building_type": "township",
        "ttl_hours": 24.0,
        "medical_needs": [],
        "mobility_issues": 15,
        "accessibility": "road_clear",
        "priority_score": 20,
        "notes": "Self-sufficient township, monitoring only",
    },
]


# =============================================================================
# Mission Solver
# =============================================================================

@dataclass
class VehicleAssignment:
    """Represents a vehicle's mission assignment."""
    vehicle_id: str
    vehicle_name: str
    targets: List[Dict[str, Any]]
    route: Optional[Dict[str, Any]]
    total_population: int
    total_distance_km: float
    estimated_time_min: float
    priority_score: float


def calculate_target_priority(target: Dict) -> float:
    """Calculate priority score for a target."""
    base_score = target.get("priority_score", 50)
    
    # TTL urgency bonus
    ttl = target.get("ttl_hours", 24)
    if ttl <= 1:
        base_score += 30
    elif ttl <= 2:
        base_score += 20
    elif ttl <= 4:
        base_score += 10
    
    # Medical needs bonus
    medical = target.get("medical_needs", [])
    if "dialysis" in medical or "critical_medical" in medical:
        base_score += 25
    if "oxygen_dependent" in medical:
        base_score += 20
    if "pregnant_woman" in medical:
        base_score += 15
    
    # Zone urgency
    zone = target.get("zone", "safe")
    if zone == "extreme":
        base_score += 25
    elif zone == "severe":
        base_score += 15
    elif zone == "moderate":
        base_score += 5
    
    return min(base_score, 150)


def match_vehicle_to_targets(
    vehicle: Dict,
    targets: List[Dict],
    depot: Dict
) -> List[Dict]:
    """
    Match a vehicle to appropriate targets based on capabilities.
    
    Prioritizes:
    - Medical vehicles for medical needs
    - High-capacity vehicles for large groups
    - Flood vehicles for flooded areas
    """
    matched = []
    remaining_capacity = vehicle["capacity"]
    
    # Sort targets by priority
    sorted_targets = sorted(
        targets,
        key=lambda t: calculate_target_priority(t),
        reverse=True
    )
    
    for target in sorted_targets:
        if remaining_capacity <= 0:
            break
        
        population = target.get("population", 0)
        if population > remaining_capacity:
            continue
        
        # Check vehicle capability match
        accessibility = target.get("accessibility", "road_clear")
        medical_needs = target.get("medical_needs", [])
        
        # Flooded areas need flood vehicles
        if accessibility == "flooded":
            if vehicle.get("water_fording_m", 0) < 1.0:
                continue
        
        # Critical medical needs prioritize ambulances
        critical_medical = any(m in medical_needs for m in 
            ["dialysis", "critical_medical", "oxygen_dependent"])
        if critical_medical and not vehicle.get("medical_equipped"):
            # Can still assign but lower priority
            pass
        
        matched.append(target)
        remaining_capacity -= population
    
    return matched


def solve_rescue_mission(
    place: str,
    targets: List[Dict],
    vehicles: List[Dict],
    depot: Dict,
    constraints: List[Dict] = None,
) -> Dict[str, Any]:
    """
    Solve the rescue mission routing problem.
    
    Parameters
    ----------
    place : str
        City/region for road network
    targets : List[Dict]
        Rescue targets with location and requirements
    vehicles : List[Dict]  
        Available vehicles with capacities
    depot : Dict
        Starting location
    constraints : List[Dict]
        Damage zone constraints
        
    Returns
    -------
    dict
        Solution with vehicle assignments, routes, and statistics
    """
    print(f"[MissionSolver] Solving mission with {len(targets)} targets, {len(vehicles)} vehicles")
    
    # Load road graph
    G = get_road_graph(place)
    
    # Apply constraints if any
    if constraints:
        G = apply_constraints(G, constraints)
    
    # Sort targets by priority
    prioritized_targets = sorted(
        targets,
        key=lambda t: calculate_target_priority(t),
        reverse=True
    )
    
    # Assign targets to vehicles
    assignments: List[VehicleAssignment] = []
    assigned_target_ids = set()
    
    # First pass: Assign medical vehicles to medical emergencies
    medical_vehicles = [v for v in vehicles if v.get("medical_equipped")]
    medical_targets = [t for t in prioritized_targets 
                       if any(m in t.get("medical_needs", []) for m in 
                              ["dialysis", "critical_medical", "oxygen_dependent", "pregnant_woman"])]
    
    for vehicle in medical_vehicles:
        if not medical_targets:
            break
        
        # Assign first available critical target
        for target in medical_targets:
            if target["id"] not in assigned_target_ids:
                matched = [target]
                for t in prioritized_targets:
                    if (t["id"] not in assigned_target_ids and 
                        t["id"] != target["id"] and
                        sum(m["population"] for m in matched) + t["population"] <= vehicle["capacity"]):
                        matched.append(t)
                        if len(matched) >= 3:  # Max 3 stops per trip
                            break
                
                # Calculate route
                stops = [(depot["lat"], depot["lon"])]
                for t in matched:
                    stops.append((t["lat"], t["lon"]))
                    assigned_target_ids.add(t["id"])
                stops.append((depot["lat"], depot["lon"]))  # Return to depot
                
                route = find_multi_stop_route(G, stops)
                
                assignment = VehicleAssignment(
                    vehicle_id=vehicle["id"],
                    vehicle_name=vehicle["name"],
                    targets=matched,
                    route=route if route.get("success") else None,
                    total_population=sum(t["population"] for t in matched),
                    total_distance_km=route.get("distance_km", 0) if route else 0,
                    estimated_time_min=route.get("travel_time_min", 0) if route else 0,
                    priority_score=sum(calculate_target_priority(t) for t in matched),
                )
                assignments.append(assignment)
                break
    
    # Second pass: Assign remaining vehicles to remaining targets
    other_vehicles = [v for v in vehicles if not v.get("medical_equipped")]
    remaining_targets = [t for t in prioritized_targets 
                        if t["id"] not in assigned_target_ids]
    
    for vehicle in other_vehicles:
        if not remaining_targets:
            break
        
        # Take highest priority targets that fit capacity
        matched = []
        current_capacity = 0
        
        for target in remaining_targets:
            if target["id"] in assigned_target_ids:
                continue
            
            # Check accessibility
            if target.get("accessibility") == "flooded":
                if vehicle.get("water_fording_m", 0) < 1.0:
                    continue
            
            if current_capacity + target["population"] <= vehicle["capacity"]:
                matched.append(target)
                current_capacity += target["population"]
                if len(matched) >= 4:  # Max 4 stops
                    break
        
        if matched:
            stops = [(depot["lat"], depot["lon"])]
            for t in matched:
                stops.append((t["lat"], t["lon"]))
                assigned_target_ids.add(t["id"])
            stops.append((depot["lat"], depot["lon"]))
            
            route = find_multi_stop_route(G, stops)
            
            assignment = VehicleAssignment(
                vehicle_id=vehicle["id"],
                vehicle_name=vehicle["name"],
                targets=matched,
                route=route if route.get("success") else None,
                total_population=sum(t["population"] for t in matched),
                total_distance_km=route.get("distance_km", 0) if route else 0,
                estimated_time_min=route.get("travel_time_min", 0) if route else 0,
                priority_score=sum(calculate_target_priority(t) for t in matched),
            )
            assignments.append(assignment)
        
        remaining_targets = [t for t in remaining_targets 
                            if t["id"] not in assigned_target_ids]
    
    # Generate solution summary
    total_rescued = sum(a.total_population for a in assignments)
    total_distance = sum(a.total_distance_km for a in assignments)
    
    unassigned = [t for t in targets if t["id"] not in assigned_target_ids]
    
    solution = {
        "success": True,
        "depot": depot,
        "assignments": [
            {
                "vehicle_id": a.vehicle_id,
                "vehicle_name": a.vehicle_name,
                "targets": [{"id": t["id"], "name": t["name"], "population": t["population"]} 
                           for t in a.targets],
                "route": a.route,
                "total_population": a.total_population,
                "distance_km": a.total_distance_km,
                "time_min": a.estimated_time_min,
                "priority_score": a.priority_score,
            }
            for a in assignments
        ],
        "summary": {
            "total_targets": len(targets),
            "targets_assigned": len(assigned_target_ids),
            "targets_unassigned": len(unassigned),
            "total_population_rescued": total_rescued,
            "total_population_at_risk": sum(t["population"] for t in targets),
            "total_distance_km": round(total_distance, 1),
            "vehicles_deployed": len(assignments),
            "vehicles_available": len(vehicles),
        },
        "unassigned_targets": [
            {"id": t["id"], "name": t["name"], "reason": "capacity_or_accessibility"}
            for t in unassigned
        ],
    }
    
    print(f"[MissionSolver] Solution: {len(assignments)} vehicles assigned, "
          f"{total_rescued} people to rescue, {round(total_distance, 1)} km total")
    
    return solution


# Export data for use in server.py
def get_mission_data() -> Dict[str, Any]:
    """Get all mission data for the workflow."""
    return {
        "depot": RESCUE_DEPOT,
        "vehicles": RESCUE_VEHICLES,
        "targets": RESCUE_TARGETS,
    }


if __name__ == "__main__":
    # Test the mission solver
    print("Testing mission solver...")
    
    solution = solve_rescue_mission(
        place="Visakhapatnam, India",
        targets=RESCUE_TARGETS,
        vehicles=RESCUE_VEHICLES,
        depot=RESCUE_DEPOT,
        constraints=[],
    )
    
    print(f"\n=== MISSION SOLUTION ===")
    print(f"Targets assigned: {solution['summary']['targets_assigned']}/{solution['summary']['total_targets']}")
    print(f"Population to rescue: {solution['summary']['total_population_rescued']}")
    print(f"Total distance: {solution['summary']['total_distance_km']} km")
    
    print(f"\n=== VEHICLE ASSIGNMENTS ===")
    for assignment in solution["assignments"]:
        print(f"\n{assignment['vehicle_name']}:")
        print(f"  Targets: {[t['name'] for t in assignment['targets']]}")
        print(f"  Population: {assignment['total_population']}")
        print(f"  Distance: {assignment['distance_km']} km")
