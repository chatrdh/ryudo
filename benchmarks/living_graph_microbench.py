#!/usr/bin/env python3
"""
LivingGraph Micro-benchmark Harness.

Benchmarks `LivingGraph.get_view_with_report()` under synthetic workloads
using deterministic graph and constraint generation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import tracemalloc
from typing import Any, Optional

import networkx as nx
from shapely.geometry import box

# Ensure repository root is importable when executing this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ryudo.core import ConstraintType, GraphConstraint, LivingGraph, TimeWindow


ROAD_TYPES = ("primary", "secondary", "tertiary", "residential", "service")
ZONE_LABELS = ("zone_alpha", "zone_beta", "zone_gamma", "zone_delta")
DEFAULT_THRESHOLD_CONFIG_PATH = REPO_ROOT / "benchmarks" / "threshold_profiles.json"


@dataclass(frozen=True)
class ScenarioConfig:
    """Benchmark scenario configuration."""

    grid_size: int
    constraint_count: int
    warmup_runs: int
    measured_runs: int
    seed: int


def build_synthetic_graph(grid_size: int) -> nx.MultiDiGraph:
    """
    Build a deterministic grid graph with attributes used by selectors.

    Nodes use x/y and lat/lon aliases to match production assumptions.
    """
    graph = nx.MultiDiGraph()

    for row in range(grid_size):
        for col in range(grid_size):
            node_id = row * grid_size + col
            x = float(col)
            y = float(row)
            graph.add_node(
                node_id,
                x=x,
                y=y,
                lat=y,
                lon=x,
            )

    for row in range(grid_size):
        for col in range(grid_size):
            node = row * grid_size + col

            if col < grid_size - 1:
                right = row * grid_size + (col + 1)
                highway = ROAD_TYPES[(row + col) % len(ROAD_TYPES)]
                surface = "paved" if (row + col) % 2 == 0 else "unpaved"
                graph.add_edge(node, right, weight=1.0, length=100.0, highway=highway, surface=surface)
                graph.add_edge(right, node, weight=1.0, length=100.0, highway=highway, surface=surface)

            if row < grid_size - 1:
                down = (row + 1) * grid_size + col
                highway = ROAD_TYPES[(row + col + 1) % len(ROAD_TYPES)]
                surface = "paved" if (row + col + 1) % 2 == 0 else "unpaved"
                graph.add_edge(node, down, weight=1.0, length=100.0, highway=highway, surface=surface)
                graph.add_edge(down, node, weight=1.0, length=100.0, highway=highway, surface=surface)

    return graph


def generate_constraints(
    graph: nx.MultiDiGraph,
    constraint_count: int,
    seed: int,
    query_time: datetime,
) -> list[GraphConstraint]:
    """
    Generate deterministic mixed constraints for benchmark scenarios.

    Distribution:
    - edge_weight: 60%
    - zone_mask: 15%
    - node_status: 20%
    - virtual_edge: 5%
    """
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    edges = list(graph.edges(keys=True))

    if not nodes or not edges:
        return []

    constraints: list[GraphConstraint] = []
    validity = TimeWindow(
        start_time=query_time - timedelta(minutes=30),
        end_time=query_time + timedelta(hours=6),
    )

    max_coord = max(1.0, math.sqrt(len(nodes)))

    for _ in range(constraint_count):
        roll = rng.random()

        if roll < 0.60:
            constraints.append(
                _build_edge_weight_constraint(rng, edges, validity, query_time)
            )
        elif roll < 0.75:
            constraints.append(
                _build_zone_mask_constraint(rng, validity, max_coord, query_time)
            )
        elif roll < 0.95:
            constraints.append(
                _build_node_status_constraint(rng, nodes, graph, validity, query_time)
            )
        else:
            constraints.append(
                _build_virtual_edge_constraint(rng, nodes, validity, query_time)
            )

    return constraints


def _build_edge_weight_constraint(
    rng: random.Random,
    edges: list[tuple[int, int, int]],
    validity: TimeWindow,
    query_time: datetime,
) -> GraphConstraint:
    selector_roll = rng.random()
    if selector_roll < 0.4:
        target = {"edge": edges[rng.randrange(len(edges))]}
    elif selector_roll < 0.7:
        target = {"road_types": [ROAD_TYPES[rng.randrange(len(ROAD_TYPES))]]}
    elif selector_roll < 0.9:
        target = {"tag_filter": {"surface": "unpaved"}}
    else:
        target = {"zone": ZONE_LABELS[rng.randrange(len(ZONE_LABELS))]}

    factor = rng.choice((1.1, 1.25, 1.5, 2.0, 5.0))
    return GraphConstraint(
        type=ConstraintType.EDGE_WEIGHT,
        target=target,
        effect={"weight_factor": factor},
        validity=validity,
        source_agent_id="bench_edge",
        metadata={"generated_at": query_time.isoformat()},
        priority=rng.randint(0, 5),
    )


def _build_zone_mask_constraint(
    rng: random.Random,
    validity: TimeWindow,
    max_coord: float,
    query_time: datetime,
) -> GraphConstraint:
    center_x = rng.uniform(0.0, max_coord)
    center_y = rng.uniform(0.0, max_coord)
    radius = rng.uniform(0.4, max(1.2, max_coord / 15.0))

    polygon = box(
        center_x - radius,
        center_y - radius,
        center_x + radius,
        center_y + radius,
    )

    effect: dict[str, Any] = {"weight_factor": rng.choice((2.0, 3.0, 5.0, 8.0))}
    if rng.random() < 0.05:
        effect["node_action"] = "disable"

    return GraphConstraint(
        type=ConstraintType.ZONE_MASK,
        target={"polygon": polygon.__geo_interface__},
        effect=effect,
        validity=validity,
        source_agent_id="bench_zone",
        metadata={
            "zone_type": ZONE_LABELS[rng.randrange(len(ZONE_LABELS))],
            "generated_at": query_time.isoformat(),
        },
        priority=rng.randint(0, 5),
    )


def _build_node_status_constraint(
    rng: random.Random,
    nodes: list[int],
    graph: nx.MultiDiGraph,
    validity: TimeWindow,
    query_time: datetime,
) -> GraphConstraint:
    selector_roll = rng.random()
    if selector_roll < 0.35:
        target = {"node_id": nodes[rng.randrange(len(nodes))]}
    elif selector_roll < 0.7:
        first = nodes[rng.randrange(len(nodes))]
        second = nodes[rng.randrange(len(nodes))]
        if first == second:
            second = nodes[(nodes.index(first) + 1) % len(nodes)]
        target = {"node_ids": [first, second]}
    else:
        chosen = nodes[rng.randrange(len(nodes))]
        attrs = graph.nodes[chosen]
        target = {"lat": attrs["y"], "lon": attrs["x"]}

    return GraphConstraint(
        type=ConstraintType.NODE_STATUS,
        target=target,
        effect={"action": "disable"},
        validity=validity,
        source_agent_id="bench_node",
        metadata={"generated_at": query_time.isoformat()},
        priority=rng.randint(0, 5),
    )


def _build_virtual_edge_constraint(
    rng: random.Random,
    nodes: list[int],
    validity: TimeWindow,
    query_time: datetime,
) -> GraphConstraint:
    from_node = nodes[rng.randrange(len(nodes))]
    to_node = nodes[rng.randrange(len(nodes))]
    if to_node == from_node:
        to_node = nodes[(nodes.index(from_node) + 1) % len(nodes)]

    return GraphConstraint(
        type=ConstraintType.VIRTUAL_EDGE,
        target={"from_node": from_node, "to_node": to_node},
        effect={
            "weight": rng.choice((0.5, 1.0, 2.0, 3.0)),
            "bidirectional": rng.random() < 0.5,
        },
        validity=validity,
        source_agent_id="bench_virtual",
        metadata={"generated_at": query_time.isoformat()},
        priority=rng.randint(0, 5),
    )


def percentile(values: list[float], percentile_rank: float) -> float:
    """Compute percentile with linear interpolation."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * percentile_rank
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]

    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def run_scenario(config: ScenarioConfig) -> dict[str, Any]:
    """Run one benchmark scenario and return structured metrics."""
    graph = build_synthetic_graph(config.grid_size)
    now = datetime.now(timezone.utc)
    constraints = generate_constraints(
        graph=graph,
        constraint_count=config.constraint_count,
        seed=config.seed,
        query_time=now,
    )

    living_graph = LivingGraph()
    living_graph.load_from_graph(graph)
    for constraint in constraints:
        living_graph.add_constraint(constraint)

    for _ in range(config.warmup_runs):
        living_graph.get_view_with_report(query_time=now)

    latencies_ms: list[float] = []
    peak_memory_mib: list[float] = []
    last_report: list[dict[str, Any]] = []

    for _ in range(config.measured_runs):
        tracemalloc.start()
        started = time.perf_counter()
        _, report = living_graph.get_view_with_report(query_time=now)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latencies_ms.append(elapsed_ms)
        peak_memory_mib.append(peak_bytes / (1024.0 * 1024.0))
        last_report = [entry.to_dict() for entry in report]

    applied = sum(1 for entry in last_report if entry["status"] == "applied")
    skipped = sum(1 for entry in last_report if entry["status"] == "skipped")
    reason_counts = Counter(entry["reason"] for entry in last_report)

    return {
        "scenario": {
            "grid_size": config.grid_size,
            "constraints": config.constraint_count,
            "seed": config.seed,
            "warmup_runs": config.warmup_runs,
            "measured_runs": config.measured_runs,
        },
        "graph": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        },
        "constraint_mix": {
            "node_status": sum(1 for c in constraints if c.type == ConstraintType.NODE_STATUS),
            "zone_mask": sum(1 for c in constraints if c.type == ConstraintType.ZONE_MASK),
            "edge_weight": sum(1 for c in constraints if c.type == ConstraintType.EDGE_WEIGHT),
            "virtual_edge": sum(1 for c in constraints if c.type == ConstraintType.VIRTUAL_EDGE),
        },
        "latency_ms": {
            "min": min(latencies_ms),
            "max": max(latencies_ms),
            "mean": statistics.mean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95": percentile(latencies_ms, 0.95),
        },
        "peak_memory_mib": {
            "min": min(peak_memory_mib),
            "max": max(peak_memory_mib),
            "mean": statistics.mean(peak_memory_mib),
            "median": statistics.median(peak_memory_mib),
            "p95": percentile(peak_memory_mib, 0.95),
        },
        "report_counts": {
            "applied": applied,
            "skipped": skipped,
            "total": len(last_report),
        },
        "report_reason_counts": dict(
            sorted(reason_counts.items(), key=lambda item: item[1], reverse=True)
        ),
    }


def print_human_summary(result: dict[str, Any]) -> None:
    """Print compact human-readable summary for CLI runs."""
    scenario = result["scenario"]
    graph = result["graph"]
    latency = result["latency_ms"]
    memory = result["peak_memory_mib"]
    report_counts = result["report_counts"]

    print(
        f"[Scenario] grid={scenario['grid_size']}x{scenario['grid_size']}, "
        f"constraints={scenario['constraints']}, runs={scenario['measured_runs']}"
    )
    print(f"  Graph: nodes={graph['nodes']}, edges={graph['edges']}")
    print(
        "  Latency(ms): "
        f"mean={latency['mean']:.2f}, median={latency['median']:.2f}, p95={latency['p95']:.2f}, "
        f"min={latency['min']:.2f}, max={latency['max']:.2f}"
    )
    print(
        "  Peak Memory(MiB): "
        f"mean={memory['mean']:.2f}, median={memory['median']:.2f}, p95={memory['p95']:.2f}, "
        f"min={memory['min']:.2f}, max={memory['max']:.2f}"
    )
    print(
        f"  Report: applied={report_counts['applied']}, "
        f"skipped={report_counts['skipped']}, total={report_counts['total']}"
    )


def evaluate_threshold_warnings(
    result: dict[str, Any],
    warn_mean_latency_ms: Optional[float],
    warn_p95_latency_ms: Optional[float],
    warn_peak_memory_mib: Optional[float],
) -> list[str]:
    """Evaluate optional warning thresholds for one scenario."""
    warnings: list[str] = []
    scenario = result["scenario"]
    latency = result["latency_ms"]
    memory = result["peak_memory_mib"]

    if warn_mean_latency_ms is not None and latency["mean"] > warn_mean_latency_ms:
        warnings.append(
            f"constraints={scenario['constraints']}: mean latency {latency['mean']:.2f} ms "
            f"exceeds {warn_mean_latency_ms:.2f} ms"
        )

    if warn_p95_latency_ms is not None and latency["p95"] > warn_p95_latency_ms:
        warnings.append(
            f"constraints={scenario['constraints']}: p95 latency {latency['p95']:.2f} ms "
            f"exceeds {warn_p95_latency_ms:.2f} ms"
        )

    if warn_peak_memory_mib is not None and memory["p95"] > warn_peak_memory_mib:
        warnings.append(
            f"constraints={scenario['constraints']}: p95 peak memory {memory['p95']:.2f} MiB "
            f"exceeds {warn_peak_memory_mib:.2f} MiB"
        )

    return warnings


def load_threshold_profile(
    config_path: Path,
    profile_name: str,
) -> dict[str, Any]:
    """Load threshold profile from JSON config."""
    if not config_path.exists():
        raise FileNotFoundError(f"Threshold config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    profiles = config.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("Threshold config must contain a 'profiles' object")

    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        available = ", ".join(sorted(profiles.keys()))
        raise ValueError(
            f"Threshold profile '{profile_name}' not found. Available: {available}"
        )

    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LivingGraph view materialization.")
    parser.add_argument(
        "--constraint-counts",
        nargs="+",
        type=int,
        default=[1000, 10000],
        help="Constraint counts to benchmark.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=15,
        help="Synthetic graph grid size (N produces N*N nodes).",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Warmup iterations per scenario.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Measured iterations per scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for deterministic fixture generation.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to write JSON results.",
    )
    parser.add_argument(
        "--warn-mean-latency-ms",
        type=float,
        default=None,
        help="Optional warning threshold for mean latency per scenario.",
    )
    parser.add_argument(
        "--warn-p95-latency-ms",
        type=float,
        default=None,
        help="Optional warning threshold for p95 latency per scenario.",
    )
    parser.add_argument(
        "--warn-peak-memory-mib",
        type=float,
        default=None,
        help="Optional warning threshold for p95 peak memory per scenario.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any warning threshold is exceeded.",
    )
    parser.add_argument(
        "--threshold-profile",
        type=str,
        default="",
        help="Optional threshold profile name from a JSON config file.",
    )
    parser.add_argument(
        "--threshold-config",
        type=str,
        default=str(DEFAULT_THRESHOLD_CONFIG_PATH),
        help="Path to threshold profile JSON config.",
    )
    args = parser.parse_args()

    warn_mean_latency_ms = args.warn_mean_latency_ms
    warn_p95_latency_ms = args.warn_p95_latency_ms
    warn_peak_memory_mib = args.warn_peak_memory_mib
    strict_mode = args.strict

    active_profile_name: Optional[str] = None
    if args.threshold_profile:
        profile = load_threshold_profile(
            config_path=Path(args.threshold_config),
            profile_name=args.threshold_profile,
        )
        active_profile_name = args.threshold_profile
        if warn_mean_latency_ms is None:
            warn_mean_latency_ms = profile.get("warn_mean_latency_ms")
        if warn_p95_latency_ms is None:
            warn_p95_latency_ms = profile.get("warn_p95_latency_ms")
        if warn_peak_memory_mib is None:
            warn_peak_memory_mib = profile.get("warn_peak_memory_mib")
        strict_mode = bool(profile.get("strict", False)) or strict_mode

    started_at = datetime.now(timezone.utc).isoformat()
    results: list[dict[str, Any]] = []
    warnings: list[str] = []

    for constraint_count in args.constraint_counts:
        scenario = ScenarioConfig(
            grid_size=args.grid_size,
            constraint_count=constraint_count,
            warmup_runs=args.warmup_runs,
            measured_runs=args.runs,
            seed=args.seed,
        )
        result = run_scenario(scenario)
        results.append(result)
        print_human_summary(result)
        warnings.extend(
            evaluate_threshold_warnings(
                result=result,
                warn_mean_latency_ms=warn_mean_latency_ms,
                warn_p95_latency_ms=warn_p95_latency_ms,
                warn_peak_memory_mib=warn_peak_memory_mib,
            )
        )

    if warnings:
        print("[Warnings]")
        for warning in warnings:
            print(f"  - {warning}")

    payload = {
        "benchmark": "living_graph_microbench",
        "started_at": started_at,
        "python": {
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "thresholds": {
            "profile": active_profile_name,
            "warn_mean_latency_ms": warn_mean_latency_ms,
            "warn_p95_latency_ms": warn_p95_latency_ms,
            "warn_peak_memory_mib": warn_peak_memory_mib,
            "strict": strict_mode,
        },
        "results": results,
        "warnings": warnings,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[Output] wrote JSON results to {args.output}")
    else:
        print(json.dumps(payload, indent=2))

    if strict_mode and warnings:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
