# LivingGraph Performance Baseline

Date: 2026-02-06

This baseline measures `LivingGraph.get_view_with_report()` using deterministic synthetic workloads.

## Benchmark Harness

Script:
- `benchmarks/living_graph_microbench.py`
- Threshold profiles: `benchmarks/threshold_profiles.json`

Command used:

```bash
.venv/bin/python benchmarks/living_graph_microbench.py \
  --constraint-counts 1000 10000 \
  --grid-size 15 \
  --warmup-runs 1 \
  --runs 3 \
  --seed 7 \
  --output output/living_graph_baseline_2026-02-06.json
```

Output artifact:
- `output/living_graph_baseline_2026-02-06.json`

Optional guardrail mode (warning-only):

```bash
.venv/bin/python benchmarks/living_graph_microbench.py \
  --threshold-profile local-warning
```

Strict guardrail mode (non-zero exit on warning):

```bash
.venv/bin/python benchmarks/living_graph_microbench.py \
  --threshold-profile ci-strict \
  --strict
```

Override profile thresholds ad-hoc:

```bash
.venv/bin/python benchmarks/living_graph_microbench.py \
  --threshold-profile ci-warning \
  --warn-mean-latency-ms 2800
```

## CI Workflow

Workflow file:
- `.github/workflows/living-graph-benchmark.yml`

Behavior:
- Pull requests run benchmark in `warning` mode (`ci-warning` profile) and do not fail the job on threshold warnings.
- Manual workflow dispatch supports `warning` or `strict` mode.
- Strict mode uses `ci-strict` profile and fails when thresholds are exceeded.
- JSON results are uploaded as workflow artifacts.

## Baseline Results

### Scenario: 1000 Constraints

- Graph: 225 nodes, 840 edges
- Constraint mix:
  - `node_status`: 204
  - `zone_mask`: 134
  - `edge_weight`: 623
  - `virtual_edge`: 39
- Latency:
  - mean: `644.49 ms`
  - median: `657.98 ms`
  - p95: `672.29 ms`
  - min/max: `601.61 / 673.88 ms`
- Peak memory:
  - mean: `0.55 MiB`
  - p95: `0.56 MiB`
- Report counts:
  - applied: `556`
  - skipped: `444`

### Scenario: 10000 Constraints

- Graph: 225 nodes, 840 edges
- Constraint mix:
  - `node_status`: 1982
  - `zone_mask`: 1426
  - `edge_weight`: 6086
  - `virtual_edge`: 506
- Latency:
  - mean: `2566.10 ms`
  - median: `2797.81 ms`
  - p95: `2880.06 ms`
  - min/max: `2011.30 / 2889.20 ms`
- Peak memory:
  - mean: `2.24 MiB`
  - p95: `2.29 MiB`
- Report counts:
  - applied: `203`
  - skipped: `9797`

## Notes

- High skip rates in the 10k scenario are expected with the current precedence model:
  - early `node_status` and `zone_mask` constraints can remove nodes/edges,
  - many later constraints then resolve to targets no longer present in the view.
- Benchmark JSON now includes `report_reason_counts` to expose skip/apply distribution by reason.
- First-pass optimization from this distribution:
  - selector cache reuse for repeated `EDGE_WEIGHT` targets within one view build,
  - early fast-skip paths when the working graph has no nodes/edges.
- This baseline is intended as a repeatable regression anchor, not an upper-bound throughput claim.
