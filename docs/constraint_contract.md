# Constraint Contract

This document defines the typed `GraphConstraint` contract used by Ryudo's Living Graph engine.

## Time Window

`TimeWindow` requirements:
- `start_time` and `end_time` are required.
- `end_time >= start_time`.

## GraphConstraint

Common requirements:
- `type`: one of `node_status`, `edge_weight`, `zone_mask`, `virtual_edge`.
- `target`: selector payload for what to mutate.
- `effect`: mutation payload.
- `validity`: `TimeWindow`.
- `source_agent_id`: non-empty string.

## Type Contracts

### `node_status`

Target must contain at least one selector:
- `node_id`
- `node_ids`
- `lat` + `lon` (nearest node selector)
- `lat` + `lon` + `radius_m` / `radius_km` (radius selector)

Effect:
- `action`: `disable` or `enable`

### `edge_weight`

Target must contain at least one selector:
- `edge` or `edges`
- `road_types`
- `tag_filter`
- `zone` or `zones`

Effect:
- `weight_factor` (numeric)

### `zone_mask`

Target:
- `polygon` (GeoJSON-like dict)

Effect must include at least one:
- `weight_factor` (numeric)
- `node_action` (`disable` or `enable`)

### `virtual_edge`

Target:
- `from_node`
- `to_node`

Effect:
- optional `weight` (numeric)
- optional `bidirectional` (boolean)

## Selector Notes

Selector behavior is implemented in `LivingGraph`:
- Node selectors resolve to concrete node IDs before mutation.
- Edge selectors resolve to concrete `(u, v, key)` edges before mutation.
- `zone_mask` writes zone labels to affected edges (`zone_labels`, `zone_type`) so follow-up `edge_weight` selectors can target zones.

## Conflict Precedence Matrix

When multiple active constraints collide, Ryudo applies them in deterministic type precedence:

1. `node_status`
2. `zone_mask`
3. `edge_weight`
4. `virtual_edge`

Notes:
- Higher-precedence types are always applied first, independent of registration order.
- Within the same type, higher `priority` runs first.
- Remaining ties preserve registration order.

## Auditing Constraint Application

Use:

```python
view, report = living_graph.get_view_with_report(query_time=now)
```

`report` entries include:
- `constraint_id`
- `constraint_type`
- `source_agent_id`
- `status` (`applied` or `skipped`)
- `reason`
- `details`

The `ConstraintApplicationRecord` object provides `to_dict()` for JSON serialization.
