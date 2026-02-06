# Replay and Event Schema

This document defines Iteration B additions for deterministic replay and normalized streaming events.

## Deterministic Run IDs

Run IDs are derived from canonical scenario payloads:

- Function: `ryudo.platform.replay.deterministic_run_id(payload, namespace=...)`
- Output format: `run_<16-hex-digest>`
- Hash source: stable JSON with sorted keys and volatile field removal (`event_id`, `timestamp`, `created_at`, `updated_at`)

Mission solves now include `run_id` in `SolutionResult.metadata`.

## Solver API Versioning

`BaseSolver` now exposes:

- `api_version` (default `1.0`)
- `implementation_version`
- `descriptor(source=...)`

`SolverRegistry` additions:

- `register(..., source="builtin")`
- `register_factory(factory, source="external")`
- `list_solver_descriptors()`
- `is_api_compatible(version)`

Compatibility policy is currently major-version match against `SolverRegistry.REGISTRY_API_VERSION`.

## Replay Endpoints

Server endpoints:

- `GET /api/replay` list replay summaries
- `GET /api/replay/latest` latest full replay record
- `GET /api/replay/{run_id}` full replay record by run ID
- `POST /api/replay/run-id` compute deterministic run ID for an external payload

## Streaming Event Envelope

All websocket broadcasts are emitted with a normalized envelope under `event`, while legacy top-level fields are preserved for compatibility.

Envelope shape:

```json
{
  "schema_version": "1.0",
  "event_id": "<uuid-hex>",
  "run_id": "run_...",
  "timestamp": "2026-02-06T00:00:00+00:00",
  "source": "FloodSentinel",
  "source_type": "agent",
  "event_type": "agent_start",
  "payload": {
    "message": "Environmental Agent analyzing cyclone data..."
  }
}
```

