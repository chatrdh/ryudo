# Ryudo Implementation Plan

Date: 2026-02-06
Owner: Core Platform

## 1. Objective

Build Ryudo into a domain-agnostic Spatial Optimization Platform with:
- real-time graph mutation from state agents,
- mission solving across multiple asset types,
- simulation and replay support,
- production-ready API, observability, and scale characteristics.

## 2. Target Architecture

1. Perception Layer
- Agent SDK with strict `WorldState -> AgentResult` contract.
- Pluggable agent runtime (sync/async execution, validation, conflict resolution).
- Reasoning and trace output for operator visibility.

2. Living Graph Layer
- Immutable base topology.
- Overlay mutation engine with typed constraints and TTL validity windows.
- Selector-based mutation targeting (IDs, spatial selectors, attribute selectors).

3. Optimization Layer
- Mission profile schema (assets, targets, objectives, constraints).
- Solver registry and pluggable optimization engines.
- Capability-aware view builder for multi-modal routing.

4. Platform Surface
- FastAPI endpoints and WebSocket event stream.
- Scenario replay and what-if simulation.
- Domain packs ("skins") for vertical use cases.

## 3. Phased Delivery

## Phase 1: Core Abstraction (Now)
Goal: stabilize the domain-agnostic kernel.

Deliverables:
- Generic constraint schema and selector semantics documented and tested.
- LivingGraph overlay engine with deterministic constraint application.
- Agent SDK contract hardening (validation, conflict rules, error boundaries).
- Mission profile and solver interfaces locked for plugin extension.

Exit criteria:
- Core tests fully green.
- Constraint selectors support ID + spatial + attribute targeting.
- No domain-specific assumptions in `ryudo/core/*`.
- Baseline performance profile captured for 1k/10k constraints.

## Phase 2: Extensibility
Goal: make Ryudo a programmable platform.

Deliverables:
- Solver API v1 for external solvers (heuristics, OR-Tools, RL).
- Agent package format and loader (local + remote registry support).
- Policy layer for permissioned signals and mutation scopes.
- Versioned scenario schema for reusable simulations.

Exit criteria:
- Third-party agent and solver can be installed without core edits.
- Compatibility tests for plugin API pass.
- Migration guide for SDK users published.

## Phase 3: Commercial Scale
Goal: production reliability and enterprise readiness.

Deliverables:
- High-volume mutation pipeline (batch apply, incremental views, caching).
- Predictive digital twin mode (Monte Carlo runs + robustness scoring).
- Edge/offline deployment profile.
- SLO-backed observability (latency, error budgets, traceability).

Exit criteria:
- Sustained real-time updates at target throughput and latency.
- Deterministic replay of historical runs.
- Deployment profiles validated for cloud and constrained hardware.

## 4. Workstreams

1. Graph Engine
- Constraint compiler, selector resolution, overlay indexing, TTL pruning.

2. Agent Runtime
- Orchestration, conflict policies, confidence propagation, reasoning traces.

3. Mission & Solvers
- Objective model, solver contracts, benchmark harness, capability constraints.

4. API & Streaming
- Event model, mission/session lifecycle, live map mutation feed.

5. Simulation
- Scenario store, time-travel controls, replay determinism.

6. Quality & Ops
- Tests, perf benchmarks, docs, CI quality gates.

## 5. Current Sprint (In Progress)

1. Constraint semantics hardening
- [x] Add selector-based mutation support in LivingGraph (`node_id`, `node_ids`, `lat/lon`, `road_types`, `tag_filter`, `zone` labels).
- [x] Add test coverage for selector targeting and chained zone -> edge selectors.
- [x] Add type-aware schema validation for each constraint class (`target`/`effect` contracts).
- [x] Publish selector contract doc in API/SDK reference.

2. Overlay engine behavior
- [x] Preserve zone labels on affected edges for downstream selector reuse.
- [x] Add `get_view_with_report()` to capture applied/skipped constraints and reasons.
- [x] Add explicit conflict precedence table and tests for mixed selector collisions.

3. Performance baseline
- [x] Add micro-benchmark for view materialization under 1k/10k constraints.
- [x] Capture memory/latency profile and identify first optimizations.

## 6. Next 2 Iterations

Iteration A:
1. [x] Add benchmark harness in CI (non-blocking initially).
2. [x] Tune warning thresholds and wire strict-mode gating per environment.
3. [x] Evaluate optimization candidates from baseline skip/apply distribution.

Iteration B:
1. [x] Solver API versioning and external registration hooks.
2. [x] Scenario replay endpoint and deterministic run IDs.
3. [x] Streaming event schema normalization across agents/solver/server.

## 7. Risks and Controls

1. Ambiguous constraint semantics
- Control: typed schema validation + selector contract tests.

2. Overlay performance degradation at scale
- Control: indexed selectors, batch mutation, benchmark gates.

3. Plugin instability
- Control: versioned interfaces and compatibility test matrix.

4. Non-deterministic mission runs
- Control: deterministic ordering rules and replay snapshots.
