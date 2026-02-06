"""
Deterministic run IDs and in-memory scenario replay storage.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import Lock
from typing import Any
from uuid import UUID


VOLATILE_KEYS = {
    "event_id",
    "timestamp",
    "created_at",
    "updated_at",
}


def deterministic_run_id(
    payload: Any,
    *,
    namespace: str = "ryudo.scenario.v1",
    prefix: str = "run",
    drop_keys: set[str] | None = None,
) -> str:
    """
    Generate a deterministic run id from a canonical payload hash.
    """
    normalized = _normalize_for_hash(payload, drop_keys=drop_keys or VOLATILE_KEYS)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(f"{namespace}|{blob}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _normalize_for_hash(value: Any, *, drop_keys: set[str]) -> Any:
    """Normalize nested values into stable, JSON-safe form."""
    if isinstance(value, dict):
        return {
            str(k): _normalize_for_hash(v, drop_keys=drop_keys)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
            if str(k) not in drop_keys
        }

    if isinstance(value, (list, tuple)):
        return [_normalize_for_hash(item, drop_keys=drop_keys) for item in value]

    if isinstance(value, (set, frozenset)):
        return sorted(_normalize_for_hash(item, drop_keys=drop_keys) for item in value)

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Enum):
        return value.value

    if hasattr(value, "model_dump"):
        return _normalize_for_hash(value.model_dump(mode="json"), drop_keys=drop_keys)

    return value


@dataclass
class ReplayRecord:
    """Stored replay data for one scenario run."""

    run_id: str
    scenario: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """Compact listing representation."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "event_count": len(self.events),
            "metadata": self.metadata,
        }

    def to_dict(self) -> dict[str, Any]:
        """Full serialization."""
        return {
            "run_id": self.run_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "scenario": self.scenario,
            "metadata": self.metadata,
            "events": self.events,
            "result": self.result,
        }


class ReplayStore:
    """Simple in-memory replay store keyed by run id."""

    def __init__(self, max_runs: int = 50, max_events_per_run: int = 5000):
        self.max_runs = max_runs
        self.max_events_per_run = max_events_per_run
        self._runs: dict[str, ReplayRecord] = {}
        self._order: list[str] = []
        self._lock = Lock()

    def start_run(
        self,
        scenario: dict[str, Any],
        *,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Create or reopen a run record and return its run id.
        """
        resolved_run_id = run_id or deterministic_run_id(scenario)

        with self._lock:
            record = self._runs.get(resolved_run_id)
            if record is None:
                record = ReplayRecord(
                    run_id=resolved_run_id,
                    scenario=scenario,
                    metadata=metadata or {},
                )
                self._runs[resolved_run_id] = record
                self._order.append(resolved_run_id)
                self._trim_runs()
            else:
                record.status = "running"
                record.completed_at = None
                record.events = []
                record.result = None
                if metadata:
                    record.metadata.update(metadata)
                if scenario:
                    record.scenario = scenario

        return resolved_run_id

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        """Append a normalized event to an existing run."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return

            record.events.append(event)
            if len(record.events) > self.max_events_per_run:
                record.events = record.events[-self.max_events_per_run:]

    def complete_run(
        self,
        run_id: str,
        *,
        result: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Mark a run completed and attach final artifacts."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return False

            record.status = "completed"
            record.completed_at = datetime.now(timezone.utc).isoformat()
            if result is not None:
                record.result = result
            if metadata:
                record.metadata.update(metadata)
            return True

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Get a replay record by run id."""
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                return None
            return record.to_dict()

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List replay summaries, newest first."""
        with self._lock:
            selected = list(reversed(self._order))[: max(limit, 0)]
            return [self._runs[run_id].summary() for run_id in selected if run_id in self._runs]

    def latest_run_id(self) -> str | None:
        """Return the most recently seen run id."""
        with self._lock:
            if not self._order:
                return None
            return self._order[-1]

    def _trim_runs(self) -> None:
        """Drop oldest runs when exceeding retention."""
        while len(self._order) > self.max_runs:
            oldest = self._order.pop(0)
            self._runs.pop(oldest, None)
