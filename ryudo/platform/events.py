"""
Normalized streaming event schema utilities.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


EVENT_SCHEMA_VERSION = "1.0"


def normalize_stream_event(
    message: dict[str, Any],
    *,
    run_id: str | None = None,
    default_source: str = "server",
) -> dict[str, Any]:
    """
    Convert a legacy event payload into a normalized event envelope.
    """
    if not isinstance(message, dict):
        raise TypeError("message must be a dictionary")

    legacy_event = message.get("event")
    if isinstance(legacy_event, dict) and legacy_event.get("schema_version"):
        return legacy_event

    source = message.get("source") or message.get("agent") or default_source
    source_type = message.get("source_type") or _infer_source_type(message, str(source))
    event_type = str(message.get("type", "unknown"))
    event_payload = {
        key: value
        for key, value in message.items()
        if key not in {"event", "schema_version", "type", "source", "source_type", "agent"}
    }

    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "event_id": uuid4().hex,
        "run_id": run_id or message.get("run_id") or "unbound",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "source_type": source_type,
        "event_type": event_type,
        "payload": event_payload,
    }


def attach_event_envelope(
    message: dict[str, Any],
    *,
    run_id: str | None = None,
    default_source: str = "server",
) -> dict[str, Any]:
    """
    Attach a normalized event envelope to an outbound message.
    """
    envelope = normalize_stream_event(
        message,
        run_id=run_id,
        default_source=default_source,
    )
    outbound = dict(message)
    outbound["schema_version"] = EVENT_SCHEMA_VERSION
    outbound["event"] = envelope
    return outbound


def _infer_source_type(message: dict[str, Any], source: str) -> str:
    """Infer source type from message fields and known agent names."""
    if message.get("agent"):
        return "agent"

    if "solver" in source.lower() or source == "MissionSolver":
        return "solver"

    if source.lower() in {"server", "system", "coordinator"}:
        return "server"

    return "component"

