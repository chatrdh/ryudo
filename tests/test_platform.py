"""
Platform module tests: replay store and event schema normalization.
"""

from ryudo.platform.events import EVENT_SCHEMA_VERSION, attach_event_envelope, normalize_stream_event
from ryudo.platform.replay import ReplayStore, deterministic_run_id


class TestDeterministicRunId:
    """Tests for deterministic run-id generation."""

    def test_payload_order_and_volatile_fields_do_not_change_id(self):
        payload_a = {
            "scenario": {"alpha": 1, "beta": [1, 2, 3]},
            "created_at": "2026-01-01T00:00:00Z",
            "event_id": "abc",
        }
        payload_b = {
            "event_id": "xyz",
            "scenario": {"beta": [1, 2, 3], "alpha": 1},
            "created_at": "2026-01-02T00:00:00Z",
        }

        assert deterministic_run_id(payload_a) == deterministic_run_id(payload_b)

    def test_namespace_changes_run_id(self):
        payload = {"mission": {"id": "m1"}, "constraints": [{"type": "zone_mask"}]}
        assert deterministic_run_id(payload, namespace="v1") != deterministic_run_id(
            payload, namespace="v2"
        )


class TestReplayStore:
    """Tests for in-memory replay storage."""

    def test_run_lifecycle(self):
        store = ReplayStore(max_runs=3, max_events_per_run=2)
        run_id = store.start_run({"scenario": "alpha"}, run_id="run_alpha")

        store.append_event(run_id, {"event_type": "a"})
        store.append_event(run_id, {"event_type": "b"})
        store.append_event(run_id, {"event_type": "c"})
        store.complete_run(run_id, result={"ok": True})

        record = store.get_run(run_id)
        assert record is not None
        assert record["status"] == "completed"
        assert len(record["events"]) == 2
        assert record["events"][0]["event_type"] == "b"
        assert record["events"][1]["event_type"] == "c"
        assert record["result"]["ok"] is True

    def test_restarting_same_run_resets_events(self):
        store = ReplayStore()
        run_id = store.start_run({"scenario": "alpha"}, run_id="run_alpha")
        store.append_event(run_id, {"event_type": "initial"})

        store.start_run({"scenario": "alpha"}, run_id="run_alpha")
        record = store.get_run(run_id)

        assert record is not None
        assert record["status"] == "running"
        assert record["events"] == []


class TestEventEnvelope:
    """Tests for event normalization and attachment."""

    def test_attach_event_envelope_preserves_legacy_fields(self):
        message = {
            "type": "agent_start",
            "agent": "FloodSentinel",
            "message": "starting analysis",
        }

        outbound = attach_event_envelope(message, run_id="run_test")
        envelope = outbound["event"]

        assert outbound["type"] == "agent_start"
        assert envelope["schema_version"] == EVENT_SCHEMA_VERSION
        assert envelope["run_id"] == "run_test"
        assert envelope["source"] == "FloodSentinel"
        assert envelope["source_type"] == "agent"
        assert envelope["event_type"] == "agent_start"
        assert envelope["payload"]["message"] == "starting analysis"

    def test_normalize_stream_event_passthrough_for_existing_envelope(self):
        envelope = {
            "schema_version": EVENT_SCHEMA_VERSION,
            "event_id": "evt_1",
            "run_id": "run_test",
            "timestamp": "2026-01-01T00:00:00Z",
            "source": "server",
            "source_type": "server",
            "event_type": "workflow_start",
            "payload": {"message": "hello"},
        }
        message = {"type": "workflow_start", "event": envelope}
        assert normalize_stream_event(message) == envelope

