import json
from pathlib import Path
import tempfile
import unittest

from crowd_safety.persistence import MemoryPersistence, PersistenceConflict, import_run


def _make_run(root: Path, *, risk: float = 0.8) -> Path:
    run = root / "run-1"
    run.mkdir()
    (run / "metadata.json").write_text(json.dumps({
        "run_id": "run-1", "config_hash": "hash-1",
        "started_at": "2026-08-30T00:00:00+00:00", "ended_at": "2026-08-30T00:00:02+00:00",
        "input": {"source_id": "camera-1", "path": "source.mp4"},
        "artifacts": {"incidents": "incidents.jsonl", "transitions": "transitions.jsonl"},
        "stages": {"fusion": {"status": "available"}},
    }))
    (run / "metrics.json").write_text(json.dumps({"incident_count": 1}))
    (run / "incidents.jsonl").write_text(json.dumps({
        "incident_id": "incident-1", "source_id": "camera-1", "region_id": "zone",
        "state": "active", "severity": "high", "started_at_s": 1.0,
        "last_updated_at_s": 2.0, "closed_at_s": None, "peak_risk": risk,
        "reason_codes": ["violence_high"],
    }) + "\n")
    (run / "transitions.jsonl").write_text("".join(json.dumps(item) + "\n" for item in (
        {"incident_id": "incident-1", "timestamp_s": 1.0, "from_state": "candidate", "to_state": "active", "severity": "medium", "cause": "persistence", "reason_codes": ["violence_high"]},
        {"incident_id": "incident-1", "timestamp_s": 2.0, "from_state": "active", "to_state": "escalating", "severity": "high", "cause": "risk_threshold", "reason_codes": ["violence_high"]},
    )))
    evidence = run / "evidence" / "incident-1"
    evidence.mkdir(parents=True)
    (evidence / "manifest.json").write_text(json.dumps({
        "run_id": "run-1", "source_id": "camera-1", "incident_id": "incident-1",
        "incident_start_s": 1.0, "incident_end_s": 2.0, "pre_event_s": 1.0, "post_event_s": 1.0,
        "reason_codes": ["violence_high"], "timeline": [], "stage_health": {},
        "artifacts": [{"kind": "snapshot", "relative_path": "run-1/incident-1/snapshot.jpg", "start_s": 1.0, "end_s": 1.0, "status": "failed", "detail": "not captured"}],
    }))
    return run


class PersistenceTest(unittest.TestCase):
    def test_import_is_idempotent_and_preserves_ordered_m4_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _make_run(Path(directory))
            store = MemoryPersistence()

            self.assertTrue(import_run(run, store))
            self.assertFalse(import_run(run, store))
            incident = store.get_incident("incident-1")

        self.assertEqual(incident["incident"]["peak_risk"], 0.8)
        self.assertEqual([item["to_state"] for item in incident["transitions"]], ["active", "escalating"])
        self.assertEqual(len(incident["evidence"]), 1)

    def test_conflicting_reimport_is_rejected_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = _make_run(root)
            store = MemoryPersistence()
            import_run(run, store)
            (run / "incidents.jsonl").write_text((run / "incidents.jsonl").read_text().replace("0.8", "0.2"))

            with self.assertRaises(PersistenceConflict):
                import_run(run, store)

        self.assertEqual(store.get_incident("incident-1")["incident"]["peak_risk"], 0.8)

    def test_identical_source_local_incident_ids_are_scoped_by_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_run = _make_run(root)
            store = MemoryPersistence()
            import_run(first_run, store)
            second = root / "run-2"
            second.mkdir()
            for name in ("metadata.json", "incidents.jsonl", "transitions.jsonl"):
                (second / name).write_text((first_run / name).read_text().replace("run-1", "run-2"))
            (second / "incidents.jsonl").write_text((second / "incidents.jsonl").read_text().replace("0.8", "0.7"))

            self.assertTrue(import_run(second, store))

        records = store.list_incidents()
        self.assertEqual(len(records), 2)
        self.assertEqual({record["record_id"] for record in records}, {"run-1--incident-1", "run-2--incident-1"})

    def test_operator_action_is_append_only_and_does_not_mutate_m4_record(self):
        with tempfile.TemporaryDirectory() as directory:
            store = MemoryPersistence()
            import_run(_make_run(Path(directory)), store)
            action = store.record_action("incident-1", "dismiss", "operator-1", "2026-08-30T00:03:00+00:00", "reviewed")
            incident = store.get_incident("incident-1")

        self.assertEqual(action["action"], "dismiss")
        self.assertEqual(len(incident["actions"]), 1)
        self.assertEqual(incident["incident"]["state"], "active")
        with self.assertRaises(ValueError):
            store.record_action("incident-1", "invalid", "operator-1", "now")

    def test_changed_fusion_timeline_conflicts_with_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = _make_run(Path(directory))
            (run / "fusion.jsonl").write_text(json.dumps({
                "source_id": "camera-1", "region_id": "zone", "timestamp_s": 1.0, "fused_risk": 0.8,
            }) + "\n")
            store = MemoryPersistence()
            import_run(run, store)
            (run / "fusion.jsonl").write_text(json.dumps({
                "source_id": "camera-1", "region_id": "zone", "timestamp_s": 1.0, "fused_risk": 0.2,
            }) + "\n")

            with self.assertRaises(PersistenceConflict):
                import_run(run, store)


if __name__ == "__main__":
    unittest.main()
