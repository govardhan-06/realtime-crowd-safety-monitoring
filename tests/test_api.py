from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from crowd_safety.api import create_app
from crowd_safety.explanations import FakeExplainer
from crowd_safety.persistence import MemoryPersistence


def seeded_store() -> MemoryPersistence:
    store = MemoryPersistence()
    store.runs["run-1"] = {
        "run_id": "run-1", "source_id": "camera-1", "config_hash": "hash",
        "input": {"source_id": "camera-1", "path": "/private/source.mp4"},
        "artifacts": {}, "stages": {"fusion": {"status": "available"}},
    }
    store.incidents["incident-1"] = {
        "incident_id": "incident-1", "source_id": "camera-1", "region_id": "zone",
        "state": "active", "severity": "high", "started_at_s": 1.0,
        "last_updated_at_s": 2.0, "closed_at_s": None, "peak_risk": 0.8,
        "reason_codes": ["violence_high"],
    }
    store.transitions["incident-1"] = [{"to_state": "active", "timestamp_s": 1.0}]
    store.timelines["incident-1"] = [{"timestamp_s": 1.0, "fused_risk": 0.8}]
    store.evidence["incident-1"] = [{"incident_id": "incident-1", "artifacts": []}]
    return store


class APITest(unittest.TestCase):
    def test_read_endpoints_return_deterministic_records_without_source_paths(self):
        client = TestClient(create_app(seeded_store()))

        self.assertEqual(client.get("/health").status_code, 200)
        self.assertEqual(client.get("/sources").json()[0]["source_id"], "camera-1")
        self.assertNotIn("path", client.get("/runs/run-1").json()["input"])
        list_response = client.get("/incidents?state=active")
        detail_response = client.get("/incidents/incident-1")

        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(detail_response.json()["incident"]["reason_codes"], ["violence_high"])
        self.assertEqual(detail_response.json()["explanation"]["status"], "disabled")
        self.assertEqual(client.get("/incidents/incident-1/timeline").json()[0]["fused_risk"], 0.8)
        self.assertEqual(client.get("/incidents/unknown").status_code, 404)
        self.assertEqual(client.get("/incidents?state=not-a-state").status_code, 422)

    def test_actions_are_validated_appended_and_cannot_mutate_incident(self):
        store = seeded_store()
        client = TestClient(create_app(store))
        payload = {"actor": "operator-1", "timestamp": datetime.now(timezone.utc).isoformat(), "note": "reviewed"}

        for action in ("acknowledge", "dismiss", "escalate"):
            response = client.post(f"/incidents/incident-1/{action}", json=payload)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["action"], action)
        forbidden = {**payload, "state": "closed", "severity": "critical"}
        self.assertEqual(client.post("/incidents/incident-1/dismiss", json=forbidden).status_code, 422)
        self.assertEqual(store.incidents["incident-1"]["state"], "active")
        self.assertEqual(len(store.actions["incident-1"]), 3)

    def test_explanation_is_labelled_and_cannot_change_decision(self):
        store = seeded_store()
        client = TestClient(create_app(store, explainer=FakeExplainer("possible altercation")))
        before = dict(store.incidents["incident-1"])

        response = client.post("/incidents/incident-1/explain")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "generated")
        self.assertEqual(client.get("/incidents/incident-1").json()["explanation"]["text"], "possible altercation")
        self.assertEqual(store.incidents["incident-1"], before)

    def test_dashboard_origin_can_submit_actions_but_origins_are_not_open(self):
        client = TestClient(create_app(seeded_store()))
        allowed = client.options(
            "/incidents/incident-1/dismiss",
            headers={"Origin": "http://127.0.0.1:3000", "Access-Control-Request-Method": "POST"},
        )
        denied = client.options(
            "/incidents/incident-1/dismiss",
            headers={"Origin": "https://untrusted.example", "Access-Control-Request-Method": "POST"},
        )

        self.assertEqual(allowed.headers.get("access-control-allow-origin"), "http://127.0.0.1:3000")
        self.assertNotEqual(denied.headers.get("access-control-allow-origin"), "https://untrusted.example")

    def test_evidence_file_serving_is_limited_to_referenced_media(self):
        store = seeded_store()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "run-1" / "incident-1" / "snapshot.jpg"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"jpeg")
            store.evidence["incident-1"] = [{"incident_id": "incident-1", "artifacts": [{
                "kind": "snapshot", "relative_path": "run-1/incident-1/snapshot.jpg", "status": "available",
            }]}]
            client = TestClient(create_app(store, root))

            response = client.get("/incidents/incident-1/evidence/snapshot")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"jpeg")
        self.assertEqual(client.get("/incidents/incident-1/evidence/post_event_clip").status_code, 404)


if __name__ == "__main__":
    unittest.main()
