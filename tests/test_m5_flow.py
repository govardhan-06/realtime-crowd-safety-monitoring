import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from crowd_safety.api import create_app
from crowd_safety.config import load_config
from crowd_safety.persistence import MemoryPersistence, import_run
from crowd_safety.runner import process_video
from crowd_safety.types import DetectionResult, PersonDetection, StageHealth, TrackObservation, TrackingResult, ViolenceEvidence
from tests.support.synthetic_video import create_video


class M5FlowTest(unittest.TestCase):
    def test_video_incident_is_captured_imported_reviewed_and_dispositioned(self):
        class Detector:
            def detect(self, packet):
                return DetectionResult((PersonDetection(packet.source_id, packet.frame_index, packet.timestamp_s, (4, 4, 14, 18), 0.9),), StageHealth("detector", "available", model="fake"))

        class Tracker:
            def update(self, packet, detections=()):
                observation = TrackObservation(packet.source_id, 1, packet.frame_index, packet.timestamp_s, (10.0 + packet.frame_index, 10.0), (4, 4, 14, 18), 0.9)
                return TrackingResult((observation,), StageHealth("tracker", "available", model="fake"))

        class Violence:
            def infer(self, window):
                return ViolenceEvidence(window.packets[0].source_id, None, window.start_s, window.end_s, 0.95, "fake", "test", (("safe", 0), ("unsafe", 1)), "available")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            create_video(source, frame_count=12, fps=3.0)
            config_path = root / "pipeline.toml"
            config_path.write_text(f"""\
[input]
path = "{source}"
[output]
directory = "{root / 'artifacts'}"
[processing]
target_fps = 3.0
resize = [32, 24]
[perception]
enabled = true
cadence_fps = 3.0
[crowd]
min_track_history = 2
[[crowd.rois]]
name = "zone"
polygon = [[0, 0], [32, 0], [32, 24], [0, 24]]
[violence]
enabled = true
clip_duration_s = 1.0
sample_count = 3
cadence_s = 0.5
[fusion]
persistence_s = 0.5
candidate_threshold = 0.2
active_threshold = 0.3
escalating_threshold = 0.6
critical_threshold = 0.8
severity_medium = 0.3
severity_high = 0.6
severity_critical = 0.8
[m5]
evidence_root = "{root / 'evidence'}"
pre_event_s = 1.0
post_event_s = 1.0
vlm_enabled = false
vlm_provider = "disabled"
""")
            config = load_config(config_path)
            result = process_video(config, detector=Detector(), tracker=Tracker(), violence_classifier=Violence())
            store = MemoryPersistence()
            self.assertTrue(import_run(result.run_directory, store, config.m5.evidence_root))
            client = TestClient(create_app(store))
            detail = client.get("/incidents").json()[0]
            incident_id = detail["incident"]["incident_id"]
            full = client.get(f"/incidents/{incident_id}").json()
            action = client.post(f"/incidents/{incident_id}/dismiss", json={"actor": "operator-1", "timestamp": "2026-08-30T00:00:00Z"})

            manifest_paths = result.evidence_paths
            metadata = json.loads(result.metadata_path.read_text())
            manifest_exists = bool(manifest_paths) and all(path.exists() for path in manifest_paths)

        self.assertEqual(action.status_code, 200)
        self.assertEqual(full["explanation"]["status"], "disabled")
        self.assertTrue(full["deterministic"]["transitions"])
        self.assertTrue(manifest_exists)
        self.assertEqual(metadata["evidence_count"], 1)


if __name__ == "__main__":
    unittest.main()
