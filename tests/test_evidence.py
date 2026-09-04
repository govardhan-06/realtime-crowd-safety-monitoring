import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2

from crowd_safety.config import load_config
from crowd_safety.evidence import capture_run_evidence
from tests.support.synthetic_video import create_video


class EvidenceTest(unittest.TestCase):
    def test_capture_writes_bounded_media_and_manifest_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run-1"
            run.mkdir()
            source = root / "source.mp4"
            create_video(source, frame_count=12, fps=2.0)
            annotated = run / "annotated.mp4"
            capture = cv2.VideoCapture(str(source))
            writer = cv2.VideoWriter(str(annotated), cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (32, 24))
            frame_times = []
            for index in range(12):
                ok, frame = capture.read()
                self.assertTrue(ok)
                writer.write(frame)
                frame_times.append({"frame_index": index, "timestamp_s": index / 2, "processed": True})
            capture.release()
            writer.release()
            (run / "frames.jsonl").write_text("".join(json.dumps(item) + "\n" for item in frame_times))
            (run / "incidents.jsonl").write_text(json.dumps({
                "incident_id": "incident-1", "source_id": "local-video", "region_id": "zone",
                "state": "active", "severity": "high", "started_at_s": 2.0,
                "last_updated_at_s": 3.0, "closed_at_s": None, "peak_risk": 0.8,
                "reason_codes": ["violence_high", "persistence_satisfied"],
            }) + "\n")
            (run / "fusion.jsonl").write_text("".join(json.dumps({
                "source_id": "local-video", "region_id": "zone", "timestamp_s": timestamp,
                "fused_risk": 0.8, "reason_codes": ["violence_high"],
            }) + "\n" for timestamp in (1.5, 2.0, 2.5, 3.0)))
            (run / "metadata.json").write_text(json.dumps({"stages": {"fusion": {"status": "available"}}}))
            config_path = root / "pipeline.toml"
            config_path.write_text(f"""\
[input]
path = "{source}"
[output]
directory = "{root / 'artifacts'}"
[processing]
resize = [32, 24]
[m5]
evidence_root = "{root / 'evidence'}"
pre_event_s = 1.0
post_event_s = 1.0
""")

            manifests = capture_run_evidence(run, load_config(config_path))
            manifest = manifests[0]

            self.assertEqual(manifest.incident_id, "incident-1")
            self.assertEqual(manifest.pre_event_s, 1.0)
            self.assertEqual(manifest.reason_codes, ("violence_high", "persistence_satisfied"))
            self.assertTrue((root / "evidence" / "run-1" / "incident-1" / "manifest.json").exists())
            self.assertEqual({item.status for item in manifest.artifacts}, {"available"})
            snapshot = root / "evidence" / manifest.artifacts[0].relative_path
            self.assertTrue(snapshot.exists())

    def test_capture_failure_is_explicit_and_does_not_require_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run-1"
            run.mkdir()
            (run / "incidents.jsonl").write_text(json.dumps({
                "incident_id": "incident-1", "source_id": "local-video", "region_id": "zone",
                "state": "active", "severity": "high", "started_at_s": 2.0,
                "last_updated_at_s": 3.0, "closed_at_s": None, "peak_risk": 0.8,
                "reason_codes": ["violence_high"],
            }) + "\n")
            config_path = root / "pipeline.toml"
            config_path.write_text(f"""\
[input]
path = "missing.mp4"
[output]
directory = "{root / 'artifacts'}"
[processing]
[m5]
evidence_root = "{root / 'evidence'}"
""")

            manifest = capture_run_evidence(run, load_config(config_path))[0]

            self.assertTrue(all(item.status != "available" for item in manifest.artifacts))
            self.assertTrue(all(item.detail for item in manifest.artifacts))

    def test_capture_prunes_expired_run_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "evidence" / "old-run"
            old.mkdir(parents=True)
            (old / "incident-1" / "manifest.json").parent.mkdir()
            (old / "incident-1" / "manifest.json").write_text(json.dumps({"run_id": "old-run", "incident_id": "incident-1"}))
            unrelated = root / "evidence" / "unrelated"
            unrelated.mkdir()
            os.utime(old, (0, 0))
            os.utime(unrelated, (0, 0))
            run = root / "run-1"
            run.mkdir()
            (run / "incidents.jsonl").write_text(json.dumps({
                "incident_id": "incident-1", "source_id": "local-video", "region_id": "zone",
                "state": "active", "severity": "high", "started_at_s": 0.0,
                "last_updated_at_s": 0.0, "closed_at_s": None, "peak_risk": 0.8,
                "reason_codes": ["violence_high"],
            }) + "\n")
            config_path = root / "pipeline.toml"
            config_path.write_text(f"""\
[input]
path = "missing.mp4"
[output]
directory = "{root / 'artifacts'}"
[processing]
[m5]
evidence_root = "{root / 'evidence'}"
retention_s = 1.0
""")

            capture_run_evidence(run, load_config(config_path))
            self.assertFalse(old.exists())
            self.assertTrue(unrelated.exists())


if __name__ == "__main__":
    unittest.main()
