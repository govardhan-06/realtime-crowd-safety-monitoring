import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from crowd_safety.config import load_config
from crowd_safety.runner import _resize_detections, process_video
from crowd_safety.detection import UltralyticsPersonDetector
from crowd_safety.tracking import ByteTrackTracker
from crowd_safety.types import DetectionResult, PersonDetection, StageHealth, TrackObservation, TrackingResult


def make_video(path: Path, frame_count: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (32, 24))
    if not writer.isOpened():
        raise RuntimeError("test video writer could not be opened")
    for value in range(frame_count):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()


class RunnerTest(unittest.TestCase):
    def test_perception_boxes_are_transformed_to_resized_roi_coordinates(self):
        detection = PersonDetection("camera-1", 0, 0.0, (10, 20, 30, 40), 0.9)

        resized = _resize_detections((detection,), 100, 100, 200, 300)

        self.assertEqual(resized[0].box_xyxy, (20.0, 60.0, 60.0, 120.0))

    def test_process_video_emits_repeatable_artifacts_and_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            config_path = root / "pipeline.toml"
            make_video(source)
            config_path.write_text(
                f"""\
[input]
path = "{source}"
[output]
directory = "{root / 'artifacts'}"
[processing]
target_fps = 3.0
resize = [16, 12]
[annotation]
enabled = false
"""
            )
            config = load_config(config_path)

            first = process_video(config)
            second = process_video(config)

            first_frames = [
                row["frame_index"]
                for line in first.frames_path.read_text().splitlines()
                if (row := json.loads(line))["processed"]
            ]
            second_frames = [
                row["frame_index"]
                for line in second.frames_path.read_text().splitlines()
                if (row := json.loads(line))["processed"]
            ]
            metrics = json.loads(first.metrics_path.read_text())
            metadata = json.loads(first.metadata_path.read_text())
            capture = cv2.VideoCapture(str(first.video_path))
            ok, frame = capture.read()
            capture.release()

        self.assertEqual(first_frames, second_frames)
        self.assertEqual(first.config_hash, second.config_hash)
        self.assertEqual(metrics["processed_frame_count"], len(first_frames))
        self.assertEqual(metrics["skipped_frame_count"], 6 - len(first_frames))
        self.assertIn("decode_seconds", metrics)
        self.assertIn("write_seconds", metrics)
        self.assertEqual(metadata["config_hash"], first.config_hash)
        self.assertTrue(ok)
        self.assertEqual(frame.shape[:2], (12, 16))

    def test_m2_fake_adapters_emit_tracks_features_health_and_overlay(self):
        class FakeDetector:
            def detect(self, packet):
                return DetectionResult(
                    (PersonDetection(packet.source_id, packet.frame_index, packet.timestamp_s, (8, 8, 18, 20), 0.9),),
                    StageHealth("detector", "available", model="fake"),
                )

        class FakeTracker:
            def __init__(self):
                self.calls = 0

            def update(self, packet, detections=()):
                self.calls += 1
                x = 10 + packet.frame_index * 2
                observation = TrackObservation(
                    packet.source_id, 1, packet.frame_index, packet.timestamp_s,
                    (x, 14), (x - 5, 8, x + 5, 20), 0.9,
                )
                return TrackingResult((observation,), StageHealth("tracker", "available", model="fake"))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            config_path = root / "pipeline.toml"
            make_video(source)
            config_path.write_text(
                f"""\
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
window_s = 1.0
min_track_history = 2
[[crowd.rois]]
name = "frame"
polygon = [[0, 0], [32, 0], [32, 24], [0, 24]]
"""
            )
            fake_tracker = FakeTracker()
            result = process_video(load_config(config_path), detector=FakeDetector(), tracker=fake_tracker)
            tracks = [json.loads(line) for line in result.tracks_path.read_text().splitlines()]
            features = [json.loads(line) for line in result.features_path.read_text().splitlines()]
            metadata = json.loads(result.metadata_path.read_text())
            metrics = json.loads(result.metrics_path.read_text())
            capture = cv2.VideoCapture(str(result.video_path))
            ok, _ = capture.read()
            capture.release()

        self.assertEqual(len(tracks), metrics["processed_frame_count"])
        self.assertEqual(len(features), len(tracks))
        self.assertEqual(features[0]["features"][0]["status"], "insufficient")
        self.assertEqual(features[-1]["features"][0]["status"], "available")
        self.assertEqual(metadata["artifacts"]["tracks"], result.tracks_path.name)
        self.assertGreater(metrics["tracker_calls"], 0)
        self.assertEqual(fake_tracker.calls, metrics["tracker_calls"])
        self.assertTrue(ok)

    def test_m2_unavailable_perception_is_exported_as_unavailable_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            config_path = root / "pipeline.toml"
            make_video(source, frame_count=2)
            config_path.write_text(
                f"""\
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
[[crowd.rois]]
name = "frame"
polygon = [[0, 0], [32, 0], [32, 24], [0, 24]]
"""
            )
            result = process_video(
                load_config(config_path),
                detector=UltralyticsPersonDetector(model_error="missing weights"),
                tracker=ByteTrackTracker(),
            )
            feature_rows = [json.loads(line) for line in result.features_path.read_text().splitlines()]

        self.assertTrue(feature_rows)
        self.assertEqual(feature_rows[0]["features"][0]["status"], "unavailable")
        self.assertEqual(feature_rows[0]["health"]["status"], "degraded")


if __name__ == "__main__":
    unittest.main()
