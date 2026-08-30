import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from crowd_safety.config import load_config
from crowd_safety.runner import process_video


def make_video(path: Path, frame_count: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (32, 24))
    if not writer.isOpened():
        raise RuntimeError("test video writer could not be opened")
    for value in range(frame_count):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()


class RunnerTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
