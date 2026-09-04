import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

from crowd_safety.artifacts import config_hash, resolved_config
from crowd_safety.config import load_config
from crowd_safety.replay import replay_run
from tests.support.synthetic_video import create_video


class CLIIntegrationTest(unittest.TestCase):
    def test_process_and_benchmark_commands_emit_inspectable_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            config = root / "pipeline.toml"
            create_video(source)
            config.write_text(
                f"""\
[input]
path = "{source}"
[output]
directory = "{root / 'artifacts'}"
[processing]
target_fps = 3.0
resize = [16, 12]
"""
            )
            processed = subprocess.run(
                [sys.executable, "-m", "crowd_safety", "process-video", "--config", str(config)],
                check=True,
                capture_output=True,
                text=True,
            )
            run_directory = Path(processed.stdout.strip())
            frames = [json.loads(line) for line in (run_directory / "frames.jsonl").read_text().splitlines()]
            metrics = json.loads((run_directory / "metrics.json").read_text())
            metadata = json.loads((run_directory / "metadata.json").read_text())
            saved_config = json.loads((run_directory / "config.json").read_text())
            repeated = subprocess.run(
                [sys.executable, "-m", "crowd_safety", "process-video", "--config", str(config)],
                check=True,
                capture_output=True,
                text=True,
            )
            repeated_directory = Path(repeated.stdout.strip())
            repeated_frames = [
                json.loads(line)
                for line in (repeated_directory / "frames.jsonl").read_text().splitlines()
            ]
            benchmark = subprocess.run(
                [sys.executable, "-m", "crowd_safety", "benchmark", "--config", str(config)],
                check=True,
                capture_output=True,
                text=True,
            )
            benchmark_path = Path(benchmark.stdout.strip())
            benchmark_values = json.loads(benchmark_path.read_text())
            output = cv2.VideoCapture(str(run_directory / "annotated.mp4"))
            output_frame_count = 0
            while output.read()[0]:
                output_frame_count += 1
            output.release()

        self.assertGreater(output_frame_count, 0)
        self.assertEqual(output_frame_count, metrics["output_frame_count"])
        self.assertEqual(len(frames), metrics["source_frame_count"])
        self.assertEqual(metrics["processed_frame_count"], sum(row["processed"] for row in frames))
        self.assertEqual(
            [row["frame_index"] for row in frames if row["processed"]],
            [row["frame_index"] for row in repeated_frames if row["processed"]],
        )
        self.assertEqual(metadata["config_hash"], saved_config["config_hash"])
        self.assertEqual(benchmark_values["status"], "success")
        self.assertIn("effective_fps", benchmark_values)
        self.assertIn("decode_seconds", benchmark_values)
        self.assertIn("write_seconds", benchmark_values)

    def test_benchmark_records_local_input_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "pipeline.toml"
            config.write_text(
                f"""\
[input]
path = "{root / 'missing.mp4'}"
[output]
directory = "{root / 'artifacts'}"
[processing]
target_fps = 3.0
resize = [16, 12]
"""
            )

            result = subprocess.run(
                [sys.executable, "-m", "crowd_safety", "benchmark", "--config", str(config)],
                check=True,
                capture_output=True,
                text=True,
            )
            values = json.loads(Path(result.stdout.strip()).read_text())

        self.assertEqual(values["status"], "failed")
        self.assertEqual(values["error_type"], "VideoIOError")
        self.assertIn("could not open video input", values["error"])

    def test_replay_command_runs_all_strategies_from_stored_signals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_directory = root / "run"
            run_directory.mkdir()
            config = root / "pipeline.toml"
            config.write_text(
                f"""\
[input]
path = "{root / 'unused.mp4'}"
[output]
directory = "{root / 'artifacts'}"
[processing]
resize = [16, 12]
[fusion]
smoothing_points = 1
"""
            )
            loaded_config = load_config(config)
            values = resolved_config(loaded_config, loaded_config.input_path)
            (run_directory / "config.json").write_text(json.dumps({
                "config_hash": config_hash(values), "config": values,
            }))
            (run_directory / "features.jsonl").write_text(json.dumps({
                "features": [{
                    "source_id": "camera-1", "roi_name": "zone", "timestamp_s": 1.0,
                    "status": "available", "occupancy": 4,
                }],
            }) + "\n")
            (run_directory / "violence.jsonl").write_text(json.dumps({
                "evidence": {
                    "source_id": "camera-1", "region_id": None, "clip_start_s": 0.0,
                    "clip_end_s": 1.0, "score": 0.9, "model": "fake", "revision": "rev",
                    "label_mapping": [["safe", 0], ["unsafe", 1]], "status": "available",
                },
            }) + "\n")
            result = subprocess.run(
                [sys.executable, "-m", "crowd_safety", "replay", "--run-directory", str(run_directory), "--config", str(config)],
                check=True, capture_output=True, text=True,
            )
            replay_directory = Path(result.stdout.strip())
            strategies = ("violence-only", "crowd-only", "naive-or", "rule-fusion", "temporal")
            self.assertEqual({path.name for path in replay_directory.iterdir()}, {*strategies, "metadata.json"})
            self.assertTrue((replay_directory / "temporal" / "fusion.jsonl").read_text())
            changed_config = root / "changed.toml"
            changed_config.write_text(config.read_text().replace("smoothing_points = 1", "smoothing_points = 2"))
            with self.assertRaisesRegex(ValueError, "config"):
                replay_run(run_directory, load_config(changed_config))


if __name__ == "__main__":
    unittest.main()
