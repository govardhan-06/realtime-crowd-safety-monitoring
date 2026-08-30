import tempfile
import unittest
from pathlib import Path

from crowd_safety.config import ConfigError, load_config


class ConfigTest(unittest.TestCase):
    def test_loads_defaults_and_resolves_paths_relative_to_config(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "pipeline.toml"
            config_path.write_text(
                """\
[input]
path = "input.mp4"

[output]
directory = "artifacts"

[processing]
target_fps = 5.0
resize = [640, 360]
"""
            )

            config = load_config(config_path)

        base = Path(directory).resolve()
        self.assertEqual(config.input_path, base / "input.mp4")
        self.assertEqual(config.output_directory, base / "artifacts")
        self.assertEqual(config.resize, (640, 360))
        self.assertTrue(config.annotation_enabled)
        self.assertTrue(config.logging_enabled)

    def test_rejects_invalid_processing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "pipeline.toml"
            config_path.write_text(
                """\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
target_fps = 0
resize = [-1, 360]
"""
            )

            with self.assertRaisesRegex(ConfigError, "target_fps"):
                load_config(config_path)

    def test_rejects_non_finite_target_fps(self):
        for value in ("nan", "inf"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "pipeline.toml"
                config_path.write_text(
                    f"""\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
target_fps = {value}
"""
                )

                with self.assertRaisesRegex(ConfigError, "target_fps"):
                    load_config(config_path)

    def test_loads_m2_settings_and_resolves_roi(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "pipeline.toml"
            config_path.write_text(
                """\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
resize = [640, 360]
[perception]
enabled = true
model = "yolo26n.pt"
confidence = 0.3
cadence_fps = 5.0
device = "auto"
person_class_id = 0
[tracking]
config = "bytetrack.yaml"
track_buffer = 30
[crowd]
window_s = 1.0
min_track_history = 2
[[crowd.rois]]
name = "full"
polygon = [[0, 0], [640, 0], [640, 360], [0, 360]]
"""
            )

            config = load_config(config_path)

        self.assertTrue(config.perception.enabled)
        self.assertEqual(config.perception.model, "yolo26n.pt")
        self.assertEqual(config.tracking.config, "bytetrack.yaml")
        self.assertEqual(config.crowd.rois[0].name, "full")
        self.assertEqual(config.crowd.rois[0].polygon[-1], (0.0, 360.0))

    def test_rejects_invalid_m2_geometry_and_windows(self):
        cases = [
            ("window_s = 0\n", "window_s"),
            ("min_track_history = 1\n", "min_track_history"),
            (
                "[[crowd.rois]]\nname = \"bad\"\npolygon = [[0, 0], [640, 0]]\n",
                "polygon",
            ),
            (
                "[[crowd.rois]]\nname = \"bad\"\npolygon = [[0, 0], [700, 0], [700, 360]]\n",
                "polygon",
            ),
        ]
        for fragment, field in cases:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "pipeline.toml"
                config_path.write_text(
                    f"""\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
resize = [640, 360]
[crowd]
{fragment}
"""
                )
                with self.assertRaisesRegex(ConfigError, field):
                    load_config(config_path)


if __name__ == "__main__":
    unittest.main()
