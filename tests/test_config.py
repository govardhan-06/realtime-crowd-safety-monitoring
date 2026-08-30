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


if __name__ == "__main__":
    unittest.main()
