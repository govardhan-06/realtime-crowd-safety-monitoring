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
        self.assertEqual(config.m5.evidence_root, base / "artifacts" / "evidence")
        self.assertFalse(config.m5.vlm_enabled)
        self.assertEqual(config.m5.database_url_env, "DATABASE_URL")

    def test_loads_m5_storage_and_explanation_settings_without_a_secret(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "pipeline.toml"
            config_path.write_text(
                """\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
[m5]
evidence_root = "retained-evidence"
pre_event_s = 4.0
post_event_s = 6.0
retention_s = 3600.0
database_url_env = "LOCAL_DATABASE_URL"
vlm_enabled = false
vlm_provider = "disabled"
vlm_model = ""
vlm_timeout_s = 8.0
"""
            )

            config = load_config(config_path)

        self.assertEqual(config.m5.evidence_root, Path(directory).resolve() / "retained-evidence")
        self.assertEqual(config.m5.pre_event_s, 4.0)
        self.assertEqual(config.m5.database_url_env, "LOCAL_DATABASE_URL")
        self.assertEqual(config.m5.vlm_provider, "disabled")

    def test_rejects_invalid_m5_settings(self):
        for fragment, field in (
            ("pre_event_s = 0\n", "m5.pre_event_s"),
            ("retention_s = -1\n", "m5.retention_s"),
            ("database_url_env = \"not-valid\"\n", "database_url_env"),
            ("vlm_provider = \"unknown\"\n", "vlm_provider"),
            ("vlm_timeout_s = 0\n", "vlm_timeout_s"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "pipeline.toml"
                config_path.write_text(
                    f"""\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
[m5]
{fragment}
"""
                )
                with self.assertRaisesRegex(ConfigError, field):
                    load_config(config_path)

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

    def test_loads_m3_violence_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "pipeline.toml"
            config_path.write_text(
                """\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
[violence]
enabled = true
model = "example/model"
revision = "abc123"
clip_duration_s = 2.0
sample_count = 8
cadence_s = 0.5
threshold = 0.7
device = "cpu"
labels = ["safe", "unsafe"]
license = "mit"
known_limitations = "unknown training data"
checkpoint_sha256 = "ff542a5aa37d4c447584523545996d7c186d87c71b70decae0a773a02f212e5c"
"""
            )

            config = load_config(config_path)

        self.assertTrue(config.violence.enabled)
        self.assertEqual(config.violence.model, "example/model")
        self.assertEqual(config.violence.revision, "abc123")
        self.assertEqual(config.violence.sample_count, 8)
        self.assertEqual(config.violence.threshold, 0.7)
        self.assertEqual(config.violence.labels, ("safe", "unsafe"))
        self.assertEqual(config.violence.license, "mit")
        self.assertEqual(config.violence.checkpoint_sha256[:8], "ff542a5a")

    def test_rejects_invalid_m3_violence_settings(self):
        for field, value in (
            ("clip_duration_s", "0"),
            ("sample_count", "1"),
            ("cadence_s", "0"),
            ("threshold", "1.1"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "pipeline.toml"
                config_path.write_text(
                    f"""\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
[violence]
{field} = {value}
"""
                )

                with self.assertRaisesRegex(ConfigError, field):
                    load_config(config_path)

    def test_loads_fusion_settings_and_rejects_invalid_policy_or_order(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "pipeline.toml"
            config_path.write_text(
                """\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
[fusion]
strategy = "temporal"
allow_crowd_only = false
smoothing_points = 4
candidate_threshold = 0.2
active_threshold = 0.4
escalating_threshold = 0.7
critical_threshold = 0.9
[fusion.normalization]
density_delta = [-2.0, 2.0]
"""
            )
            config = load_config(config_path)
            self.assertEqual(config.fusion.strategy, "temporal")
            self.assertFalse(config.fusion.allow_crowd_only)
            self.assertEqual(config.fusion.smoothing_points, 4)
            self.assertEqual(config.fusion.normalization.density_delta, (-2.0, 2.0))

        for fragment, field in (
            ('strategy = "unknown"\n', "fusion.strategy"),
            ("candidate_threshold = 0.8\nactive_threshold = 0.2\n", "lifecycle thresholds"),
            ("violence_weight = nan\n", "violence_weight"),
            ("[fusion.normalization]\ndensity_delta = [1.0, 1.0]\n", "density_delta"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                config_path = Path(directory) / "pipeline.toml"
                config_path.write_text(
                    f"""\
[input]
path = "input.mp4"
[output]
directory = "artifacts"
[processing]
[fusion]
{fragment}"""
                )
                with self.assertRaisesRegex(ConfigError, field):
                    load_config(config_path)


if __name__ == "__main__":
    unittest.main()
