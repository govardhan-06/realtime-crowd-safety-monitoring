import unittest
from pathlib import Path
import tempfile
from unittest import mock

import torch

from crowd_safety.types import FramePacket
from crowd_safety.config import ViolenceConfig
from crowd_safety.violence import (
    ClipWindow,
    VideoMAEViolenceClassifier,
    X3DViolenceClassifier,
    _load_x3d_checkpoint,
    _normalize_x3d_state_dict,
    create_violence_classifier,
)


def window() -> ClipWindow:
    packets = tuple(
        FramePacket("camera-1", index, float(index), torch.zeros((2, 2, 3), dtype=torch.uint8))
        for index in range(3)
    )
    return ClipWindow(packets, packets, 0.0, 2.0)


class FakeProcessor:
    def __call__(self, frames, return_tensors):
        self.frame_count = len(frames)
        return {"pixel_values": torch.zeros((1, len(frames), 3, 2, 2))}


class FakeModel:
    class Config:
        id2label = {"0": "safe", "1": "unsafe"}

    config = Config()

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **inputs):
        return type("Output", (), {"logits": torch.tensor([[0.0, 2.0]])})()


class FakeX3DModel:
    def __init__(self):
        self.inputs = None

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def __call__(self, inputs):
        self.inputs = inputs
        return torch.tensor([[0.0, 2.0]])


class ViolenceAdapterTest(unittest.TestCase):
    def test_x3d_checkpoint_state_dict_strips_backbone_wrapper(self):
        state_dict = {
            "backbone.blocks.0.weight": torch.ones(1),
            "backbone.blocks.5.proj.weight": torch.ones(2),
        }

        normalized = _normalize_x3d_state_dict(state_dict)

        self.assertEqual(set(normalized), {"blocks.0.weight", "blocks.5.proj.weight"})

    def test_x3d_checkpoint_load_allowlists_numpy_scalar_with_weights_only(self):
        import numpy as np

        safe_globals = mock.Mock()
        safe_globals.return_value.__enter__ = mock.Mock(return_value=None)
        safe_globals.return_value.__exit__ = mock.Mock(return_value=None)
        fake_torch = mock.Mock()
        fake_torch.serialization.safe_globals = safe_globals
        fake_torch.load.return_value = {"model": {}}

        checkpoint = _load_x3d_checkpoint(Path("checkpoint.pt"), fake_torch)

        self.assertEqual(checkpoint, {"model": {}})
        safe_globals.assert_called_once()
        allowed_globals = safe_globals.call_args.args[0]
        self.assertIn(type(np.dtype(np.float64)), allowed_globals)
        fake_torch.load.assert_called_once_with(Path("checkpoint.pt"), map_location="cpu", weights_only=True)

    def test_x3d_checkpoint_load_accepts_numpy_float64_dtype(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            torch.save({"dtype": np.dtype(np.float64)}, path)

            checkpoint = _load_x3d_checkpoint(path, torch)

        self.assertEqual(checkpoint["dtype"], np.dtype(np.float64))

    def test_maps_confirmed_unsafe_label_to_generic_score(self):
        classifier = VideoMAEViolenceClassifier(
            "model", "revision", device="cpu", processor=FakeProcessor(), model_instance=FakeModel()
        )

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "available")
        self.assertGreater(evidence.score, 0.8)
        self.assertEqual(evidence.label_mapping, (("safe", 0), ("unsafe", 1)))
        self.assertEqual(classifier.health.status, "available")

    def test_model_failure_is_unavailable_without_a_zero_score(self):
        classifier = VideoMAEViolenceClassifier("model", "revision", load_error="weights missing")

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "unavailable")
        self.assertIsNone(evidence.score)
        self.assertIn("weights missing", evidence.detail)
        self.assertEqual(classifier.health.status, "unavailable")

    def test_malformed_label_mapping_is_unavailable(self):
        class MalformedModel(FakeModel):
            class Config:
                id2label = {"0": "safe", "1": "unknown"}

            config = Config()

        classifier = VideoMAEViolenceClassifier(
            "model", "revision", device="cpu", processor=FakeProcessor(), model_instance=MalformedModel()
        )

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "unavailable")
        self.assertIsNone(evidence.score)
        self.assertIn("label", evidence.detail.lower())

    def test_multiclass_label_mapping_is_unavailable(self):
        class MulticlassModel(FakeModel):
            class Config:
                id2label = {"0": "safe", "1": "violent", "2": "other"}

            config = Config()

        classifier = VideoMAEViolenceClassifier(
            "model", "revision", device="cpu", processor=FakeProcessor(), model_instance=MulticlassModel()
        )

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "unavailable")
        self.assertIsNone(evidence.score)
        self.assertIn("two output labels", evidence.detail)

    def test_configured_labels_must_match_model_mapping(self):
        classifier = VideoMAEViolenceClassifier(
            "model", "revision", device="cpu", labels=("safe", "violent"),
            processor=FakeProcessor(), model_instance=FakeModel(),
        )

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "unavailable")
        self.assertIn("do not match", evidence.detail)

    def test_explicit_configured_labels_cover_models_without_id2label(self):
        class ModelWithoutLabels(FakeModel):
            class Config:
                pass

            config = Config()

        classifier = VideoMAEViolenceClassifier(
            "model", "revision", device="cpu", labels=("Non-Violent Incident", "Violent Crime"),
            processor=FakeProcessor(), model_instance=ModelWithoutLabels(),
        )

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "available")
        self.assertEqual(evidence.label_mapping, (("Non-Violent Incident", 0), ("Violent Crime", 1)))

    def test_inference_error_is_degraded_without_a_score(self):
        class BrokenModel(FakeModel):
            def __call__(self, **inputs):
                raise RuntimeError("inference failed")

        classifier = VideoMAEViolenceClassifier(
            "model", "revision", device="cpu", processor=FakeProcessor(), model_instance=BrokenModel()
        )

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "degraded")
        self.assertIsNone(evidence.score)
        self.assertIn("inference failed", evidence.detail)

    def test_x3d_adapter_samples_rgb_frames_and_returns_provenance(self):
        model = FakeX3DModel()
        packets = tuple(
            FramePacket(
                "camera-1", index, float(index) / 5.0,
                torch.tensor([[[0, 0, 255]]], dtype=torch.uint8).repeat(2, 2, 1),
            )
            for index in range(20)
        )
        x3d_window = ClipWindow(packets, packets[:16], packets[0].timestamp_s, packets[-1].timestamp_s)
        classifier = X3DViolenceClassifier(
            "visionlab-ai/school-violence-detection-models",
            "final/final_x3d_realtime.pt",
            "a744b6af7496f0cbfa4f0ba32acd46b65e52d4e1",
            architecture="x3d_m", device="cpu", sample_count=16,
            labels=("non-violent", "violent"), license_name="mit",
            checkpoint_sha256="e833f69d110f167cad4a6c38d385564bdb2f6de63d246e45cb03ff9aa17f0349",
            model_instance=model,
        )

        evidence = classifier.infer(x3d_window)

        self.assertEqual(evidence.status, "available")
        self.assertGreater(evidence.score, 0.8)
        self.assertEqual(evidence.label_mapping, (("non-violent", 0), ("violent", 1)))
        self.assertEqual(tuple(model.inputs.shape), (1, 3, 16, 224, 224))
        self.assertGreater(float(model.inputs[0, 0].mean()), float(model.inputs[0, 1].mean()))
        self.assertEqual(classifier.provenance["backend"], "x3d")
        self.assertEqual(classifier.provenance["architecture"], "x3d_m")
        self.assertEqual(classifier.provenance["sample_count"], 16)

    def test_x3d_checkpoint_hash_mismatch_is_unavailable_without_zero(self):
        with tempfile.NamedTemporaryFile() as checkpoint:
            checkpoint.write(b"checkpoint")
            checkpoint.flush()
            classifier = X3DViolenceClassifier(
                "repo", "checkpoint", "revision", device="cpu",
                labels=("non-violent", "violent"), checkpoint_sha256="0" * 64,
                checkpoint_path=Path(checkpoint.name), model_instance=FakeX3DModel(),
            )

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "unavailable")
        self.assertIsNone(evidence.score)
        self.assertIn("checksum", evidence.detail.lower())

    def test_x3d_inference_failure_is_degraded_without_zero(self):
        class BrokenX3DModel(FakeX3DModel):
            def __call__(self, inputs):
                raise RuntimeError("x3d inference failed")

        classifier = X3DViolenceClassifier(
            "repo", "checkpoint", "revision", device="cpu",
            labels=("non-violent", "violent"), sample_count=3, model_instance=BrokenX3DModel(),
        )

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "degraded")
        self.assertIsNone(evidence.score)
        self.assertIn("x3d inference failed", evidence.detail)

    def test_x3d_non_binary_output_is_degraded_without_zero(self):
        class MulticlassX3DModel(FakeX3DModel):
            def __call__(self, inputs):
                return torch.tensor([[0.0, 1.0, 2.0]])

        classifier = X3DViolenceClassifier(
            "repo", "checkpoint", "revision", device="cpu",
            labels=("non-violent", "violent"), sample_count=3, model_instance=MulticlassX3DModel(),
        )

        evidence = classifier.infer(window())

        self.assertEqual(evidence.status, "degraded")
        self.assertIsNone(evidence.score)
        self.assertIn("binary", evidence.detail)

    def test_factory_selects_x3d_backend(self):
        config = ViolenceConfig(
            enabled=True, backend="x3d", repository="repo", checkpoint="checkpoint",
            revision="revision", architecture="x3d_m", labels=("non-violent", "violent"),
        )
        with mock.patch("crowd_safety.violence.X3DViolenceClassifier") as adapter:
            create_violence_classifier(config)

        adapter.assert_called_once()


if __name__ == "__main__":
    unittest.main()
