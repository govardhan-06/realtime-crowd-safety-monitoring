import unittest

import torch

from crowd_safety.types import FramePacket
from crowd_safety.violence import ClipWindow, VideoMAEViolenceClassifier


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


class ViolenceAdapterTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
