import unittest

from crowd_safety.detection import UltralyticsPersonDetector
from crowd_safety.types import FramePacket


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[1, 2, 11, 22], [3, 4, 13, 24]])
    conf = FakeTensor([0.9, 0.8])
    cls = FakeTensor([0, 2])


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    names = {0: "person", 2: "car"}

    def __call__(self, image, **kwargs):
        return [FakeResult()]


class DetectionTest(unittest.TestCase):
    def test_adapter_emits_only_project_owned_person_records(self):
        adapter = UltralyticsPersonDetector(model_object=FakeModel(), person_class_id=0)
        result = adapter.detect(FramePacket("camera-1", 4, 0.5, object()))

        self.assertEqual(len(result.detections), 1)
        self.assertEqual(result.detections[0].box_xyxy, (1.0, 2.0, 11.0, 22.0))
        self.assertEqual(result.health.status, "available")
        self.assertEqual(result.health.model, "yolo26n.pt")

    def test_missing_dependency_or_model_is_explicitly_unavailable(self):
        adapter = UltralyticsPersonDetector(model_error="weights unavailable")
        result = adapter.detect(FramePacket("camera-1", 4, 0.5, object()))

        self.assertEqual(result.detections, ())
        self.assertEqual(result.health.status, "unavailable")
        self.assertIn("weights unavailable", result.health.detail)


if __name__ == "__main__":
    unittest.main()
