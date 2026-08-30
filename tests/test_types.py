import unittest

from crowd_safety.types import (
    CrowdFeatureRecord,
    PersonDetection,
    StageHealth,
    TrackObservation,
)


class TypesTest(unittest.TestCase):
    def test_domain_records_do_not_require_vendor_objects(self):
        detection = PersonDetection("camera-1", 3, 0.5, (1.0, 2.0, 5.0, 8.0), 0.9)
        track = TrackObservation("camera-1", 7, 3, 0.5, (3.0, 5.0), detection.box_xyxy, 0.9)
        health = StageHealth("detector", "available", model="yolo26n.pt", latency_ms=2.5)
        feature = CrowdFeatureRecord(
            "camera-1", "zone", 0.5, "insufficient", occupancy=1, density_proxy=0.01
        )

        self.assertEqual(detection.source_id, track.source_id)
        self.assertEqual(health.status, "available")
        self.assertIsNone(feature.mean_speed_px_s)

    def test_unavailable_feature_cannot_look_like_normal_zero_feature(self):
        feature = CrowdFeatureRecord("camera-1", "zone", 0.5, "unavailable")
        self.assertEqual(feature.status, "unavailable")
        self.assertIsNone(feature.occupancy)
        self.assertIsNone(feature.mean_speed_px_s)

    def test_health_rejects_unknown_status(self):
        with self.assertRaises(ValueError):
            StageHealth("detector", "normal")


if __name__ == "__main__":
    unittest.main()
