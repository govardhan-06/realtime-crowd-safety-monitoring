import unittest

from crowd_safety.types import (
    CrowdFeatureRecord,
    PersonDetection,
    StageHealth,
    TrackObservation,
    ViolenceEvidence,
    EvidenceManifest,
    EvidenceReference,
    IncidentExplanation,
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

    def test_violence_evidence_keeps_model_unavailability_distinct_from_zero(self):
        evidence = ViolenceEvidence(
            "camera-1", None, 1.0, 3.0, None, "model", "revision",
            (("NON_VIOLENCE", 0), ("VIOLENCE", 1)), "unavailable",
        )

        self.assertIsNone(evidence.score)
        self.assertEqual(evidence.status, "unavailable")

    def test_available_violence_evidence_requires_a_score(self):
        with self.assertRaises(ValueError):
            ViolenceEvidence(
                "camera-1", None, 1.0, 3.0, None, "model", "revision",
                (), "available",
            )

    def test_m5_records_reject_invalid_paths_and_secret_metadata(self):
        reference = EvidenceReference("snapshot", "incidents/i1/snapshot.jpg", 1.0, 1.0, "available")
        manifest = EvidenceManifest(
            "run-1", "camera-1", "i1", 1.0, 2.0, 3.0, 4.0,
            ("violence",), ({"timestamp_s": 1.0, "fused_risk": 0.8},),
            {"violence": {"status": "available"}}, (reference,),
        )
        self.assertEqual(manifest.artifacts[0].relative_path, "incidents/i1/snapshot.jpg")
        with self.assertRaises(ValueError):
            EvidenceReference("snapshot", "../outside.jpg", 1.0, 1.0, "available")
        with self.assertRaises(ValueError):
            IncidentExplanation("i1", "generated", "fake", "model", "token=secret")


if __name__ == "__main__":
    unittest.main()
