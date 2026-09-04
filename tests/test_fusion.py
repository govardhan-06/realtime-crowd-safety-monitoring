import unittest

from crowd_safety.config import FusionConfig, FusionNormalizationConfig, ROIConfig
from crowd_safety.fusion import build_fusion_points
from crowd_safety.types import CrowdFeatureRecord, ViolenceEvidence


def crowd(timestamp, *, density=0.0, speed=0.0, convergence=0.0, status="available"):
    return CrowdFeatureRecord(
        "camera-1", "zone", timestamp, status, occupancy=10, density_proxy=0.1,
        density_delta=density, mean_speed_px_s=speed, acceleration_px_s2=0.0,
        speed_variance_px_s2=0.0, direction_disorder=0.2, convergence=convergence,
        dispersal=0.0, counter_flow=0.0, congestion=1.0,
    )


def violence(start, end, score, status="available", region=None):
    return ViolenceEvidence(
        "camera-1", region, start, end, score, "fake", "rev", (("safe", 0), ("unsafe", 1)), status,
    )


class FusionTest(unittest.TestCase):
    def test_aligns_without_interpolation_and_keeps_unavailable_distinct(self):
        config = FusionConfig(smoothing_points=1, violence_stale_after_s=0.5)
        rows = build_fusion_points(
            [crowd(1.0), crowd(2.0)],
            [violence(0.0, 1.0, 0.8)],
            config,
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].violence_score, 0.8)
        self.assertEqual(rows[1].violence_score, 0.8)
        self.assertEqual(rows[1].violence_status, "available")
        self.assertIsNone(rows[1].effective_violence_score)
        self.assertTrue(rows[1].violence_stale)
        self.assertIn("violence_stale", rows[1].reason_codes)

    def test_global_violence_is_associated_to_same_source_roi(self):
        config = FusionConfig(smoothing_points=1)
        other = CrowdFeatureRecord("camera-2", "zone", 1.0, "available", occupancy=1)
        rows = build_fusion_points([crowd(1.0), other], [violence(0.0, 1.0, 0.8)], config)
        self.assertEqual(rows[0].violence_score, 0.8)
        self.assertIsNone(rows[1].violence_score)

    def test_normalisation_clamps_and_replay_is_stable(self):
        config = FusionConfig(
            smoothing_points=2,
            normalization=FusionNormalizationConfig(density_delta=(0.0, 1.0), mean_speed_px_s=(0.0, 10.0)),
        )
        stream = [crowd(1.0, density=2.0, speed=20.0), crowd(2.0, density=0.5, speed=5.0)]
        first = build_fusion_points(stream, [], config)
        second = build_fusion_points(stream, [], config)
        self.assertEqual(first, second)
        self.assertEqual(first[0].normalized_crowd["density_delta"], 1.0)
        self.assertEqual(first[0].normalized_crowd["mean_speed_px_s"], 1.0)

    def test_strategies_use_same_inputs_but_different_policy(self):
        stream = [crowd(1.0, density=1.0, speed=30.0, convergence=1.0)]
        evidence = [violence(0.0, 1.0, 0.8)]
        risks = {
            strategy: build_fusion_points(stream, evidence, FusionConfig(strategy=strategy, smoothing_points=1))[0].fused_risk
            for strategy in ("violence-only", "crowd-only", "naive-or", "rule-fusion", "temporal")
        }
        self.assertEqual(risks["violence-only"], 0.8)
        self.assertEqual(risks["naive-or"], max(risks["violence-only"], risks["crowd-only"]))
        self.assertNotEqual(risks["temporal"], risks["violence-only"])

    def test_replay_preserves_stored_roi_order(self):
        first = crowd(1.0)
        second = CrowdFeatureRecord("camera-1", "a-first", 1.0, "available", occupancy=2)
        rows = build_fusion_points([first, second], [], FusionConfig(smoothing_points=1))
        self.assertEqual([row.region_id for row in rows], ["zone", "a-first"])


if __name__ == "__main__":
    unittest.main()
