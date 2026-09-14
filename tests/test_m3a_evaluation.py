import unittest

from crowd_safety.m3a_evaluation import (
    INSUFFICIENT_REASONS,
    aggregate_video_rows,
    apply_selection,
    coverage_summary,
    diagnostic_row,
    failure_rows,
    select_validation_selection,
    threshold_curve,
    validate_selection,
    crowd_signal_ablation_rows,
)


def row(video_id, split, label, score, status="available", reason=None):
    return diagnostic_row(
        video_id=video_id, split=split, label=label, video_duration_s=4.0, source_fps=6.0,
        decoded_frames=24, processed_frames=12, buffered_frames=12, required_frames=4,
        clip_duration_s=1.0, sample_count=4, status=status, insufficient_reason=reason,
        score=score, clip_start_s=0.0, clip_end_s=1.0, source_run="run-1", timestamp_s=1.0,
    )


class M3AEvaluationTest(unittest.TestCase):
    def test_diagnostics_preserve_reason_taxonomy_and_coverage_denominator(self):
        rows = [row("normal", "test", "normal", 0.1), row("short", "test", "violent", None, "insufficient", "video_too_short")]
        summary = coverage_summary(rows, 0.5)

        self.assertEqual(summary["insufficient_reason_counts"], {"video_too_short": 1})
        self.assertEqual(summary["total_videos"], 2)
        self.assertEqual(summary["available_videos"], 1)
        self.assertEqual(summary["window_coverage"], 0.5)
        self.assertEqual(summary["recall_on_available"], 0.0)

    def test_selection_is_validation_only_and_applies_unchanged(self):
        validation = [row("normal", "validation", "normal", 0.1), row("violent", "validation", "violent", 0.9)]
        selection = select_validation_selection(validation, provenance={"model": "fixture", "config": "v1"}, thresholds=[0.1, 0.5, 0.9])
        validate_selection(selection, expected_provenance={"model": "fixture", "config": "v1"}, split="test")
        test_rows = [row("test-normal", "test", "normal", 0.9), row("test-violent", "test", "violent", 0.1)]
        scored = apply_selection(test_rows, selection, split="test")
        self.assertEqual(len(scored), 2)
        failures = failure_rows(test_rows, selection)
        self.assertEqual(len(failures), 2)
        self.assertTrue(set(failures[0]) >= {"ground_truth", "predicted", "aggregation_rule", "clip_start_s", "clip_end_s"})
        self.assertTrue(all(set(item) >= {"precision", "recall", "f1", "false_positive_rate", "false_negative_rate"} for item in threshold_curve(scored)))

    def test_aggregation_rules_and_invalid_selection(self):
        rows = [row("v", "validation", "violent", 0.2), row("v", "validation", "violent", 0.9)]
        self.assertEqual(aggregate_video_rows(rows, {"name": "max_window_score"})[0]["score"], 0.9)
        self.assertEqual(aggregate_video_rows(rows, {"name": "mean_window_score"})[0]["score"], 0.55)
        with self.assertRaises(ValueError):
            validate_selection({"schema_version": "1.0", "selection_split": "validation"}, split="test")

    def test_selection_fails_closed_without_available_validation_scores(self):
        missing = row("validation-missing", "validation", "violent", None, "unavailable", "model_unavailable")
        with self.assertRaisesRegex(ValueError, "available validation scores"):
            select_validation_selection([missing], provenance={"model": "fixture"})

    def test_ablation_rows_keep_traceability_and_zero_promotion(self):
        rows = crowd_signal_ablation_rows(
            {"precision": 0.5, "recall": 0.4, "f1": 0.44, "false_alerts_per_camera_hour": 1.0,
             "detection_delay_s_mean": 2.0, "duplicate_alerts_per_true_event": 0.1},
            run_id="run", config_hash="config", feature_version="features-v1", m3a_selection_id="selection",
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1]["m3a_selection_id"], "selection")
        self.assertTrue(all("not_promoted" in row["promotion_status"] for row in rows))


if __name__ == "__main__":
    unittest.main()
