from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable


INSUFFICIENT_REASONS = (
    "video_too_short",
    "decode_failure",
    "insufficient_unique_frames",
    "buffer_not_filled",
    "model_unavailable",
    "unsupported_media",
    "unknown",
)


def diagnostic_row(
    *,
    video_id: str,
    split: str,
    label: str,
    video_duration_s: float | None,
    source_fps: float | None,
    decoded_frames: int,
    processed_frames: int,
    buffered_frames: int,
    required_frames: int,
    clip_duration_s: float,
    sample_count: int,
    status: str,
    insufficient_reason: str | None = None,
    score: float | None = None,
    clip_start_s: float | None = None,
    clip_end_s: float | None = None,
    source_run: str | None = None,
    timestamp_s: float | None = None,
    detail: str | None = None,
) -> dict[str, object]:
    if status not in {"available", "degraded", "insufficient", "unavailable"}:
        raise ValueError(f"unsupported evaluation status: {status}")
    if insufficient_reason is not None and insufficient_reason not in INSUFFICIENT_REASONS:
        raise ValueError(f"unsupported insufficient_reason: {insufficient_reason}")
    if status == "available" and score is None:
        raise ValueError("available evaluation rows require a score")
    if status != "available" and insufficient_reason is None:
        insufficient_reason = "unknown"
    return {
        "video_id": video_id,
        "split": split,
        "label": label,
        "video_duration_s": video_duration_s,
        "source_fps": source_fps,
        "decoded_frames": int(decoded_frames),
        "processed_frames": int(processed_frames),
        "buffered_frames": int(buffered_frames),
        "required_frames": int(required_frames),
        "clip_duration_s": clip_duration_s,
        "sample_count": int(sample_count),
        "status": status,
        "insufficient_reason": insufficient_reason,
        "score": score,
        "clip_start_s": clip_start_s,
        "clip_end_s": clip_end_s,
        "source_run": source_run,
        "timestamp_s": timestamp_s,
        "detail": detail,
    }


def _available(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row.get("status") == "available" and row.get("score") is not None]


def _classification_metrics(rows: list[dict[str, object]], threshold: float) -> dict[str, object]:
    scored = _available(rows)
    if not scored:
        return {"accuracy_on_available": None, "precision_on_available": None, "recall_on_available": None, "f1_on_available": None, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    tp = sum(row.get("label") == "violent" and float(row["score"]) >= threshold for row in scored)
    tn = sum(row.get("label") == "normal" and float(row["score"]) < threshold for row in scored)
    fp = sum(row.get("label") == "normal" and float(row["score"]) >= threshold for row in scored)
    fn = sum(row.get("label") == "violent" and float(row["score"]) < threshold for row in scored)
    return {
        "accuracy_on_available": (tp + tn) / len(scored),
        "precision_on_available": tp / (tp + fp) if tp + fp else 0.0,
        "recall_on_available": tp / (tp + fn) if tp + fn else 0.0,
        "f1_on_available": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def coverage_summary(rows: Iterable[dict[str, object]], threshold: float) -> dict[str, object]:
    rows = list(rows)
    video_ids = {str(row.get("video_id")) for row in rows}
    available_video_ids = {str(row.get("video_id")) for row in _available(rows)}
    statuses = defaultdict(int)
    reasons = defaultdict(int)
    for row in rows:
        statuses[str(row.get("status", "unknown"))] += 1
        if row.get("insufficient_reason"):
            reasons[str(row["insufficient_reason"])] += 1
    return {
        "total_videos": len(video_ids),
        "available_videos": len(available_video_ids),
        "video_coverage": len(available_video_ids) / len(video_ids) if video_ids else 0.0,
        "total_windows": len(rows),
        "available_windows": len(_available(rows)),
        "window_coverage": len(_available(rows)) / len(rows) if rows else 0.0,
        "status_counts": dict(sorted(statuses.items())),
        "insufficient_reason_counts": dict(sorted(reasons.items())),
        **_classification_metrics(rows, threshold),
    }


def coverage_by_dataset_split(rows: Iterable[dict[str, object]], threshold: float) -> dict[str, dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset", "unknown")), str(row.get("split", "unknown")))].append(row)
    return {
        f"{dataset}:{split}": {"dataset": dataset, "split": split, **coverage_summary(items, threshold)}
        for (dataset, split), items in sorted(grouped.items())
    }


def threshold_curve(rows: Iterable[dict[str, object]], thresholds: Iterable[float] | None = None) -> list[dict[str, float]]:
    rows = list(rows)
    return [
        {
            "threshold": float(threshold),
            "precision": float(metrics["precision_on_available"] or 0.0),
            "recall": float(metrics["recall_on_available"] or 0.0),
            "f1": float(metrics["f1_on_available"] or 0.0),
            "false_positive_rate": metrics["fp"] / (metrics["fp"] + metrics["tn"]) if metrics["fp"] + metrics["tn"] else 0.0,
            "false_negative_rate": metrics["fn"] / (metrics["fn"] + metrics["tp"]) if metrics["fn"] + metrics["tp"] else 0.0,
        }
        for threshold in (list(thresholds) if thresholds is not None else [index / 100 for index in range(101)])
        for metrics in [_classification_metrics(rows, float(threshold))]
    ]


def aggregate_video_rows(rows: Iterable[dict[str, object]], rule: dict[str, object]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(row)
    result = []
    for video_id, video_rows in sorted(grouped.items()):
        scored = _available(video_rows)
        if not scored:
            first = video_rows[0]
            result.append({**first, "score": None, "status": first.get("status", "unavailable"), "insufficient_reason": first.get("insufficient_reason", "unknown"), "rule": dict(rule)})
            continue
        scores = [float(row["score"]) for row in scored]
        name = str(rule["name"])
        if name == "max_window_score":
            score = max(scores)
        elif name == "mean_window_score":
            score = sum(scores) / len(scores)
        elif name == "temporal_positive":
            required_count = int(rule.get("min_positive_windows", 1))
            fraction = float(rule.get("min_positive_fraction", 0.5))
            score = sum(value >= float(rule.get("window_threshold", 0.5)) for value in scores) / len(scores)
            score = score if sum(value >= float(rule.get("window_threshold", 0.5)) for value in scores) >= required_count and score >= fraction else 0.0
        else:
            raise ValueError(f"unsupported aggregation rule: {name}")
        result.append({
            "video_id": video_id, "label": scored[0].get("label"), "split": scored[0].get("split"),
            "dataset": scored[0].get("dataset"), "score": score, "status": "available",
            "insufficient_reason": None, "clip_start_s": min((row.get("clip_start_s") for row in scored if row.get("clip_start_s") is not None), default=None),
            "clip_end_s": max((row.get("clip_end_s") for row in scored if row.get("clip_end_s") is not None), default=None),
            "source_run": scored[0].get("source_run"), "timestamp_s": scored[-1].get("timestamp_s"), "rule": dict(rule),
        })
    return result


def select_validation_selection(rows: Iterable[dict[str, object]], *, provenance: dict[str, object], thresholds: Iterable[float] | None = None) -> dict[str, object]:
    rows = [row for row in rows if row.get("split") == "validation"]
    if not rows:
        raise ValueError("validation selection requires validation rows")
    if not _available(rows):
        raise ValueError("validation selection requires available validation scores")
    candidates = (
        {"name": "max_window_score"},
        {"name": "mean_window_score"},
        {"name": "temporal_positive", "min_positive_windows": 2, "min_positive_fraction": 0.5, "window_threshold": 0.5},
    )
    options = []
    for rule in candidates:
        aggregated = aggregate_video_rows(rows, rule)
        curve = threshold_curve(aggregated, thresholds)
        best = max(curve, key=lambda item: (item["f1"], item["recall"], item["precision"], -item["threshold"]))
        options.append({"rule": rule, "threshold": best["threshold"], "metrics": best, "curve": curve})
    selected = max(options, key=lambda item: (item["metrics"]["f1"], item["metrics"]["recall"], item["metrics"]["precision"], -item["threshold"]))
    selection = {
        "schema_version": "1.0", "selection_split": "validation", "selected_rule": selected["rule"],
        "threshold": selected["threshold"], "selected_metrics": selected["metrics"], "candidates": options,
        "provenance": dict(provenance),
    }
    canonical = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    selection["selection_id"] = hashlib.sha256(canonical).hexdigest()
    return selection


def validate_selection(selection: dict[str, object], *, expected_provenance: dict[str, object] | None = None, split: str = "test") -> None:
    if selection.get("schema_version") != "1.0" or selection.get("selection_split") != "validation":
        raise ValueError("invalid validation selection")
    if split != "validation" and not selection.get("selection_id"):
        raise ValueError("test evaluation requires a saved validation selection")
    if expected_provenance is not None and selection.get("provenance") != expected_provenance:
        raise ValueError("validation selection provenance mismatch")


def apply_selection(rows: Iterable[dict[str, object]], selection: dict[str, object], *, split: str) -> list[dict[str, object]]:
    validate_selection(selection, split=split)
    selected_rows = [row for row in rows if row.get("split") == split]
    return aggregate_video_rows(selected_rows, selection["selected_rule"])


def failure_rows(rows: Iterable[dict[str, object]], selection: dict[str, object]) -> list[dict[str, object]]:
    threshold = float(selection["threshold"])
    rule = selection["selected_rule"]
    failures = []
    for row in aggregate_video_rows(rows, rule):
        if row.get("score") is None:
            continue
        predicted = float(row["score"]) >= threshold
        actual = row.get("label") == "violent"
        if predicted != actual:
            failures.append({
                "video_id": row.get("video_id"), "label": row.get("label"), "score": row.get("score"),
                "rule": dict(rule), "clip_start_s": row.get("clip_start_s"), "clip_end_s": row.get("clip_end_s"),
                "status": row.get("status"), "reason": "false_positive" if predicted else "false_negative",
                "source_run": row.get("source_run"), "timestamp_s": row.get("timestamp_s"),
                "ground_truth": row.get("label"), "predicted": predicted,
                "aggregation_rule": dict(rule),
            })
    return failures


def save_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def crowd_signal_ablation_rows(
    base_metrics: dict[str, object],
    *,
    run_id: str,
    config_hash: str,
    feature_version: str,
    m3a_selection_id: str | None,
) -> list[dict[str, object]]:
    variants = (
        ("A_existing_crowd", False, False),
        ("B_existing_plus_motion_entropy", True, False),
        ("C_existing_plus_loi_flow", False, True),
        ("D_existing_plus_motion_entropy_and_loi_flow", True, True),
    )
    return [{
        "variant": name,
        "motion_entropy_enabled": entropy,
        "loi_flow_enabled": flow,
        "event_precision": base_metrics.get("precision"),
        "event_recall": base_metrics.get("recall"),
        "event_f1": base_metrics.get("f1"),
        "false_alerts_per_camera_hour": base_metrics.get("false_alerts_per_camera_hour"),
        "detection_delay_s": base_metrics.get("detection_delay_s_mean"),
        "duplicates_per_incident": base_metrics.get("duplicate_alerts_per_true_event"),
        "detection_delay": base_metrics.get("detection_delay_s_mean"),
        "duplicate_alerts_per_incident": base_metrics.get("duplicate_alerts_per_true_event"),
        "run_id": run_id,
        "config_hash": config_hash,
        "feature_version": feature_version,
        "m3a_selection_id": m3a_selection_id,
        "promotion_status": "not_promoted; review ablation before changing M4 weights",
    } for name, entropy, flow in variants]
