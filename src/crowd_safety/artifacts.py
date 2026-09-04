import hashlib
import json
from pathlib import Path
from typing import Any

from .fusion import FUSION_VERSION


def resolved_config(config: Any, input_path: Path) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_directory": str(config.output_directory),
        "resize": list(config.resize),
        "target_fps": config.target_fps,
        "annotation_enabled": config.annotation_enabled,
        "logging_enabled": config.logging_enabled,
        "fusion_version": FUSION_VERSION,
        "perception": {
            "enabled": config.perception.enabled,
            "model": config.perception.model,
            "confidence": config.perception.confidence,
            "cadence_fps": config.perception.cadence_fps,
            "device": config.perception.device,
            "person_class_id": config.perception.person_class_id,
        },
        "tracking": {
            "config": config.tracking.config,
            "track_buffer": config.tracking.track_buffer,
        },
        "crowd": {
            "window_s": config.crowd.window_s,
            "min_track_history": config.crowd.min_track_history,
            "min_speed_px_s": config.crowd.min_speed_px_s,
            "congestion_occupancy": config.crowd.congestion_occupancy,
            "congestion_speed_px_s": config.crowd.congestion_speed_px_s,
            "rois": [
                {"name": roi.name, "polygon": [list(point) for point in roi.polygon]}
                for roi in config.crowd.rois
            ],
        },
        "violence": {
            "enabled": config.violence.enabled,
            "model": config.violence.model,
            "revision": config.violence.revision,
            "clip_duration_s": config.violence.clip_duration_s,
            "sample_count": config.violence.sample_count,
            "cadence_s": config.violence.cadence_s,
            "threshold": config.violence.threshold,
            "device": config.violence.device,
            "labels": list(config.violence.labels),
            "license": config.violence.license,
            "known_limitations": config.violence.known_limitations,
            "checkpoint_sha256": config.violence.checkpoint_sha256,
        },
        "fusion": {
            "strategy": config.fusion.strategy,
            "source_roi_policy": config.fusion.source_roi_policy,
            "violence_stale_after_s": config.fusion.violence_stale_after_s,
            "smoothing_points": config.fusion.smoothing_points,
            "allow_crowd_only": config.fusion.allow_crowd_only,
            "violence_weight": config.fusion.violence_weight,
            "density_weight": config.fusion.density_weight,
            "movement_weight": config.fusion.movement_weight,
            "context_weight": config.fusion.context_weight,
            "persistence_weight": config.fusion.persistence_weight,
            "candidate_threshold": config.fusion.candidate_threshold,
            "active_threshold": config.fusion.active_threshold,
            "escalating_threshold": config.fusion.escalating_threshold,
            "critical_threshold": config.fusion.critical_threshold,
            "persistence_s": config.fusion.persistence_s,
            "hysteresis": config.fusion.hysteresis,
            "decay_s": config.fusion.decay_s,
            "quiet_period_s": config.fusion.quiet_period_s,
            "severity_medium": config.fusion.severity_medium,
            "severity_high": config.fusion.severity_high,
            "severity_critical": config.fusion.severity_critical,
            "normalization": {
                name: list(bounds) for name, bounds in vars(config.fusion.normalization).items()
            },
        },
        "m5": {
            "evidence_root": str(config.m5.evidence_root),
            "pre_event_s": config.m5.pre_event_s,
            "post_event_s": config.m5.post_event_s,
            "retention_s": config.m5.retention_s,
            "database_url_env": config.m5.database_url_env,
            "vlm_enabled": config.m5.vlm_enabled,
            "vlm_provider": config.m5.vlm_provider,
            "vlm_model": config.m5.vlm_model,
            "vlm_timeout_s": config.m5.vlm_timeout_s,
        },
    }


def config_hash(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
