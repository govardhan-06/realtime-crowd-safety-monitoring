import hashlib
import json
from pathlib import Path
from typing import Any


def resolved_config(config: Any, input_path: Path) -> dict[str, Any]:
    return {
        "input_path": str(input_path),
        "output_directory": str(config.output_directory),
        "resize": list(config.resize),
        "target_fps": config.target_fps,
        "annotation_enabled": config.annotation_enabled,
        "logging_enabled": config.logging_enabled,
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
    }


def config_hash(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
