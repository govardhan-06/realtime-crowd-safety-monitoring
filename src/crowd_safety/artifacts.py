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
    }


def config_hash(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")
