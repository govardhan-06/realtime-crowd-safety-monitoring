from dataclasses import dataclass
import math
from pathlib import Path
import tomllib


class ConfigError(ValueError):
    """Raised when a pipeline TOML file is invalid."""


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_directory: Path
    resize: tuple[int, int]
    target_fps: float
    annotation_enabled: bool
    logging_enabled: bool


def _path(value: object, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        values = tomllib.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file does not exist: {config_path}") from exc
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"could not read config {config_path}: {exc}") from exc

    try:
        input_values = values["input"]
        output_values = values["output"]
        processing_values = values["processing"]
    except (KeyError, TypeError) as exc:
        raise ConfigError("config must contain [input], [output], and [processing] sections") from exc
    if not all(isinstance(section, dict) for section in (input_values, output_values, processing_values)):
        raise ConfigError("[input], [output], and [processing] must be TOML tables")

    annotation_values = values.get("annotation", {})
    logging_values = values.get("logging", {})
    if not isinstance(annotation_values, dict) or not isinstance(logging_values, dict):
        raise ConfigError("[annotation] and [logging] must be TOML tables")

    target_fps = processing_values.get("target_fps", 5.0)
    if (
        not isinstance(target_fps, (int, float))
        or isinstance(target_fps, bool)
        or not math.isfinite(target_fps)
        or target_fps <= 0
    ):
        raise ConfigError("processing.target_fps must be greater than zero")

    resize_value = processing_values.get("resize", [640, 360])
    if (
        not isinstance(resize_value, list)
        or len(resize_value) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in resize_value)
    ):
        raise ConfigError("processing.resize must contain two positive integer dimensions")

    annotation = annotation_values.get("enabled", True)
    logging = logging_values.get("enabled", True)
    if not isinstance(annotation, bool):
        raise ConfigError("annotation.enabled must be boolean")
    if not isinstance(logging, bool):
        raise ConfigError("logging.enabled must be boolean")

    return PipelineConfig(
        input_path=_path(input_values.get("path"), "input.path", config_path.parent),
        output_directory=_path(output_values.get("directory"), "output.directory", config_path.parent),
        resize=(resize_value[0], resize_value[1]),
        target_fps=float(target_fps),
        annotation_enabled=annotation,
        logging_enabled=logging,
    )
