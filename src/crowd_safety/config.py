from dataclasses import asdict, dataclass
import math
from pathlib import Path
import tomllib
from typing import Any


class ConfigError(ValueError):
    """Raised when a pipeline TOML file is invalid."""


@dataclass(frozen=True)
class ROIConfig:
    name: str
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class PerceptionConfig:
    enabled: bool = False
    model: str = "yolo26n.pt"
    confidence: float = 0.25
    cadence_fps: float = 5.0
    device: str = "auto"
    person_class_id: int = 0


@dataclass(frozen=True)
class TrackingConfig:
    config: str = "bytetrack.yaml"
    track_buffer: int = 30


@dataclass(frozen=True)
class CrowdConfig:
    window_s: float = 1.0
    min_track_history: int = 2
    min_speed_px_s: float = 1.0
    congestion_occupancy: int = 5
    congestion_speed_px_s: float = 2.0
    rois: tuple[ROIConfig, ...] = ()


@dataclass(frozen=True)
class PipelineConfig:
    input_path: Path
    output_directory: Path
    resize: tuple[int, int]
    target_fps: float
    annotation_enabled: bool
    logging_enabled: bool
    perception: PerceptionConfig = PerceptionConfig()
    tracking: TrackingConfig = TrackingConfig()
    crowd: CrowdConfig = CrowdConfig()

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["input_path"] = str(self.input_path)
        values["output_directory"] = str(self.output_directory)
        return values


def _path(value: object, name: str, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    return path if path.is_absolute() else base / path


def _positive_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ConfigError(f"{name} must be greater than zero")
    return float(value)


def _non_negative_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ConfigError(f"{name} must be non-negative")
    return float(value)


def _positive_int(value: object, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def _roi_values(value: object, resize: tuple[int, int]) -> tuple[ROIConfig, ...]:
    if value is None:
        width, height = resize
        return (ROIConfig("full-frame", ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height)))),)
    if not isinstance(value, list) or not value:
        raise ConfigError("crowd.rois must be a non-empty array of tables")
    rois: list[ROIConfig] = []
    names: set[str] = set()
    width, height = resize
    for index, raw_roi in enumerate(value):
        if not isinstance(raw_roi, dict):
            raise ConfigError(f"crowd.rois[{index}] must be a TOML table")
        name = raw_roi.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ConfigError(f"crowd.rois[{index}].name must be unique and non-empty")
        polygon = raw_roi.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ConfigError(f"crowd.rois[{index}].polygon must contain at least three points")
        points: list[tuple[float, float]] = []
        for point in polygon:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or any(not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool) for coordinate in point)
            ):
                raise ConfigError(f"crowd.rois[{index}].polygon points must contain two numbers")
            x, y = float(point[0]), float(point[1])
            if not math.isfinite(x) or not math.isfinite(y) or not (0 <= x <= width and 0 <= y <= height):
                raise ConfigError(f"crowd.rois[{index}].polygon coordinates must fit inside processing.resize")
            points.append((x, y))
        area = sum(
            points[i][0] * points[(i + 1) % len(points)][1]
            - points[(i + 1) % len(points)][0] * points[i][1]
            for i in range(len(points))
        )
        if abs(area) <= 1e-9:
            raise ConfigError(f"crowd.rois[{index}].polygon must enclose an area")
        names.add(name)
        rois.append(ROIConfig(name, tuple(points)))
    return tuple(rois)


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

    perception_values = values.get("perception", {})
    tracking_values = values.get("tracking", {})
    crowd_values = values.get("crowd", {})
    if not all(isinstance(section, dict) for section in (perception_values, tracking_values, crowd_values)):
        raise ConfigError("[perception], [tracking], and [crowd] must be TOML tables")
    enabled = perception_values.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError("perception.enabled must be boolean")
    model = perception_values.get("model", "yolo26n.pt")
    if not isinstance(model, str) or not model.strip():
        raise ConfigError("perception.model must be a non-empty string")
    if "/" in model or "\\" in model:
        model = str(_path(model, "perception.model", config_path.parent))
    confidence = perception_values.get("confidence", 0.25)
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ConfigError("perception.confidence must be between zero and one")
    cadence_fps = _positive_number(perception_values.get("cadence_fps", target_fps), "perception.cadence_fps")
    device = perception_values.get("device", "auto")
    if not isinstance(device, str) or not device.strip():
        raise ConfigError("perception.device must be a non-empty string")
    person_class_id = _positive_int(perception_values.get("person_class_id", 0), "perception.person_class_id", 0)

    tracker_config = tracking_values.get("config", "bytetrack.yaml")
    if not isinstance(tracker_config, str) or not tracker_config.strip():
        raise ConfigError("tracking.config must be a non-empty string")
    track_buffer = _positive_int(tracking_values.get("track_buffer", 30), "tracking.track_buffer")

    window_s = _positive_number(crowd_values.get("window_s", 1.0), "crowd.window_s")
    min_track_history = _positive_int(crowd_values.get("min_track_history", 2), "crowd.min_track_history", 2)
    min_speed_px_s = _non_negative_number(crowd_values.get("min_speed_px_s", 1.0), "crowd.min_speed_px_s")
    congestion_occupancy = _positive_int(crowd_values.get("congestion_occupancy", 5), "crowd.congestion_occupancy")
    congestion_speed_px_s = _non_negative_number(crowd_values.get("congestion_speed_px_s", 2.0), "crowd.congestion_speed_px_s")

    return PipelineConfig(
        input_path=_path(input_values.get("path"), "input.path", config_path.parent),
        output_directory=_path(output_values.get("directory"), "output.directory", config_path.parent),
        resize=(resize_value[0], resize_value[1]),
        target_fps=float(target_fps),
        annotation_enabled=annotation,
        logging_enabled=logging,
        perception=PerceptionConfig(
            enabled=enabled,
            model=model,
            confidence=float(confidence),
            cadence_fps=cadence_fps,
            device=device,
            person_class_id=person_class_id,
        ),
        tracking=TrackingConfig(config=tracker_config, track_buffer=track_buffer),
        crowd=CrowdConfig(
            window_s=window_s,
            min_track_history=min_track_history,
            min_speed_px_s=min_speed_px_s,
            congestion_occupancy=congestion_occupancy,
            congestion_speed_px_s=congestion_speed_px_s,
            rois=_roi_values(crowd_values.get("rois"), tuple(resize_value)),
        ),
    )
