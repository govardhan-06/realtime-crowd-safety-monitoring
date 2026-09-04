from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
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
class ViolenceConfig:
    enabled: bool = False
    model: str = "mitegvg/videomae-small-kinetics-binary-finetuned-xd-violence"
    revision: str = "main"
    clip_duration_s: float = 2.0
    sample_count: int = 16
    cadence_s: float = 1.0
    threshold: float = 0.5
    device: str = "auto"
    labels: tuple[str, ...] = ("safe", "unsafe")
    license: str = ""
    known_limitations: str = ""
    checkpoint_sha256: str | None = None


@dataclass(frozen=True)
class FusionNormalizationConfig:
    density_delta: tuple[float, float] = (-1.0, 2.0)
    mean_speed_px_s: tuple[float, float] = (0.0, 100.0)
    acceleration_px_s2: tuple[float, float] = (-100.0, 100.0)
    direction_disorder: tuple[float, float] = (0.0, 1.0)
    convergence: tuple[float, float] = (0.0, 1.0)
    dispersal: tuple[float, float] = (0.0, 1.0)
    counter_flow: tuple[float, float] = (0.0, 1.0)
    congestion: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class FusionConfig:
    strategy: str = "temporal"
    source_roi_policy: str = "same-source-configured-rois"
    violence_stale_after_s: float = 2.0
    smoothing_points: int = 3
    allow_crowd_only: bool = True
    violence_weight: float = 0.35
    density_weight: float = 0.15
    movement_weight: float = 0.2
    context_weight: float = 0.15
    persistence_weight: float = 0.15
    candidate_threshold: float = 0.35
    active_threshold: float = 0.5
    escalating_threshold: float = 0.7
    critical_threshold: float = 0.85
    persistence_s: float = 2.0
    hysteresis: float = 0.05
    decay_s: float = 2.0
    quiet_period_s: float = 2.0
    severity_medium: float = 0.5
    severity_high: float = 0.7
    severity_critical: float = 0.85
    normalization: FusionNormalizationConfig = FusionNormalizationConfig()


@dataclass(frozen=True)
class M5Config:
    evidence_root: Path = Path("evidence")
    pre_event_s: float = 5.0
    post_event_s: float = 5.0
    retention_s: float = 86400.0
    database_url_env: str = "DATABASE_URL"
    vlm_enabled: bool = False
    vlm_provider: str = "disabled"
    vlm_model: str = ""
    vlm_timeout_s: float = 10.0


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
    violence: ViolenceConfig = ViolenceConfig()
    fusion: FusionConfig = FusionConfig()
    m5: M5Config = M5Config()

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["input_path"] = str(self.input_path)
        values["output_directory"] = str(self.output_directory)
        values["m5"]["evidence_root"] = str(self.m5.evidence_root)
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
    violence_values = values.get("violence", {})
    if not all(isinstance(section, dict) for section in (perception_values, tracking_values, crowd_values, violence_values)):
        raise ConfigError("[perception], [tracking], [crowd], and [violence] must be TOML tables")
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

    violence_enabled = violence_values.get("enabled", False)
    if not isinstance(violence_enabled, bool):
        raise ConfigError("violence.enabled must be boolean")
    violence_model = violence_values.get("model", ViolenceConfig.model)
    if not isinstance(violence_model, str) or not violence_model.strip():
        raise ConfigError("violence.model must be a non-empty string")
    violence_revision = violence_values.get("revision", ViolenceConfig.revision)
    if not isinstance(violence_revision, str) or not violence_revision.strip():
        raise ConfigError("violence.revision must be a non-empty string")
    clip_duration_s = _positive_number(violence_values.get("clip_duration_s", 2.0), "violence.clip_duration_s")
    sample_count = _positive_int(violence_values.get("sample_count", 16), "violence.sample_count", 2)
    cadence_s = _positive_number(violence_values.get("cadence_s", 1.0), "violence.cadence_s")
    threshold = violence_values.get("threshold", 0.5)
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(threshold)
        or not 0 <= threshold <= 1
    ):
        raise ConfigError("violence.threshold must be between zero and one")
    violence_device = violence_values.get("device", "auto")
    if not isinstance(violence_device, str) or not violence_device.strip():
        raise ConfigError("violence.device must be a non-empty string")
    violence_labels = violence_values.get("labels", list(ViolenceConfig.labels))
    if (
        not isinstance(violence_labels, list)
        or len(violence_labels) < 2
        or any(not isinstance(label, str) or not label.strip() for label in violence_labels)
    ):
        raise ConfigError("violence.labels must contain at least two non-empty strings")
    violence_license = violence_values.get("license", "")
    if not isinstance(violence_license, str):
        raise ConfigError("violence.license must be a string")
    known_limitations = violence_values.get("known_limitations", "")
    if not isinstance(known_limitations, str):
        raise ConfigError("violence.known_limitations must be a string")
    checkpoint_sha256 = violence_values.get("checkpoint_sha256")
    if checkpoint_sha256 is not None and (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in checkpoint_sha256)
    ):
        raise ConfigError("violence.checkpoint_sha256 must be a 64-character hexadecimal string")

    fusion_values = values.get("fusion", {})
    if not isinstance(fusion_values, dict):
        raise ConfigError("[fusion] must be a TOML table")
    strategy = fusion_values.get("strategy", "temporal")
    strategies = {"violence-only", "crowd-only", "naive-or", "rule-fusion", "temporal"}
    if not isinstance(strategy, str) or strategy not in strategies:
        raise ConfigError(f"fusion.strategy must be one of {sorted(strategies)}")
    source_roi_policy = fusion_values.get("source_roi_policy", "same-source-configured-rois")
    if source_roi_policy != "same-source-configured-rois":
        raise ConfigError("fusion.source_roi_policy must be same-source-configured-rois")
    violence_stale_after_s = _positive_number(
        fusion_values.get("violence_stale_after_s", 2.0), "fusion.violence_stale_after_s"
    )
    smoothing_points = _positive_int(fusion_values.get("smoothing_points", 3), "fusion.smoothing_points")
    allow_crowd_only = fusion_values.get("allow_crowd_only", True)
    if not isinstance(allow_crowd_only, bool):
        raise ConfigError("fusion.allow_crowd_only must be boolean")
    weight_names = ("violence_weight", "density_weight", "movement_weight", "context_weight", "persistence_weight")
    weights = {}
    for name in weight_names:
        weights[name] = _non_negative_number(fusion_values.get(name, getattr(FusionConfig, name)), f"fusion.{name}")
    if sum(weights.values()) <= 0:
        raise ConfigError("fusion weights must have a positive sum")
    threshold_names = ("candidate_threshold", "active_threshold", "escalating_threshold", "critical_threshold")
    thresholds = {name: fusion_values.get(name, getattr(FusionConfig, name)) for name in threshold_names}
    for name, value in thresholds.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ConfigError(f"fusion.{name} must be between zero and one")
        thresholds[name] = float(value)
    if not thresholds["candidate_threshold"] <= thresholds["active_threshold"] <= thresholds["escalating_threshold"] <= thresholds["critical_threshold"]:
        raise ConfigError("fusion lifecycle thresholds must be ordered")
    persistence_s = _positive_number(fusion_values.get("persistence_s", 2.0), "fusion.persistence_s")
    hysteresis = _non_negative_number(fusion_values.get("hysteresis", 0.05), "fusion.hysteresis")
    if hysteresis > 1.0:
        raise ConfigError("fusion.hysteresis must not exceed one")
    decay_s = _positive_number(fusion_values.get("decay_s", 2.0), "fusion.decay_s")
    quiet_period_s = _positive_number(fusion_values.get("quiet_period_s", 2.0), "fusion.quiet_period_s")
    severity_names = ("severity_medium", "severity_high", "severity_critical")
    severities = {name: fusion_values.get(name, getattr(FusionConfig, name)) for name in severity_names}
    for name, value in severities.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ConfigError(f"fusion.{name} must be between zero and one")
        severities[name] = float(value)
    if not severities["severity_medium"] <= severities["severity_high"] <= severities["severity_critical"]:
        raise ConfigError("fusion severity boundaries must be ordered")
    normalization_values = fusion_values.get("normalization", {})
    if not isinstance(normalization_values, dict):
        raise ConfigError("[fusion.normalization] must be a TOML table")
    normalization = {}
    for name in FusionNormalizationConfig.__dataclass_fields__:
        value = normalization_values.get(name, getattr(FusionNormalizationConfig, name))
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ConfigError(f"fusion.normalization.{name} must contain two numbers")
        lower, upper = value
        if any(not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(item) for item in value) or lower >= upper:
            raise ConfigError(f"fusion.normalization.{name} must contain finite ordered bounds")
        normalization[name] = (float(lower), float(upper))

    m5_values = values.get("m5", {})
    if not isinstance(m5_values, dict):
        raise ConfigError("[m5] must be a TOML table")
    evidence_root_value = m5_values.get("evidence_root")
    if evidence_root_value is None:
        evidence_root = _path(output_values.get("directory"), "output.directory", config_path.parent) / "evidence"
    else:
        evidence_root = _path(evidence_root_value, "m5.evidence_root", config_path.parent)
    pre_event_s = _positive_number(m5_values.get("pre_event_s", 5.0), "m5.pre_event_s")
    post_event_s = _positive_number(m5_values.get("post_event_s", 5.0), "m5.post_event_s")
    retention_s = _positive_number(m5_values.get("retention_s", 86400.0), "m5.retention_s")
    database_url_env = m5_values.get("database_url_env", "DATABASE_URL")
    if not isinstance(database_url_env, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", database_url_env) is None:
        raise ConfigError("m5.database_url_env must be a valid environment variable name")
    vlm_enabled = m5_values.get("vlm_enabled", False)
    if not isinstance(vlm_enabled, bool):
        raise ConfigError("m5.vlm_enabled must be boolean")
    vlm_provider = m5_values.get("vlm_provider", "disabled")
    if vlm_provider not in {"disabled", "fake", "gemini"}:
        raise ConfigError("m5.vlm_provider must be disabled, fake, or gemini")
    if vlm_enabled and vlm_provider == "disabled":
        raise ConfigError("m5.vlm_enabled requires a configured provider")
    vlm_model = m5_values.get("vlm_model", "")
    if not isinstance(vlm_model, str) or (vlm_enabled and not vlm_model.strip()):
        raise ConfigError("m5.vlm_model must be a non-empty string when VLM is enabled")
    vlm_timeout_s = _positive_number(m5_values.get("vlm_timeout_s", 10.0), "m5.vlm_timeout_s")

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
        violence=ViolenceConfig(
            enabled=violence_enabled,
            model=violence_model,
            revision=violence_revision,
            clip_duration_s=clip_duration_s,
            sample_count=sample_count,
            cadence_s=cadence_s,
            threshold=float(threshold),
            device=violence_device,
            labels=tuple(violence_labels),
            license=violence_license,
            known_limitations=known_limitations,
            checkpoint_sha256=checkpoint_sha256,
        ),
        fusion=FusionConfig(
            strategy=strategy,
            source_roi_policy=source_roi_policy,
            violence_stale_after_s=violence_stale_after_s,
            smoothing_points=smoothing_points,
            allow_crowd_only=allow_crowd_only,
            **weights,
            **thresholds,
            persistence_s=persistence_s,
            hysteresis=hysteresis,
            decay_s=decay_s,
            quiet_period_s=quiet_period_s,
            **severities,
            normalization=FusionNormalizationConfig(**normalization),
        ),
        m5=M5Config(
            evidence_root=evidence_root,
            pre_event_s=pre_event_s,
            post_event_s=post_event_s,
            retention_s=retention_s,
            database_url_env=database_url_env,
            vlm_enabled=vlm_enabled,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            vlm_timeout_s=vlm_timeout_s,
        ),
    )
