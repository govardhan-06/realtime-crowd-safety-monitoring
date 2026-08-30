from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal


BoxXYXY = tuple[float, float, float, float]
PointXY = tuple[float, float]
StageStatus = Literal["available", "degraded", "unavailable"]
FeatureStatus = Literal["available", "insufficient", "unavailable"]


@dataclass(frozen=True)
class FramePacket:
    source_id: str
    frame_index: int
    timestamp_s: float
    image: Any


@dataclass(frozen=True)
class StageHealth:
    stage: str
    status: StageStatus
    model: str | None = None
    device: str | None = None
    detail: str | None = None
    latency_ms: float | None = None
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"available", "degraded", "unavailable"}:
            raise ValueError(f"invalid stage status: {self.status}")
        if self.latency_ms is not None and (not math.isfinite(self.latency_ms) or self.latency_ms < 0):
            raise ValueError("latency_ms must be finite and non-negative")


@dataclass(frozen=True)
class PersonDetection:
    source_id: str
    frame_index: int
    timestamp_s: float
    box_xyxy: BoxXYXY
    confidence: float

    def __post_init__(self) -> None:
        if len(self.box_xyxy) != 4 or any(not math.isfinite(value) for value in self.box_xyxy):
            raise ValueError("box_xyxy must contain four finite values")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")

    @property
    def box(self) -> BoxXYXY:
        return self.box_xyxy


@dataclass(frozen=True)
class DetectionResult:
    detections: tuple[PersonDetection, ...]
    health: StageHealth


@dataclass(frozen=True)
class TrackingResult:
    observations: tuple[TrackObservation, ...]
    health: StageHealth


@dataclass(frozen=True)
class TrackObservation:
    source_id: str
    track_id: int
    frame_index: int
    timestamp_s: float
    center_xy: PointXY
    box_xyxy: BoxXYXY
    confidence: float


@dataclass(frozen=True)
class CrowdFeatureRecord:
    source_id: str
    roi_name: str
    timestamp_s: float
    status: FeatureStatus
    occupancy: int | None = None
    density_proxy: float | None = None
    density_delta: float | None = None
    mean_speed_px_s: float | None = None
    acceleration_px_s2: float | None = None
    speed_variance_px_s2: float | None = None
    direction_disorder: float | None = None
    convergence: float | None = None
    dispersal: float | None = None
    counter_flow: float | None = None
    congestion: float | None = None
    track_count: int = 0
    detail: str | None = None


@dataclass(frozen=True)
class ViolenceEvidence:
    source_id: str
    region_id: str | None
    clip_start_s: float
    clip_end_s: float
    score: float | None
    model: str
    revision: str
    label_mapping: tuple[tuple[str, int], ...]
    status: StageStatus
    latency_ms: float | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.model.strip() or not self.revision.strip():
            raise ValueError("source_id, model, and revision must be non-empty")
        if not math.isfinite(self.clip_start_s) or not math.isfinite(self.clip_end_s) or self.clip_end_s <= self.clip_start_s:
            raise ValueError("violence clip timestamps must be finite and increasing")
        if self.status not in {"available", "degraded", "unavailable"}:
            raise ValueError(f"invalid violence status: {self.status}")
        if self.score is not None and (not math.isfinite(self.score) or not 0.0 <= self.score <= 1.0):
            raise ValueError("violence score must be between zero and one")
        if self.status == "available" and self.score is None:
            raise ValueError("available violence evidence requires a score")
        if self.status == "unavailable" and self.score is not None:
            raise ValueError("unavailable violence evidence cannot include a score")
        if self.latency_ms is not None and (not math.isfinite(self.latency_ms) or self.latency_ms < 0):
            raise ValueError("violence latency_ms must be finite and non-negative")
        if any(not label.strip() or not isinstance(index, int) for label, index in self.label_mapping):
            raise ValueError("label_mapping must contain non-empty labels and integer indexes")
