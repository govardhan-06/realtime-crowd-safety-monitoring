from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import PurePosixPath
from typing import Any, Literal


BoxXYXY = tuple[float, float, float, float]
PointXY = tuple[float, float]
StageStatus = Literal["available", "degraded", "unavailable"]
FeatureStatus = Literal["available", "insufficient", "unavailable"]
FusionStrategy = Literal["violence-only", "crowd-only", "naive-or", "rule-fusion", "temporal"]
IncidentState = Literal["candidate", "active", "escalating", "critical", "resolving", "closed"]
Severity = Literal["low", "medium", "high", "critical"]
EvidenceKind = Literal["snapshot", "pre_event_clip", "post_event_clip", "combined_clip"]
EvidenceStatus = Literal["available", "failed", "unavailable"]
ExplanationStatus = Literal["generated", "disabled", "unavailable", "failed"]
OperatorActionKind = Literal["acknowledge", "dismiss", "escalate"]


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


@dataclass(frozen=True)
class FusionPoint:
    source_id: str
    region_id: str
    timestamp_s: float
    strategy: FusionStrategy
    crowd_features: CrowdFeatureRecord
    crowd_status: FeatureStatus
    violence_score: float | None
    violence_status: StageStatus
    violence_clip_start_s: float | None
    violence_clip_end_s: float | None
    effective_violence_score: float | None
    violence_stale: bool
    normalized_crowd: dict[str, float]
    smoothed_violence: float | None
    smoothed_crowd: float | None
    persistence_s: float
    fused_risk: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class Incident:
    incident_id: str
    source_id: str
    region_id: str
    state: IncidentState
    severity: Severity
    started_at_s: float
    last_updated_at_s: float
    closed_at_s: float | None
    peak_risk: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class IncidentTransition:
    incident_id: str
    source_id: str
    region_id: str
    timestamp_s: float
    from_state: IncidentState
    to_state: IncidentState
    severity: Severity
    cause: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class EvidenceReference:
    kind: EvidenceKind
    relative_path: str
    start_s: float
    end_s: float
    status: EvidenceStatus
    detail: str | None = None

    def __post_init__(self) -> None:
        path = PurePosixPath(self.relative_path)
        if (self.relative_path and path.is_absolute()) or ".." in path.parts:
            raise ValueError("evidence relative_path must stay inside the configured storage root")
        if not math.isfinite(self.start_s) or not math.isfinite(self.end_s) or self.end_s < self.start_s:
            raise ValueError("evidence timestamps must be finite and ordered")
        if self.status == "available" and not self.relative_path:
            raise ValueError("available evidence requires a path")


@dataclass(frozen=True)
class EvidenceManifest:
    run_id: str
    source_id: str
    incident_id: str
    incident_start_s: float
    incident_end_s: float
    pre_event_s: float
    post_event_s: float
    reason_codes: tuple[str, ...]
    timeline: tuple[dict[str, Any], ...]
    stage_health: dict[str, Any]
    artifacts: tuple[EvidenceReference, ...]

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.run_id, self.source_id, self.incident_id)):
            raise ValueError("evidence manifest identifiers must be non-empty")
        if self.incident_end_s < self.incident_start_s or self.pre_event_s < 0 or self.post_event_s < 0:
            raise ValueError("evidence manifest timestamps and bounds are invalid")
        if any(not code.strip() for code in self.reason_codes):
            raise ValueError("evidence reason codes must be non-empty")
        _reject_secret_keys(self.timeline)
        _reject_secret_keys(self.stage_health)


@dataclass(frozen=True)
class IncidentExplanation:
    incident_id: str
    status: ExplanationStatus
    provider: str
    model: str
    text: str = ""
    detail: str | None = None

    def __post_init__(self) -> None:
        if not self.incident_id.strip() or not self.provider.strip():
            raise ValueError("incident explanation identifiers must be non-empty")
        if self.status == "generated" and not self.text.strip():
            raise ValueError("generated explanation requires text")
        if any(token in self.text.lower() for token in ("api_key", "token=", "password", "secret")):
            raise ValueError("explanation text must not contain secret metadata")


@dataclass(frozen=True)
class OperatorAction:
    incident_id: str
    action: OperatorActionKind
    actor: str
    timestamp: str
    note: str | None = None

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.incident_id, self.actor, self.timestamp)):
            raise ValueError("operator action requires incident_id, actor, and timestamp")
        if self.action not in {"acknowledge", "dismiss", "escalate"}:
            raise ValueError(f"unsupported operator action: {self.action}")


def _reject_secret_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if any(token in str(key).lower() for token in ("password", "secret", "token", "api_key", "database_url")):
                raise ValueError("persisted metadata must not contain secrets")
            _reject_secret_keys(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_secret_keys(nested)
