from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import FusionConfig
from .types import CrowdFeatureRecord, FusionPoint, FusionStrategy, ViolenceEvidence


_FEATURES = (
    "density_delta",
    "mean_speed_px_s",
    "acceleration_px_s2",
    "direction_disorder",
    "convergence",
    "dispersal",
    "counter_flow",
    "congestion",
)
FUSION_VERSION = "m4-temporal-fusion-v1"
FUSION_STRATEGIES = ("violence-only", "crowd-only", "naive-or", "rule-fusion", "temporal")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _normalise(value: float | None, bounds: tuple[float, float]) -> float | None:
    if value is None:
        return None
    lower, upper = bounds
    return _clamp((value - lower) / (upper - lower))


def _violence_for(
    record: CrowdFeatureRecord,
    evidence: list[ViolenceEvidence],
    stale_after_s: float,
) -> tuple[ViolenceEvidence | None, bool]:
    matching = [
        item for item in evidence
        if item.source_id == record.source_id
        and (item.region_id is None or item.region_id == record.roi_name)
        and item.clip_end_s <= record.timestamp_s + 1e-9
    ]
    if not matching:
        return None, False
    selected = max(matching, key=lambda item: (item.clip_end_s, item.clip_start_s))
    return selected, record.timestamp_s - selected.clip_end_s > stale_after_s


def _components(record: CrowdFeatureRecord, config: FusionConfig) -> tuple[dict[str, float], float | None]:
    normalized = {
        name: _normalise(getattr(record, name), getattr(config.normalization, name))
        for name in _FEATURES
    }
    values = {name: value for name, value in normalized.items() if value is not None}
    if record.status != "available" or not values:
        return {}, None
    density = max(0.0, values.get("density_delta", 0.0))
    movement_values = [values[name] for name in ("mean_speed_px_s", "acceleration_px_s2", "direction_disorder", "counter_flow") if name in values]
    context_values = [values[name] for name in ("convergence", "dispersal", "congestion") if name in values]
    normalized.update({
        "density_risk": density,
        "movement_risk": sum(movement_values) / len(movement_values) if movement_values else 0.0,
        "context_risk": max(context_values, default=0.0),
    })
    risk = (density + normalized["movement_risk"] + normalized["context_risk"]) / 3.0
    return normalized, risk


def _weighted(parts: list[tuple[float, float]]) -> float:
    available = [(value, weight) for value, weight in parts if weight > 0]
    total_weight = sum(weight for _, weight in available)
    return sum(value * weight for value, weight in available) / total_weight if total_weight else 0.0


@dataclass
class FusionBuilder:
    config: FusionConfig
    strategy: FusionStrategy | None = None

    def __post_init__(self) -> None:
        self.strategy = self.strategy or self.config.strategy
        if self.strategy not in FUSION_STRATEGIES:
            raise ValueError(f"unknown fusion strategy: {self.strategy}")
        self._seen: set[tuple[str, str, float]] = set()
        self._violence: list[ViolenceEvidence] = []
        self._history: dict[tuple[str, str], list[tuple[float, float | None, float | None]]] = {}
        self._positive_since: dict[tuple[str, str], float | None] = {}

    def add(
        self,
        record: CrowdFeatureRecord,
        violence: ViolenceEvidence | None = None,
    ) -> FusionPoint | None:
        key = (record.source_id, record.roi_name, record.timestamp_s)
        if violence is not None and violence not in self._violence:
            self._violence.append(violence)
        if key in self._seen:
            return None
        self._seen.add(key)
        matched, stale = _violence_for(record, self._violence, self.config.violence_stale_after_s)
        normalized, crowd_risk = _components(record, self.config)
        violence_score = matched.score if matched else None
        violence_status = matched.status if matched else "unavailable"
        effective_violence_score = violence_score if matched and not stale else None
        violence_history = self._history.setdefault(key[:2], [])
        previous = [item[1] for item in violence_history[-self.config.smoothing_points + 1:] if item[1] is not None]
        smoothed_violence = (sum(previous + [effective_violence_score]) / len(previous + [effective_violence_score])) if effective_violence_score is not None else None
        crowd_history = [item[2] for item in violence_history[-self.config.smoothing_points + 1:] if item[2] is not None]
        smoothed_crowd = (sum(crowd_history + [crowd_risk]) / len(crowd_history + [crowd_risk])) if crowd_risk is not None else None
        if matched is None:
            reasons = ["violence_unavailable"]
        elif stale:
            reasons = ["violence_stale"]
        else:
            reasons = ["violence_degraded" if matched.status == "degraded" else "violence_available"]
        if record.status == "unavailable":
            reasons.append("crowd_unavailable")
        elif record.status == "insufficient":
            reasons.append("crowd_insufficient")
        elif normalized:
            if normalized["density_risk"] >= self.config.candidate_threshold:
                reasons.append("crowd_density")
            if normalized["movement_risk"] >= self.config.candidate_threshold:
                reasons.append("crowd_movement")
            if normalized["context_risk"] >= self.config.candidate_threshold:
                reasons.append("crowd_context")
        if effective_violence_score is not None and smoothed_crowd is not None:
            reasons.append("combined_evidence")
        seed = max(smoothed_violence or 0.0, smoothed_crowd or 0.0)
        positive_since = self._positive_since.get(key[:2])
        if seed >= self.config.candidate_threshold:
            positive_since = record.timestamp_s if positive_since is None else positive_since
        elif positive_since is not None and record.timestamp_s - positive_since > self.config.decay_s:
            positive_since = None
        self._positive_since[key[:2]] = positive_since
        persistence = (
            _clamp((record.timestamp_s - positive_since) / self.config.persistence_s)
            if positive_since is not None else 0.0
        )
        if persistence > 0:
            reasons.append("persistent_signal")
        strategy = self.strategy
        if strategy == "violence-only":
            risk = effective_violence_score or 0.0
        elif strategy == "crowd-only":
            risk = crowd_risk or 0.0
        elif strategy == "naive-or":
            risk = max(effective_violence_score or 0.0, crowd_risk or 0.0)
        elif strategy == "rule-fusion":
            risk = max(effective_violence_score or 0.0, crowd_risk or 0.0)
            if effective_violence_score is not None and crowd_risk is not None:
                risk = _clamp(risk + 0.15)
        else:
            if not self.config.allow_crowd_only and effective_violence_score is None:
                risk = 0.0
                reasons.append("violence_unavailable_suppressed")
            else:
                risk = _weighted([
                    (smoothed_violence, self.config.violence_weight) for smoothed_violence in [smoothed_violence] if smoothed_violence is not None
                ] + [
                    (normalized.get("density_risk", 0.0), self.config.density_weight),
                    (normalized.get("movement_risk", 0.0), self.config.movement_weight),
                    (normalized.get("context_risk", 0.0), self.config.context_weight),
                    (persistence, self.config.persistence_weight),
                ])
        point = FusionPoint(
            record.source_id, record.roi_name, record.timestamp_s, strategy, record, record.status,
            violence_score, violence_status,
            matched.clip_start_s if matched else None,
            matched.clip_end_s if matched else None,
            effective_violence_score, stale,
            normalized, smoothed_violence, smoothed_crowd, persistence, _clamp(risk), tuple(reasons),
        )
        violence_history.append((record.timestamp_s, effective_violence_score, crowd_risk))
        if len(violence_history) > self.config.smoothing_points:
            del violence_history[:-self.config.smoothing_points]
        return point


def build_fusion_points(
    crowd_records: Iterable[CrowdFeatureRecord],
    violence_evidence: Iterable[ViolenceEvidence],
    config: FusionConfig,
    strategy: FusionStrategy | None = None,
) -> tuple[FusionPoint, ...]:
    builder = FusionBuilder(config, strategy)
    evidence = sorted(violence_evidence, key=lambda item: (item.source_id, item.clip_end_s, item.clip_start_s))
    builder._violence.extend(evidence)
    rows: list[FusionPoint] = []
    for record in crowd_records:
        point = builder.add(record)
        if point is not None:
            rows.append(point)
    return tuple(rows)
