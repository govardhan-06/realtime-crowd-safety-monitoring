from dataclasses import dataclass
import math

from .config import LOIConfig
from .types import CrowdFlowRecord, TrackObservation


def _side(point: tuple[float, float], line: LOIConfig) -> float:
    (x1, y1), (x2, y2) = line.start_xy, line.end_xy
    return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)


@dataclass
class CrowdFlowTracker:
    source_id: str
    roi_name: str
    lois: tuple[LOIConfig, ...]
    interval_s: float

    def __post_init__(self) -> None:
        if not self.source_id or not self.roi_name or self.interval_s <= 0:
            raise ValueError("flow identifiers and interval must be valid")
        self._sides: dict[tuple[str, int], int] = {}
        self._interval_start: float | None = None
        self._counts: dict[str, list[int]] = {loi.name: [0, 0] for loi in self.lois}

    def _record(self, loi: LOIConfig, timestamp_s: float) -> CrowdFlowRecord:
        inflow, outflow = self._counts[loi.name]
        elapsed = max(self.interval_s, timestamp_s - (self._interval_start or timestamp_s))
        factor = 60.0 / elapsed
        return CrowdFlowRecord(
            self.source_id, self.roi_name, loi.name, timestamp_s, "available",
            inflow, outflow, inflow * factor, outflow * factor, (inflow - outflow) * factor,
        )

    def update(self, timestamp_s: float, observations: list[TrackObservation] | tuple[TrackObservation, ...]) -> tuple[CrowdFlowRecord, ...]:
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._interval_start is None:
            self._interval_start = timestamp_s
        if timestamp_s < self._interval_start:
            raise ValueError("flow timestamps must be ordered")
        for observation in observations:
            for loi in self.lois:
                key = (loi.name, observation.track_id)
                current_side = _side(observation.center_xy, loi)
                previous_side = self._sides.get(key)
                if previous_side is not None and current_side * previous_side < 0:
                    # Positive-to-negative is inflow; negative-to-positive is outflow.
                    self._counts[loi.name][0 if previous_side > 0 else 1] += 1
                if current_side > 0:
                    self._sides[key] = 1
                elif current_side < 0:
                    self._sides[key] = -1
        if timestamp_s - self._interval_start < self.interval_s:
            return ()
        records = tuple(self._record(loi, timestamp_s) for loi in self.lois)
        self._interval_start = timestamp_s
        self._counts = {loi.name: [0, 0] for loi in self.lois}
        return records

    def flush(self, timestamp_s: float) -> tuple[CrowdFlowRecord, ...]:
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._interval_start is None or timestamp_s <= self._interval_start:
            return ()
        records = tuple(self._record(loi, timestamp_s) for loi in self.lois)
        self._interval_start = timestamp_s
        self._counts = {loi.name: [0, 0] for loi in self.lois}
        return records
