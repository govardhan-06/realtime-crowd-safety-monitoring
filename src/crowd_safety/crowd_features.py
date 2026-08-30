from collections import defaultdict
import math
from typing import Iterable

from .config import ROIConfig
from .types import CrowdFeatureRecord, TrackObservation


def _point_in_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    x, y = point
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if abs(cross) <= 1e-9 and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def _area(polygon: tuple[tuple[float, float], ...]) -> float:
    return abs(sum(
        polygon[index][0] * polygon[(index + 1) % len(polygon)][1]
        - polygon[(index + 1) % len(polygon)][0] * polygon[index][1]
        for index in range(len(polygon))
    )) / 2.0


def _velocity_pairs(history: list[TrackObservation]) -> list[tuple[float, float, float]]:
    pairs = []
    for previous, current in zip(history, history[1:]):
        elapsed = current.timestamp_s - previous.timestamp_s
        if elapsed <= 0:
            continue
        vx = (current.center_xy[0] - previous.center_xy[0]) / elapsed
        vy = (current.center_xy[1] - previous.center_xy[1]) / elapsed
        pairs.append((vx, vy, elapsed))
    return pairs


def compute_crowd_features(
    tracks: Iterable[TrackObservation],
    roi: ROIConfig,
    timestamp_s: float,
    *,
    window_s: float,
    min_track_history: int,
    min_speed_px_s: float = 1.0,
    congestion_occupancy: int = 5,
    congestion_speed_px_s: float = 2.0,
    source_id: str | None = None,
) -> CrowdFeatureRecord:
    cutoff = timestamp_s - window_s
    by_track: dict[int, list[TrackObservation]] = defaultdict(list)
    for observation in tracks:
        if observation.source_id and observation.timestamp_s <= timestamp_s:
            by_track[observation.track_id].append(observation)

    histories: list[list[TrackObservation]] = []
    for history in by_track.values():
        history.sort(key=lambda observation: observation.timestamp_s)
        recent = [observation for observation in history if observation.timestamp_s >= cutoff]
        prior = [observation for observation in history if observation.timestamp_s < cutoff]
        if recent and _point_in_polygon(recent[-1].center_xy, roi.polygon):
            histories.append(([prior[-1]] if prior else []) + recent)

    occupancy = len(histories)
    density_proxy = occupancy / max(_area(roi.polygon), 1.0)
    baseline_seen = False
    baseline_occupancy = 0
    for history in by_track.values():
        prior = [observation for observation in history if observation.timestamp_s <= cutoff]
        if prior:
            baseline_seen = True
            baseline_occupancy += int(_point_in_polygon(prior[-1].center_xy, roi.polygon))
    density_delta = (
        (occupancy - baseline_occupancy) / max(baseline_occupancy, 1)
        if baseline_seen else None
    )
    source_id = source_id or (next(iter(by_track.values()))[0].source_id if by_track else "")
    if not histories:
        return CrowdFeatureRecord(
            source_id, roi.name, timestamp_s, "insufficient", occupancy=0, density_proxy=density_proxy,
            density_delta=density_delta,
            detail="no current tracks in ROI",
        )
    velocities: list[tuple[float, float]] = []
    accelerations: list[float] = []
    directions: list[tuple[float, float]] = []
    for history in histories:
        pairs = _velocity_pairs(history)
        recent_count = sum(observation.timestamp_s >= cutoff for observation in history)
        if recent_count < min_track_history or not pairs:
            continue
        velocities.extend((vx, vy) for vx, vy, _ in pairs)
        directions.extend(
            (vx / speed, vy / speed)
            for vx, vy, _ in pairs
            if (speed := math.hypot(vx, vy)) >= min_speed_px_s
        )
        speeds = [math.hypot(vx, vy) for vx, vy, _ in pairs]
        accelerations.extend(
            (speeds[index] - speeds[index - 1]) / pairs[index][2]
            for index in range(1, len(speeds))
        )

    if len(velocities) == 0 or any(
        sum(observation.timestamp_s >= cutoff for observation in history) < min_track_history
        for history in histories
    ):
        return CrowdFeatureRecord(
            histories[0][0].source_id, roi.name, timestamp_s, "insufficient", occupancy=occupancy,
            density_proxy=density_proxy, density_delta=density_delta,
            detail="track history is shorter than configured feature window",
        )

    speeds = [math.hypot(vx, vy) for vx, vy in velocities]
    mean_speed = sum(speeds) / len(speeds)
    mean_speed_sq = sum(speed * speed for speed in speeds) / len(speeds)
    speed_variance = max(0.0, mean_speed_sq - mean_speed * mean_speed)
    if directions:
        resultant = math.hypot(sum(x for x, _ in directions), sum(y for _, y in directions)) / len(directions)
        direction_disorder = 1.0 - resultant
    else:
        direction_disorder = 0.0

    current_centroid = (
        sum(history[-1].center_xy[0] for history in histories) / occupancy,
        sum(history[-1].center_xy[1] for history in histories) / occupancy,
    )
    toward = 0
    away = 0
    for history in histories:
        vx, vy, _ = _velocity_pairs(history)[-1]
        dx = current_centroid[0] - history[-1].center_xy[0]
        dy = current_centroid[1] - history[-1].center_xy[1]
        dot = vx * dx + vy * dy
        if dot > 0:
            toward += 1
        elif dot < 0:
            away += 1
    convergence = toward / occupancy
    dispersal = away / occupancy

    horizontal = [vx for vx, vy in velocities if abs(vx) >= abs(vy) and abs(vx) >= min_speed_px_s]
    vertical = [vy for vx, vy in velocities if abs(vy) > abs(vx) and abs(vy) >= min_speed_px_s]
    dominant = horizontal or vertical
    positive = sum(value > 0 for value in dominant)
    negative = sum(value < 0 for value in dominant)
    counter_flow = 2.0 * min(positive, negative) / len(dominant) if dominant else 0.0
    congestion = 1.0 if occupancy >= congestion_occupancy and mean_speed <= congestion_speed_px_s else 0.0

    source_id = histories[0][0].source_id
    return CrowdFeatureRecord(
        source_id, roi.name, timestamp_s, "available", occupancy, density_proxy, density_delta,
        mean_speed, sum(accelerations) / len(accelerations) if accelerations else None,
        speed_variance, direction_disorder, convergence, dispersal, counter_flow, congestion,
        occupancy,
    )
