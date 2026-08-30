import time
from typing import Any, Callable, Protocol

from .types import FramePacket, PersonDetection, StageHealth, TrackObservation, TrackingResult


class Tracker(Protocol):
    def update(self, packet: FramePacket, detections: tuple[PersonDetection, ...] = ()) -> TrackingResult: ...


class ByteTrackTracker:
    """Project-owned boundary for Ultralytics' persistent ByteTrack results."""

    def __init__(
        self,
        model: Any | None = None,
        tracker_config: str = "bytetrack.yaml",
        track_buffer: int = 30,
        confidence: float = 0.25,
        device: str = "auto",
        person_class_id: int = 0,
        *,
        track_fn: Callable[[FramePacket], Any] | None = None,
    ) -> None:
        self.model = model
        self.tracker_config = tracker_config
        self.track_buffer = track_buffer
        self.confidence = confidence
        self.device = device
        self.person_class_id = person_class_id
        self.track_fn = track_fn
        self._tracker = self._build_tracker() if model is not None and track_fn is None else None

    def _build_tracker(self) -> Any | None:
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker
            from ultralytics.utils import IterableSimpleNamespace, YAML
            from ultralytics.utils.checks import check_yaml

            values = YAML.load(check_yaml(self.tracker_config))
            values["track_buffer"] = self.track_buffer
            args = IterableSimpleNamespace(**values)
            args.device = self.device
            return BYTETracker(args=args)
        except Exception:
            return None

    def update(self, packet: FramePacket, detections: tuple[PersonDetection, ...] = ()) -> TrackingResult:
        started = time.perf_counter()
        if self.track_fn is not None:
            try:
                raw_tracks = self.track_fn(packet)
                observations = tuple(self._observations(raw_tracks, packet))
                return TrackingResult(observations, StageHealth(
                    "tracker", "available", "ByteTrack", self.device,
                    latency_ms=_elapsed_ms(started),
                ))
            except Exception as exc:
                return TrackingResult((), StageHealth(
                    "tracker", "degraded", "ByteTrack", self.device,
                    f"tracking failed: {exc}", _elapsed_ms(started),
                ))
        if self._tracker is None:
            return TrackingResult((), StageHealth(
                "tracker", "unavailable", "ByteTrack", self.device,
                "Ultralytics tracking model is unavailable", _elapsed_ms(started),
            ))
        try:
            import torch
            from ultralytics.engine.results import Boxes

            data = [
                [*detection.box_xyxy, detection.confidence, float(self.person_class_id)]
                for detection in detections
            ]
            tensor = torch.tensor(data, dtype=torch.float32) if data else torch.zeros((0, 6), dtype=torch.float32)
            boxes = Boxes(tensor, packet.image.shape[:2])
            rows = self._tracker.update(boxes, packet.image)
            observations = tuple(self._row_observations(rows, packet))
            return TrackingResult(observations, StageHealth(
                "tracker", "available", "ByteTrack", self.device,
                latency_ms=_elapsed_ms(started),
            ))
        except Exception as exc:
            return TrackingResult((), StageHealth(
                "tracker", "degraded", "ByteTrack", self.device,
                f"tracking failed: {exc}", _elapsed_ms(started),
            ))

    def _observations(self, results: Any, packet: FramePacket) -> list[TrackObservation]:
        if isinstance(results, (list, tuple)) and (not results or not _is_simple_track(results[0])):
            boxes = results[0].boxes if results else None
            if boxes is not None:
                coordinates = _values(boxes.xyxy)
                confidences = _values(boxes.conf)
                ids = _values(boxes.id) if getattr(boxes, "id", None) is not None else []
                return [self._observation(packet, track_id, box, confidence) for track_id, box, confidence in zip(ids, coordinates, confidences)]
        return [self._observation(packet, track_id, box, confidence) for track_id, box, confidence in results]

    def _row_observations(self, rows: Any, packet: FramePacket) -> list[TrackObservation]:
        return [
            self._observation(packet, row[4], row[:4], row[5])
            for row in rows
            if int(row[6]) == self.person_class_id
        ]

    @staticmethod
    def _observation(packet: FramePacket, track_id: Any, box: Any, confidence: Any) -> TrackObservation:
        x1, y1, x2, y2 = map(float, box)
        return TrackObservation(
            packet.source_id, int(track_id), packet.frame_index, packet.timestamp_s,
            ((x1 + x2) / 2, (y1 + y2) / 2), (x1, y1, x2, y2), float(confidence),
        )


def _is_simple_track(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3


def _values(value: Any) -> list[Any]:
    return value.cpu().tolist() if hasattr(value, "cpu") else value.tolist()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
