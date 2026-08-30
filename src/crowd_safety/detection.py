from hashlib import sha256
from pathlib import Path
import time
from typing import Any, Protocol

from .types import DetectionResult, FramePacket, PersonDetection, StageHealth


class PersonDetector(Protocol):
    def detect(self, packet: FramePacket) -> DetectionResult: ...


def _checkpoint_hash(model: str) -> str | None:
    path = Path(model).expanduser()
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UltralyticsPersonDetector:
    def __init__(
        self,
        model: str = "yolo26n.pt",
        confidence: float = 0.25,
        device: str = "auto",
        person_class_id: int = 0,
        *,
        model_object: Any | None = None,
        model_error: str | None = None,
    ) -> None:
        self.model_name = model
        self.confidence = confidence
        self.device = self._select_device(device)
        self.person_class_id = person_class_id
        self.model = model_object
        self.model_error = model_error
        if self.model is None and self.model_error is None:
            try:
                from ultralytics import YOLO

                self.model = YOLO(model)
            except Exception as exc:  # model import/download is an optional runtime capability
                self.model_error = f"could not load detector model: {exc}"
        self.checkpoint_sha256 = _checkpoint_hash(self.model_name)

    @property
    def available(self) -> bool:
        return self.model is not None and self.model_error is None

    @staticmethod
    def _select_device(configured: str) -> str:
        if configured != "auto":
            return configured
        try:
            import torch

            return "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            return "cpu"

    def detect(self, packet: FramePacket) -> DetectionResult:
        started = time.perf_counter()
        if not self.available:
            return DetectionResult((), StageHealth(
                "detector", "unavailable", self.model_name, self.device,
                self.model_error or "detector model unavailable", _elapsed_ms(started),
                self.checkpoint_sha256,
            ))
        try:
            results = self.model(
                packet.image, conf=self.confidence, classes=[self.person_class_id],
                device=self.device, verbose=False,
            )
            detections = tuple(self._detections(results, packet))
            health = StageHealth(
                "detector", "available", self.model_name, self.device,
                latency_ms=_elapsed_ms(started), checkpoint_sha256=self.checkpoint_sha256,
            )
            return DetectionResult(detections, health)
        except Exception as exc:
            return DetectionResult((), StageHealth(
                "detector", "degraded", self.model_name, self.device,
                f"inference failed: {exc}", _elapsed_ms(started), self.checkpoint_sha256,
            ))

    def _detections(self, results: Any, packet: FramePacket) -> list[PersonDetection]:
        boxes = results[0].boxes if isinstance(results, (list, tuple)) else results.boxes
        coordinates = _values(boxes.xyxy)
        confidences = _values(boxes.conf)
        classes = _values(boxes.cls) if hasattr(boxes, "cls") else [self.person_class_id] * len(coordinates)
        return [
            PersonDetection(packet.source_id, packet.frame_index, packet.timestamp_s, tuple(map(float, box)), float(confidence))
            for box, confidence, class_id in zip(coordinates, confidences, classes)
            if int(class_id) == self.person_class_id
        ]


def _values(value: Any) -> list[Any]:
    return value.cpu().tolist() if hasattr(value, "cpu") else value.tolist()


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000.0
