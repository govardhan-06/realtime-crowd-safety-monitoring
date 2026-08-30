from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Any, Protocol

from .types import FramePacket, StageHealth, ViolenceEvidence


@dataclass(frozen=True)
class ClipWindow:
    packets: tuple[FramePacket, ...]
    sampled_packets: tuple[FramePacket, ...]
    start_s: float
    end_s: float


def _sample_packets(packets: tuple[FramePacket, ...], sample_count: int) -> tuple[FramePacket, ...]:
    if len(packets) <= sample_count:
        return packets
    return tuple(
        packets[int(index * (len(packets) - 1) / (sample_count - 1) + 0.5)]
        for index in range(sample_count)
    )


class RollingClipBuffer:
    def __init__(self, duration_s: float, sample_count: int) -> None:
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("duration_s must be greater than zero")
        if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 2:
            raise ValueError("sample_count must be an integer greater than or equal to two")
        self.duration_s = duration_s
        self.sample_count = sample_count
        self._packets: deque[FramePacket] = deque()

    @property
    def packets(self) -> tuple[FramePacket, ...]:
        return tuple(self._packets)

    def append(self, packet: FramePacket) -> None:
        if self._packets and packet.source_id != self._packets[-1].source_id:
            raise ValueError("a clip buffer can contain only one source")
        if self._packets and packet.timestamp_s < self._packets[-1].timestamp_s:
            raise ValueError("clip packet timestamps must be monotonic")
        if self._packets and packet.timestamp_s == self._packets[-1].timestamp_s:
            return
        self._packets.append(packet)
        cutoff = packet.timestamp_s - self.duration_s
        # ponytail: retain one boundary anchor for timestamp jitter; exact-duration pruning can never form a clip.
        while len(self._packets) > 1 and self._packets[1].timestamp_s < cutoff:
            self._packets.popleft()

    def complete_window(self, end_s: float | None = None) -> ClipWindow | None:
        if not self._packets:
            return None
        end = self._packets[-1].timestamp_s if end_s is None else end_s
        packets = tuple(packet for packet in self._packets if packet.timestamp_s <= end)
        if len(packets) < 2 or packets[-1].timestamp_s - packets[0].timestamp_s < self.duration_s:
            return None
        return ClipWindow(
            packets=packets,
            sampled_packets=_sample_packets(packets, self.sample_count),
            start_s=packets[0].timestamp_s,
            end_s=packets[-1].timestamp_s,
        )


class ViolenceCadence:
    def __init__(self, interval_s: float) -> None:
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("interval_s must be greater than zero")
        self.interval_s = interval_s
        self._last_inference_s: float | None = None

    def is_due(self, timestamp_s: float) -> bool:
        if not math.isfinite(timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if self._last_inference_s is not None and timestamp_s < self._last_inference_s:
            raise ValueError("cadence timestamps must be monotonic")
        if self._last_inference_s is not None and timestamp_s - self._last_inference_s < self.interval_s:
            return False
        self._last_inference_s = timestamp_s
        return True


class ViolenceClassifier(Protocol):
    def infer(self, window: ClipWindow) -> ViolenceEvidence:
        ...


class VideoMAEViolenceClassifier:
    """Project-owned adapter for a binary Hugging Face video classifier."""

    def __init__(
        self,
        model: str,
        revision: str,
        *,
        device: str = "auto",
        labels: tuple[str, ...] = (),
        license_name: str = "",
        known_limitations: str = "",
        checkpoint_sha256: str | None = None,
        processor: Any | None = None,
        model_instance: Any | None = None,
        load_error: str | None = None,
    ) -> None:
        self.model_name = model
        self.revision = revision
        self.device = device
        self.fallback_labels = labels
        self.license_name = license_name
        self.known_limitations = known_limitations
        self.checkpoint_sha256 = checkpoint_sha256
        self.processor = processor
        self.model_instance = model_instance
        self._load_error = load_error
        self._label_mapping: tuple[tuple[str, int], ...] = ()
        self._violent_index: int | None = None
        self._health = StageHealth(
            "violence", "unavailable", model=model, device=device,
            detail=load_error, checkpoint_sha256=checkpoint_sha256,
        )
        if self._load_error is None and (processor is None or model_instance is None):
            self._load()
        if self._load_error is None:
            try:
                self._label_mapping, self._violent_index = self._resolve_labels(model_instance, labels)
                self.device = self._resolve_device(device)
                if hasattr(self.model_instance, "eval"):
                    self.model_instance.eval()
                if hasattr(self.model_instance, "to"):
                    self.model_instance.to(self.device)
                self._health = StageHealth(
                    "violence", "available", model=model, device=self.device,
                    detail="model loaded; inference not yet measured", checkpoint_sha256=checkpoint_sha256,
                )
            except Exception as exc:
                self._load_error = str(exc)
                self._health = StageHealth(
                    "violence", "unavailable", model=model, device=device,
                    detail=str(exc), checkpoint_sha256=checkpoint_sha256,
                )

    @property
    def health(self) -> StageHealth:
        return self._health

    @property
    def label_mapping(self) -> tuple[tuple[str, int], ...]:
        return self._label_mapping

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "revision": self.revision,
            "license": self.license_name,
            "known_limitations": self.known_limitations,
            "checkpoint_sha256": self.checkpoint_sha256,
            "label_mapping": [list(item) for item in self._label_mapping],
        }

    def _load(self) -> None:
        try:
            from transformers import AutoImageProcessor, AutoModelForVideoClassification
            import torch

            self.processor = AutoImageProcessor.from_pretrained(self.model_name, revision=self.revision)
            self.model_instance = AutoModelForVideoClassification.from_pretrained(
                self.model_name, revision=self.revision
            )
            self._torch = torch
        except Exception as exc:
            self._load_error = f"could not load violence model: {exc}"
            self._health = StageHealth(
                "violence", "unavailable", model=self.model_name, device=self.device,
                detail=self._load_error, checkpoint_sha256=self.checkpoint_sha256,
            )

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        torch = getattr(self, "_torch", None)
        if torch is None:
            import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _resolve_labels(model: Any, fallback_labels: tuple[str, ...]) -> tuple[tuple[tuple[str, int], ...], int]:
        raw_mapping = getattr(getattr(model, "config", None), "id2label", None)
        if isinstance(raw_mapping, dict) and raw_mapping:
            mapping = tuple(
                (str(label), int(index))
                for index, label in sorted(raw_mapping.items(), key=lambda item: int(item[0]))
            )
            if fallback_labels and mapping != tuple((label, index) for index, label in enumerate(fallback_labels)):
                raise ValueError("configured violence labels do not match model id2label mapping")
        elif fallback_labels:
            mapping = tuple((label, index) for index, label in enumerate(fallback_labels))
        else:
            raise ValueError("violence model has no id2label mapping or configured labels")
        if len(mapping) != 2 or {index for _, index in mapping} != {0, 1}:
            raise ValueError("violence model label mapping must contain exactly two output labels indexed zero and one")
        violent = [
            index for label, index in mapping
            if any(token in label.lower().replace("_", " ").split() for token in ("unsafe", "violence", "violent"))
        ]
        if len(violent) != 1:
            raise ValueError("violence model label mapping must contain exactly one unsafe/violence label")
        return mapping, violent[0]

    @staticmethod
    def _rgb_frame(image: Any) -> Any:
        from PIL import Image

        if hasattr(image, "convert"):
            return image.convert("RGB")
        if hasattr(image, "detach"):
            image = image.detach().cpu().numpy()
        if getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
            raise ValueError("violence frames must be HxWx3 images")
        return Image.fromarray(image[..., ::-1].copy()).convert("RGB")

    def _evidence(
        self,
        window: ClipWindow,
        score: float | None,
        status: str,
        latency_ms: float | None,
        detail: str | None = None,
    ) -> ViolenceEvidence:
        return ViolenceEvidence(
            window.packets[0].source_id,
            None,
            window.start_s,
            window.end_s,
            score,
            self.model_name,
            self.revision,
            self._label_mapping,
            status,
            latency_ms,
            detail,
        )

    def infer(self, window: ClipWindow) -> ViolenceEvidence:
        if self._load_error is not None:
            return self._evidence(window, None, "unavailable", None, self._load_error)
        started = time.perf_counter()
        try:
            frames = [self._rgb_frame(packet.image) for packet in window.sampled_packets]
            inputs = self.processor(frames, return_tensors="pt")
            inputs = {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            torch = getattr(self, "_torch", None)
            if torch is None:
                import torch
            with torch.inference_mode():
                output = self.model_instance(**inputs)
                probabilities = torch.softmax(output.logits, dim=-1)[0]
            score = float(probabilities[self._violent_index].detach().cpu().item())
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._health = StageHealth(
                "violence", "available", model=self.model_name, device=self.device,
                latency_ms=latency_ms, detail="binary unsafe-label probability",
                checkpoint_sha256=self.checkpoint_sha256,
            )
            return self._evidence(window, score, "available", latency_ms)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._health = StageHealth(
                "violence", "degraded", model=self.model_name, device=self.device,
                latency_ms=latency_ms, detail=str(exc), checkpoint_sha256=self.checkpoint_sha256,
            )
            return self._evidence(window, None, "degraded", latency_ms, str(exc))
