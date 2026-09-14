from collections import deque
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Protocol

from .types import FramePacket, StageHealth, ViolenceEvidence


def _load_x3d_checkpoint(path: Path, torch: Any) -> Any:
    import numpy as np

    numpy_core = np._core if hasattr(np, "_core") else np.core
    safe_globals = [
        numpy_core.multiarray.scalar,
        np.dtype,
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
    ]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(path, map_location="cpu", weights_only=True)


def _normalize_x3d_state_dict(state_dict: Any) -> Any:
    if not isinstance(state_dict, dict) or not state_dict:
        return state_dict
    if all(isinstance(key, str) and key.startswith("backbone.") for key in state_dict):
        return {key.removeprefix("backbone."): value for key, value in state_dict.items()}
    return state_dict


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


class X3DViolenceClassifier:
    """Adapter for the verified X3D-M binary checkpoint."""

    def __init__(
        self,
        repository: str,
        checkpoint: str,
        revision: str,
        *,
        architecture: str = "x3d_m",
        device: str = "auto",
        sample_count: int = 16,
        labels: tuple[str, ...] = ("non-violent", "violent"),
        license_name: str = "",
        known_limitations: str = "",
        checkpoint_sha256: str | None = None,
        checkpoint_path: str | Path | None = None,
        model_instance: Any | None = None,
        load_error: str | None = None,
    ) -> None:
        self.repository = repository
        self.checkpoint = checkpoint
        self.revision = revision
        self.architecture = architecture
        self.device = device
        self.sample_count = sample_count
        self.fallback_labels = labels
        self.license_name = license_name
        self.known_limitations = known_limitations
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.model_instance = model_instance
        self.model_name = f"{repository}:{checkpoint}"
        self._load_error = load_error
        self._label_mapping: tuple[tuple[str, int], ...] = ()
        self._violent_index: int | None = None
        self._health = StageHealth(
            "violence", "unavailable", model=self.model_name, device=device,
            detail=load_error, checkpoint_sha256=checkpoint_sha256,
        )
        if self._load_error is None and self.checkpoint_path is not None:
            try:
                self._verify_checkpoint(self.checkpoint_path)
            except Exception as exc:
                self._load_error = str(exc)
        if self._load_error is None and self.model_instance is None:
            self._load()
        if self._load_error is None:
            try:
                self._label_mapping, self._violent_index = VideoMAEViolenceClassifier._resolve_labels(
                    self.model_instance, labels
                )
                self.device = self._resolve_device(device)
                if hasattr(self.model_instance, "eval"):
                    self.model_instance.eval()
                if hasattr(self.model_instance, "to"):
                    self.model_instance.to(self.device)
                self._health = StageHealth(
                    "violence", "available", model=self.model_name, device=self.device,
                    detail="X3D-M model loaded; inference not yet measured",
                    checkpoint_sha256=checkpoint_sha256,
                )
            except Exception as exc:
                self._load_error = str(exc)
        if self._load_error is not None:
            self._health = StageHealth(
                "violence", "unavailable", model=self.model_name, device=device,
                detail=self._load_error, checkpoint_sha256=checkpoint_sha256,
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
            "backend": "x3d",
            "repository": self.repository,
            "checkpoint": self.checkpoint,
            "model": self.model_name,
            "revision": self.revision,
            "architecture": self.architecture,
            "sample_count": self.sample_count,
            "labels": list(self.fallback_labels),
            "license": self.license_name,
            "known_limitations": self.known_limitations,
            "checkpoint_sha256": self.checkpoint_sha256,
            "label_mapping": [list(item) for item in self._label_mapping],
        }

    def _verify_checkpoint(self, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"X3D checkpoint does not exist: {path}")
        if self.checkpoint_sha256 is None:
            raise ValueError("X3D checkpoint checksum is required")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != self.checkpoint_sha256.lower():
            raise ValueError(
                f"X3D checkpoint checksum mismatch: expected {self.checkpoint_sha256}, got {digest.hexdigest()}"
            )

    def _load(self) -> None:
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from pytorchvideo.models.hub import x3d_m

            path = self.checkpoint_path
            if path is None:
                path = Path(hf_hub_download(
                    repo_id=self.repository,
                    filename=self.checkpoint,
                    revision=self.revision,
                ))
            self._verify_checkpoint(path)
            model = x3d_m(pretrained=False)
            model.blocks[5].proj = torch.nn.Linear(2048, 2)
            checkpoint = _load_x3d_checkpoint(path, torch)
            state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
            state_dict = _normalize_x3d_state_dict(state_dict)
            model.load_state_dict(state_dict)
            self.checkpoint_path = path
            self.model_instance = model
            self._torch = torch
        except Exception as exc:
            self._load_error = f"could not load X3D violence model: {exc}"

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _preprocess(window: ClipWindow, sample_count: int, device: str) -> Any:
        if len(window.sampled_packets) != sample_count:
            raise ValueError(f"X3D requires exactly {sample_count} sampled frames")
        import numpy as np
        import torch
        from PIL import Image

        frames = []
        for packet in window.sampled_packets:
            image = packet.image
            if hasattr(image, "detach"):
                image = image.detach().cpu().numpy()
            if getattr(image, "ndim", 0) != 3 or image.shape[2] != 3:
                raise ValueError("violence frames must be HxWx3 images")
            rgb = Image.fromarray(np.asarray(image[..., ::-1].copy(), dtype=np.uint8), mode="RGB")
            rgb = rgb.resize((224, 224), Image.Resampling.BILINEAR)
            frame = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0).permute(2, 0, 1)
            frames.append(frame)
        tensor = torch.stack(frames).permute(1, 0, 2, 3)
        tensor = (tensor - 0.45) / 0.225
        return tensor.unsqueeze(0).to(device)

    def _evidence(
        self,
        window: ClipWindow,
        score: float | None,
        status: str,
        latency_ms: float | None,
        detail: str | None = None,
    ) -> ViolenceEvidence:
        return ViolenceEvidence(
            window.packets[0].source_id, None, window.start_s, window.end_s, score,
            self.model_name, self.revision, self._label_mapping, status, latency_ms, detail,
        )

    def infer(self, window: ClipWindow) -> ViolenceEvidence:
        if self._load_error is not None:
            return self._evidence(window, None, "unavailable", None, self._load_error)
        started = time.perf_counter()
        try:
            import torch

            inputs = self._preprocess(window, self.sample_count, self.device)
            with torch.inference_mode():
                output = self.model_instance(inputs)
                logits = output.logits if hasattr(output, "logits") else output
                if getattr(logits, "ndim", None) != 2 or tuple(logits.shape) != (1, 2):
                    raise ValueError("X3D model output must have binary logits with shape [1, 2]")
                probabilities = torch.softmax(logits, dim=-1)[0]
            score = float(probabilities[self._violent_index].detach().cpu().item())
            if not 0.0 <= score <= 1.0:
                raise ValueError("X3D violence probability is outside [0, 1]")
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._health = StageHealth(
                "violence", "available", model=self.model_name, device=self.device,
                latency_ms=latency_ms, detail="X3D-M violent-label probability",
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


def create_violence_classifier(config: Any) -> ViolenceClassifier:
    from .config import ConfigError

    if config.backend == "x3d":
        return X3DViolenceClassifier(
            config.repository, config.checkpoint, config.revision,
            architecture=config.architecture, device=config.device,
            sample_count=config.sample_count,
            labels=config.labels, license_name=config.license,
            known_limitations=config.known_limitations,
            checkpoint_sha256=config.checkpoint_sha256,
        )
    if config.backend == "huggingface":
        return VideoMAEViolenceClassifier(
            config.model, config.revision, device=config.device, labels=config.labels,
            license_name=config.license, known_limitations=config.known_limitations,
            checkpoint_sha256=config.checkpoint_sha256,
        )
    raise ConfigError(f"unsupported violence backend: {config.backend}")
