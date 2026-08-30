from pathlib import Path
import math
from typing import Iterator

import cv2

from .annotations import annotate_frame
from .types import FramePacket


class VideoIOError(RuntimeError):
    """Raised when local video input/output cannot be opened or used."""


class VideoReader:
    def __init__(self, path: str | Path, source_id: str = "local-video") -> None:
        self.path = Path(path)
        self.source_id = source_id
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            self.capture.release()
            raise VideoIOError(f"could not open video input: {self.path}")
        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __iter__(self) -> Iterator[FramePacket]:
        last_timestamp = -math.inf
        frame_index = 0
        while True:
            ok, image = self.capture.read()
            if not ok:
                break
            reported_timestamp = float(self.capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            fallback_timestamp = frame_index / self.fps if self.fps > 0 else float(frame_index)
            timestamp = (
                reported_timestamp
                if math.isfinite(reported_timestamp)
                and (frame_index == 0 or reported_timestamp > last_timestamp)
                else fallback_timestamp
            )
            if timestamp < last_timestamp:
                timestamp = last_timestamp
            yield FramePacket(self.source_id, frame_index, timestamp, image)
            last_timestamp = timestamp
            frame_index += 1

    def close(self) -> None:
        self.capture.release()


class VideoWriter:
    def __init__(
        self,
        path: str | Path,
        size: tuple[int, int],
        fps: float,
        annotate: bool = True,
    ) -> None:
        if len(size) != 2 or any(value <= 0 for value in size):
            raise VideoIOError("output size must contain two positive dimensions")
        if not math.isfinite(fps) or fps <= 0:
            raise VideoIOError("output FPS must be greater than zero")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.size = size
        self.fps = float(fps)
        self.annotate = annotate
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size
        )
        if not self.writer.isOpened():
            self.writer.release()
            raise VideoIOError(f"could not open video output: {self.path}")

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def write(self, packet: FramePacket, image: object | None = None) -> None:
        image = image if image is not None else packet.image
        if (image.shape[1], image.shape[0]) != self.size:
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
        if self.annotate:
            image = annotate_frame(image, packet.frame_index, packet.timestamp_s)
        self.writer.write(image)

    def close(self) -> None:
        self.writer.release()
