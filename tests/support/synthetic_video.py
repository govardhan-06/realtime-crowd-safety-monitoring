from pathlib import Path

import cv2
import numpy as np


def create_video(path: Path, frame_count: int = 6, fps: float = 6.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 24))
    if not writer.isOpened():
        raise RuntimeError("synthetic video writer could not be opened")
    for value in range(frame_count):
        writer.write(np.full((24, 32, 3), value * 20, dtype=np.uint8))
    writer.release()
