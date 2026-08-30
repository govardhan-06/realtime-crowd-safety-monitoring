from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FramePacket:
    source_id: str
    frame_index: int
    timestamp_s: float
    image: Any
