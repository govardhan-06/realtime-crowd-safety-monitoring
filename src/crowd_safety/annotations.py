from typing import Any

import cv2


def annotate_frame(image: Any, frame_index: int, timestamp_s: float) -> Any:
    annotated = image.copy()
    cv2.putText(
        annotated,
        f"frame={frame_index} time={timestamp_s:.3f}s",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return annotated
