from typing import Any

import cv2
import numpy as np

from .config import ROIConfig
from .types import CrowdFeatureRecord, TrackObservation, ViolenceEvidence


def _put_contrasted_text(image: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def annotate_frame(
    image: Any,
    frame_index: int,
    timestamp_s: float,
    *,
    tracks: tuple[TrackObservation, ...] = (),
    histories: tuple[tuple[TrackObservation, ...], ...] = (),
    rois: tuple[ROIConfig, ...] = (),
    features: tuple[CrowdFeatureRecord, ...] = (),
    violence: ViolenceEvidence | None = None,
) -> Any:
    annotated = image.copy()
    _put_contrasted_text(annotated, f"frame={frame_index} time={timestamp_s:.3f}s", (8, 20), (0, 140, 140))
    for roi in rois:
        points = [(int(x), int(y)) for x, y in roi.polygon]
        cv2.polylines(annotated, [np.array(points)], True, (255, 180, 0), 1)
        cv2.putText(annotated, roi.name, points[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 0), 1)
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.box_xyxy)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 1)
        cv2.putText(annotated, f"id={track.track_id}", (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 0), 1)
    for history in histories:
        points = [(int(x), int(y)) for x, y in (observation.center_xy for observation in history)]
        if len(points) > 1:
            cv2.polylines(annotated, [np.array(points)], False, (0, 180, 255), 1)
    for index, feature in enumerate(features):
        status = feature.status
        count = feature.occupancy if feature.occupancy is not None else "-"
        entropy = "-" if feature.motion_entropy is None else f"{feature.motion_entropy:.2f}"
        text = f"crowd {feature.roi_name}: count={count} entropy={entropy} status={status}"
        _put_contrasted_text(annotated, text, (8, 38 + index * 16), (0, 120, 120))
    if violence is None:
        violence_text = "violence: warming up (need 16 frames)"
    elif violence.score is None:
        violence_text = f"violence: {violence.status}"
    else:
        violence_text = f"violence: {violence.status} | score={violence.score:.2f}"
    _put_contrasted_text(annotated, violence_text, (8, 38 + len(features) * 16), (0, 0, 150))
    return annotated
