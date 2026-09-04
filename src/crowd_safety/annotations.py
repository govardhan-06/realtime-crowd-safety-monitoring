from typing import Any

import cv2
import numpy as np

from .config import ROIConfig
from .types import CrowdFeatureRecord, TrackObservation, ViolenceEvidence


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
        text = f"{feature.roi_name}: n={feature.occupancy if feature.occupancy is not None else '-'} {status}"
        cv2.putText(annotated, text, (8, 38 + index * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    if violence is None:
        violence_text = "violence: warming-up"
    elif violence.score is None:
        violence_text = f"violence: {violence.status}"
    else:
        violence_text = f"violence: {violence.status} score={violence.score:.2f}"
    cv2.putText(
        annotated,
        violence_text,
        (8, 38 + len(features) * 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )
    return annotated
