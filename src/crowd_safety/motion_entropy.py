from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from .config import MotionEntropyConfig, ROIConfig
from .types import FeatureStatus


@dataclass(frozen=True)
class MotionEntropyResult:
    value: float | None
    status: FeatureStatus
    detail: str | None = None


def _mask(shape: tuple[int, int], roi: ROIConfig) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    points = np.asarray(roi.polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def compute_motion_entropy(
    previous: Any,
    current: Any,
    roi: ROIConfig,
    config: MotionEntropyConfig,
) -> MotionEntropyResult:
    if not config.enabled:
        return MotionEntropyResult(None, "unavailable", "motion entropy is disabled")
    if not isinstance(previous, np.ndarray) or not isinstance(current, np.ndarray):
        return MotionEntropyResult(None, "unavailable", "consecutive frames are required")
    if previous.shape != current.shape or previous.ndim not in {2, 3} or previous.size == 0:
        return MotionEntropyResult(None, "unavailable", "frame pair is malformed or incompatible")
    try:
        previous_gray = previous if previous.ndim == 2 else cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        current_gray = current if current.ndim == 2 else cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray, current_gray, None, config.pyr_scale, config.levels,
            config.winsize, config.iterations, config.poly_n, config.poly_sigma, 0,
        )
        magnitude, direction = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        selected = _mask(previous_gray.shape, roi) & (magnitude >= config.min_magnitude)
        magnitudes = magnitude[selected]
        directions = direction[selected]
    except (cv2.error, ValueError) as exc:
        return MotionEntropyResult(None, "unavailable", f"optical flow failed: {exc}")
    if magnitudes.size == 0 or not np.any(magnitudes > 1e-6):
        return MotionEntropyResult(0.0, "available", "no measurable motion in ROI")
    max_magnitude = max(float(np.max(magnitudes)), config.min_magnitude + 1e-6)
    magnitude_edges = np.linspace(0.0, max_magnitude + 1e-6, config.magnitude_bins + 1)
    direction_edges = np.linspace(0.0, 2 * math.pi, config.direction_bins + 1)
    histogram, _, _ = np.histogram2d(magnitudes, directions, bins=(magnitude_edges, direction_edges), weights=magnitudes)
    probabilities = histogram.ravel()
    probabilities = probabilities[probabilities > 0]
    probabilities /= probabilities.sum()
    entropy = float(-(probabilities * np.log(probabilities)).sum() / math.log(config.magnitude_bins * config.direction_bins))
    return MotionEntropyResult(max(0.0, min(1.0, entropy)), "available")
