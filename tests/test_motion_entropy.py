import unittest

import cv2
import numpy as np

from crowd_safety.config import MotionEntropyConfig, ROIConfig
from crowd_safety.motion_entropy import compute_motion_entropy


ROI = ROIConfig("full", ((0.0, 0.0), (64.0, 0.0), (64.0, 48.0), (0.0, 48.0)))


def shifted_frame(dx=0, dy=0):
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    cv2.rectangle(frame, (8 + dx, 16 + dy), (24 + dx, 32 + dy), (255, 255, 255), -1)
    return frame


class MotionEntropyTest(unittest.TestCase):
    def test_static_and_coherent_motion_are_less_entropic_than_conflicting_motion(self):
        config = MotionEntropyConfig()
        static = compute_motion_entropy(shifted_frame(), shifted_frame(), ROI, config)
        coherent = compute_motion_entropy(shifted_frame(), shifted_frame(2), ROI, config)
        conflicting = compute_motion_entropy(shifted_frame(), np.roll(shifted_frame(2), 20, axis=1), ROI, config)

        self.assertEqual(static.status, "available")
        self.assertEqual(coherent.status, "available")
        self.assertEqual(conflicting.status, "available")
        self.assertLessEqual(static.value, coherent.value)
        self.assertLess(coherent.value, conflicting.value)

    def test_invalid_frames_are_not_zero(self):
        result = compute_motion_entropy(None, np.zeros((48, 64, 3), dtype=np.uint8), ROI, MotionEntropyConfig())
        self.assertIsNone(result.value)
        self.assertIn(result.status, {"insufficient", "unavailable"})


if __name__ == "__main__":
    unittest.main()
