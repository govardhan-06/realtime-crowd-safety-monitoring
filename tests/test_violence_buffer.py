import unittest

import numpy as np

from crowd_safety.types import FramePacket
from crowd_safety.violence import RollingClipBuffer, ViolenceCadence


def packet(timestamp: float, frame_index: int | None = None) -> FramePacket:
    return FramePacket("camera-1", frame_index if frame_index is not None else int(timestamp * 10), timestamp, np.zeros((2, 2, 3)))


class ViolenceBufferTest(unittest.TestCase):
    def test_buffer_does_not_emit_an_incomplete_window(self):
        buffer = RollingClipBuffer(duration_s=2.0, sample_count=3)
        buffer.append(packet(0.0))
        buffer.append(packet(1.5))

        self.assertIsNone(buffer.complete_window())

    def test_window_preserves_true_bounds_and_samples_without_duplicate_timestamps(self):
        buffer = RollingClipBuffer(duration_s=2.0, sample_count=3)
        for timestamp in (0.0, 0.4, 1.1, 2.0, 2.4):
            buffer.append(packet(timestamp))

        window = buffer.complete_window()

        self.assertIsNotNone(window)
        self.assertEqual((window.start_s, window.end_s), (0.0, 2.4))
        self.assertEqual([item.timestamp_s for item in window.sampled_packets], [0.0, 1.1, 2.4])

    def test_expiry_keeps_only_the_configured_time_window(self):
        buffer = RollingClipBuffer(duration_s=2.0, sample_count=2)
        for timestamp in (0.0, 0.5, 1.0, 2.1, 2.6):
            buffer.append(packet(timestamp))

        self.assertEqual([item.timestamp_s for item in buffer.packets], [0.5, 1.0, 2.1, 2.6])
        self.assertEqual((buffer.complete_window().start_s, buffer.complete_window().end_s), (0.5, 2.6))

    def test_cadence_has_no_duplicate_due_timestamps(self):
        cadence = ViolenceCadence(interval_s=1.0)

        due = [timestamp for timestamp in (2.0, 2.2, 3.0, 3.1, 4.1) if cadence.is_due(timestamp)]

        self.assertEqual(due, [2.0, 3.0, 4.1])

    def test_buffer_rejects_non_monotonic_packets(self):
        buffer = RollingClipBuffer(duration_s=2.0, sample_count=2)
        buffer.append(packet(1.0))

        with self.assertRaises(ValueError):
            buffer.append(packet(0.5))

    def test_duplicate_timestamps_are_not_emitted(self):
        buffer = RollingClipBuffer(duration_s=1.0, sample_count=2)
        buffer.append(packet(0.0, 1))
        buffer.append(packet(0.0, 2))
        buffer.append(packet(1.0, 3))

        self.assertEqual([item.timestamp_s for item in buffer.packets], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
