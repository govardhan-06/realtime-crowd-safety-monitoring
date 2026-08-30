import unittest

from crowd_safety.tracking import ByteTrackTracker
from crowd_safety.types import FramePacket, PersonDetection


class TrackingTest(unittest.TestCase):
    def test_adapter_returns_source_local_persistent_observations(self):
        frames = iter([
            [(1, (10, 10, 20, 30), 0.9)],
            [(1, (12, 10, 22, 30), 0.8)],
        ])

        def fake_track(packet):
            return next(frames)

        tracker = ByteTrackTracker(track_fn=fake_track)
        first = tracker.update(FramePacket("camera-1", 0, 0.0, object()))
        second = tracker.update(FramePacket("camera-1", 1, 0.5, object()))

        self.assertEqual(first.observations[0].track_id, 1)
        self.assertEqual(second.observations[0].track_id, 1)
        self.assertEqual(second.observations[0].source_id, "camera-1")
        self.assertEqual(second.health.status, "available")

    def test_unavailable_tracker_is_not_an_empty_normal_result(self):
        tracker = ByteTrackTracker()
        result = tracker.update(FramePacket("camera-1", 0, 0.0, object()))

        self.assertEqual(result.observations, ())
        self.assertEqual(result.health.status, "unavailable")


if __name__ == "__main__":
    unittest.main()
