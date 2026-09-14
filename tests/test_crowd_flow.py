import unittest

from crowd_safety.config import LOIConfig
from crowd_safety.crowd_flow import CrowdFlowTracker
from crowd_safety.types import TrackObservation


def observation(track_id, x, timestamp):
    return TrackObservation("camera-1", track_id, int(timestamp * 10), timestamp, (x, 5.0), (x - 1, 4, x + 1, 6), 0.9)


class CrowdFlowTest(unittest.TestCase):
    def test_counts_both_directions_once(self):
        tracker = CrowdFlowTracker("camera-1", "zone", (LOIConfig("door", (10.0, 0.0), (10.0, 20.0)),), 3.0)
        tracker.update(0.0, [observation(1, 5, 0.0)])
        tracker.update(1.0, [observation(1, 15, 1.0), observation(2, 15, 1.0)])
        tracker.update(2.0, [observation(2, 5, 2.0)])
        record = tracker.update(3.0, [observation(1, 15, 3.0)]) [0]

        self.assertEqual(record.inflow_count, 1)
        self.assertEqual(record.outflow_count, 1)
        self.assertEqual(record.net_flow_per_min, 0.0)

    def test_touch_and_oscillation_do_not_duplicate_crossings(self):
        tracker = CrowdFlowTracker("camera-1", "zone", (LOIConfig("door", (10.0, 0.0), (10.0, 20.0)),), 6.0)
        for timestamp, x in enumerate((5, 10, 5, 10, 5, 15)):
            tracker.update(float(timestamp), [observation(1, x, float(timestamp))])
        record = tracker.update(6.0, [])[0]

        self.assertEqual(record.inflow_count, 1)
        self.assertEqual(record.outflow_count, 0)

    def test_reappearance_keeps_track_state(self):
        tracker = CrowdFlowTracker("camera-1", "zone", (LOIConfig("door", (10.0, 0.0), (10.0, 20.0)),), 3.0)
        tracker.update(0.0, [observation(1, 5, 0.0)])
        tracker.update(1.0, [observation(1, 15, 1.0)])
        self.assertEqual(tracker.update(2.0, [observation(1, 15, 2.0)]), ())
        record = tracker.update(3.0, [observation(1, 15, 3.0)])[0]
        self.assertEqual(record.inflow_count, 1)


if __name__ == "__main__":
    unittest.main()
