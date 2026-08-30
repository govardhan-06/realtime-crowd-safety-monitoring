import unittest

from crowd_safety.config import ROIConfig
from crowd_safety.crowd_features import compute_crowd_features
from crowd_safety.types import TrackObservation


ROI = ROIConfig("test", ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)))


def tracks_for_positions(positions_by_track, timestamp=1.0):
    observations = []
    for track_id, positions in positions_by_track.items():
        for index, (x, y) in enumerate(positions):
            observations.append(
                TrackObservation(
                    "camera-1", track_id, index, index / max(1, len(positions) - 1),
                    (float(x), float(y)), (x - 1, y - 1, x + 1, y + 1), 0.9
                )
            )
    return compute_crowd_features(observations, ROI, timestamp, window_s=1.0, min_track_history=2)


class CrowdFeaturesTest(unittest.TestCase):
    def test_rightward_tracks_have_low_disorder(self):
        result = tracks_for_positions({1: [(10, 20), (20, 20)], 2: [(10, 40), (20, 40)]})

        self.assertEqual(result.status, "available")
        self.assertLess(result.direction_disorder, 0.1)
        self.assertEqual(result.counter_flow, 0.0)
        self.assertIsNone(result.acceleration_px_s2)

    def test_opposing_tracks_have_counter_flow(self):
        result = tracks_for_positions({1: [(10, 20), (20, 20)], 2: [(90, 40), (80, 40)]})

        self.assertGreater(result.counter_flow, 0.9)
        self.assertGreater(result.direction_disorder, 0.9)

    def test_inward_and_outward_groups_have_convergence_and_dispersal(self):
        inward = tracks_for_positions({1: [(20, 50), (45, 50)], 2: [(80, 50), (55, 50)]})
        outward = tracks_for_positions({1: [(45, 50), (20, 50)], 2: [(55, 50), (80, 50)]})

        self.assertGreater(inward.convergence, 0.9)
        self.assertLess(inward.dispersal, 0.1)
        self.assertGreater(outward.dispersal, 0.9)
        self.assertLess(outward.convergence, 0.1)

    def test_stationary_dense_tracks_are_congested(self):
        result = tracks_for_positions({i: [(10 + i * 5, 40), (10 + i * 5, 40)] for i in range(5)})

        self.assertEqual(result.occupancy, 5)
        self.assertEqual(result.congestion, 1.0)
        self.assertEqual(result.mean_speed_px_s, 0.0)

    def test_empty_roi_and_short_history_are_explicit(self):
        empty = tracks_for_positions({1: [(150, 150), (160, 160)]})
        short = tracks_for_positions({1: [(10, 10)]})

        self.assertEqual(empty.status, "insufficient")
        self.assertEqual(empty.occupancy, 0)
        self.assertEqual(empty.density_delta, 0.0)
        self.assertIsNone(empty.mean_speed_px_s)
        self.assertEqual(short.status, "insufficient")
        self.assertIsNone(short.mean_speed_px_s)

    def test_expired_tracks_do_not_create_movement(self):
        observations = [
            TrackObservation("camera-1", 1, 0, 0.0, (10.0, 10.0), (9, 9, 11, 11), 0.9),
            TrackObservation("camera-1", 1, 1, 0.2, (20.0, 10.0), (19, 9, 21, 11), 0.9),
        ]
        result = compute_crowd_features(observations, ROI, 2.0, window_s=1.0, min_track_history=2)

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.occupancy, 0)
        self.assertIsNone(result.mean_speed_px_s)

    def test_irregular_timestamps_use_elapsed_time(self):
        observations = [
            TrackObservation("camera-1", 1, 0, 0.0, (10.0, 10.0), (9, 9, 11, 11), 0.9),
            TrackObservation("camera-1", 1, 1, 0.5, (20.0, 10.0), (19, 9, 21, 11), 0.9),
            TrackObservation("camera-1", 1, 2, 1.0, (40.0, 10.0), (39, 9, 41, 11), 0.9),
        ]
        result = compute_crowd_features(observations, ROI, 1.0, window_s=1.0, min_track_history=2)

        self.assertEqual(result.mean_speed_px_s, 30.0)
        self.assertEqual(result.acceleration_px_s2, 40.0)

    def test_density_delta_uses_latest_pre_window_occupancy(self):
        observations = [
            TrackObservation("camera-1", 1, 0, -0.1, (10, 10), (9, 9, 11, 11), 0.9),
            TrackObservation("camera-1", 1, 1, 0.5, (20, 10), (19, 9, 21, 11), 0.9),
            TrackObservation("camera-1", 1, 2, 1.0, (30, 10), (29, 9, 31, 11), 0.9),
            TrackObservation("camera-1", 2, 3, 0.5, (40, 10), (39, 9, 41, 11), 0.9),
            TrackObservation("camera-1", 2, 4, 1.0, (50, 10), (49, 9, 51, 11), 0.9),
        ]
        result = compute_crowd_features(observations, ROI, 1.0, window_s=1.0, min_track_history=2)

        self.assertEqual(result.status, "available")
        self.assertEqual(result.density_delta, 1.0)

    def test_density_delta_counts_entry_from_empty_roi_baseline(self):
        observations = [
            TrackObservation("camera-1", 1, 0, 0.0, (150, 10), (149, 9, 151, 11), 0.9),
            TrackObservation("camera-1", 1, 1, 0.5, (10, 10), (9, 9, 11, 11), 0.9),
            TrackObservation("camera-1", 1, 2, 1.0, (20, 10), (19, 9, 21, 11), 0.9),
        ]
        result = compute_crowd_features(observations, ROI, 1.0, window_s=1.0, min_track_history=2)

        self.assertEqual(result.status, "available")
        self.assertEqual(result.occupancy, 1)
        self.assertEqual(result.density_delta, 1.0)

    def test_density_delta_counts_exit_from_roi_baseline(self):
        observations = [
            TrackObservation("camera-1", 1, 0, 0.0, (10, 10), (9, 9, 11, 11), 0.9),
            TrackObservation("camera-1", 1, 1, 0.5, (150, 10), (149, 9, 151, 11), 0.9),
        ]
        result = compute_crowd_features(observations, ROI, 1.0, window_s=1.0, min_track_history=2)

        self.assertEqual(result.status, "insufficient")
        self.assertEqual(result.occupancy, 0)
        self.assertEqual(result.density_delta, -1.0)


if __name__ == "__main__":
    unittest.main()
