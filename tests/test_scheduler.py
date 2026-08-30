import unittest

from crowd_safety.scheduling import SchedulerError, schedule_frames


class SchedulerTest(unittest.TestCase):
    def test_selects_first_and_then_target_fps_slots(self):
        result = schedule_frames([0.0, 0.033, 0.100, 0.200, 0.233, 0.400], 5.0)

        self.assertEqual(result.processed_indices, (0, 3, 5))
        self.assertEqual(result.skipped_indices, (1, 2, 4))
        self.assertEqual(result.schedule_times, (0.0, 0.2, 0.4))

    def test_high_target_fps_processes_each_source_frame_without_duplicate_times(self):
        result = schedule_frames([0.0, 0.2, 0.4], 100.0)

        self.assertEqual(result.processed_indices, (0, 1, 2))
        self.assertEqual(len(result.schedule_times), len(set(result.schedule_times)))

    def test_irregular_timestamps_are_deterministic(self):
        timestamps = [10.0, 10.07, 10.21, 10.22, 10.51]

        self.assertEqual(schedule_frames(timestamps, 5.0), schedule_frames(timestamps, 5.0))
        self.assertEqual(schedule_frames(timestamps, 5.0).processed_indices, (0, 2, 4))

    def test_rejects_non_monotonic_timestamps_and_invalid_fps(self):
        with self.assertRaises(SchedulerError):
            schedule_frames([0.0, 0.1, 0.05], 5.0)
        with self.assertRaises(SchedulerError):
            schedule_frames([0.0], 0.0)


if __name__ == "__main__":
    unittest.main()
