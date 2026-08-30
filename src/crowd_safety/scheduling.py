from dataclasses import dataclass
import math


class SchedulerError(ValueError):
    """Raised when source timestamps or target FPS cannot be scheduled."""


@dataclass(frozen=True)
class ScheduleResult:
    processed_indices: tuple[int, ...]
    skipped_indices: tuple[int, ...]
    schedule_times: tuple[float, ...]


class FrameScheduler:
    def __init__(self, target_fps: float) -> None:
        if (
            not isinstance(target_fps, (int, float))
            or isinstance(target_fps, bool)
            or not math.isfinite(target_fps)
            or target_fps <= 0
        ):
            raise SchedulerError("target_fps must be greater than zero")
        self.interval = 1.0 / target_fps
        self.first_timestamp: float | None = None
        self.last_timestamp: float | None = None
        self.next_slot = 0

    def decide(self, timestamp: float) -> tuple[bool, float | None]:
        if not math.isfinite(timestamp):
            raise SchedulerError("source timestamps must be finite")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise SchedulerError("source timestamps must be monotonic")
        if self.first_timestamp is None:
            self.first_timestamp = timestamp
        scheduled_time = self.first_timestamp + self.next_slot * self.interval
        if timestamp + 1e-9 < scheduled_time:
            self.last_timestamp = timestamp
            return False, None
        self.next_slot = max(
            self.next_slot + 1,
            math.floor((timestamp - self.first_timestamp) / self.interval) + 1,
        )
        self.last_timestamp = timestamp
        return True, scheduled_time


def schedule_frames(timestamps: list[float] | tuple[float, ...], target_fps: float) -> ScheduleResult:
    if any(not math.isfinite(timestamp) for timestamp in timestamps):
        raise SchedulerError("source timestamps must be finite")
    if any(current < previous for previous, current in zip(timestamps, timestamps[1:])):
        raise SchedulerError("source timestamps must be monotonic")
    scheduler = FrameScheduler(target_fps)
    processed: list[int] = []
    skipped: list[int] = []
    schedule_times: list[float] = []

    for frame_index, timestamp in enumerate(timestamps):
        process, scheduled_time = scheduler.decide(timestamp)
        if process:
            processed.append(frame_index)
            assert scheduled_time is not None
            schedule_times.append(scheduled_time)
        else:
            skipped.append(frame_index)

    return ScheduleResult(tuple(processed), tuple(skipped), tuple(schedule_times))
