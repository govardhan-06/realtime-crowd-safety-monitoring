import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from crowd_safety.types import FramePacket
from crowd_safety.video import VideoIOError, VideoReader, VideoWriter


def make_video(path: Path, frame_count: int = 5, size: tuple[int, int] = (32, 24)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, size)
    if not writer.isOpened():
        raise RuntimeError("test video writer could not be opened")
    for value in range(frame_count):
        writer.write(np.full((size[1], size[0], 3), value * 20, dtype=np.uint8))
    writer.release()


class VideoIOTest(unittest.TestCase):
    def test_reader_emits_monotonic_timestamped_packets(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            make_video(source)

            with VideoReader(source, source_id="camera-1") as reader:
                packets = list(reader)

        self.assertEqual([packet.frame_index for packet in packets], list(range(5)))
        self.assertEqual([packet.source_id for packet in packets], ["camera-1"] * 5)
        self.assertEqual(
            [packet.timestamp_s for packet in packets],
            sorted(packet.timestamp_s for packet in packets),
        )
        self.assertTrue(all(isinstance(packet, FramePacket) for packet in packets))

    def test_writer_resizes_and_can_disable_operational_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            output = Path(directory) / "output.mp4"
            make_video(source)
            with VideoReader(source) as reader, VideoWriter(output, (16, 12), 5.0, annotate=False) as writer:
                for packet in reader:
                    writer.write(packet)

            capture = cv2.VideoCapture(str(output))
            ok, frame = capture.read()
            count = 1 if ok else 0
            while ok:
                ok, _ = capture.read()
                count += int(ok)
            capture.release()

        self.assertEqual(frame.shape[:2], (12, 16))
        self.assertEqual(count, 5)

    def test_reader_rejects_unreadable_input(self):
        with self.assertRaisesRegex(VideoIOError, "could not open video"):
            VideoReader("missing.mp4")


if __name__ == "__main__":
    unittest.main()
