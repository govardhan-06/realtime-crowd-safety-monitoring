import tempfile
import unittest
from pathlib import Path


class VideoIOSmokeTest(unittest.TestCase):
    def test_opencv_round_trips_a_small_mp4(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.mp4"
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24)
            )
            self.assertTrue(writer.isOpened())
            for value in range(3):
                writer.write(np.full((24, 32, 3), value * 40, dtype=np.uint8))
            writer.release()

            reader = cv2.VideoCapture(str(path))
            frames = 0
            while True:
                ok, _ = reader.read()
                if not ok:
                    break
                frames += 1
            reader.release()

        self.assertGreater(frames, 0)


if __name__ == "__main__":
    unittest.main()
