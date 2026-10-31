"""Small real-encoder checks; no model checkpoints or GPU are needed."""

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from cgd import script_util


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "needs ffmpeg/ffprobe")
class TestVideoExport(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.frames = []
        for index, color in enumerate(("red", "green", "blue")):
            frame = self.root / f"{index}.png"
            Image.new("RGB", (17, 19), color).save(frame)
            self.frames.append(frame)

    def probe(self, path):
        result = subprocess.run([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames",
            "-of", "json", str(path),
        ], check=True, capture_output=True, text=True)
        return json.loads(result.stdout)["streams"][0]

    def test_video_contains_every_frame_and_pads_odd_dimensions(self):
        output = self.root / "animation.mp4"
        self.assertEqual(script_util.create_video_ffmpeg(self.frames, output), str(output))
        stream = self.probe(output)
        self.assertEqual(stream["codec_name"], "h264")
        self.assertEqual(stream["pix_fmt"], "yuv420p")
        self.assertEqual((stream["width"], stream["height"]), (18, 20))
        self.assertEqual(stream["nb_read_frames"], "3")
        self.assertTrue(all(frame.exists() for frame in self.frames))
        self.assertEqual(list(self.root.glob("*.tmp.mp4")), [])

    def test_gif_still_contains_every_frame(self):
        output = self.root / "animation.gif"
        self.assertEqual(script_util.create_gif_ffmpeg(self.frames, output), str(output))
        self.assertEqual(self.probe(output)["nb_read_frames"], "3")

    def test_encoder_failure_preserves_old_output_and_source_frames(self):
        output = self.root / "animation.mp4"
        output.write_bytes(b"previous export")
        invalid = self.root / "invalid.png"
        invalid.write_bytes(b"not a PNG")
        self.assertIsNone(script_util.create_video_ffmpeg([invalid], output))
        self.assertEqual(output.read_bytes(), b"previous export")
        self.assertTrue(invalid.exists())
        self.assertEqual(list(self.root.glob("*.tmp.mp4")), [])

    def test_missing_ffmpeg_preserves_frames(self):
        with patch("cgd.script_util.subprocess.Popen", side_effect=FileNotFoundError("ffmpeg")):
            self.assertIsNone(script_util.create_video_ffmpeg(self.frames, self.root / "out.mp4"))
        self.assertTrue(all(frame.exists() for frame in self.frames))

    def test_empty_frames_or_bad_frame_rate_are_rejected(self):
        for frames, fps in (([], 10), (self.frames, 0)):
            with self.assertRaises(ValueError):
                script_util.create_video_ffmpeg(frames, self.root / "out.mp4", fps=fps)


if __name__ == "__main__":
    unittest.main()
