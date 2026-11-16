"""
Unit Tests for Audio & Video Processing Pipelines
Tests validators, format checks, temporary file lifecycle, and FFmpeg detection.
"""

import unittest
from pathlib import Path
from utils.validators import validate_audio_file, validate_video_file
from utils.file_manager import cleanup_file, save_uploaded_file
from modules.video_processor import VideoProcessor
import config

class DummyUpload:
    def __init__(self, name: str, size: int):
        self.name = name
        self.size = size

class TestAudioVideoPipelines(unittest.TestCase):
    def test_audio_validator_formats(self):
        # Supported formats
        for ext in [".mp3", ".wav", ".m4a"]:
            dummy = DummyUpload(f"lecture{ext}", 1024 * 1024)
            valid, msg = validate_audio_file(dummy)
            self.assertTrue(valid, msg)

        # Unsupported formats
        for bad_ext in [".flac", ".ogg", ".exe", ".txt"]:
            dummy = DummyUpload(f"lecture{bad_ext}", 1024 * 1024)
            valid, msg = validate_audio_file(dummy)
            self.assertFalse(valid)

        # Empty file
        empty = DummyUpload("empty.mp3", 0)
        valid, msg = validate_audio_file(empty)
        self.assertFalse(valid)

        # Exceeds limit
        too_big = DummyUpload("huge.mp3", config.MAX_AUDIO_SIZE_BYTES + 100)
        valid, msg = validate_audio_file(too_big)
        self.assertFalse(valid)

    def test_video_validator_formats(self):
        for ext in [".mp4", ".mov", ".mkv"]:
            dummy = DummyUpload(f"lecture{ext}", 10 * 1024 * 1024)
            valid, msg = validate_video_file(dummy)
            self.assertTrue(valid, msg)

        for bad_ext in [".avi", ".wmv", ".flv"]:
            dummy = DummyUpload(f"lecture{bad_ext}", 10 * 1024 * 1024)
            valid, msg = validate_video_file(dummy)
            self.assertFalse(valid)

    def test_file_cleanup(self):
        test_file = config.TEMP_DIR / "temp_to_delete.txt"
        with open(test_file, "w") as f:
            f.write("temporary test content")
        self.assertTrue(test_file.exists())
        cleanup_file(test_file)
        self.assertFalse(test_file.exists())

    def test_ffmpeg_detection_logic(self):
        # Checks whether ffmpeg is detected or reports status cleanly
        is_avail = VideoProcessor.is_ffmpeg_available()
        self.assertIsInstance(is_avail, bool)

if __name__ == "__main__":
    unittest.main()
