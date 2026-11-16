"""
Unit Tests for Video Processor Module (Phase 2)
Tests video file validation, FFmpeg audio extraction handling,
guaranteed temporary audio file cleanup, and LectureContent standardization.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.video_processor import VideoProcessor
from modules.content_models import LectureContent
import config

class TestVideoProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = VideoProcessor()
        self.test_dir = config.TEMP_DIR / "test_video_artifacts"
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.synthetic_mp4 = self.test_dir / "sample_lecture.mp4"
        with open(self.synthetic_mp4, "wb") as f:
            f.write(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00isomiso2mp41")

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_video_validation_formats(self):
        # 1. Valid MP4
        valid, msg = self.processor.validate_video(self.synthetic_mp4)
        self.assertTrue(valid, msg)

        # 2. Unsupported format
        bad_video = self.test_dir / "lecture.avi"
        bad_video.touch()
        valid, msg = self.processor.validate_video(bad_video)
        self.assertFalse(valid)
        self.assertIn("Unsupported", msg)

        # 3. Non-existent file
        missing = self.test_dir / "missing.mov"
        valid, msg = self.processor.validate_video(missing)
        self.assertFalse(valid)
        self.assertIn("does not exist", msg)

        # 4. Zero-byte file
        empty_mp4 = self.test_dir / "empty.mp4"
        empty_mp4.touch()
        valid, msg = self.processor.validate_video(empty_mp4)
        self.assertFalse(valid)
        self.assertIn("empty", msg)

    def test_ffmpeg_availability_check(self):
        is_avail = VideoProcessor.is_ffmpeg_available()
        self.assertIsInstance(is_avail, bool)

    def test_video_processing_pipeline_and_cleanup(self):
        # Mock extract_audio_from_video to write a dummy extracted WAV
        temp_extracted_wav = config.TEMP_DIR / f"extracted_{self.synthetic_mp4.stem}.wav"

        def mock_extract(video_p, out_p):
            with open(out_p, "w") as f:
                f.write("mock_audio_bytes")
            return out_p

        # Mock audio_processor.transcribe_audio
        mock_content = LectureContent(
            content_id="lec_video_test",
            source_type="audio",
            source_name=self.synthetic_mp4.name,
            raw_text="Machine learning video lecture transcript.",
            cleaned_text="Machine learning video lecture transcript.",
            word_count=5,
            reading_time="1 min read",
            sentences=["Machine learning video lecture transcript."],
            chunks=[{"chunk_id": 0, "text": "Machine learning video lecture transcript.", "source_ref": "Section 1"}],
            segments=[{"segment_id": 1, "text": "Machine learning video lecture transcript.", "source_ref": "00:00 - 00:05"}],
            metadata={}
        )

        progress_calls = []
        def track_progress(pct, msg):
            progress_calls.append((pct, msg))

        with patch.object(self.processor, "extract_audio_from_video", side_effect=mock_extract) as mock_ext:
            with patch.object(self.processor.audio_processor, "transcribe_audio", return_value=mock_content) as mock_trans:
                result = self.processor.process_video(
                    video_path=self.synthetic_mp4,
                    progress_callback=track_progress
                )

        # Verify result is updated to source_type="video"
        self.assertEqual(result.source_type, "video")
        self.assertEqual(result.metadata["original_video_name"], self.synthetic_mp4.name)

        # Strictly verify cleanup: temp extracted WAV must be deleted!
        self.assertFalse(temp_extracted_wav.exists(), "Temporary audio extraction file was not cleaned up!")

        # Verify progress callback was invoked
        self.assertGreater(len(progress_calls), 0)
        self.assertEqual(progress_calls[-1][0], 1.0)

    def test_video_without_audio_handling(self):
        # Simulate video without audio track raising ValueError
        with patch.object(self.processor, "extract_audio_from_video", side_effect=ValueError("The uploaded video file contains no audio track")):
            with self.assertRaises(ValueError) as ctx:
                self.processor.process_video(self.synthetic_mp4)
            self.assertIn("no audio track", str(ctx.exception).lower())

if __name__ == "__main__":
    unittest.main()
