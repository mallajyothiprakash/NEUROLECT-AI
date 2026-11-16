"""
Unit Tests for Audio Processor Module (Phase 2)
Tests audio file validation, Whisper pipeline integration, timestamp preservation,
progress callback reporting, and standardized LectureContent generation.
"""

import unittest
import wave
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch
from modules.audio_processor import AudioProcessor
from modules.content_models import LectureContent
import config

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessor()
        self.test_dir = config.TEMP_DIR / "test_audio_artifacts"
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Generate a synthetic 1-second silence WAV file
        self.synthetic_wav = self.test_dir / "sample_speech.wav"
        with wave.open(str(self.synthetic_wav), "w") as wf:
            wf.setnchannels(1)        # Mono
            wf.setsampwidth(2)        # 16-bit PCM
            wf.setframerate(16000)    # 16kHz
            # 1 second of audio (16,000 samples)
            data = struct.pack("<16000h", *([0] * 16000))
            wf.writeframes(data)

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_audio_validation_formats_and_sizes(self):
        # 1. Valid WAV
        valid, msg = self.processor.validate_audio(self.synthetic_wav)
        self.assertTrue(valid, msg)

        # 2. Non-existent file
        non_existent = self.test_dir / "missing.mp3"
        valid, msg = self.processor.validate_audio(non_existent)
        self.assertFalse(valid)
        self.assertIn("does not exist", msg)

        # 3. Unsupported format
        bad_format = self.test_dir / "audio.flac"
        bad_format.touch()
        valid, msg = self.processor.validate_audio(bad_format)
        self.assertFalse(valid)
        self.assertIn("Unsupported", msg)

        # 4. Zero-byte empty file
        empty_file = self.test_dir / "empty.wav"
        empty_file.touch()
        valid, msg = self.processor.validate_audio(empty_file)
        self.assertFalse(valid)
        self.assertIn("empty", msg)

    def test_transcribe_audio_with_timestamps(self):
        # Mock Whisper ASR pipeline output with timestamped chunks
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = {
            "text": "Hello world. Welcome to the artificial intelligence lecture.",
            "chunks": [
                {"timestamp": (0.0, 1.5), "text": "Hello world."},
                {"timestamp": (1.5, 4.2), "text": "Welcome to the artificial intelligence lecture."}
            ]
        }

        # Track progress callback invocations
        progress_calls = []
        def track_progress(pct, msg):
            progress_calls.append((pct, msg))

        with patch.object(self.processor.loader, "get_whisper_pipeline", return_value=mock_pipeline):
            content = self.processor.transcribe_audio(
                audio_path=self.synthetic_wav,
                source_name="Lecture_Audio.wav",
                progress_callback=track_progress
            )

        # Verify output is instance of standardized LectureContent
        self.assertIsInstance(content, LectureContent)
        self.assertEqual(content.source_type, "audio")
        self.assertEqual(content.source_name, "Lecture_Audio.wav")
        self.assertIn("artificial intelligence", content.cleaned_text)
        
        # Verify segment and timestamp structure
        self.assertEqual(len(content.segments), 2)
        self.assertEqual(content.segments[0]["source_ref"], "00:00 - 00:02")
        self.assertEqual(content.segments[1]["source_ref"], "00:02 - 00:04")
        self.assertGreater(content.metadata["duration_seconds"], 4.0)

        # Verify progress callback was invoked across steps
        self.assertGreaterEqual(len(progress_calls), 4)
        self.assertEqual(progress_calls[-1][0], 1.0)

if __name__ == "__main__":
    unittest.main()
