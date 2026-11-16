"""
NEUROLECT AI - Video Processing Module
Handles MP4, MOV, and MKV video validation, FFmpeg audio track extraction,
Whisper speech transcription, progress reporting, guaranteed temporary cleanup,
and standardization into LectureContent.
Strictly RAG-free.
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple
from modules.audio_processor import AudioProcessor
from modules.content_models import LectureContent
from utils.file_manager import cleanup_file
import config
import logging

logger = logging.getLogger("VideoProcessor")

class VideoProcessor:
    """Processes video lecture files by extracting speech audio and running transcription."""

    def __init__(self):
        self.audio_processor = AudioProcessor()

    @staticmethod
    def is_ffmpeg_available() -> bool:
        """Checks if FFmpeg binary is available on system PATH."""
        return shutil.which("ffmpeg") is not None

    def validate_video(self, video_path: Path) -> Tuple[bool, str]:
        """Validates video file path, extension, and size."""
        if not video_path.exists():
            return False, f"Video file does not exist at '{video_path}'."
        
        ext = video_path.suffix.lower()
        if ext not in config.SUPPORTED_VIDEO_EXTENSIONS:
            return False, f"Unsupported video format '{ext}'. Supported: {', '.join(config.SUPPORTED_VIDEO_EXTENSIONS)}"
        
        size = video_path.stat().st_size
        if size == 0:
            return False, "Video file is empty (0 bytes)."
        
        if size > config.MAX_VIDEO_SIZE_BYTES:
            max_mb = config.MAX_VIDEO_SIZE_BYTES // (1024 * 1024)
            return False, f"Video file exceeds maximum size limit of {max_mb} MB."
            
        return True, "Valid video file."

    def extract_audio_from_video(
        self,
        video_path: Path,
        output_audio_path: Path
    ) -> Path:
        """
        Extracts the audio track from video using FFmpeg.
        Converts to 16kHz mono PCM WAV for optimal Whisper compatibility.
        """
        if not self.is_ffmpeg_available():
            raise RuntimeError(
                "FFmpeg is required for video audio extraction but was not found on your system PATH.\n"
                "Please install FFmpeg:\n"
                "  • macOS: brew install ffmpeg\n"
                "  • Windows: winget install Gyan.FFmpeg (or download from https://ffmpeg.org)\n"
                "  • Linux: sudo apt-get install ffmpeg"
            )

        cmd = [
            "ffmpeg",
            "-y",                  # Overwrite without asking
            "-i", str(video_path), # Input video
            "-vn",                 # Disable video streams
            "-acodec", "pcm_s16le",# Standard PCM 16-bit
            "-ar", "16000",        # 16kHz sample rate (Whisper native)
            "-ac", "1",            # Mono channel
            str(output_audio_path)
        ]

        logger.info(f"Executing audio extraction: {' '.join(cmd)}")
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            err_msg = result.stderr.lower()
            if "does not contain any stream" in err_msg or "audio:0kb" in err_msg or "no audio" in err_msg or "invalid data" in err_msg:
                raise ValueError("The uploaded video file contains no audio track or the video format is corrupted.")
            raise RuntimeError(f"FFmpeg audio extraction failed (Code {result.returncode}): {result.stderr[:300]}")

        if not output_audio_path.exists() or output_audio_path.stat().st_size == 0:
            raise ValueError("Audio extraction produced an empty audio file. The video likely has no sound track.")

        return output_audio_path

    def process_video(
        self,
        video_path: Path,
        source_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> LectureContent:
        """
        Full end-to-end pipeline:
        Video Validation -> FFmpeg Audio Extraction -> Whisper Transcription -> Cleanup -> LectureContent
        """
        name = source_name or video_path.name

        # 1. Validation
        if progress_callback:
            progress_callback(0.05, f"Validating video file: {name}...")

        is_valid, val_msg = self.validate_video(video_path)
        if not is_valid:
            logger.error(f"Video validation failed for {name}: {val_msg}")
            raise ValueError(val_msg)

        temp_audio_path = config.TEMP_DIR / f"extracted_{video_path.stem}.wav"

        try:
            # 2. Extract audio track via FFmpeg
            if progress_callback:
                progress_callback(0.2, "Step 1/2: Extracting 16kHz mono audio track via FFmpeg...")

            logger.info(f"Extracting audio track from video: {name}...")
            self.extract_audio_from_video(video_path, temp_audio_path)

            # 3. Transcribe via Whisper
            if progress_callback:
                progress_callback(0.5, "Step 2/2: Transcribing extracted speech with Whisper...")

            logger.info("Transcribing extracted audio track...")
            lecture_content = self.audio_processor.transcribe_audio(
                audio_path=temp_audio_path,
                source_name=name,
                progress_callback=lambda p, msg: progress_callback(0.5 + (p * 0.45), msg) if progress_callback else None
            )

            # 4. Update source metadata to reflect video origin
            lecture_content.source_type = "video"
            lecture_content.metadata["original_video_name"] = name
            lecture_content.metadata["audio_extraction"] = "ffmpeg_pcm16le_16khz"

            if progress_callback:
                progress_callback(1.0, "Video lecture transcription and processing complete!")

            return lecture_content

        finally:
            # 5. Guaranteed cleanup of temporary extracted audio file
            cleanup_file(temp_audio_path)
            logger.info(f"Cleaned up temporary audio extraction file: {temp_audio_path}")
