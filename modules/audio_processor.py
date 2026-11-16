"""
NEUROLECT AI - Audio Processing Module
Handles MP3, WAV, and M4A audio validation, Whisper speech transcription,
progress reporting, timestamp preservation, and standardization into LectureContent.
Strictly RAG-free.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from models.model_loader import model_loader
from modules.content_models import LectureContent
from modules.content_engine import UnifiedContentEngine
from utils.helpers import format_timestamp
from utils.validators import validate_audio_file
import config
import logging

logger = logging.getLogger("AudioProcessor")

class AudioProcessor:
    """Processes educational audio recordings into formatted transcripts with timestamps."""

    def __init__(self):
        self.loader = model_loader

    def validate_audio(self, audio_path: Path) -> Tuple[bool, str]:
        """
        Validates the audio file path, format, and size against system configuration.
        """
        if not audio_path.exists():
            return False, f"Audio file does not exist at '{audio_path}'."
        
        ext = audio_path.suffix.lower()
        if ext not in config.SUPPORTED_AUDIO_EXTENSIONS:
            return False, f"Unsupported audio format '{ext}'. Supported: {', '.join(config.SUPPORTED_AUDIO_EXTENSIONS)}"
        
        size = audio_path.stat().st_size
        if size == 0:
            return False, "Audio file is empty (0 bytes)."
        
        if size > config.MAX_AUDIO_SIZE_BYTES:
            max_mb = config.MAX_AUDIO_SIZE_BYTES // (1024 * 1024)
            return False, f"Audio file exceeds maximum size limit of {max_mb} MB."
            
        return True, "Valid audio file."

    def transcribe_audio(
        self,
        audio_path: Path,
        source_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> LectureContent:
        """
        Transcribes an audio file using the cached Whisper pipeline.
        Extracts speech, segments, and timestamps.
        Reports progress if a callback is supplied.
        """
        name = source_name or audio_path.name

        # 1. Validation
        if progress_callback:
            progress_callback(0.1, f"Validating audio format: {name}...")
        
        is_valid, val_msg = self.validate_audio(audio_path)
        if not is_valid:
            logger.error(f"Audio validation failed for {name}: {val_msg}")
            raise ValueError(val_msg)

        logger.info(f"Initiating speech recognition for {name}...")

        # 2. Lazy Model Loading
        if progress_callback:
            progress_callback(0.3, "Loading Whisper automatic speech recognition model...")
        
        asr_pipeline = self.loader.get_whisper_pipeline()

        # 3. Speech Recognition Inference
        if progress_callback:
            progress_callback(0.6, "Transcribing speech segments and extracting timestamps...")

        try:
            result = asr_pipeline(
                str(audio_path),
                return_timestamps=True,
                generate_kwargs={"task": "transcribe", "language": "english"}
            )
        except Exception as e:
            logger.error(f"Whisper inference error on {name}: {e}")
            raise RuntimeError(f"Speech transcription failed: {e}")

        full_text = result.get("text", "").strip()
        raw_chunks = result.get("chunks", [])

        if not full_text:
            logger.warning(f"Whisper produced empty transcript for {name}.")
            full_text = "[No audible speech detected in the audio file]"

        # 4. Segment & Timestamp Processing
        if progress_callback:
            progress_callback(0.85, "Structuring transcript segments with timestamps...")

        segments: List[Dict[str, Any]] = []
        for idx, chunk in enumerate(raw_chunks):
            ts = chunk.get("timestamp", (0.0, 0.0))
            start_t = ts[0] if ts and ts[0] is not None else 0.0
            end_t = ts[1] if ts and len(ts) > 1 and ts[1] is not None else start_t
            
            time_ref = f"{format_timestamp(start_t)} - {format_timestamp(end_t)}"
            chunk_txt = chunk.get("text", "").strip()
            
            if chunk_txt:
                segments.append({
                    "segment_id": idx + 1,
                    "text": chunk_txt,
                    "source_ref": time_ref,
                    "start_time": start_t,
                    "end_time": end_t
                })

        duration_sec = segments[-1]["end_time"] if segments else 0.0

        # 5. Delivery to Common Content Engine
        if progress_callback:
            progress_callback(0.95, "Synthesizing standardized LectureContent object...")

        content = UnifiedContentEngine.process_content(
            source_type="audio",
            source_name=name,
            raw_text=full_text,
            segments=segments,
            custom_metadata={
                "duration_seconds": duration_sec,
                "duration_formatted": format_timestamp(duration_sec),
                "total_audio_segments": len(segments),
                "model_used": config.MODELS_CONFIG["whisper"]["model_id"]
            }
        )

        if progress_callback:
            progress_callback(1.0, "Transcription complete!")

        return content
