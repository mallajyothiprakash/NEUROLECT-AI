"""
NEUROLECT AI - Input Validation Utility
Validates audio, video, document files, and direct text inputs.
"""

from pathlib import Path
from typing import Tuple
import config

def validate_audio_file(uploaded_file) -> Tuple[bool, str]:
    """Validates uploaded audio file format and size limits."""
    if uploaded_file is None:
        return False, "No audio file provided."
    
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in config.SUPPORTED_AUDIO_EXTENSIONS:
        return False, f"Unsupported audio format '{file_ext}'. Supported formats: {', '.join(config.SUPPORTED_AUDIO_EXTENSIONS)}"
    
    # Check size
    file_size = getattr(uploaded_file, "size", 0)
    if file_size <= 0:
        return False, "The uploaded audio file is empty (0 bytes)."
    if file_size > config.MAX_AUDIO_SIZE_BYTES:
        max_mb = config.MAX_AUDIO_SIZE_BYTES // (1024 * 1024)
        return False, f"Audio file size exceeds the maximum allowed limit of {max_mb} MB."
    
    return True, "Valid audio file."

def validate_video_file(uploaded_file) -> Tuple[bool, str]:
    """Validates uploaded video file format and size limits."""
    if uploaded_file is None:
        return False, "No video file provided."
    
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in config.SUPPORTED_VIDEO_EXTENSIONS:
        return False, f"Unsupported video format '{file_ext}'. Supported formats: {', '.join(config.SUPPORTED_VIDEO_EXTENSIONS)}"
    
    file_size = getattr(uploaded_file, "size", 0)
    if file_size <= 0:
        return False, "The uploaded video file is empty (0 bytes)."
    if file_size > config.MAX_VIDEO_SIZE_BYTES:
        max_mb = config.MAX_VIDEO_SIZE_BYTES // (1024 * 1024)
        return False, f"Video file size exceeds the maximum allowed limit of {max_mb} MB."
    
    return True, "Valid video file."

def validate_document_file(uploaded_file) -> Tuple[bool, str]:
    """Validates uploaded document (PDF/TXT) format and size limits."""
    if uploaded_file is None:
        return False, "No document file provided."
    
    file_ext = Path(uploaded_file.name).suffix.lower()
    if file_ext not in config.SUPPORTED_DOCUMENT_EXTENSIONS:
        return False, f"Unsupported document format '{file_ext}'. Supported formats: {', '.join(config.SUPPORTED_DOCUMENT_EXTENSIONS)}"
    
    file_size = getattr(uploaded_file, "size", 0)
    if file_size <= 0:
        return False, "The uploaded document file is empty (0 bytes)."
    if file_size > config.MAX_DOCUMENT_SIZE_BYTES:
        max_mb = config.MAX_DOCUMENT_SIZE_BYTES // (1024 * 1024)
        return False, f"Document file size exceeds the maximum allowed limit of {max_mb} MB."
    
    return True, "Valid document file."

def validate_text_content(text: str, min_words: int = 15) -> Tuple[bool, str]:
    """Validates raw text input to ensure sufficient educational content for processing."""
    if not text or not text.strip():
        return False, "Text input is empty. Please paste or enter educational content."
    
    words = text.strip().split()
    if len(words) < min_words:
        return False, f"Content is too brief ({len(words)} words). Please provide at least {min_words} words for meaningful NLP analysis."
    
    return True, "Valid text content."
