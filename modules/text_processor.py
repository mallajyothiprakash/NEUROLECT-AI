"""
NEUROLECT AI - Text Processing Module
Handles raw text input and TXT file uploads, performs text normalization,
encoding detection, and constructs standardized LectureContent objects.
Strictly RAG-free.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from modules.content_models import LectureContent
from modules.content_engine import UnifiedContentEngine
from utils.validators import validate_text_content
import config
import logging

logger = logging.getLogger("TextProcessor")

class TextProcessor:
    """Processes plain text and TXT files into unified educational lecture content."""

    def validate_txt_file(self, txt_path: Path) -> Tuple[bool, str]:
        """Validates TXT file existence, extension, and non-empty status."""
        if not txt_path.exists():
            return False, f"Text file does not exist at '{txt_path}'."
        
        ext = txt_path.suffix.lower()
        if ext != ".txt":
            return False, f"Unsupported text format '{ext}'. Expected '.txt'."
        
        size = txt_path.stat().st_size
        if size == 0:
            return False, "Text file is empty (0 bytes)."
        
        if size > config.MAX_DOCUMENT_SIZE_BYTES:
            max_mb = config.MAX_DOCUMENT_SIZE_BYTES // (1024 * 1024)
            return False, f"Text file exceeds maximum allowed limit of {max_mb} MB."
            
        return True, "Valid text file."

    def process_raw_text(
        self,
        text: str,
        lecture_title: Optional[str] = "Pasted Lecture Notes"
    ) -> LectureContent:
        """Processes directly pasted educational lecture text."""
        is_valid, msg = validate_text_content(text)
        if not is_valid:
            logger.error(f"Pasted text validation failed: {msg}")
            raise ValueError(msg)

        title = lecture_title.strip() if lecture_title and lecture_title.strip() else "Direct Lecture Notes"
        
        logger.info(f"Processing raw text for: {title} ({len(text.split())} words)...")

        return UnifiedContentEngine.process_content(
            source_type="text",
            source_name=title,
            raw_text=text,
            segments=None,
            custom_metadata={
                "input_method": "direct_paste"
            }
        )

    def process_txt_file(
        self,
        txt_path: Path,
        source_name: Optional[str] = None
    ) -> LectureContent:
        """
        Processes an uploaded plain text (.txt) file with multi-encoding fallback:
        UTF-8 -> UTF-8-sig -> Latin-1 -> CP1252.
        """
        name = source_name or txt_path.name

        is_valid, val_msg = self.validate_txt_file(txt_path)
        if not is_valid:
            logger.error(f"Text file validation failed for {name}: {val_msg}")
            raise ValueError(val_msg)

        logger.info(f"Reading text file: {name}...")

        # Multi-encoding fallback sequence
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        content = None
        used_encoding = "utf-8"

        for enc in encodings:
            try:
                with open(txt_path, "r", encoding=enc) as f:
                    content = f.read()
                used_encoding = enc
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            raise ValueError(f"Could not decode text file '{name}'. Please save the file in UTF-8 encoding.")

        is_valid, msg = validate_text_content(content)
        if not is_valid:
            raise ValueError(f"Error in {name}: {msg}")

        return UnifiedContentEngine.process_content(
            source_type="text",
            source_name=name,
            raw_text=content,
            segments=None,
            custom_metadata={
                "input_method": "txt_file_upload",
                "encoding": used_encoding
            }
        )
