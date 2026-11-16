"""
NEUROLECT AI - PDF Processing Module
Extracts educational text from PDF lecture notes and slides using PyMuPDF,
preserves page-level metadata, detects empty or corrupted documents,
and standardizes into LectureContent.
Strictly RAG-free.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
import pymupdf
from modules.content_models import LectureContent
from modules.content_engine import UnifiedContentEngine
from utils.helpers import clean_text
import config
import logging

logger = logging.getLogger("PDFProcessor")

class PDFProcessor:
    """Extracts structured text and page segments from PDF documents."""

    def validate_pdf(self, pdf_path: Path) -> Tuple[bool, str]:
        """Validates PDF file existence, extension, and size limits."""
        if not pdf_path.exists():
            return False, f"PDF file does not exist at '{pdf_path}'."
        
        ext = pdf_path.suffix.lower()
        if ext != ".pdf":
            return False, f"Unsupported document format '{ext}'. Expected '.pdf'."
        
        size = pdf_path.stat().st_size
        if size == 0:
            return False, "PDF file is empty (0 bytes)."
        
        if size > config.MAX_DOCUMENT_SIZE_BYTES:
            max_mb = config.MAX_DOCUMENT_SIZE_BYTES // (1024 * 1024)
            return False, f"PDF file exceeds maximum allowed limit of {max_mb} MB."
            
        return True, "Valid PDF file."

    def extract_text_from_pdf(
        self,
        pdf_path: Path,
        source_name: Optional[str] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> LectureContent:
        """
        Extracts textual content page-by-page using PyMuPDF.
        Tracks page boundaries, validates readability, and preserves source metadata.
        """
        name = source_name or pdf_path.name

        # 1. Validation
        if progress_callback:
            progress_callback(0.05, f"Validating PDF file: {name}...")

        is_valid, val_msg = self.validate_pdf(pdf_path)
        if not is_valid:
            logger.error(f"PDF validation failed for {name}: {val_msg}")
            raise ValueError(val_msg)

        logger.info(f"Extracting text from PDF: {name}...")

        # 2. PyMuPDF Document Loading
        try:
            doc = pymupdf.open(str(pdf_path))
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable PDF document: {e}")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise ValueError("The uploaded PDF has 0 pages (empty document).")

        full_text_parts: List[str] = []
        segments: List[Dict[str, Any]] = []

        # 3. Page-by-page Extraction with Progress Reporting
        for page_idx in range(total_pages):
            page = doc[page_idx]
            page_num = page_idx + 1
            
            if progress_callback:
                frac = 0.1 + ((page_num / total_pages) * 0.75)
                progress_callback(frac, f"Extracting text from Page {page_num} of {total_pages}...")

            page_text = page.get_text("text").strip()

            if page_text:
                full_text_parts.append(page_text)
                segments.append({
                    "segment_id": page_num,
                    "text": clean_text(page_text),
                    "source_ref": f"Page {page_num}",
                    "page_number": page_num
                })

        doc.close()

        # 4. Check for Empty / Scanned Document
        combined_text = "\n\n".join(full_text_parts).strip()
        if not combined_text:
            raise ValueError(
                "No readable text could be extracted from this PDF.\n"
                "The document may consist purely of scanned images or encrypted vector drawings without an extractable text layer."
            )

        # 5. Standardize into Common LectureContent
        if progress_callback:
            progress_callback(0.95, "Structuring lecture content and text chunks...")

        lecture_content = UnifiedContentEngine.process_content(
            source_type="pdf",
            source_name=name,
            raw_text=combined_text,
            segments=segments,
            custom_metadata={
                "total_pages": total_pages,
                "pages_with_text": len(segments),
                "extraction_engine": "PyMuPDF"
            }
        )

        if progress_callback:
            progress_callback(1.0, f"Successfully extracted {len(segments)} pages from PDF!")

        return lecture_content
