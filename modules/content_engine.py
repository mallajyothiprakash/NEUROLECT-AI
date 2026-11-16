"""
NEUROLECT AI - Unified Content Processing Engine
Standardizes input from Audio, Video, PDF, and Raw Text into a single
compatible internal representation without any RAG or vector database.
"""

import uuid
import hashlib
from typing import List, Dict, Any, Optional
from utils.helpers import clean_text, split_into_sentences, chunk_text, calculate_reading_time
from modules.content_models import ContentSegment, LectureContent, ProcessedLectureContent
import config

class UnifiedContentEngine:
    """
    Standardization pipeline that digests raw input strings and segments from any
    supported modality into the unified LectureContent container.
    """

    @staticmethod
    def process_content(
        source_type: str,
        source_name: str,
        raw_text: str,
        segments: Optional[List[Dict[str, Any]]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> LectureContent:
        """
        Cleans, segments, and chunks raw content while preserving source tracking.
        Does NOT index into vector databases (strictly RAG-free).
        """
        cleaned = clean_text(raw_text)
        sentences = split_into_sentences(cleaned)
        word_count = len(cleaned.split())
        reading_time = calculate_reading_time(word_count)
        
        # Transformer-friendly chunking (overlap for sentence continuity)
        chunks = chunk_text(
            cleaned,
            chunk_size_words=config.CHUNK_SIZE_WORDS,
            overlap_words=config.CHUNK_OVERLAP_WORDS
        )
        
        # Attach source tracking to chunks
        if segments:
            # Map chunk boundaries to corresponding segment tags
            for chunk in chunks:
                chunk_snippet = chunk["text"][:60].lower()
                matched_ref = None
                for seg in segments:
                    if seg.get("text") and seg["text"][:30].lower() in chunk_snippet:
                        matched_ref = seg.get("source_ref")
                        break
                chunk["source_ref"] = matched_ref or segments[0].get("source_ref", "General")
        else:
            for chunk in chunks:
                chunk["source_ref"] = f"Section {chunk['chunk_id'] + 1}"

        # Generate deterministic or unique ID
        hash_digest = hashlib.md5(f"{source_name}_{cleaned[:100]}".encode("utf-8")).hexdigest()[:12]
        content_id = f"lec_{hash_digest}"

        meta = custom_metadata or {}
        meta["total_chunks"] = len(chunks)
        meta["total_sentences"] = len(sentences)

        return LectureContent(
            content_id=content_id,
            source_type=source_type,
            source_name=source_name,
            raw_text=raw_text,
            cleaned_text=cleaned,
            word_count=word_count,
            reading_time=reading_time,
            sentences=sentences,
            chunks=chunks,
            segments=segments or [],
            metadata=meta
        )
