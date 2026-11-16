"""
NEUROLECT AI - Unified Content Models
Provides standardized data structures for representing lecture materials
across Audio, Video, PDF, and Raw Text modalities.
Strictly RAG-free.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

@dataclass
class ContentSegment:
    """Represents a localized slice of content with page number or timestamp metadata."""
    segment_id: int
    text: str
    source_ref: str  # e.g., "Page 2" or "02:15 - 02:45"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    page_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ContentChunk:
    """Represents a bounded text chunk sized for transformer context limits without RAG."""
    chunk_id: int
    text: str
    word_count: int
    source_ref: str  # Source reference mapped from segments (e.g. Page 1 or 00:00 - 00:30)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class LectureContent:
    """
    Unified educational lecture content representation.
    Produced by Audio, Video, PDF, and Text processors alike.
    All modules in the common NLP intelligence engine consume this model.
    """
    content_id: str
    source_type: str        # 'audio', 'video', 'pdf', 'text'
    source_name: str        # Original filename or user-given lecture title
    raw_text: str
    cleaned_text: str
    word_count: int
    reading_time: str
    sentences: List[str] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LectureContent":
        return cls(**data)

# Backward-compatibility alias
ProcessedLectureContent = LectureContent
