"""
NEUROLECT AI - Automatic Lecture Summarization Module
Transformer-based hierarchical summarization supporting DistilBART and FLAN-T5 checkpoints.
Supports concise summaries, detailed summaries, section-wise summaries,
topic-wise summaries, and structured lecture notes.
Guarantees full-text coverage with no silent truncation of trailing chunks.
Strictly RAG-free: operates on sequential text chunks without vector retrieval.
"""

from typing import Dict, Any, List, Optional
from models.model_loader import model_loader
from modules.content_models import LectureContent
from utils.helpers import chunk_text
import config
import logging

logger = logging.getLogger("Summarizer")

class LectureSummarizer:
    """Performs hierarchical transformer-based summarization for short and long educational texts."""

    def __init__(self):
        self.loader = model_loader

    def _format_model_input(self, text_block: str, model_id: str) -> str:
        """Applies task-specific prefixes for instruction-tuned models like T5/FLAN-T5."""
        if "t5" in model_id.lower():
            return f"summarize: {text_block.strip()}"
        return text_block.strip()

    def _summarize_text_block(
        self,
        text_block: str,
        pipeline: Any,
        max_length: int = 150,
        min_length: int = 40
    ) -> str:
        """Runs a single chunk through the transformer summarization pipeline."""
        words = text_block.split()
        if len(words) < 25:
            return text_block.strip()

        # Dynamic length bounds respecting input token size
        adjusted_max = min(max_length, max(30, int(len(words) * 0.75)))
        adjusted_min = min(min_length, max(15, int(adjusted_max * 0.35)))

        model_id = getattr(pipeline.model, "name_or_path", config.MODELS_CONFIG["summarizer"]["model_id"])
        formatted_input = self._format_model_input(text_block, str(model_id))

        try:
            summary_output = pipeline(
                formatted_input,
                max_length=adjusted_max,
                min_length=adjusted_min,
                do_sample=False,
                truncation=True
            )
            return summary_output[0]["summary_text"].strip()
        except Exception as e:
            logger.error(f"Transformer summarization step failed: {e}")
            raise RuntimeError(f"Summarization model error: {e}")

    def generate_all_summaries(
        self,
        lecture_content: LectureContent,
        topics_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates comprehensive summaries across all chunks without truncation:
        1. Concise Summary: Synthesized iteratively via hierarchical passes.
        2. Detailed Summary: Complete concatenated overview of all chunk summaries.
        3. Section-wise Summaries: Granular summaries tagged to source pages or timestamps.
        4. Structured Lecture Notes: Bulleted takeaways per chunk.
        5. Topic-wise Summaries: Grouped by discovered educational topics.
        """
        pipeline = self.loader.get_summarizer_pipeline()
        chunks = lecture_content.chunks

        if not chunks:
            chunks = chunk_text(lecture_content.cleaned_text)

        section_summaries: List[Dict[str, Any]] = []
        chunk_summary_texts: List[str] = []

        logger.info(f"Summarizing all {len(chunks)} chunks sequentially using transformer pipeline...")
        
        # Guarantee every single chunk is processed
        for chunk in chunks:
            summary_txt = self._summarize_text_block(
                chunk["text"],
                pipeline=pipeline,
                max_length=160,
                min_length=35
            )
            chunk_summary_texts.append(summary_txt)
            section_summaries.append({
                "chunk_id": chunk.get("chunk_id", 0),
                "source_ref": chunk.get("source_ref", "General"),
                "summary": summary_txt,
                "original_word_count": chunk.get("word_count", len(chunk["text"].split())),
                "summary_word_count": len(summary_txt.split())
            })

        # Detailed summary combines all section summaries from start to finish
        detailed_summary = "\n\n".join(chunk_summary_texts)

        # Hierarchical Concise Summary Synthesis (no silent truncation on long texts)
        concise_summary = self._hierarchical_concise_synthesis(chunk_summary_texts, pipeline)

        # Structured Lecture Notes: bulleted key takeaways
        structured_notes = self._build_structured_notes(section_summaries, lecture_content)

        # Topic-wise summaries
        topic_summaries = self.generate_topic_summaries(lecture_content, topics_data, pipeline=pipeline)

        compression_ratio = round((1 - (len(concise_summary.split()) / max(1, lecture_content.word_count))) * 100, 1)

        return {
            "concise_summary": concise_summary,
            "detailed_summary": detailed_summary,
            "section_summaries": section_summaries,
            "structured_notes": structured_notes,
            "topic_summaries": topic_summaries,
            "total_chunks_processed": len(chunks),
            "original_word_count": lecture_content.word_count,
            "concise_word_count": len(concise_summary.split()),
            "compression_ratio": f"{compression_ratio}%",
            "model_used": config.MODELS_CONFIG["summarizer"]["model_id"]
        }

    def _hierarchical_concise_synthesis(self, chunk_summaries: List[str], pipeline: Any) -> str:
        """
        Synthesizes a unified concise summary from multiple chunk summaries.
        If chunk summaries collectively exceed 300 words, it summarizes them in
        sub-batches to prevent token truncation until a single coherent summary is formed.
        """
        if not chunk_summaries:
            return ""
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]

        current_layer = chunk_summaries
        
        # Iteratively reduce until the combined length fits into a single concise summary
        while len(" ".join(current_layer).split()) > 350 and len(current_layer) > 2:
            next_layer = []
            # Group into pairs or triplets
            group_size = 3
            for i in range(0, len(current_layer), group_size):
                sub_group = " ".join(current_layer[i:i+group_size])
                reduced = self._summarize_text_block(sub_group, pipeline=pipeline, max_length=140, min_length=40)
                next_layer.append(reduced)
            current_layer = next_layer

        # Final pass over the combined layer
        final_input = " ".join(current_layer)
        return self._summarize_text_block(final_input, pipeline=pipeline, max_length=150, min_length=50)

    def generate_topic_summaries(
        self,
        lecture_content: LectureContent,
        topics_data: Optional[Dict[str, Any]] = None,
        pipeline: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Generates individual summaries for each detected topic cluster."""
        if not topics_data or "topics" not in topics_data:
            return []

        pipe = pipeline or self.loader.get_summarizer_pipeline()
        topic_summaries: List[Dict[str, Any]] = []

        for topic in topics_data.get("topics", []):
            topic_title = topic.get("title", "Topic")
            topic_sentences = topic.get("sentences", [])
            combined_topic_text = " ".join(topic_sentences)

            if len(combined_topic_text.split()) > 30:
                summary_txt = self._summarize_text_block(combined_topic_text, pipe, max_length=120, min_length=30)
            else:
                summary_txt = combined_topic_text.strip()

            topic_summaries.append({
                "topic_id": topic.get("topic_id", 1),
                "title": topic_title,
                "summary": summary_txt,
                "source_segments": topic.get("source_segments", []),
                "sentence_count": len(topic_sentences)
            })

        return topic_summaries

    def _build_structured_notes(
        self,
        section_summaries: List[Dict[str, Any]],
        content: LectureContent
    ) -> List[Dict[str, Any]]:
        """Organizes summaries into structured note blocks with bullet points and source references."""
        notes = []
        for idx, sec in enumerate(section_summaries):
            text = sec["summary"]
            points = [p.strip() for p in text.replace(". ", ".\n").split("\n") if len(p.strip()) > 10]
            notes.append({
                "header": f"Core Concept {idx + 1} ({sec['source_ref']})",
                "source_ref": sec["source_ref"],
                "key_points": points
            })
        return notes
