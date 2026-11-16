"""
Unit Tests for Summarizer Module (Phase 3)
Tests concise, detailed, section-wise, topic-wise summaries, and structured notes.
Verifies long-text handling, oversized paragraph splitting, and completeness of final text chunks.
"""

import unittest
from unittest.mock import MagicMock, patch
from modules.summarizer import LectureSummarizer
from modules.content_models import LectureContent
from utils.helpers import chunk_text

SAMPLE_LECTURE_TEXT = (
    "Natural language processing is a crucial branch of artificial intelligence. "
    "It enables computers to understand, interpret, and generate human languages. "
    "Transformers have revolutionized modern natural language processing by using self-attention mechanisms. "
    "Unlike recurrent neural networks which process tokens sequentially, transformers process all tokens simultaneously. "
    "This allows for superior parallelization and captures long-range linguistic dependencies efficiently."
)

class TestSummarizer(unittest.TestCase):
    def setUp(self):
        self.summarizer = LectureSummarizer()
        self.content = LectureContent(
            content_id="lec_sum_test",
            source_type="text",
            source_name="NLP Transformers Lecture",
            raw_text=SAMPLE_LECTURE_TEXT,
            cleaned_text=SAMPLE_LECTURE_TEXT,
            word_count=len(SAMPLE_LECTURE_TEXT.split()),
            reading_time="1 min read",
            sentences=SAMPLE_LECTURE_TEXT.split(". "),
            chunks=[
                {
                    "chunk_id": 0,
                    "text": SAMPLE_LECTURE_TEXT,
                    "word_count": len(SAMPLE_LECTURE_TEXT.split()),
                    "source_ref": "Page 1"
                }
            ],
            segments=[{"segment_id": 1, "text": SAMPLE_LECTURE_TEXT, "source_ref": "Page 1"}],
            metadata={}
        )

    def test_summarizer_with_mocked_pipeline(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "Transformers revolutionized NLP using self-attention for parallel token processing."}]

        with patch.object(self.summarizer.loader, "get_summarizer_pipeline", return_value=mock_pipe):
            summaries = self.summarizer.generate_all_summaries(self.content)

        self.assertIn("concise_summary", summaries)
        self.assertIn("detailed_summary", summaries)
        self.assertIn("section_summaries", summaries)
        self.assertIn("structured_notes", summaries)
        self.assertEqual(len(summaries["section_summaries"]), 1)
        self.assertEqual(summaries["section_summaries"][0]["source_ref"], "Page 1")
        self.assertIn("Transformers", summaries["concise_summary"])

    def test_long_text_and_final_chunk_completeness(self):
        """
        Verifies that long educational texts with multiple chunks are processed completely,
        and that the final chunk is not silently dropped.
        """
        # Construct a synthetic multi-chunk lecture
        chunk1_text = "Section 1 Introduction to Neural Networks. " * 35  # ~140 words
        chunk2_text = "Section 2 Deep Convolutional Architectures. " * 35  # ~140 words
        chunk3_text = "Section 3 Transformers and Attention. " * 35      # ~140 words
        chunk4_text = "Section 4 Final Remarks on Ethical AI and Responsible Deployment. " * 30 # Final trailing text

        full_long_text = f"{chunk1_text}\n\n{chunk2_text}\n\n{chunk3_text}\n\n{chunk4_text}"
        
        # Chunk with small chunk_size to generate at least 4 distinct chunks
        chunks = chunk_text(full_long_text, chunk_size_words=120, overlap_words=20)
        self.assertGreaterEqual(len(chunks), 4, "Failed to create multi-chunk test document")

        long_content = LectureContent(
            content_id="lec_long_test",
            source_type="text",
            source_name="Comprehensive_Deep_Learning.txt",
            raw_text=full_long_text,
            cleaned_text=full_long_text,
            word_count=len(full_long_text.split()),
            reading_time="3 min read",
            sentences=[s for s in full_long_text.split(". ") if s.strip()],
            chunks=chunks,
            segments=[{"segment_id": i+1, "text": c["text"], "source_ref": f"Part {i+1}"} for i, c in enumerate(chunks)],
            metadata={}
        )

        # Mock pipeline returning distinct summary per chunk
        mock_pipe = MagicMock()
        def mock_summarize(text, **kw):
            if "Ethical AI" in text or "Responsible Deployment" in text:
                return [{"summary_text": "Final chunk summarized: Ethical AI and responsible deployment principles."}]
            return [{"summary_text": f"Summarized intermediate section containing {len(text.split())} words."}]
        
        mock_pipe.side_effect = mock_summarize

        with patch.object(self.summarizer.loader, "get_summarizer_pipeline", return_value=mock_pipe):
            summaries = self.summarizer.generate_all_summaries(long_content)

        # 1. Verify EVERY chunk was processed
        self.assertEqual(len(summaries["section_summaries"]), len(chunks))
        self.assertEqual(summaries["total_chunks_processed"], len(chunks))

        # 2. Verify that the FINAL chunk was NOT silently discarded in the detailed summary
        detailed_summary = summaries["detailed_summary"]
        self.assertIn("Ethical AI and responsible deployment", detailed_summary)

        # 3. Verify section summaries have correct IDs
        self.assertEqual(summaries["section_summaries"][-1]["chunk_id"], len(chunks) - 1)

    def test_topic_wise_summarization(self):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"summary_text": "Topic summary on transformer attention mechanisms."}]

        topics_data = {
            "topics": [
                {
                    "topic_id": 1,
                    "title": "Transformer Architecture",
                    "sentences": [
                        "Transformers use self-attention mechanisms to process tokens in parallel.",
                        "This replaces recurrent neural networks in sequence-to-sequence translation."
                    ],
                    "source_segments": ["Page 1"]
                }
            ]
        }

        topic_summaries = self.summarizer.generate_topic_summaries(
            lecture_content=self.content,
            topics_data=topics_data,
            pipeline=mock_pipe
        )

        self.assertEqual(len(topic_summaries), 1)
        self.assertEqual(topic_summaries[0]["title"], "Transformer Architecture")
        self.assertEqual(topic_summaries[0]["source_segments"], ["Page 1"])

if __name__ == "__main__":
    unittest.main()
