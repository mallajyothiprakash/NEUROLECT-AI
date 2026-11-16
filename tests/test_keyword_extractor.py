"""
Unit Tests for Keyword Extractor Module (Phase 3)
Tests KeyBERT-style candidate phrase generation, MMR deduplication,
relevance ranking, and topic association.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from modules.keyword_extractor import KeywordExtractor
from modules.content_models import LectureContent

SAMPLE_TEXT = (
    "Supervised learning trains models on labeled input-output pairs. "
    "Neural network backpropagation computes loss gradients across parameter matrices. "
    "Gradient descent updates weights iteratively to minimize the objective loss function. "
    "Regularization techniques such as dropout prevent model overfitting."
)

class TestKeywordExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = KeywordExtractor()
        self.content = LectureContent(
            content_id="lec_kw_test",
            source_type="text",
            source_name="ML Fundamentals",
            raw_text=SAMPLE_TEXT,
            cleaned_text=SAMPLE_TEXT,
            word_count=len(SAMPLE_TEXT.split()),
            reading_time="1 min read",
            sentences=SAMPLE_TEXT.split(". "),
            chunks=[{"chunk_id": 0, "text": SAMPLE_TEXT, "source_ref": "Section 1"}],
            segments=[],
            metadata={}
        )

    def test_keyword_extraction_with_mmr_and_topics(self):
        mock_st = MagicMock()
        # Return random embedding vectors for document and candidates
        mock_st.encode.side_effect = lambda texts, **kw: np.random.randn(len(texts), 16)

        topics_data = {
            "topics": [
                {
                    "title": "Supervised Learning & Neural Networks",
                    "content_preview": "Supervised learning trains models on labeled input-output pairs.",
                    "sentences": ["Supervised learning trains models on labeled input-output pairs."]
                }
            ]
        }

        with patch.object(self.extractor.loader, "get_sentence_transformer", return_value=mock_st):
            keywords = self.extractor.extract_keywords(
                self.content,
                top_n=5,
                topics_data=topics_data
            )

        self.assertGreater(len(keywords), 0)
        for kw in keywords:
            self.assertIn("keyword", kw)
            self.assertIn("relevance_score", kw)
            self.assertIn("frequency", kw)
            self.assertIn("associated_topic", kw)
            self.assertGreaterEqual(kw["frequency"], 1)

    def test_keyword_extraction_fallback(self):
        with patch.object(self.extractor.loader, "get_sentence_transformer", side_effect=RuntimeError("Offline")):
            keywords = self.extractor.extract_keywords(self.content, top_n=4)

        self.assertGreater(len(keywords), 0)
        self.assertIn("keyword", keywords[0])
        self.assertIn("relevance_score", keywords[0])

if __name__ == "__main__":
    unittest.main()
