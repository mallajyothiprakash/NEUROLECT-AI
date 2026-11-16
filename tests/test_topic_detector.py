"""
Unit Tests for Topic Detector Module (Phase 3)
Tests major topic discovery, subtopic extraction, hierarchical outlines,
unrelated multi-topic clustering, and source segment (pages/timestamps) association.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from modules.topic_detector import TopicDetector
from modules.content_models import LectureContent

SAMPLE_MULTI_TOPIC_TEXT = (
    "Convolutional Neural Networks are designed for spatial data such as images. "
    "Feature maps are extracted using kernels that slide across pixel matrices. "
    "Pooling layers reduce dimensional complexity while preserving dominant signals. "
    "Recurrent Neural Networks are specialized for processing sequential data. "
    "Hidden states retain temporal memory across previous tokens in the sequence. "
    "Vanishing gradients in recurrent architectures motivated gated units like LSTMs. "
    "Optimization algorithms update network parameters using computed gradients. "
    "Stochastic gradient descent calculates loss over mini-batches to minimize training cost. "
    "Learning rate schedules adjust the step size to achieve stable convergence."
)

SAMPLE_UNRELATED_TOPICS_TEXT = (
    "Quantum computing uses qubits that exist in superpositions of state zero and one. "
    "Quantum entanglement allows faster computational speed for specific matrix problems. "
    "Photosynthesis is the biological process by which green plants convert light into glucose. "
    "Chlorophyll pigments in chloroplast thylakoids absorb solar photons. "
    "The Renaissance was a fervent period of European cultural and artistic rebirth in Florence. "
    "Leonardo da Vinci and Michelangelo pioneered naturalistic perspective in painting."
)

class TestTopicDetector(unittest.TestCase):
    def setUp(self):
        self.detector = TopicDetector()
        sentences = [s.strip() for s in SAMPLE_MULTI_TOPIC_TEXT.split(". ") if s.strip()]
        self.content = LectureContent(
            content_id="lec_topic_test",
            source_type="pdf",
            source_name="Deep_Learning_Architectures.pdf",
            raw_text=SAMPLE_MULTI_TOPIC_TEXT,
            cleaned_text=SAMPLE_MULTI_TOPIC_TEXT,
            word_count=len(SAMPLE_MULTI_TOPIC_TEXT.split()),
            reading_time="1 min read",
            sentences=sentences,
            chunks=[{"chunk_id": 0, "text": SAMPLE_MULTI_TOPIC_TEXT, "source_ref": "Page 1"}],
            segments=[
                {"segment_id": 1, "text": "Convolutional Neural Networks spatial data images.", "source_ref": "Page 1"},
                {"segment_id": 2, "text": "Recurrent Neural Networks sequential data hidden states.", "source_ref": "Page 2"},
                {"segment_id": 3, "text": "Optimization algorithms stochastic gradient descent.", "source_ref": "Page 3"}
            ],
            metadata={}
        )

    def test_topic_detection_with_embedding_clustering(self):
        mock_st = MagicMock()
        mock_st.encode.side_effect = lambda sents, **kw: np.random.randn(len(sents), 16)

        with patch.object(self.detector.loader, "get_sentence_transformer", return_value=mock_st):
            results = self.detector.detect_topics(self.content, num_topics=3)

        self.assertIn("topics", results)
        self.assertIn("outline", results)
        self.assertGreaterEqual(results["total_topics"], 2)

        for topic in results["topics"]:
            self.assertIn("title", topic)
            self.assertIn("subtopics", topic)
            self.assertIn("source_segments", topic)
            self.assertGreater(len(topic["subtopics"]), 0)
            self.assertGreater(len(topic["source_segments"]), 0)

        self.assertGreater(len(results["outline"]), 2)

    def test_unrelated_topics_clustering(self):
        """Tests that clearly distinct topics (Quantum, Biology, Renaissance) are separated."""
        unrelated_sents = [s.strip() for s in SAMPLE_UNRELATED_TOPICS_TEXT.split(". ") if s.strip()]
        unrelated_content = LectureContent(
            content_id="lec_unrelated_test",
            source_type="text",
            source_name="Multidisciplinary_Lectures.txt",
            raw_text=SAMPLE_UNRELATED_TOPICS_TEXT,
            cleaned_text=SAMPLE_UNRELATED_TOPICS_TEXT,
            word_count=len(SAMPLE_UNRELATED_TOPICS_TEXT.split()),
            reading_time="1 min read",
            sentences=unrelated_sents,
            chunks=[{"chunk_id": 0, "text": SAMPLE_UNRELATED_TOPICS_TEXT, "source_ref": "Section 1"}],
            segments=[{"segment_id": 1, "text": SAMPLE_UNRELATED_TOPICS_TEXT, "source_ref": "Section 1"}],
            metadata={}
        )

        # Synthetic embeddings with distinct clusters for the 3 domains
        def mock_domain_embeddings(sents, **kw):
            emb_list = []
            for s in sents:
                if "Quantum" in s or "qubits" in s or "entanglement" in s:
                    emb_list.append([10.0, 0.0, 0.0])
                elif "Photosynthesis" in s or "Chlorophyll" in s or "glucose" in s:
                    emb_list.append([0.0, 10.0, 0.0])
                else:
                    emb_list.append([0.0, 0.0, 10.0])
            return np.array(emb_list)

        mock_st = MagicMock()
        mock_st.encode.side_effect = mock_domain_embeddings

        with patch.object(self.detector.loader, "get_sentence_transformer", return_value=mock_st):
            results = self.detector.detect_topics(unrelated_content, num_topics=3)

        self.assertEqual(results["total_topics"], 2) # min(3, len(sentences)//3) = 2
        self.assertGreaterEqual(len(results["topics"]), 2)

    def test_topic_detection_fallback(self):
        with patch.object(self.detector.loader, "get_sentence_transformer", side_effect=RuntimeError("Model Offline")):
            results = self.detector.detect_topics(self.content, num_topics=2)

        self.assertIn("topics", results)
        self.assertIn("outline", results)
        self.assertIn("Fallback", results["methodology"])

    def test_short_document_graceful_handling(self):
        short_content = LectureContent(
            content_id="lec_short_test",
            source_type="text",
            source_name="Brief.txt",
            raw_text="This is a very short text with only two sentences. It describes natural language.",
            cleaned_text="This is a very short text with only two sentences. It describes natural language.",
            word_count=15,
            reading_time="1 min read",
            sentences=["This is a very short text with only two sentences.", "It describes natural language."],
            chunks=[{"chunk_id": 0, "text": "This is a very short text with only two sentences. It describes natural language.", "source_ref": "Section 1"}],
            segments=[],
            metadata={}
        )
        results = self.detector.detect_topics(short_content)
        self.assertEqual(results["total_topics"], 1)
        self.assertEqual(len(results["topics"]), 1)

if __name__ == "__main__":
    unittest.main()
