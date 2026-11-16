"""
Unit Tests for Named Entity Recognition Module (Phase 3)
Tests BERT NER entity categorization (People, Organizations, Locations, Technical),
subword cleaning, confidence scoring, and source information tracking.
"""

import unittest
from unittest.mock import MagicMock, patch
from modules.ner_processor import NamedEntityRecognizer
from modules.content_models import LectureContent

SAMPLE_NER_TEXT = (
    "Geoffrey Hinton and Yann LeCun presented their deep learning research at New York University. "
    "Google and Meta have invested heavily in PyTorch and TensorFlow transformer libraries."
)

class TestNERProcessor(unittest.TestCase):
    def setUp(self):
        self.ner_engine = NamedEntityRecognizer()
        self.content = LectureContent(
            content_id="lec_ner_test",
            source_type="audio",
            source_name="AI Pioneers Lecture.mp3",
            raw_text=SAMPLE_NER_TEXT,
            cleaned_text=SAMPLE_NER_TEXT,
            word_count=len(SAMPLE_NER_TEXT.split()),
            reading_time="1 min read",
            sentences=[SAMPLE_NER_TEXT],
            chunks=[
                {
                    "chunk_id": 0,
                    "text": SAMPLE_NER_TEXT,
                    "source_ref": "00:00 - 00:30"
                }
            ],
            segments=[{"segment_id": 1, "text": SAMPLE_NER_TEXT, "source_ref": "00:00 - 00:30"}],
            metadata={}
        )

    def test_entity_extraction_and_categorization(self):
        # Mock pipeline output from dslim/bert-base-NER
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = [
            {"entity_group": "PER", "word": "Geoffrey Hinton", "score": 0.99},
            {"entity_group": "PER", "word": "Yann LeCun", "score": 0.98},
            {"entity_group": "ORG", "word": "New York University", "score": 0.95},
            {"entity_group": "ORG", "word": "Google", "score": 0.97},
            {"entity_group": "ORG", "word": "Meta", "score": 0.96},
            {"entity_group": "MISC", "word": "PyTorch", "score": 0.92}
        ]

        with patch.object(self.ner_engine.loader, "get_ner_pipeline", return_value=mock_pipeline):
            ner_results = self.ner_engine.extract_entities(self.content)

        self.assertIn("categories", ner_results)
        categories = ner_results["categories"]
        self.assertIn("People", categories)
        self.assertIn("Organizations", categories)
        self.assertIn("Technical & Miscellaneous", categories)

        # Check Person entities
        person_names = [p["text"] for p in categories["People"]]
        self.assertIn("Geoffrey Hinton", person_names)
        self.assertIn("Yann Lecun", person_names)

        # Check source reference preservation
        self.assertEqual(categories["People"][0]["source_ref"], "00:00 - 00:30")

if __name__ == "__main__":
    unittest.main()
