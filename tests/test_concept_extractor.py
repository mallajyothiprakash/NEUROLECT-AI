"""
Unit Tests for Concept Extractor Module (Phase 3)
Tests definition extraction, technical concept mining, explanation pattern detection,
and source page/timestamp preservation.
"""

import unittest
from modules.concept_extractor import ConceptExtractor
from modules.content_models import LectureContent

SAMPLE_CONCEPTS_TEXT = (
    "Supervised Learning is defined as a machine learning method where models are trained using labeled examples. "
    "Recurrent Neural Networks refer to a class of artificial neural networks where connections between nodes form a directed sequence. "
    "The primary purpose of regularization is to prevent model overfitting on noisy training data. "
    "The key difference between classification and regression lies in the nature of the target variable."
)

class TestConceptExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = ConceptExtractor()
        self.content = LectureContent(
            content_id="lec_concept_test",
            source_type="pdf",
            source_name="ML_Definitions.pdf",
            raw_text=SAMPLE_CONCEPTS_TEXT,
            cleaned_text=SAMPLE_CONCEPTS_TEXT,
            word_count=len(SAMPLE_CONCEPTS_TEXT.split()),
            reading_time="1 min read",
            sentences=SAMPLE_CONCEPTS_TEXT.split(". "),
            chunks=[
                {
                    "chunk_id": 0,
                    "text": SAMPLE_CONCEPTS_TEXT,
                    "source_ref": "Page 3"
                }
            ],
            segments=[{"segment_id": 1, "text": SAMPLE_CONCEPTS_TEXT, "source_ref": "Page 3"}],
            metadata={}
        )

    def test_definition_and_principle_extraction(self):
        concepts = self.extractor.extract_concepts(self.content)
        self.assertGreaterEqual(len(concepts), 3)

        types = [c["type"] for c in concepts]
        terms = [c["term"].lower() for c in concepts]

        # Verify definitions found
        self.assertIn("Definition", types)
        self.assertTrue(any("supervised" in t or "recurrent" in t for t in terms))

        # Verify explanations / principles found
        self.assertIn("Key Principle / Explanation", types)

        # Verify source reference preservation
        for c in concepts:
            self.assertEqual(c["source_ref"], "Page 3")
            self.assertIn(c["importance"], ["High", "Medium", "Standard"])

if __name__ == "__main__":
    unittest.main()
