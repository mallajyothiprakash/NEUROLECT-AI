"""
Unit Tests for NLP Intelligence Modules
Tests Bloom's Taxonomy Classifier, Difficulty Classifier, Concept Extractor, and Question Validator.
"""

import unittest
from modules.bloom_classifier import BloomsTaxonomyClassifier
from modules.difficulty_classifier import DifficultyClassifier
from modules.concept_extractor import ConceptExtractor
from modules.question_validator import QuestionValidator
from modules.content_engine import UnifiedContentEngine

class TestNLPModules(unittest.TestCase):
    def setUp(self):
        self.bloom_clf = BloomsTaxonomyClassifier()
        self.diff_clf = DifficultyClassifier()
        self.concept_ext = ConceptExtractor()
        self.validator = QuestionValidator()

    def test_blooms_taxonomy_classification(self):
        q_remember = self.bloom_clf.classify_question("What is the definition of backpropagation?", "MCQ")
        self.assertEqual(q_remember["bloom_level"], "Remember")

        q_understand = self.bloom_clf.classify_question("Explain why self-attention scales quadratically with sequence length.", "Descriptive")
        self.assertEqual(q_understand["bloom_level"], "Understand")

        q_analyze = self.bloom_clf.classify_question("Compare and contrast Recurrent Neural Networks with Transformer architectures.", "Analytical")
        self.assertEqual(q_analyze["bloom_level"], "Analyze")

        q_apply = self.bloom_clf.classify_question("Calculate the gradient update step given a learning rate of 0.01.", "Application")
        self.assertEqual(q_apply["bloom_level"], "Apply")

    def test_difficulty_classification(self):
        easy = self.diff_clf.classify_difficulty("What is AI?", "Remember", "MCQ")
        self.assertEqual(easy["difficulty"], "Easy")

        hard = self.diff_clf.classify_difficulty(
            "Analyze and critically evaluate the trade-offs between dense matrix multiplication and sparse attention in large multi-modal transformer networks.",
            "Evaluate",
            "Analytical"
        )
        self.assertEqual(hard["difficulty"], "Hard")

    def test_concept_extractor(self):
        lecture_text = (
            "Supervised Learning is defined as a machine learning paradigm where models learn from labeled pairs. "
            "Neural Networks are a type of computing system inspired by biological neural networks. "
            "The primary purpose of regularization is to prevent overfitting during training."
        )
        content = UnifiedContentEngine.process_content("text", "ML Intro", lecture_text)
        concepts = self.concept_ext.extract_concepts(content)

        self.assertGreaterEqual(len(concepts), 2)
        terms = [c["term"].lower() for c in concepts]
        self.assertTrue(any("supervised" in t or "neural" in t for t in terms))

    def test_question_validator(self):
        source = "Convolutional neural networks are commonly used in image classification."
        
        valid_q = [{
            "question": "What is commonly used in image classification?",
            "correct_answer": "Convolutional neural networks",
            "type": "Short Answer"
        }]
        
        valid_out, warnings = self.validator.validate_question_batch(valid_q, source)
        self.assertEqual(len(valid_out), 1)

        # Test duplicate and empty rejection
        bad_q = [
            {"question": "", "correct_answer": "bad", "type": "Short Answer"},
            {"question": "Valid Q?", "correct_answer": "", "type": "Short Answer"},
            {"question": "Same question text?", "correct_answer": "ans1", "type": "Short Answer"},
            {"question": "Same question text?", "correct_answer": "ans2", "type": "Short Answer"}
        ]
        bad_out, bad_warnings = self.validator.validate_question_batch(bad_q, source)
        self.assertEqual(len(bad_out), 1) # Only one unique non-empty passed
        self.assertGreater(len(bad_warnings), 1)

if __name__ == "__main__":
    unittest.main()
