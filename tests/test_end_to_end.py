"""
End-to-End Integration Test for NEUROLECT AI
Validates the complete pipeline from lecture ingestion to assessment scoring and analytics.
Strictly RAG-free.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from modules.content_engine import UnifiedContentEngine
from modules.concept_extractor import ConceptExtractor
from modules.topic_detector import TopicDetector
from modules.keyword_extractor import KeywordExtractor
from modules.bloom_classifier import BloomsTaxonomyClassifier
from modules.difficulty_classifier import DifficultyClassifier
from modules.distractor_generator import DistractorGenerator
from modules.question_validator import QuestionValidator
from modules.quiz_engine import QuizEngine
from modules.analytics import LearningAnalytics
from database.db_manager import DatabaseManager
import config

SAMPLE_LECTURE = """
Artificial Intelligence and Machine Learning Fundamentals.
Machine Learning is defined as a field of study that gives computers the ability to learn without being explicitly programmed.
Supervised Learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs.
Deep Neural Networks are composed of multiple layers to progressively extract higher-level features from the raw input.
The primary purpose of backpropagation is to calculate the gradient of the loss function with respect to each weight.
Optimization methods like Stochastic Gradient Descent update model parameters iteratively to minimize training error.
Evaluation metrics such as precision, recall, and F1-score are essential for assessing classification models.
Overfitting occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts performance on new data.
Regularization techniques including L1, L2, and Dropout are widely used to prevent overfitting in deep learning.
"""

class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.test_db_path = config.TEMP_DIR / "e2e_test.db"
        self.db = DatabaseManager(db_path=self.test_db_path)

    def tearDown(self):
        if self.test_db_path.exists():
            self.test_db_path.unlink()

    def test_full_pipeline_flow(self):
        # 1. Unified Content Ingestion
        content = UnifiedContentEngine.process_content(
            source_type="text",
            source_name="AI Foundations Lecture",
            raw_text=SAMPLE_LECTURE
        )
        self.assertGreater(content.word_count, 100)
        self.assertGreaterEqual(len(content.chunks), 1)

        # 2. Concept Extraction
        c_extractor = ConceptExtractor()
        concepts = c_extractor.extract_concepts(content)
        self.assertGreaterEqual(len(concepts), 2)

        # Mock SentenceTransformer for offline testing
        mock_st = MagicMock()
        mock_st.encode.side_effect = lambda texts, **kw: np.random.randn(len(texts), 8)

        # 3. Topic Detection
        with patch("models.model_loader.model_loader.get_sentence_transformer", return_value=mock_st):
            t_detector = TopicDetector()
            topics = t_detector.detect_topics(content)
            self.assertIn("topics", topics)
            self.assertGreaterEqual(len(topics["topics"]), 1)

            # 4. Keyword Extraction
            kw_extractor = KeywordExtractor()
            keywords = kw_extractor.extract_keywords(content, top_n=8)
            self.assertGreaterEqual(len(keywords), 3)

        # 5. Question Synthesis & Validation
        bloom_clf = BloomsTaxonomyClassifier()
        diff_clf = DifficultyClassifier()
        distractor_gen = DistractorGenerator()
        validator = QuestionValidator()

        # Build synthetic questions
        raw_items = [
            {
                "id": "q1",
                "question": "What is the primary purpose of backpropagation in deep neural networks?",
                "correct_answer": "Calculate the gradient of the loss function with respect to weights",
                "type": "Short Answer",
                "source_ref": "Section 1"
            },
            {
                "id": "q2",
                "question": "Which regularization technique is commonly used to prevent overfitting?",
                "correct_answer": "Dropout",
                "type": "Multiple Choice (MCQ)",
                "source_ref": "Section 1"
            }
        ]

        # Generate MCQ distractors for Q2 (mocking offline model to test fallback)
        with patch.object(distractor_gen.loader, "get_question_generator", side_effect=RuntimeError("Offline unit test")):
            mcq_res = distractor_gen.generate_options(
                question=raw_items[1]["question"],
                correct_answer=raw_items[1]["correct_answer"],
                context_text=SAMPLE_LECTURE,
                candidate_terms=["Stochastic Gradient Descent", "Backpropagation", "Linear Activation"]
            )
        raw_items[1]["options"] = mcq_res["options"]
        raw_items[1]["correct_letter"] = mcq_res["correct_letter"]

        for item in raw_items:
            b_info = bloom_clf.classify_question(item["question"], item["type"])
            item["bloom_level"] = b_info["bloom_level"]
            d_info = diff_clf.classify_difficulty(item["question"], item["bloom_level"], item["type"])
            item["difficulty"] = d_info["difficulty"]
            item["topic"] = "Neural Networks"
            item["explanation"] = "Direct principle from foundational AI concepts."

        valid_questions, warnings = validator.validate_question_batch(raw_items, SAMPLE_LECTURE)
        self.assertEqual(len(valid_questions), 2)

        # 6. Interactive Quiz Assessment
        quiz_engine = QuizEngine()
        quiz_engine.db = self.db

        # Controlled vectors for evaluation
        v_student = np.array([0.9, 0.8, 0.1, 0.05])
        v_ref = np.array([0.92, 0.81, 0.12, 0.04])
        mock_eval_st = MagicMock()
        mock_eval_st.encode.return_value = np.array([v_student, v_ref])

        with patch.object(quiz_engine.evaluator.loader, "get_sentence_transformer", return_value=mock_eval_st):
            submission = {
                "q1": "It calculates the gradient of the loss function to update weights.",
                "q2": mcq_res["correct_letter"] # Correct MCQ letter
            }
            result = quiz_engine.evaluate_quiz_submission(
                lecture_id=content.content_id,
                questions=valid_questions,
                student_answers=submission
            )

        self.assertEqual(result["total_questions"], 2)
        self.assertEqual(result["correct_answers"], 2)
        self.assertGreaterEqual(result["overall_percentage"], 80.0)

        # 7. Learning Analytics
        analytics = LearningAnalytics()
        analytics.db = self.db
        dashboard_data = analytics.generate_dashboard_metrics()

        self.assertEqual(dashboard_data["summary"]["total_attempts"], 1)
        self.assertIn("Neural Networks", dashboard_data["strong_topics"])
        self.assertGreater(len(dashboard_data["recommendations"]), 0)

if __name__ == "__main__":
    unittest.main()
