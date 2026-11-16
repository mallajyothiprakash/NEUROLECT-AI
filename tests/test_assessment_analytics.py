"""
Unit Tests for Assessment Engine, Semantic Evaluation, SQLite DB, and Analytics
Tests quiz grading, semantic scoring, SQLite persistence, and learning analytics.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path
from database.db_manager import DatabaseManager
from modules.answer_evaluator import SemanticAnswerEvaluator
from modules.quiz_engine import QuizEngine
from modules.analytics import LearningAnalytics
import config

class TestAssessmentAndAnalytics(unittest.TestCase):
    def setUp(self):
        # Use a temporary SQLite database for tests
        self.test_db_path = config.TEMP_DIR / "test_neurolect.db"
        self.db = DatabaseManager(db_path=self.test_db_path)
        self.evaluator = SemanticAnswerEvaluator()

    def tearDown(self):
        if self.test_db_path.exists():
            self.test_db_path.unlink()

    def test_semantic_answer_evaluator(self):
        reference = "Gradient descent is an optimization algorithm used to minimize the loss function by updating weights iteratively."
        
        # Test close student answer
        good_student = "It is an optimization method that minimizes loss by iteratively adjusting parameters."
        
        # Mock transformer encoder to return realistic cosine similarity embeddings
        mock_model = MagicMock()
        # Vector 1 and Vector 2 with high cosine similarity (0.85)
        v1 = np.array([1.0, 0.8, 0.2, 0.1])
        v2 = np.array([0.9, 0.85, 0.15, 0.1])
        mock_model.encode.return_value = np.array([v1, v2])

        with patch.object(self.evaluator.loader, "get_sentence_transformer", return_value=mock_model):
            eval_good = self.evaluator.evaluate_answer(good_student, reference)
            self.assertGreaterEqual(eval_good["score_percentage"], 65.0)
            self.assertTrue(eval_good["is_correct"])

        # Test poor student answer with low similarity
        bad_student = "Photosynthesis is the process plants use to convert light into energy."
        v_bad = np.array([-0.5, -0.2, 0.9, -0.4])
        mock_model.encode.return_value = np.array([v_bad, v2])

        with patch.object(self.evaluator.loader, "get_sentence_transformer", return_value=mock_model):
            eval_bad = self.evaluator.evaluate_answer(bad_student, reference)
            self.assertLess(eval_bad["score_percentage"], 45.0)
            self.assertFalse(eval_bad["is_correct"])

    def test_database_persistence_and_analytics(self):
        # Save lecture
        self.db.save_lecture("lec_1", "Intro to Deep Learning", "text", 250)

        # Save quiz attempt
        answers = [
            {
                "question": "What is backpropagation?",
                "type": "Short Answer",
                "student_answer": "Algorithm for calculating gradients",
                "correct_answer": "Gradient computation algorithm",
                "is_correct": True,
                "score": 90.0,
                "feedback": "Great answer",
                "difficulty": "Medium",
                "bloom_level": "Understand",
                "topic": "Neural Networks"
            },
            {
                "question": "What does CNN stand for?",
                "type": "MCQ",
                "student_answer": "Convolutional Neural Network",
                "correct_answer": "Convolutional Neural Network",
                "is_correct": True,
                "score": 100.0,
                "feedback": "Correct",
                "difficulty": "Easy",
                "bloom_level": "Remember",
                "topic": "Computer Vision"
            }
        ]

        saved = self.db.save_quiz_attempt(
            attempt_id="atm_test_1",
            lecture_id="lec_1",
            total_questions=2,
            correct_answers=2,
            total_score=190.0,
            percentage=95.0,
            difficulty_selected="Mixed",
            bloom_summary={"Understand": {"count": 1, "correct": 1}},
            answers=answers
        )
        self.assertTrue(saved)

        # Retrieve history
        history = self.db.get_quiz_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["percentage"], 95.0)

        # Verify Analytics calculation
        analytics = LearningAnalytics()
        analytics.db = self.db
        metrics = analytics.generate_dashboard_metrics()
        
        self.assertEqual(metrics["summary"]["total_attempts"], 1)
        self.assertEqual(metrics["summary"]["total_questions"], 2)
        self.assertIn("Neural Networks", metrics["strong_topics"])

if __name__ == "__main__":
    unittest.main()
