"""
NEUROLECT AI - Dedicated Phase 5 Automated Assessment Engine Tests
Tests deterministic MCQ grading, semantic free-text evaluation, concept coverage,
partial-credit scoring, negation contrast detection, human review flags,
quiz filtering, and SQLite database persistence.
Strictly RAG-free.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from pathlib import Path

from modules.answer_evaluator import (
    SemanticAnswerEvaluator,
    extract_technical_concepts,
    detect_polarity_inversion
)
from modules.quiz_engine import QuizEngine
from database.db_manager import DatabaseManager
import config

class TestAutomatedAssessmentEngine(unittest.TestCase):
    def setUp(self):
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.test_db_path = config.TEMP_DIR / "test_assessment_phase5.db"
        self.db = DatabaseManager(db_path=self.test_db_path)
        self.evaluator = SemanticAnswerEvaluator()
        self.quiz_engine = QuizEngine()
        self.quiz_engine.db = self.db
        self.quiz_engine.evaluator = self.evaluator

        # Mock sentence transformer for offline unit testing
        self.mock_st = MagicMock()
        # Default encode behavior: return unit vectors with realistic cosine similarity
        def mock_encode(texts, **kwargs):
            if len(texts) == 2:
                # If texts are exact match, return identical vectors
                if texts[0].strip().lower() == texts[1].strip().lower():
                    return np.array([[1.0, 0.5, 0.2], [1.0, 0.5, 0.2]])
                # Otherwise return high-similarity default vectors
                return np.array([[0.9, 0.8, 0.1], [0.85, 0.82, 0.12]])
            return np.ones((len(texts), 3))
        self.mock_st.encode.side_effect = mock_encode

        # Patch loader.get_sentence_transformer by default
        self.st_patcher = patch.object(self.evaluator.loader, "get_sentence_transformer", return_value=self.mock_st)
        self.st_patcher.start()

    def tearDown(self):
        self.st_patcher.stop()
        if self.test_db_path.exists():
            self.test_db_path.unlink()

    # ------------------------------------------------------------------------
    # 1. Deterministic MCQ Evaluation
    # ------------------------------------------------------------------------
    def test_mcq_evaluation_correct_and_incorrect(self):
        """Verifies deterministic MCQ grading across letters, option prefixes, and text."""
        mcq_q = [
            {
                "id": "q_mcq_1",
                "type": "Multiple Choice (MCQ)",
                "question": "Which algorithm computes gradients in neural networks?",
                "options": {
                    "A": "K-Means",
                    "B": "Backpropagation",
                    "C": "Principal Component Analysis",
                    "D": "DBSCAN"
                },
                "correct_letter": "B",
                "correct_answer": "Backpropagation",
                "difficulty": "Easy",
                "bloom_level": "Remember",
                "topic": "Neural Networks"
            }
        ]

        # Case 1: Exact letter 'B'
        res1 = self.quiz_engine.evaluate_quiz_submission(None, mcq_q, {"q_mcq_1": "B"})
        item1 = res1["items"][0]
        self.assertTrue(item1["is_correct"])
        self.assertEqual(item1["score"], 100.0)
        self.assertEqual(item1["awarded_marks"], 1.0)
        self.assertEqual(item1["evaluation_method"], "Deterministic Exact Option Match")
        self.assertFalse(item1["human_review_recommended"])

        # Case 2: Lowercase letter 'b'
        res2 = self.quiz_engine.evaluate_quiz_submission(None, mcq_q, {"q_mcq_1": "b"})
        self.assertTrue(res2["items"][0]["is_correct"])

        # Case 3: Prefixed option string 'B: Backpropagation'
        res3 = self.quiz_engine.evaluate_quiz_submission(None, mcq_q, {"q_mcq_1": "B: Backpropagation"})
        self.assertTrue(res3["items"][0]["is_correct"])

        # Case 4: Incorrect option 'A'
        res4 = self.quiz_engine.evaluate_quiz_submission(None, mcq_q, {"q_mcq_1": "A"})
        item4 = res4["items"][0]
        self.assertFalse(item4["is_correct"])
        self.assertEqual(item4["score"], 0.0)
        self.assertEqual(item4["awarded_marks"], 0.0)
        self.assertIn("option B: 'Backpropagation'", item4["feedback"])

    # ------------------------------------------------------------------------
    # 2. Empty and Trivial Answers
    # ------------------------------------------------------------------------
    def test_empty_and_trivial_answers(self):
        """Verifies zero score and helpful feedback for empty or whitespace-only answers."""
        ref = "Supervised learning algorithms build mathematical models from labeled training data."
        
        # Test empty string
        res_empty = self.evaluator.evaluate_answer("", ref, question_type="Short Answer", max_marks=2.0)
        self.assertEqual(res_empty["score_percentage"], 0.0)
        self.assertEqual(res_empty["awarded_marks"], 0.0)
        self.assertFalse(res_empty["is_correct"])
        self.assertEqual(res_empty["grade"], "Unsatisfactory")

        # Test single word trivial response
        res_trivial = self.evaluator.evaluate_answer("data", ref, question_type="Short Answer", max_marks=2.0)
        self.assertEqual(res_trivial["score_percentage"], 0.0)
        self.assertFalse(res_trivial["is_correct"])

    # ------------------------------------------------------------------------
    # 3. Exact and Semantically Equivalent Answers
    # ------------------------------------------------------------------------
    def test_exact_and_semantically_equivalent_answers(self):
        """Verifies high scoring for exact and semantically paraphrased student answers."""
        reference = "Backpropagation calculates the gradient of the loss function with respect to layer weights."

        # Case 1: Exact string match
        res_exact = self.evaluator.evaluate_answer(
            student_answer=reference,
            reference_answer=reference,
            question_type="Short Answer",
            max_marks=2.0
        )
        self.assertGreaterEqual(res_exact["score_percentage"], 90.0)
        self.assertEqual(res_exact["grade"], "Excellent")
        self.assertTrue(res_exact["is_correct"])

        # Case 2: Semantically equivalent paraphrase with mocked SentenceTransformer (similarity = 0.88)
        paraphrase = "It computes gradients of the error loss to iteratively adjust the neural network weights."
        v_student = np.array([0.9, 0.85, 0.1, 0.05])
        v_ref = np.array([0.92, 0.82, 0.12, 0.04])
        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([v_student, v_ref])

        with patch.object(self.evaluator.loader, "get_sentence_transformer", return_value=mock_st):
            res_para = self.evaluator.evaluate_answer(
                student_answer=paraphrase,
                reference_answer=reference,
                question_type="Short Answer",
                max_marks=2.0
            )
            self.assertGreaterEqual(res_para["score_percentage"], 75.0)
            self.assertTrue(res_para["is_correct"])
            self.assertIn(res_para["grade"], ["Excellent", "Good"])

    # ------------------------------------------------------------------------
    # 4. Partially Correct Answers and Partial Credit
    # ------------------------------------------------------------------------
    def test_partially_correct_answers(self):
        """Verifies partial credit scoring and human review recommendation on borderline answers."""
        reference = "Overfitting occurs when a model learns noise in training data instead of generalizing to new data."
        partial_student = "Overfitting is when a model has issues with training data."

        # Mock medium similarity (0.50)
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.5, 0.866, 0.0])
        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([v1, v2])

        with patch.object(self.evaluator.loader, "get_sentence_transformer", return_value=mock_st):
            res = self.evaluator.evaluate_answer(
                student_answer=partial_student,
                reference_answer=reference,
                question_type="Conceptual",
                max_marks=5.0
            )
            # Should receive partial credit
            self.assertGreaterEqual(res["score_percentage"], 35.0)
            self.assertLess(res["score_percentage"], 75.0)
            self.assertGreater(res["awarded_marks"], 0.0)
            self.assertLess(res["awarded_marks"], 5.0)
            # Borderline score triggers human review
            self.assertTrue(res["human_review_recommended"])

    # ------------------------------------------------------------------------
    # 5. Incorrect and Off-Topic Answers
    # ------------------------------------------------------------------------
    def test_incorrect_and_off_topic_answers(self):
        """Verifies low scores and improvement suggestions for completely off-topic answers."""
        reference = "Convolutional neural networks use kernel convolutions for feature extraction from images."
        off_topic = "Photosynthesis converts solar energy into chemical sugars in green plant leaves."

        v_bad = np.array([-0.8, -0.6, 0.7, 0.5])
        v_ref = np.array([0.9, 0.8, 0.1, 0.05])
        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([v_bad, v_ref])

        with patch.object(self.evaluator.loader, "get_sentence_transformer", return_value=mock_st):
            res = self.evaluator.evaluate_answer(
                student_answer=off_topic,
                reference_answer=reference,
                question_type="Descriptive",
                max_marks=10.0
            )
            self.assertLess(res["score_percentage"], 35.0)
            self.assertEqual(res["awarded_marks"], round((res["score_percentage"] / 100.0) * 10.0, 1))
            self.assertFalse(res["is_correct"])
            self.assertEqual(res["grade"], "Needs Improvement")

    # ------------------------------------------------------------------------
    # 6. Concept Coverage Analysis
    # ------------------------------------------------------------------------
    def test_concept_coverage_analysis(self):
        """Verifies accurate identification of covered vs missing technical concepts."""
        reference = "Regularization methods such as dropout and weight decay prevent overfitting."
        student = "Dropout is a method to prevent overfitting in models."

        res = self.evaluator.evaluate_answer(
            student_answer=student,
            reference_answer=reference,
            question_type="Short Answer"
        )
        self.assertIn("dropout", res["covered_concepts"])
        self.assertIn("overfitting", res["covered_concepts"])
        self.assertIn("decay", res["missing_concepts"])

    # ------------------------------------------------------------------------
    # 7. Polarity Inversion and Negation Detection
    # ------------------------------------------------------------------------
    def test_negation_inversion_detection(self):
        """Verifies that inserting a negation token triggers a penalty and human review."""
        ref = "Gradient descent is an optimization algorithm that iteratively minimizes loss."
        # Student negates the core premise
        student_negated = "Gradient descent is not an optimization algorithm that minimizes loss."

        self.assertTrue(detect_polarity_inversion(student_negated, ref))

        v1 = np.array([0.9, 0.85, 0.1, 0.05])
        v2 = np.array([0.92, 0.82, 0.12, 0.04])
        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([v1, v2])

        with patch.object(self.evaluator.loader, "get_sentence_transformer", return_value=mock_st):
            res = self.evaluator.evaluate_answer(
                student_answer=student_negated,
                reference_answer=ref,
                question_type="Descriptive",
                max_marks=10.0
            )
            # Score should be penalized by polarity discount and flagged for instructor review
            self.assertTrue(res["human_review_recommended"])
            self.assertIn("negation", res["review_reason"].lower())

    # ------------------------------------------------------------------------
    # 8. Missing Reference Answer Handling
    # ------------------------------------------------------------------------
    def test_missing_reference_answer(self):
        """Verifies graceful handling when question lacks a reference answer."""
        res = self.evaluator.evaluate_answer("Some student response", "")
        self.assertEqual(res["score_percentage"], 0.0)
        self.assertEqual(res["grade"], "Unrated")
        self.assertTrue(res["human_review_recommended"])
        self.assertIn("Missing Reference", res["evaluation_method"])

    # ------------------------------------------------------------------------
    # 9. Quiz Question Filtering by Topic, Type, Difficulty, and Bloom
    # ------------------------------------------------------------------------
    def test_quiz_question_filtering(self):
        """Verifies filtering by question type, topic, difficulty, and Bloom's taxonomy."""
        pool = [
            {"id": "1", "type": "MCQ", "topic": "Deep Learning", "difficulty": "Easy", "bloom_level": "Remember"},
            {"id": "2", "type": "Short Answer", "topic": "Deep Learning", "difficulty": "Medium", "bloom_level": "Understand"},
            {"id": "3", "type": "Analytical", "topic": "Optimization", "difficulty": "Hard", "bloom_level": "Analyze"},
            {"id": "4", "type": "MCQ", "topic": "Optimization", "difficulty": "Easy", "bloom_level": "Apply"},
        ]

        # Filter by Topic
        f_topic = self.quiz_engine.filter_questions(pool, num_questions=5, selected_topics=["Optimization"])
        self.assertEqual(len(f_topic), 2)
        self.assertTrue(all(q["topic"] == "Optimization" for q in f_topic))

        # Filter by Difficulty and Type
        f_diff_type = self.quiz_engine.filter_questions(
            pool,
            num_questions=5,
            selected_difficulties=["Easy"],
            selected_types=["MCQ"]
        )
        self.assertEqual(len(f_diff_type), 2)

    # ------------------------------------------------------------------------
    # 10. Database Persistence & History Verification
    # ------------------------------------------------------------------------
    def test_database_persistence_and_attempt_details(self):
        """Verifies persistence of enriched assessment fields and accurate retrieval."""
        self.db.save_lecture("lec_audit", "NLP Assessment Systems", "text", 500)

        questions = [
            {
                "id": "q1",
                "type": "Multiple Choice (MCQ)",
                "question": "What does NLP stand for?",
                "options": {"A": "Natural Language Processing", "B": "Neural Logic", "C": "Network Layer", "D": "Non-Linear"},
                "correct_letter": "A",
                "correct_answer": "Natural Language Processing",
                "difficulty": "Easy",
                "bloom_level": "Remember",
                "topic": "NLP Foundations"
            },
            {
                "id": "q2",
                "type": "Short Answer",
                "question": "What is tokenization?",
                "correct_answer": "Tokenization is the process of breaking text into constituent words or subwords.",
                "difficulty": "Medium",
                "bloom_level": "Understand",
                "topic": "Text Processing"
            }
        ]

        submission = {
            "q1": "A",
            "q2": "It is breaking raw text down into word units or tokens."
        }

        # Mock sentence transformer
        mock_st = MagicMock()
        mock_st.encode.return_value = np.array([[0.9, 0.8, 0.1], [0.88, 0.82, 0.09]])

        with patch.object(self.evaluator.loader, "get_sentence_transformer", return_value=mock_st):
            quiz_result = self.quiz_engine.evaluate_quiz_submission(
                lecture_id="lec_audit",
                questions=questions,
                student_answers=submission
            )

        attempt_id = quiz_result["attempt_id"]
        self.assertIsNotNone(attempt_id)
        self.assertEqual(quiz_result["total_questions"], 2)

        # Retrieve from DB
        details = self.db.get_attempt_details(attempt_id)
        self.assertIsNotNone(details)
        self.assertEqual(len(details["answers"]), 2)
        
        # Verify schema columns are preserved
        ans1 = details["answers"][0]
        self.assertIn("evaluation_method", ans1)
        self.assertIn("awarded_marks", ans1)
        self.assertIn("max_marks", ans1)
        self.assertIn("human_review_recommended", ans1)

        # Verify history
        history = self.db.get_quiz_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["lecture_title"], "NLP Assessment Systems")

if __name__ == "__main__":
    unittest.main()
