"""
NEUROLECT AI - Phase 7 Comprehensive End-to-End System Integration Test
Validates the entire continuous workflow across all 6 phases:
Ingestion -> Understanding -> Question Intelligence -> Assessment -> Persistence -> Learning Analytics.
Strictly RAG-free.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import json
from pathlib import Path

from modules.content_engine import UnifiedContentEngine
from modules.summarizer import LectureSummarizer
from modules.topic_detector import TopicDetector
from modules.keyword_extractor import KeywordExtractor
from modules.ner_processor import NamedEntityRecognizer
from modules.concept_extractor import ConceptExtractor
from modules.question_generator import QuestionGenerator
from modules.quiz_engine import QuizEngine
from modules.analytics import LearningAnalytics
from database.db_manager import DatabaseManager
import config

SAMPLE_COMPREHENSIVE_LECTURE = """
Deep Learning and Neural Network Fundamentals.
Artificial Intelligence is the broad field of computer science dedicated to creating systems capable of performing tasks that typically require human intelligence.
Machine Learning is defined as a subset of artificial intelligence where algorithms learn from data to optimize decision functions without explicit rule programming.
Supervised Learning algorithms train mathematical models using paired input features and target output labels.
Neural Networks are composed of layers of interconnected artificial neurons that apply linear transformations followed by non-linear activation functions.
The backpropagation algorithm calculates gradients of the loss function with respect to weights using the mathematical chain rule of calculus.
Optimization algorithms such as Stochastic Gradient Descent and Adam iteratively update network parameters to minimize training error.
Overfitting occurs when a neural network memorizes noise in the training data, degrading its generalization performance on unseen test data.
Regularization techniques including Dropout and L2 weight decay are essential methods to reduce overfitting in deep networks.
Convolutional Neural Networks utilize local receptive fields and weight sharing to extract spatial features from computer vision inputs.
Recurrent Neural Networks and Transformer architectures process sequential data for natural language processing applications.
Evaluation metrics such as precision, recall, and F1-score are standard criteria for assessing classification performance.
"""

class TestEndToEndSystemIntegration(unittest.TestCase):
    def setUp(self):
        self.test_db_path = config.TEMP_DIR / "test_final_e2e.db"
        self.db = DatabaseManager(db_path=self.test_db_path)

        # Set offline flags to prevent accidental external network attempts
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def tearDown(self):
        if self.test_db_path.exists():
            self.test_db_path.unlink()

    def test_complete_end_to_end_lifecycle(self):
        """
        Executes the complete continuous workflow across all 6 phases:
        1. Ingestion -> 2. Summarization -> 3. Topics/Keywords -> 4. NER/Concepts ->
        5. Question Intelligence -> 6. Assessment -> 7. Persistence -> 8. Analytics
        """
        # -------------------------------------------------------------
        # STEP 1: Multimodal Ingestion to Unified LectureContent
        # -------------------------------------------------------------
        content = UnifiedContentEngine.process_content(
            source_type="text",
            source_name="AI Foundations Lecture",
            raw_text=SAMPLE_COMPREHENSIVE_LECTURE
        )
        self.assertIsNotNone(content.content_id)
        self.assertGreater(content.word_count, 150)
        self.assertGreaterEqual(len(content.chunks), 1)

        # Persist into database as done during UI ingestion
        self.db.save_lecture(
            lecture_id=content.content_id,
            title=content.source_name,
            source_type=content.source_type,
            word_count=content.word_count
        )

        # -------------------------------------------------------------
        # STEP 2: Lecture Understanding & NLP Analysis
        # -------------------------------------------------------------
        # 2a. Concept Extraction (linguistic regex definitions)
        c_extractor = ConceptExtractor()
        concepts = c_extractor.extract_concepts(content)
        self.assertGreaterEqual(len(concepts), 2)
        self.assertTrue(any(c.get("term") for c in concepts))

        # 2b. Mock SentenceTransformer for Topic Detection & Keyword Extraction
        mock_st = MagicMock()
        mock_st.encode.side_effect = lambda texts, **kw: np.random.randn(len(texts), 4)

        with patch("models.model_loader.model_loader.get_sentence_transformer", return_value=mock_st):
            # Topic Detection
            t_detector = TopicDetector()
            topics_data = t_detector.detect_topics(content)
            self.assertIn("topics", topics_data)
            self.assertGreaterEqual(len(topics_data["topics"]), 1)
            topic_titles = [t["title"] for t in topics_data["topics"]]

            # Keyword Extraction
            kw_extractor = KeywordExtractor()
            keywords = kw_extractor.extract_keywords(content, topics_data=topics_data, top_n=6)
            self.assertGreaterEqual(len(keywords), 3)
            candidate_terms = [k["keyword"] for k in keywords]

        # -------------------------------------------------------------
        # STEP 3: Question Intelligence Generation
        # -------------------------------------------------------------
        q_gen = QuestionGenerator()
        
        # Mock Flan-T5 output generator
        def mock_generate_side_effect(prompt, max_tokens=150):
            p = prompt.lower()
            if "multiple-choice" in p:
                return (
                    "Question: What algorithm calculates loss gradients to update neural network weights?\n"
                    "Answer: Backpropagation\n"
                    "Explanation: Backpropagation uses the chain rule to compute gradients."
                )
            elif "short-answer" in p:
                return (
                    "Question: What is supervised learning?\n"
                    "Answer: Learning algorithms trained on paired features and target labels.\n"
                    "Explanation: Foundational definition of supervised learning."
                )
            else:
                return (
                    "Question: Explain in detail how dropout regularization reduces overfitting.\n"
                    "Answer: Dropout deactivates random neurons during training to prevent noise co-adaptation.\n"
                    "Explanation: Mechanism of stochastic neuron deactivation."
                )

        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_tok.return_value = {"input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))}
        mock_tok.encode.side_effect = lambda t, **kw: list(range(len(t.split())))
        mock_tok.decode.return_value = "Mocked Output"

        with patch.object(q_gen.loader, "get_question_generator", return_value=(mock_tok, mock_model)):
            with patch.object(q_gen, "_generate_text", side_effect=mock_generate_side_effect):
                with patch.object(q_gen.distractor_gen, "generate_options", return_value={
                    "success": True,
                    "options": {"A": "Gradient Descent", "B": "Backpropagation", "C": "Dropout", "D": "Adam"},
                    "correct_letter": "B",
                    "correct_answer": "Backpropagation",
                    "distractors": ["Gradient Descent", "Dropout", "Adam"]
                }):
                    q_res = q_gen.generate_questions(
                        lecture_content=content,
                        question_types=["Multiple Choice (MCQ)", "Short Answer", "Descriptive"],
                        num_questions=3,
                        candidate_keywords=candidate_terms,
                        topics_data=topics_data
                    )

        questions = q_res["questions"]
        self.assertEqual(len(questions), 3)
        for q in questions:
            self.assertIn("id", q)
            self.assertIn("bloom_level", q)
            self.assertIn("difficulty", q)
            self.assertIn("topic", q)
            self.assertIn("source_ref", q)

        # -------------------------------------------------------------
        # STEP 4: Interactive Assessment & Evaluation
        # -------------------------------------------------------------
        quiz_engine = QuizEngine()
        quiz_engine.db = self.db

        # Student submissions:
        # Q1 (MCQ): Selects correct option 'B'
        # Q2 (Short Answer): Good conceptual paraphrase
        # Q3 (Descriptive): Partial explanation
        student_answers = {
            questions[0]["id"]: "B",
            questions[1]["id"]: "Algorithms that train mathematical models using labeled input-output pairs.",
            questions[2]["id"]: "It deactivates neurons during training."
        }

        # Mock sentence transformer embeddings for student responses
        v_good = np.array([0.9, 0.85, 0.1])
        v_partial = np.array([0.6, 0.5, 0.3])
        v_ref = np.array([0.92, 0.82, 0.11])

        def mock_eval_encode(texts, **kw):
            if any("labeled" in t for t in texts):
                return np.array([v_good, v_ref])
            return np.array([v_partial, v_ref])

        mock_eval_st = MagicMock()
        mock_eval_st.encode.side_effect = mock_eval_encode

        with patch.object(quiz_engine.evaluator.loader, "get_sentence_transformer", return_value=mock_eval_st):
            quiz_result = quiz_engine.evaluate_quiz_submission(
                lecture_id=content.content_id,
                questions=questions,
                student_answers=student_answers
            )

        self.assertIsNotNone(quiz_result["attempt_id"])
        self.assertEqual(quiz_result["total_questions"], 3)
        self.assertGreaterEqual(quiz_result["overall_percentage"], 60.0)

        # -------------------------------------------------------------
        # STEP 5: Persistence & History Verification
        # -------------------------------------------------------------
        history = self.db.get_quiz_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["lecture_title"], "AI Foundations Lecture")
        self.assertEqual(history[0]["source_type"], "text")

        details = self.db.get_attempt_details(quiz_result["attempt_id"])
        self.assertIsNotNone(details)
        self.assertEqual(len(details["answers"]), 3)
        self.assertIn("evaluation_method", details["answers"][0])
        self.assertIn("awarded_marks", details["answers"][0])

        # -------------------------------------------------------------
        # STEP 6: Learning Analytics & Diagnostics
        # -------------------------------------------------------------
        analytics = LearningAnalytics(db=self.db)
        metrics = analytics.generate_dashboard_metrics()

        self.assertEqual(metrics["summary"]["total_attempts"], 1)
        self.assertEqual(metrics["summary"]["total_questions"], 3)
        self.assertGreaterEqual(metrics["summary"]["average_score"], 60.0)
        self.assertGreater(len(metrics["topic_stats"]), 0)
        self.assertGreater(len(metrics["recommendations"]), 0)
        self.assertEqual(metrics["progress_tracking"]["total_attempts"], 1)

if __name__ == "__main__":
    unittest.main()
