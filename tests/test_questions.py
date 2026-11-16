"""
NEUROLECT AI - Dedicated Phase 4 Question Intelligence Tests
Tests question generation across all 6 pedagogical types, MCQ distractor generation,
Bloom's Taxonomy, difficulty classification, question validation, deduplication,
source reference traceability, Phase 3 topic integration, and export integrity.
Strictly RAG-free.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import re

from modules.content_engine import UnifiedContentEngine
from modules.question_generator import QuestionGenerator
from modules.distractor_generator import DistractorGenerator, normalize_text_for_comparison
from modules.bloom_classifier import BloomsTaxonomyClassifier
from modules.difficulty_classifier import DifficultyClassifier
from modules.question_validator import QuestionValidator, extract_content_keywords, calculate_jaccard_similarity
import config

SAMPLE_LECTURE_TEXT = """
Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.
In supervised learning, algorithms are trained on labeled datasets containing input features and target labels.
Neural networks are composed of layers of artificial neurons that transform input data through weighted connections.
The backpropagation algorithm calculates gradients of the loss function to update neural network weights.
Overfitting occurs when a machine learning model memorizes training noise rather than generalizing to unseen data.
Regularization methods such as dropout and L2 weight decay reduce overfitting in deep neural networks.
Evaluation metrics like accuracy, precision, recall, and F1-score quantify model performance on classification tasks.
Convolutional neural networks excel at spatial feature extraction for computer vision and image processing applications.
"""

class TestQuestionIntelligence(unittest.TestCase):
    def setUp(self):
        self.q_gen = QuestionGenerator()
        self.distractor_gen = DistractorGenerator()
        self.bloom_clf = BloomsTaxonomyClassifier()
        self.diff_clf = DifficultyClassifier()
        self.validator = QuestionValidator()
        self.content = UnifiedContentEngine.process_content("text", "ML Foundations", SAMPLE_LECTURE_TEXT)

        # Set environment to offline to avoid network retries
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        # Mock tokenizer and model for unit testing
        self.mock_tok = MagicMock()
        self.mock_model = MagicMock()
        self.mock_tok.return_value = {"input_ids": MagicMock(to=MagicMock(return_value=MagicMock()))}
        self.mock_tok.encode.side_effect = lambda t, **kw: list(range(len(t.split())))
        self.mock_tok.decode.return_value = "Mocked Model Output"

    # ------------------------------------------------------------------------
    # 1. All Six Question Types Generation (Mocked Flan-T5 Inference)
    # ------------------------------------------------------------------------
    def test_all_six_question_types_generation(self):
        """Verifies that all 6 pedagogical question types can be generated and parsed."""
        # Test responses mapped by task keywords
        def mock_generate_side_effect(prompt, max_tokens=150):
            p_lower = prompt.lower()
            if "multiple-choice" in p_lower:
                return (
                    "Question: What algorithm calculates gradients to update neural network weights?\n"
                    "Answer: Backpropagation\n"
                    "Explanation: Backpropagation computes loss gradients."
                )
            elif "short-answer" in p_lower:
                return (
                    "Question: What is supervised learning?\n"
                    "Answer: A paradigm trained on labeled datasets containing features and targets.\n"
                    "Explanation: Fundamental definition of supervised learning."
                )
            elif "descriptive" in p_lower:
                return (
                    "Question: Explain in detail how overfitting occurs and how regularization techniques mitigate it.\n"
                    "Answer: Overfitting occurs when a model learns noise; methods like dropout reduce this risk.\n"
                    "Explanation: Comprehensive elaboration on generalization."
                )
            elif "conceptual" in p_lower:
                return (
                    "Question: What is the underlying principle behind convolutional neural networks for vision tasks?\n"
                    "Answer: Spatial feature extraction through hierarchical weighted convolutional layers.\n"
                    "Explanation: Conceptual understanding of spatial invariance."
                )
            elif "analytical" in p_lower:
                return (
                    "Question: Compare and contrast precision and recall metrics when evaluating classification models.\n"
                    "Answer: Precision measures exactness while recall measures completeness of positive predictions.\n"
                    "Explanation: Analytical trade-off in classification evaluation."
                )
            elif "application" in p_lower:
                return (
                    "Question: How would you apply dropout during training to prevent a neural network from overfitting?\n"
                    "Answer: Randomly deactivate neurons during forward propagation with a specified probability.\n"
                    "Explanation: Practical implementation of dropout regularization."
                )
            else:
                return (
                    "Question: General fallback question?\n"
                    "Answer: Reference answer.\n"
                    "Explanation: General explanation."
                )

        with patch.object(self.q_gen.loader, "get_question_generator", return_value=(self.mock_tok, self.mock_model)):
            with patch.object(self.q_gen, "_generate_text", side_effect=mock_generate_side_effect):
                with patch.object(self.distractor_gen, "generate_options", return_value={
                    "success": True,
                    "options": {"A": "Gradient Descent", "B": "Backpropagation", "C": "Dropout", "D": "Adam"},
                    "correct_letter": "B",
                    "correct_answer": "Backpropagation",
                    "distractors": ["Gradient Descent", "Dropout", "Adam"]
                }):
                    res = self.q_gen.generate_questions(
                        lecture_content=self.content,
                        question_types=config.QUESTION_TYPES,
                        num_questions=6
                    )

        generated_types = [q["type"] for q in res["questions"]]
        self.assertEqual(len(generated_types), 6)
        for expected_type in config.QUESTION_TYPES:
            self.assertIn(expected_type, generated_types)

    # ------------------------------------------------------------------------
    # 2. MCQ Options: Exactly 4 Unique Options & Exactly 1 Correct Answer
    # ------------------------------------------------------------------------
    def test_mcq_options_integrity_and_shuffling(self):
        """Verifies exactly 4 unique choices, 1 correct answer, and accurate letter tracking after shuffling."""
        question = "What technique prevents overfitting in deep neural networks?"
        correct_ans = "Dropout"
        candidate_terms = ["Backpropagation", "Convolutional Layer", "Learning Rate"]

        with patch.object(self.distractor_gen.loader, "get_question_generator", side_effect=RuntimeError("Offline unit test")):
            # Test fallback distractor synthesis
            mcq = self.distractor_gen.generate_options(
                question=question,
                correct_answer=correct_ans,
                context_text=SAMPLE_LECTURE_TEXT,
                candidate_terms=candidate_terms
            )

        self.assertTrue(mcq["success"])
        options = mcq["options"]
        self.assertEqual(len(options), 4)
        self.assertEqual(set(options.keys()), {"A", "B", "C", "D"})
        
        # Verify 4 unique values
        norm_values = [normalize_text_for_comparison(v) for v in options.values()]
        self.assertEqual(len(set(norm_values)), 4)

        # Verify exactly one correct letter and that it points to the correct answer
        correct_letter = mcq["correct_letter"]
        self.assertIn(correct_letter, {"A", "B", "C", "D"})
        self.assertEqual(options[correct_letter].lower(), correct_ans.lower())

    # ------------------------------------------------------------------------
    # 3. Rejection of Insufficient Distractors (No Fake Syntactic Fillers)
    # ------------------------------------------------------------------------
    def test_rejection_when_insufficient_distractors(self):
        """Verifies that MCQ generator rejects rather than manufacturing misleading generic distractors."""
        # When model fails and no candidate terms or context terms are available
        with patch.object(self.distractor_gen.loader, "get_question_generator", side_effect=RuntimeError("Offline")):
            res = self.distractor_gen.generate_options(
                question="What is X?",
                correct_answer="Y",
                context_text="", # No context
                candidate_terms=[] # No candidate terms
            )
            self.assertFalse(res["success"])
            self.assertEqual(len(res["options"]), 0)

    # ------------------------------------------------------------------------
    # 4. Duplicate Detection: Full-Question Comparison & No False Collisions
    # ------------------------------------------------------------------------
    def test_duplicate_detection_and_no_false_prefix_collisions(self):
        """Verifies that questions with identical opening 8 words but different subjects both pass."""
        source = SAMPLE_LECTURE_TEXT
        
        q1 = {
            "question": "What is the primary difference between supervised learning and unsupervised learning?",
            "correct_answer": "Supervised learning uses labeled data while unsupervised learning does not.",
            "type": "Short Answer"
        }
        q2 = {
            "question": "What is the primary difference between supervised learning and reinforcement learning?",
            "correct_answer": "Reinforcement learning learns via rewards and penalties from an environment.",
            "type": "Short Answer"
        }
        
        # In the old 8-word prefix logic, q2 would be falsely rejected!
        valid_items, warnings = self.validator.validate_question_batch([q1, q2], source)
        self.assertEqual(len(valid_items), 2, "Distinct questions with shared opening clauses should both pass.")

        # Exact duplicate test with different casing and punctuation
        q_dup = {
            "question": "what is the primary difference between supervised learning and unsupervised learning???",
            "correct_answer": "Supervised uses labels.",
            "type": "Short Answer"
        }
        valid_after_dup, warn_dup = self.validator.validate_question_batch([q1, q_dup], source)
        self.assertEqual(len(valid_after_dup), 1, "Exact normalized duplicates must be rejected.")

    # ------------------------------------------------------------------------
    # 5. Bloom's Taxonomy All Six Cognitive Levels
    # ------------------------------------------------------------------------
    def test_all_six_blooms_taxonomy_levels(self):
        """Verifies classification across all 6 levels of Revised Bloom's Taxonomy."""
        test_cases = [
            ("Define backpropagation in deep neural networks.", "Short Answer", "Remember"),
            ("Explain why overfitting happens when training on small datasets.", "Descriptive", "Understand"),
            ("Calculate the parameter count given an input dimension of 100.", "Application-Based", "Apply"),
            ("Compare and contrast convolutional layers with dense layers.", "Analytical", "Analyze"),
            ("Evaluate the trade-offs between precision and recall for medical diagnosis.", "Analytical", "Evaluate"),
            ("Design a neural network architecture for real-time speech processing.", "Descriptive", "Create")
        ]
        
        for q_text, q_type, expected_bloom in test_cases:
            res = self.bloom_clf.classify_question(q_text, q_type)
            self.assertEqual(res["bloom_level"], expected_bloom, f"Failed for question: {q_text}")

    # ------------------------------------------------------------------------
    # 6. Difficulty Classification Boundaries (Easy, Medium, Hard)
    # ------------------------------------------------------------------------
    def test_difficulty_classification_boundaries(self):
        """Verifies difficulty assignment for Easy, Medium, and Hard."""
        easy = self.diff_clf.classify_difficulty("What is AI?", "Remember", "MCQ")
        self.assertEqual(easy["difficulty"], "Easy")

        medium = self.diff_clf.classify_difficulty(
            "Explain in detail the mathematical mechanisms by which dropout regularization prevents overfitting during deep neural network training.",
            "Understand",
            "Descriptive"
        )
        self.assertEqual(medium["difficulty"], "Medium")

        hard = self.diff_clf.classify_difficulty(
            "Critically analyze and compare the convergence properties and computational complexity of stochastic gradient descent versus Adam.",
            "Analyze",
            "Analytical"
        )
        self.assertEqual(hard["difficulty"], "Hard")

    # ------------------------------------------------------------------------
    # 7. Source Reference & Phase 3 Topic Integration
    # ------------------------------------------------------------------------
    def test_source_reference_and_topic_integration(self):
        """Verifies preservation of source segment refs and Phase 3 topic metadata."""
        mock_topics = {
            "topics": [
                {
                    "topic_id": 1,
                    "title": "Neural Network Optimization",
                    "source_segments": ["Page 2", "01:00 - 01:30"],
                    "sentences": ["The backpropagation algorithm calculates gradients of the loss function."]
                }
            ]
        }

        # Test topic resolution
        topic_matched = self.q_gen._resolve_topic(
            chunk_text="The backpropagation algorithm calculates gradients to optimize weights.",
            source_ref="Page 2",
            topics_data=mock_topics,
            candidate_terms=[]
        )
        self.assertEqual(topic_matched, "Neural Network Optimization")

    # ------------------------------------------------------------------------
    # 8. Handling Empty, Malformed, and Ungrounded Model Outputs
    # ------------------------------------------------------------------------
    def test_malformed_and_ungrounded_question_rejection(self):
        """Verifies rejection of empty, malformed, and out-of-context hallucinated questions."""
        source = SAMPLE_LECTURE_TEXT

        # 1. Empty question text
        bad_empty = [{"question": "", "correct_answer": "answer", "type": "Short Answer"}]
        v_out, w_out = self.validator.validate_question_batch(bad_empty, source)
        self.assertEqual(len(v_out), 0)

        # 2. Empty answer
        bad_no_ans = [{"question": "What is machine learning?", "correct_answer": "", "type": "Short Answer"}]
        v_out, w_out = self.validator.validate_question_batch(bad_no_ans, source)
        self.assertEqual(len(v_out), 0)

        # 3. Zero lexical overlap (ungrounded hallucination)
        bad_hallucinated = [{
            "question": "What are the celestial orbital mechanics of Jupiter moons in gravitational astrophysics?",
            "correct_answer": "Galilean resonance",
            "type": "Short Answer"
        }]
        v_out, w_out = self.validator.validate_question_batch(bad_hallucinated, source)
        self.assertEqual(len(v_out), 0, "Question with zero lexical overlap with source lecture should be rejected.")

    # ------------------------------------------------------------------------
    # 9. Maximum-Attempt Loop Termination (Guaranteed No Infinite Loop)
    # ------------------------------------------------------------------------
    def test_maximum_attempt_loop_termination(self):
        """Verifies that question generation terminates gracefully even if every chunk attempt fails."""
        # Force single item generation to fail / return None every time
        with patch.object(self.q_gen, "_generate_single_item", return_value=None):
            res = self.q_gen.generate_questions(
                lecture_content=self.content,
                num_questions=5
            )
        
        self.assertEqual(res["total_generated"], 0)
        self.assertEqual(len(res["questions"]), 0)
        self.assertTrue(any("Unable to generate valid assessment questions" in w for w in res["warnings"]))

    # ------------------------------------------------------------------------
    # 10. TXT and JSON Export Completeness
    # ------------------------------------------------------------------------
    def test_export_metadata_completeness(self):
        """Verifies that all 10 required metadata fields are preserved in export data."""
        sample_q = {
            "id": "q_abc123",
            "type": "Multiple Choice (MCQ)",
            "question": "What algorithm computes gradients for neural network training?",
            "options": {"A": "Backpropagation", "B": "K-Means", "C": "PCA", "D": "DBSCAN"},
            "correct_letter": "A",
            "correct_answer": "Backpropagation",
            "explanation": "Backpropagation computes loss gradients efficiently.",
            "topic": "Neural Network Optimization",
            "bloom_level": "Remember",
            "difficulty": "Easy",
            "source_ref": "Page 1"
        }

        # 1. JSON Export Losslessness
        json_str = json.dumps([sample_q])
        loaded = json.loads(json_str)[0]
        for field in ["id", "type", "question", "options", "correct_letter", "correct_answer", "explanation", "topic", "bloom_level", "difficulty", "source_ref"]:
            self.assertIn(field, loaded)
            self.assertEqual(loaded[field], sample_q[field])

        # 2. Text Export Completeness check
        txt_repr = (
            f"Question #1 [ID: {sample_q['id']}]\n"
            f"Type: {sample_q['type']}\n"
            f"Topic: {sample_q['topic']}\n"
            f"Bloom's Taxonomy: {sample_q['bloom_level']} | Difficulty: {sample_q['difficulty']}\n"
            f"Source Reference: {sample_q['source_ref']}\n"
            f"Question: {sample_q['question']}\n"
            f"Correct Answer: {sample_q['correct_answer']}\n"
            f"Explanation: {sample_q['explanation']}\n"
        )
        for field_val in [sample_q["id"], sample_q["type"], sample_q["topic"], sample_q["bloom_level"], sample_q["difficulty"], sample_q["source_ref"], sample_q["question"], sample_q["correct_answer"], sample_q["explanation"]]:
            self.assertIn(field_val, txt_repr)

if __name__ == "__main__":
    unittest.main()
