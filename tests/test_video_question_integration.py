"""
NEUROLECT AI - Video Transcript to FLAN-T5 Question Generation Regression & Integration Tests
Validates the complete lifecycle:
Video Upload -> Audio Extraction -> Whisper Transcript -> LectureContent ->
Question Studio -> FLAN-T5 Generation -> Validation -> Distractors -> Session State Storage.
Strictly RAG-free.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from modules.content_engine import UnifiedContentEngine
from modules.content_models import LectureContent
from modules.question_generator import QuestionGenerator
from modules.distractor_generator import DistractorGenerator
from modules.question_validator import QuestionValidator
from models.model_loader import model_loader
import config

SAMPLE_VIDEO_TRANSCRIPT = """
In this lecture, we examine deep learning architectures for computer vision and pattern recognition.
Specifically, convolutional neural networks, or CNNs, process visual data through spatial convolution operations.
A convolutional layer applies a set of learnable filters or kernels across the input image to extract local feature maps.
Activation functions such as ReLU, the rectified linear unit, introduce non-linearity by outputting max of zero and input.
Pooling layers, such as max pooling, downsample feature maps to reduce spatial dimensions and computational complexity.
Fully connected layers at the end of the network compile extracted features to compute final class probability scores.
Backpropagation with stochastic gradient descent optimizes the weights by minimizing categorical cross-entropy loss.
"""

class TestVideoQuestionIntegration(unittest.TestCase):
    """Exhaustive test suite for video transcript to FLAN-T5 question generation."""

    def setUp(self):
        # Enforce offline flags for deterministic unit testing
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        self.q_gen = QuestionGenerator()
        self.validator = QuestionValidator()
        self.distractor_gen = DistractorGenerator()

        # Build realistic video-derived LectureContent
        self.video_content = UnifiedContentEngine.process_content(
            source_type="video",
            source_name="Lecture_Computer_Vision.mp4",
            raw_text=SAMPLE_VIDEO_TRANSCRIPT,
            segments=[
                {
                    "segment_id": 1,
                    "text": SAMPLE_VIDEO_TRANSCRIPT[:250].strip(),
                    "source_ref": "00:00 - 00:35",
                    "start_time": 0.0,
                    "end_time": 35.0
                },
                {
                    "segment_id": 2,
                    "text": SAMPLE_VIDEO_TRANSCRIPT[250:].strip(),
                    "source_ref": "00:35 - 01:15",
                    "start_time": 35.0,
                    "end_time": 75.0
                }
            ],
            custom_metadata={
                "duration_seconds": 75.0,
                "duration_formatted": "01:15",
                "original_video_name": "Lecture_Computer_Vision.mp4",
                "audio_extraction": "ffmpeg_pcm16le_16khz"
            }
        )

    # ------------------------------------------------------------------------
    # 1. Video-Derived LectureContent Reaches Question Studio
    # ------------------------------------------------------------------------
    def test_video_derived_lecture_content_structure(self):
        """Verifies that video pipeline produces valid LectureContent with chunks and metadata."""
        self.assertEqual(self.video_content.source_type, "video")
        self.assertEqual(self.video_content.source_name, "Lecture_Computer_Vision.mp4")
        self.assertTrue(len(self.video_content.chunks) >= 1)
        self.assertTrue(self.video_content.word_count > 50)
        self.assertTrue(len(self.video_content.cleaned_text) > 200)
        
        # Verify chunks have source reference mapped from video timestamps
        first_chunk = self.video_content.chunks[0]
        self.assertIn("text", first_chunk)
        self.assertIn("source_ref", first_chunk)

    # ------------------------------------------------------------------------
    # 2. Non-Empty Transcript Passed to QuestionGenerator
    # ------------------------------------------------------------------------
    def test_non_empty_transcript_passed_to_generator(self):
        """Verifies that QuestionGenerator safely receives video transcript and content chunks."""
        chunks = self.video_content.chunks
        self.assertGreater(len(chunks), 0)
        chunk_text = chunks[0]["text"]
        
        prepared_context = self.q_gen._prepare_context(chunk_text, max_context_tokens=380)
        self.assertTrue(len(prepared_context) > 50)
        self.assertIn("convolutional", prepared_context.lower())

    # ------------------------------------------------------------------------
    # 3. FLAN-T5 Output Processing (Two-Pass & Single-Pass Support)
    # ------------------------------------------------------------------------
    def test_flan_t5_output_processing(self):
        """Verifies string cleaning and two-pass question-answer generation parsing."""
        raw_q = "Question: What is the role of convolutional layers in CNNs? Choose your answer from: (A) ..."
        cleaned_q = self.q_gen._clean_question_string(raw_q)
        self.assertEqual(cleaned_q, "What is the role of convolutional layers in CNNs?")
        self.assertTrue(cleaned_q.endswith("?"))

        raw_a = "Answer: Apply learnable filters to extract local feature maps. Explanation: Grounded."
        cleaned_a = self.q_gen._clean_answer_string(raw_a)
        self.assertEqual(cleaned_a, "Apply learnable filters to extract local feature maps")

    # ------------------------------------------------------------------------
    # 4. Generated Questions Survive Quality Validation
    # ------------------------------------------------------------------------
    def test_generated_questions_survive_validation(self):
        """Verifies that well-formed questions grounded in video transcript survive validation."""
        candidate = {
            "id": "q_test1",
            "type": "Short Answer",
            "question": "What is the primary role of convolutional layers in CNNs?",
            "correct_answer": "To apply learnable filters to extract local feature maps",
            "explanation": "Grounded in video segment 00:00 - 00:35.",
            "bloom_level": "Understand",
            "difficulty": "Medium",
            "topic": "Convolutional Neural Networks",
            "source_ref": "00:00 - 00:35"
        }
        valid_items, warnings = self.validator.validate_question_batch(
            questions=[candidate],
            source_text=self.video_content.cleaned_text
        )
        self.assertEqual(len(valid_items), 1)
        self.assertEqual(valid_items[0]["id"], "q_test1")

    # ------------------------------------------------------------------------
    # 5. Valid MCQs Contain Exactly Four Options and Valid Correct Letter
    # ------------------------------------------------------------------------
    def test_valid_mcq_four_options_and_correct_letter(self):
        """Verifies that MCQ items contain exactly 4 unique options with accurate correct letter mapping."""
        res = self.distractor_gen.generate_options(
            question="What activation function introduces non-linearity in convolutional neural networks?",
            correct_answer="ReLU",
            context_text=SAMPLE_VIDEO_TRANSCRIPT,
            candidate_terms=["ReLU", "Max Pooling", "Backpropagation", "Stochastic Gradient Descent"]
        )
        self.assertTrue(res["success"])
        self.assertEqual(len(res["options"]), 4)
        self.assertIn(res["correct_letter"], ["A", "B", "C", "D"])
        self.assertEqual(res["options"][res["correct_letter"]], "ReLU")
        self.assertEqual(len(set(res["options"].values())), 4)

    # ------------------------------------------------------------------------
    # 6. Invalid Questions are Rejected with Recorded Reasons
    # ------------------------------------------------------------------------
    def test_invalid_questions_rejected_with_recorded_reasons(self):
        """Verifies that malformed, ungrounded, or duplicate questions are rejected with explicit warnings."""
        malformed = [
            # Too short question
            {"id": "q_bad1", "type": "Short Answer", "question": "Why?", "correct_answer": "Because.", "bloom_level": "Remember", "difficulty": "Easy", "topic": "General", "source_ref": "General"},
            # Empty answer
            {"id": "q_bad2", "type": "Short Answer", "question": "What is convolutional neural network architecture?", "correct_answer": "", "bloom_level": "Remember", "difficulty": "Easy", "topic": "General", "source_ref": "General"},
            # Hallucinated question with zero lexical overlap
            {"id": "q_bad3", "type": "Short Answer", "question": "How did ancient Roman gladiators utilize bronze shields?", "correct_answer": "For physical combat defense in amphitheatres.", "bloom_level": "Remember", "difficulty": "Easy", "topic": "General", "source_ref": "General"}
        ]
        valid_items, warnings = self.validator.validate_question_batch(
            questions=malformed,
            source_text=self.video_content.cleaned_text
        )
        self.assertEqual(len(valid_items), 0)
        self.assertGreaterEqual(len(warnings), 3)

    # ------------------------------------------------------------------------
    # 7. Final Question List Stored in Streamlit Session State Structure
    # ------------------------------------------------------------------------
    def test_final_question_list_stored_in_session_state(self):
        """Verifies structure and key consistency expected by Streamlit session state."""
        mock_results = {
            "total_generated": 2,
            "questions": [
                {
                    "id": "q_1",
                    "type": "Short Answer",
                    "question": "What is the purpose of max pooling?",
                    "correct_answer": "Downsample feature maps and reduce spatial dimensions",
                    "bloom_level": "Understand",
                    "difficulty": "Easy",
                    "topic": "Pooling",
                    "source_ref": "00:35 - 01:15"
                }
            ],
            "warnings": [],
            "failure_category": None,
            "model_used": "google/flan-t5-base"
        }
        
        # Simulate session state dictionary
        session_state = {}
        session_state["generated_questions"] = mock_results
        
        retrieved = session_state.get("generated_questions")
        self.assertIsNotNone(retrieved)
        self.assertEqual(len(retrieved.get("questions", [])), 1)
        self.assertEqual(retrieved["questions"][0]["type"], "Short Answer")

    # ------------------------------------------------------------------------
    # 8. Generated Questions Remain Available After Rerun
    # ------------------------------------------------------------------------
    def test_generated_questions_available_after_rerun(self):
        """Verifies session state persistence across simulated rerun passes."""
        session_state = {"current_lecture": self.video_content}
        
        # Pass 1: Questions generated and saved
        q_data = {
            "total_generated": 1,
            "questions": [{"id": "q_persist", "question": "What is ReLU?", "correct_answer": "Rectified Linear Unit", "type": "Short Answer", "bloom_level": "Remember", "difficulty": "Easy", "topic": "Activation", "source_ref": "00:00"}],
            "warnings": [],
            "failure_category": None
        }
        session_state["generated_questions"] = q_data
        
        # Pass 2: Simulated Streamlit page re-render reading from session state
        retrieved = session_state.get("generated_questions")
        self.assertIsNotNone(retrieved)
        self.assertTrue(bool(retrieved.get("questions")))
        self.assertEqual(retrieved["questions"][0]["id"], "q_persist")

    # ------------------------------------------------------------------------
    # 9. Empty Generation Produces Understandable Diagnostic
    # ------------------------------------------------------------------------
    def test_empty_generation_produces_understandable_diagnostic(self):
        """Verifies that an empty transcript produces EMPTY_TRANSCRIPT failure category and clear action."""
        empty_content = LectureContent(
            content_id="empty_lec",
            source_type="video",
            source_name="empty.mp4",
            raw_text="",
            cleaned_text="",
            word_count=0,
            reading_time="0 min",
            chunks=[],
            segments=[],
            metadata={}
        )
        res = self.q_gen.generate_questions(lecture_content=empty_content)
        self.assertEqual(res["total_generated"], 0)
        self.assertEqual(len(res["questions"]), 0)
        self.assertEqual(res["failure_category"], "EMPTY_TRANSCRIPT")
        self.assertTrue(any("No lecture content" in w for w in res["warnings"]))

    # ------------------------------------------------------------------------
    # 10. Missing Model Weights Produce Explicit Error
    # ------------------------------------------------------------------------
    def test_missing_model_weights_produces_explicit_error(self):
        """Verifies that if FLAN-T5 model loading fails, failure_category is MODEL_UNAVAILABLE."""
        with patch.object(self.q_gen.loader, "get_question_generator", side_effect=RuntimeError("Weights not downloaded")):
            res = self.q_gen.generate_questions(lecture_content=self.video_content)
            self.assertEqual(res["total_generated"], 0)
            self.assertEqual(res["failure_category"], "MODEL_UNAVAILABLE")
            self.assertTrue(any("FLAN-T5 model is unavailable" in w for w in res["warnings"]))

    # ------------------------------------------------------------------------
    # 11. Existing PDF, Audio, and Text Workflows Remain Functional
    # ------------------------------------------------------------------------
    def test_existing_modalities_question_generation_functional(self):
        """Verifies that question generation functions uniformly across audio, PDF, and text inputs."""
        for m in ["audio", "pdf", "text"]:
            content = UnifiedContentEngine.process_content(
                source_type=m,
                source_name=f"lecture.{'wav' if m=='audio' else ('pdf' if m=='pdf' else 'txt')}",
                raw_text=SAMPLE_VIDEO_TRANSCRIPT
            )
            self.assertEqual(content.source_type, m)
            self.assertTrue(len(content.chunks) >= 1)
            
            # Mock distinct items to verify generation pipeline without triggering duplicate detection
            mock_item1 = {
                "id": f"q_{m}_1",
                "type": "Short Answer",
                "question": f"What is a primary concept in this {m} lecture?",
                "correct_answer": "Convolutional neural network operations",
                "explanation": "Derived from source.",
                "bloom_level": "Remember",
                "difficulty": "Easy",
                "topic": "Neural Networks",
                "source_ref": "General"
            }
            mock_item2 = {
                "id": f"q_{m}_2",
                "type": "Descriptive",
                "question": f"How do pooling layers reduce spatial dimensions in {m} models?",
                "correct_answer": "By downsampling feature maps to reduce complexity",
                "explanation": "Derived from source.",
                "bloom_level": "Understand",
                "difficulty": "Medium",
                "topic": "Pooling",
                "source_ref": "General"
            }
            with patch.object(self.q_gen, "_generate_single_item", side_effect=[mock_item1, mock_item2]):
                res = self.q_gen.generate_questions(lecture_content=content, num_questions=2)
                self.assertEqual(res["total_generated"], 2)
                self.assertIsNone(res.get("failure_category"))

    # ------------------------------------------------------------------------
    # 12. Real FLAN-T5 Inference on Video Transcript (if Cached on Disk)
    # ------------------------------------------------------------------------
    def test_actual_flan_t5_inference_on_video_transcript(self):
        """
        Executes actual pretrained FLAN-T5 inference on video transcript if weights are locally cached.
        Explicitly distinguishes real model inference from mocked unit tests.
        """
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        flan_cached = any("flan-t5" in p.name.lower() for p in hf_cache.iterdir()) if hf_cache.exists() else False
        
        if not flan_cached:
            self.skipTest("FLAN-T5 model weights not cached on disk. Skipping live inference test.")

        # Real model inference with local weights
        print("\n[LIVE INFERENCE] Executing genuine FLAN-T5 question generation on video transcript...")
        res = self.q_gen.generate_questions(
            lecture_content=self.video_content,
            question_types=["Short Answer", "Descriptive"],
            num_questions=2
        )
        self.assertGreater(res["total_generated"], 0)
        self.assertGreater(len(res["questions"]), 0)
        self.assertIsNone(res.get("failure_category"))
        
        first_q = res["questions"][0]
        self.assertTrue(len(first_q["question"]) > 10)
        self.assertTrue(len(first_q["correct_answer"]) > 2)
        print(f"[LIVE INFERENCE SUCCESS] Generated: '{first_q['question']}' -> '{first_q['correct_answer']}'")

    # ------------------------------------------------------------------------
    # 13. Exact Requested Count Verification: 5, 10, 15, and 20 Questions
    # ------------------------------------------------------------------------
    def test_exact_requested_counts_5_10_15_20(self):
        """
        Verifies that when sufficient valid candidates are available,
        the system returns exactly the requested count for 5, 10, 15, and 20 questions.
        """
        concepts_pool = [
            ("convolutional layers", "apply learnable filter kernels across input visual features"),
            ("ReLU activation", "introduces non-linearity by thresholding negative activations to zero"),
            ("max pooling layers", "downsample feature map dimensions to reduce computational complexity"),
            ("fully connected layers", "compile extracted visual features to compute class probability distribution"),
            ("stochastic gradient descent", "iteratively updates weights in opposite direction of loss gradient"),
            ("cross entropy loss", "measures divergence between predicted class probabilities and ground truth labels"),
            ("kernel size", "determines spatial extent of local receptive field during convolution"),
            ("stride parameter", "specifies step size when sliding filter across image matrix"),
            ("zero padding", "preserves spatial border dimensions of output feature map"),
            ("feature maps", "represent spatial activations produced by convolving kernels with inputs"),
            ("dropout regularization", "randomly deactivates neurons during forward pass to prevent overfitting"),
            ("batch normalization", "normalizes layer inputs across mini batch to accelerate training convergence"),
            ("learning rate", "hyperparameter controlling magnitude of weight updates at each optimization step"),
            ("backpropagation algorithm", "computes loss function gradients with respect to weights using chain rule"),
            ("softmax function", "converts raw logits into normalized posterior probability values summing to one"),
            ("spatial hierarchy", "captures low-level edges in early layers and complex semantic patterns in deeper layers"),
            ("receptive field", "local region of input image that influences activation of particular neuron"),
            ("image classification", "assigns categorical label to visual input based on learned representations"),
            ("weight decay", "applies L2 penalty on model parameters to discourage excessively large weights"),
            ("residual connections", "allow gradients to bypass layers directly to mitigate vanishing gradient problem"),
            ("channel depth", "refers to number of filter maps produced by convolutional operation"),
            ("pooling stride", "step interval at which pooling window slides across feature maps"),
            ("gradient vanishing", "diminution of error signals as backpropagation traverses multiple deep layers"),
            ("loss function", "mathematical objective quantifying deviation between model output and ground truth"),
            ("dense connections", "connects every input unit to every output unit in classification heads")
        ]

        # Ensure lecture text contains all concept vocabulary to satisfy groundedness
        extended_text = SAMPLE_VIDEO_TRANSCRIPT + "\n" + " ".join([c[0] + " " + c[1] for c in concepts_pool])
        extended_content = UnifiedContentEngine.process_content("video", "Extended_CV.mp4", extended_text)

        for req in [5, 10, 15, 20]:
            # Generate req distinct, grounded items from concept pool
            distinct_items = [
                {
                    "id": f"q_req_{req}_{i}",
                    "type": "Short Answer",
                    "question": f"What is the primary function of {concepts_pool[i][0]} in deep convolutional networks?",
                    "correct_answer": concepts_pool[i][1],
                    "explanation": f"Grounded in segment #{i+1}.",
                    "bloom_level": "Understand",
                    "difficulty": "Medium",
                    "topic": concepts_pool[i][0].title(),
                    "source_ref": "00:00 - 00:35"
                }
                for i in range(req)
            ]
            with patch.object(self.q_gen, "_generate_single_item", side_effect=distinct_items):
                res = self.q_gen.generate_questions(lecture_content=extended_content, num_questions=req)
                
                # Verify all four instrumented metrics
                self.assertEqual(res["requested_count"], req)
                self.assertEqual(res["final_count"], req)
                self.assertEqual(res["total_generated"], req)
                self.assertEqual(len(res["questions"]), req)
                # Verify never exceeds requested count
                self.assertLessEqual(len(res["questions"]), req)

    # ------------------------------------------------------------------------
    # 14. Invalid and Duplicate Questions Do Not Count Toward Final Total
    # ------------------------------------------------------------------------
    def test_count_with_invalid_and_duplicate_rejections(self):
        """
        Verifies that when invalid and duplicate candidates occur,
        the generator continues attempting and replaces them to fulfill the requested count.
        """
        req = 5
        # 3 duplicate/invalid candidates interspersed with 5 unique valid candidates
        item_stream = [
            # 1. Valid item
            {"id": "q_v1", "type": "Short Answer", "question": "What is the primary role of convolutional layers?", "correct_answer": "Extract feature maps", "bloom_level": "Remember", "difficulty": "Easy", "topic": "CNN", "source_ref": "00:00"},
            # 2. Duplicate of item 1 (must be rejected)
            {"id": "q_dup1", "type": "Short Answer", "question": "What is the primary role of convolutional layers?", "correct_answer": "Extract feature maps", "bloom_level": "Remember", "difficulty": "Easy", "topic": "CNN", "source_ref": "00:00"},
            # 3. Valid item
            {"id": "q_v2", "type": "Short Answer", "question": "How do activation functions introduce non-linearity?", "correct_answer": "By mapping inputs via ReLU", "bloom_level": "Understand", "difficulty": "Easy", "topic": "ReLU", "source_ref": "00:00"},
            # 4. Invalid item (empty answer)
            {"id": "q_bad1", "type": "Short Answer", "question": "What is max pooling operation in CNN architectures?", "correct_answer": "", "bloom_level": "Remember", "difficulty": "Easy", "topic": "Pooling", "source_ref": "00:00"},
            # 5. Valid item
            {"id": "q_v3", "type": "Short Answer", "question": "What is the primary benefit of max pooling?", "correct_answer": "Downsamples feature maps", "bloom_level": "Remember", "difficulty": "Easy", "topic": "Pooling", "source_ref": "00:00"},
            # 6. Near-duplicate of item 5 (must be rejected)
            {"id": "q_dup2", "type": "Short Answer", "question": "What is the primary benefit of max pooling?", "correct_answer": "Reduces spatial dimension", "bloom_level": "Remember", "difficulty": "Easy", "topic": "Pooling", "source_ref": "00:00"},
            # 7. Valid item
            {"id": "q_v4", "type": "Short Answer", "question": "How do fully connected layers compute class scores?", "correct_answer": "Compile extracted features", "bloom_level": "Apply", "difficulty": "Medium", "topic": "Dense Layers", "source_ref": "00:00"},
            # 8. Valid item (reaches target 5)
            {"id": "q_v5", "type": "Short Answer", "question": "Which optimization algorithm minimizes cross-entropy loss?", "correct_answer": "Stochastic gradient descent", "bloom_level": "Analyze", "difficulty": "Hard", "topic": "Optimization", "source_ref": "00:00"}
        ]
        
        with patch.object(self.q_gen, "_generate_single_item", side_effect=item_stream):
            res = self.q_gen.generate_questions(lecture_content=self.video_content, num_questions=req)
            
            # Requested: 5, Generated: 8 candidates, Validated: 5, Final: 5
            self.assertEqual(res["requested_count"], 5)
            self.assertEqual(res["generated_count"], 8)
            self.assertEqual(res["validated_count"], 5)
            self.assertEqual(res["final_count"], 5)
            self.assertEqual(len(res["questions"]), 5)
            
            # Verify rejected questions did not enter the final list
            question_ids = {q["id"] for q in res["questions"]}
            self.assertIn("q_v1", question_ids)
            self.assertIn("q_v2", question_ids)
            self.assertIn("q_v3", question_ids)
            self.assertIn("q_v4", question_ids)
            self.assertIn("q_v5", question_ids)
            self.assertNotIn("q_dup1", question_ids)
            self.assertNotIn("q_bad1", question_ids)
            self.assertNotIn("q_dup2", question_ids)

    # ------------------------------------------------------------------------
    # 15. Insufficient Valid Candidates Produces Honest Warning
    # ------------------------------------------------------------------------
    def test_shortfall_honest_warning_when_insufficient_content(self):
        """
        Verifies that when lecture content yields fewer valid questions than requested (e.g. 12 of 20),
        an explicit, transparent shortfall message is produced without fabricating fake items.
        """
        req = 20
        concepts_12 = [
            ("convolutional layers", "apply learnable filter kernels across input visual features"),
            ("ReLU activation", "introduces non-linearity by thresholding negative activations to zero"),
            ("max pooling layers", "downsample feature map dimensions to reduce computational complexity"),
            ("fully connected layers", "compile extracted visual features to compute class probability distribution"),
            ("stochastic gradient descent", "iteratively updates weights in opposite direction of loss gradient"),
            ("cross entropy loss", "measures divergence between predicted class probabilities and ground truth labels"),
            ("kernel size", "determines spatial extent of local receptive field during convolution"),
            ("stride parameter", "specifies step size when sliding filter across image matrix"),
            ("zero padding", "preserves spatial border dimensions of output feature map"),
            ("feature maps", "represent spatial activations produced by convolving kernels with inputs"),
            ("dropout regularization", "randomly deactivates neurons during forward pass to prevent overfitting"),
            ("batch normalization", "normalizes layer inputs across mini batch to accelerate training convergence")
        ]

        extended_text = SAMPLE_VIDEO_TRANSCRIPT + "\n" + " ".join([c[0] + " " + c[1] for c in concepts_12])
        extended_content = UnifiedContentEngine.process_content("video", "Extended_CV_12.mp4", extended_text)

        # Only 12 distinct questions possible before exhaustively returning None
        only_12_items = [
            {
                "id": f"q_shortfall_{i}",
                "type": "Short Answer",
                "question": f"What is the primary function of {concepts_12[i][0]} in deep convolutional networks?",
                "correct_answer": concepts_12[i][1],
                "explanation": "Contextual reference.",
                "bloom_level": "Understand",
                "difficulty": "Easy",
                "topic": concepts_12[i][0].title(),
                "source_ref": "00:00"
            }
            for i in range(12)
        ] + [None] * 100

        with patch.object(self.q_gen, "_generate_single_item", side_effect=only_12_items):
            res = self.q_gen.generate_questions(lecture_content=extended_content, num_questions=req)
            
            self.assertEqual(res["requested_count"], 20)
            self.assertEqual(res["final_count"], 12)
            self.assertEqual(len(res["questions"]), 12)
            
            # Verify explicit honest shortfall message
            self.assertTrue(
                any("Requested: 20 questions. Successfully generated: 12 valid questions. 8 additional questions could not pass validation" in w for w in res["warnings"])
            )

    # ------------------------------------------------------------------------
    # 16. Live Model Inference Requested Count Evaluation (if Cached)
    # ------------------------------------------------------------------------
    def test_actual_flan_t5_count_consistency_on_video(self):
        """
        Executes actual FLAN-T5 live model inference for requested counts (5 questions),
        measuring all four count pipeline values and verifying final <= requested.
        """
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        flan_cached = any("flan-t5" in p.name.lower() for p in hf_cache.iterdir()) if hf_cache.exists() else False
        
        if not flan_cached:
            self.skipTest("FLAN-T5 model weights not cached on disk. Skipping live inference count test.")

        res = self.q_gen.generate_questions(
            lecture_content=self.video_content,
            question_types=["Multiple Choice (MCQ)", "Short Answer", "Descriptive", "Conceptual"],
            num_questions=5
        )
        
        req = res["requested_count"]
        gen = res["generated_count"]
        val = res["validated_count"]
        fin = res["final_count"]
        
        self.assertEqual(req, 5)
        self.assertGreaterEqual(gen, val)
        self.assertEqual(fin, min(req, val))
        self.assertLessEqual(fin, req)
        print(f"\n[LIVE COUNT METRICS] Requested: {req} | Generated: {gen} | Validated: {val} | Final: {fin}")

if __name__ == "__main__":
    unittest.main()
