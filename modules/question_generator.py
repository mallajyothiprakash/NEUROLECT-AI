"""
NEUROLECT AI - Pretrained Transformer Question Generation Engine
Generates multi-type educational questions using instruction-tuned FLAN-T5.
Supports MCQ, Short-Answer, Descriptive, Conceptual, Application, and Analytical questions.
Strictly RAG-free: operates directly on sequential lecture chunks.
"""

import re
import uuid
from typing import List, Dict, Any, Optional
from models.model_loader import model_loader
from modules.content_engine import ProcessedLectureContent
from modules.distractor_generator import DistractorGenerator
from modules.bloom_classifier import BloomsTaxonomyClassifier
from modules.difficulty_classifier import DifficultyClassifier
from modules.question_validator import QuestionValidator
import config
import logging

logger = logging.getLogger("QuestionGenerator")

class QuestionGenerator:
    """Generates varied, high-quality assessment items using pretrained sequence-to-sequence models."""

    def __init__(self):
        self.loader = model_loader
        self.distractor_gen = DistractorGenerator()
        self.bloom_clf = BloomsTaxonomyClassifier()
        self.diff_clf = DifficultyClassifier()
        self.validator = QuestionValidator()

    def _prepare_context(self, chunk_text: str, max_context_tokens: int = 380) -> str:
        """
        Prepares tokenizer-aware context slice for prompt insertion.
        Preserves as much chunk context as safely permitted by FLAN-T5's 512 input limit.
        """
        try:
            tokenizer, _ = self.loader.get_question_generator()
            tokens = tokenizer.encode(chunk_text, truncation=False)
            if len(tokens) > max_context_tokens:
                tokens = tokens[:max_context_tokens]
                return tokenizer.decode(tokens, skip_special_tokens=True).strip()
            return chunk_text.strip()
        except Exception:
            # Word-level fallback (approx. 1.3 tokens per word -> ~300 words)
            words = chunk_text.split()
            if len(words) > 300:
                return " ".join(words[:300])
            return chunk_text.strip()

    def _generate_text(self, prompt: str, max_tokens: int = 150) -> str:
        """Executes text generation using cached Flan-T5 model on Apple M5 MPS or CPU fallback."""
        import torch
        tokenizer, model = self.loader.get_question_generator()
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                num_beams=2,
                early_stopping=True
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def _clean_question_string(self, text: str) -> str:
        """Cleans and standardizes raw question string from model output."""
        if not text:
            return ""
        q = text.strip()
        # Remove leading Question:, Task:, etc.
        q = re.sub(r'^(?:Question|Task|Q|Generate\s*(?:a\s*)?question):\s*', '', q, flags=re.IGNORECASE).strip()
        # Remove trailing options or multiple choice directives
        q = re.sub(r'(?:Choose your answer from|Options?|Choices?):\s*.*$', '', q, flags=re.IGNORECASE).strip()
        q = re.sub(r'\s*\([A-D]\).*$', '', q, flags=re.IGNORECASE).strip()
        # Strip trailing punctuation before re-adding question mark
        q = q.rstrip(' .:,;')
        if q and not q.endswith('?'):
            q += '?'
        return q

    def _clean_answer_string(self, text: str) -> str:
        """Cleans and standardizes raw answer string from model output."""
        if not text:
            return ""
        a = text.strip()
        a = re.sub(r'^(?:Answer|A|Model Answer|Reference Answer):\s*', '', a, flags=re.IGNORECASE).strip()
        # Strip trailing explanation if concatenated
        a = re.sub(r'(?:Explanation|Rationale):\s*.*$', '', a, flags=re.IGNORECASE).strip()
        return a.strip('"\'. ')

    def generate_questions(
        self,
        lecture_content: ProcessedLectureContent,
        question_types: Optional[List[str]] = None,
        num_questions: int = 6,
        candidate_keywords: Optional[List[str]] = None,
        topics_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates assessment questions across specified types from lecture chunks.
        Applies distractor generation, Bloom classification, difficulty scoring, and quality validation.
        Guarantees loop termination with a maximum attempt safeguard.
        """
        target_types = question_types or config.QUESTION_TYPES
        chunks = getattr(lecture_content, "chunks", [])
        transcript_text = getattr(lecture_content, "cleaned_text", "") or getattr(lecture_content, "raw_text", "")

        # Structured Diagnostic Logging (Step 3)
        logger.info("Question generation started")
        logger.info("Lecture transcript length: %s", len(transcript_text))
        logger.info("Number of content chunks: %s", len(chunks))

        # Check 1: Empty transcript or chunks guard
        if not chunks or not transcript_text.strip():
            logger.warning("No lecture content received by the question generator.")
            return {
                "total_generated": 0,
                "questions": [],
                "warnings": ["No lecture content was received by the question generator."],
                "failure_category": "EMPTY_TRANSCRIPT",
                "diagnostic_counts": {"attempts": 0, "raw_candidates": 0, "validated": 0, "rejections": 1},
                "model_used": config.MODELS_CONFIG["question_generator"]["model_id"]
            }

        # Check 2: Model loader availability guard
        try:
            _, model = self.loader.get_question_generator()
            logger.info("FLAN-T5 model loaded: %s", model is not None)
        except Exception as e:
            logger.error(f"FLAN-T5 model loading failed: {e}")
            return {
                "total_generated": 0,
                "questions": [],
                "warnings": [f"FLAN-T5 model is unavailable: {e}. Please ensure model weights are cached."],
                "failure_category": "MODEL_UNAVAILABLE",
                "diagnostic_counts": {"attempts": 0, "raw_candidates": 0, "validated": 0, "rejections": 1},
                "model_used": config.MODELS_CONFIG["question_generator"]["model_id"]
            }

        raw_questions: List[Dict[str, Any]] = []
        valid_questions: List[Dict[str, Any]] = []
        validation_warnings: List[str] = []
        kw_terms = candidate_keywords or [w for w in transcript_text.split() if len(w) > 5 and w.isalpha()][:25]

        logger.info(f"Generating {num_questions} questions across types: {target_types}")

        # Maximum attempt safeguard to guarantee termination and avoid infinite loops
        max_attempts = max(num_questions * 4, len(chunks) * 4, 25)
        attempt_count = 0
        chunk_idx = 0
        mcq_distractor_failures = 0

        # Continue generating until the requested number of VALID questions is reached, or max_attempts exhausted
        while len(valid_questions) < num_questions and attempt_count < max_attempts:
            attempt_count += 1
            chunk = chunks[chunk_idx]
            chunk_text = chunk.get("text", "")
            source_ref = chunk.get("source_ref", "General")
            
            # Select question type cyclically based on current valid count
            chosen_type = target_types[len(valid_questions) % len(target_types)]
            # Cycle through key concepts for prompt diversity
            focus_concept = kw_terms[attempt_count % len(kw_terms)] if kw_terms else None
            
            try:
                item = self._generate_single_item(
                    chunk_text=chunk_text,
                    q_type=chosen_type,
                    source_ref=source_ref,
                    candidate_terms=kw_terms,
                    topics_data=topics_data,
                    focus_term=focus_concept
                )
                if item:
                    raw_questions.append(item)
                    # Validate accumulated candidates to ensure replacements are generated if rejected
                    valid_questions, validation_warnings = self.validator.validate_question_batch(
                        questions=raw_questions,
                        source_text=transcript_text
                    )
                elif "MCQ" in chosen_type and getattr(self, "_last_mcq_distractor_failed", False):
                    mcq_distractor_failures += 1
            except Exception as e:
                logger.warning(f"Error generating question attempt {attempt_count} on chunk {chunk_idx}: {e}")

            chunk_idx = (chunk_idx + 1) % len(chunks)

        # Step 2: Quality validation metrics and boundary capping
        requested_count = num_questions
        generated_count = len(raw_questions)
        validated_count = len(valid_questions)
        final_questions = valid_questions[:requested_count]
        final_count = len(final_questions)

        logger.info(
            "Question count metrics: requested=%s, generated=%s, validated=%s, final=%s",
            requested_count, generated_count, validated_count, final_count
        )

        all_warnings = list(validation_warnings)

        # Determine diagnostic failure category and shortfall reporting
        failure_category = None
        if final_count == 0:
            if generated_count == 0:
                if mcq_distractor_failures > 0 and len(target_types) == 1 and "MCQ" in target_types[0]:
                    failure_category = "DISTRACTOR_FAILURE"
                    all_warnings.append(
                        f"Unable to generate valid assessment questions: MCQ distractors could not be generated with sufficient quality after {attempt_count} attempts."
                    )
                else:
                    failure_category = "EMPTY_GENERATION"
                    all_warnings.append(
                        f"Unable to generate valid assessment questions from the provided content after {attempt_count} attempts."
                    )
            else:
                failure_category = "VALIDATION_FAILURE"
                all_warnings.append(
                    f"Unable to generate valid assessment questions: All {generated_count} candidates failed quality validation after {attempt_count} attempts."
                )
        elif final_count < requested_count:
            shortfall = requested_count - final_count
            all_warnings.append(
                f"Requested: {requested_count} questions. Successfully generated: {final_count} valid questions. {shortfall} additional questions could not pass validation after {attempt_count} attempts."
            )

        return {
            "requested_count": requested_count,
            "generated_count": generated_count,
            "validated_count": validated_count,
            "final_count": final_count,
            "total_generated": final_count,
            "questions": final_questions,
            "warnings": all_warnings,
            "failure_category": failure_category,
            "diagnostic_counts": {
                "requested": requested_count,
                "generated": generated_count,
                "validated": validated_count,
                "final": final_count,
                "attempts": attempt_count,
                "rejections": generated_count - validated_count
            },
            "model_used": config.MODELS_CONFIG["question_generator"]["model_id"]
        }

    def _generate_single_item(
        self,
        chunk_text: str,
        q_type: str,
        source_ref: str,
        candidate_terms: List[str],
        topics_data: Optional[Dict[str, Any]] = None,
        focus_term: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generates a single question item matching the requested pedagogical type.
        Supports both single-pass (compound output) and robust two-pass seq2seq generation.
        Returns: item_dict or None
        """
        self._last_mcq_distractor_failed = False
        context_slice = self._prepare_context(chunk_text, max_context_tokens=380)
        
        # Build prompt tailored to FLAN-T5 seq2seq instruction dataset
        term_focus_clause = f" about {focus_term}" if focus_term else ""
        if "MCQ" in q_type:
            prompt_q = (
                f"Generate a factual multiple-choice question{term_focus_clause} based on this text:\n\n"
                f"{context_slice}\n\nQuestion:"
            )
        elif "Short Answer" in q_type:
            prompt_q = (
                f"Ask a concise short-answer question{term_focus_clause} based on this text:\n\n"
                f"{context_slice}\n\nQuestion:"
            )
        elif "Descriptive" in q_type:
            prompt_q = (
                f"Ask a comprehensive descriptive question explaining{term_focus_clause} based on this text:\n\n"
                f"{context_slice}\n\nQuestion:"
            )
        elif "Conceptual" in q_type:
            prompt_q = (
                f"Ask a conceptual question about the underlying principles{term_focus_clause} in this text:\n\n"
                f"{context_slice}\n\nQuestion:"
            )
        elif "Application" in q_type:
            prompt_q = (
                f"Ask a practical application question{term_focus_clause} based on this text:\n\n"
                f"{context_slice}\n\nQuestion:"
            )
        else: # Analytical
            prompt_q = (
                f"Ask an analytical question comparing concepts{term_focus_clause} in this text:\n\n"
                f"{context_slice}\n\nQuestion:"
            )

        gen_output = self._generate_text(prompt_q, max_tokens=100)
        
        # Check if the output already has compound fields (e.g., from unit test mocks)
        parsed = self._parse_generated_fields(gen_output, default_type=q_type)
        if parsed.get("question") and parsed.get("answer"):
            q_text = self._clean_question_string(parsed["question"])
            correct_answer = self._clean_answer_string(parsed["answer"])
            explanation = parsed.get("explanation") or f"Derived directly from {source_ref}."
        else:
            # Clean question string from standard seq2seq output
            q_text = self._clean_question_string(gen_output)
            if not q_text or len(q_text) < 10:
                return None

            # Pass 2: Generate Answer grounded in the context
            prompt_a = (
                f"Answer the following question based strictly on the text.\n\n"
                f"Context: {context_slice}\n\n"
                f"Question: {q_text}\n\nAnswer:"
            )
            ans_output = self._generate_text(prompt_a, max_tokens=80)
            parsed_a = self._parse_generated_fields(ans_output, default_type=q_type)
            correct_answer = self._clean_answer_string(parsed_a.get("answer") or ans_output)

            if not correct_answer or len(correct_answer) < 2:
                return None

            explanation = (
                f"Grounded directly in lecture discussion of {focus_term} at {source_ref}."
                if focus_term else f"Derived directly from lecture source material at {source_ref}."
            )

        # Classify Bloom's Taxonomy
        bloom_info = self.bloom_clf.classify_question(q_text, q_type)
        bloom_level = bloom_info["bloom_level"]

        # Classify Difficulty
        diff_info = self.diff_clf.classify_difficulty(q_text, bloom_level, q_type)
        difficulty = diff_info["difficulty"]

        # Resolve topic using Phase 3 Topic Detector integration
        topic_title = self._resolve_topic(
            chunk_text=chunk_text,
            source_ref=source_ref,
            topics_data=topics_data,
            candidate_terms=candidate_terms
        )

        item_id = f"q_{uuid.uuid4().hex[:8]}"

        item: Dict[str, Any] = {
            "id": item_id,
            "type": q_type,
            "question": q_text,
            "correct_answer": correct_answer,
            "explanation": explanation,
            "bloom_level": bloom_level,
            "difficulty": difficulty,
            "topic": topic_title,
            "source_ref": source_ref
        }

        # For MCQs, invoke distractor generator
        if "MCQ" in q_type:
            mcq_data = self.distractor_gen.generate_options(
                question=q_text,
                correct_answer=correct_answer,
                context_text=chunk_text,
                candidate_terms=candidate_terms
            )
            # If 3 credible distractors could not be generated, reject MCQ rather than manufacturing misleading choices
            if not mcq_data.get("success", False) or len(mcq_data.get("distractors", [])) < 3:
                logger.warning(f"Rejecting MCQ due to insufficient credible distractors for chunk anchor: {source_ref}")
                self._last_mcq_distractor_failed = True
                return None

            item["options"] = mcq_data["options"]
            item["correct_letter"] = mcq_data["correct_letter"]
            item["distractors"] = mcq_data["distractors"]

        return item

    def _parse_generated_fields(self, text: str, default_type: str) -> Dict[str, str]:
        """Extracts Question, Answer, and Explanation fields from model output."""
        q_match = re.search(r'Question:\s*(.*?)(?=(?:Answer|Explanation|$))', text, re.DOTALL | re.IGNORECASE)
        a_match = re.search(r'Answer:\s*(.*?)(?=(?:Explanation|$))', text, re.DOTALL | re.IGNORECASE)
        e_match = re.search(r'Explanation:\s*(.*)', text, re.DOTALL | re.IGNORECASE)

        question = q_match.group(1).strip() if q_match else ""
        answer = a_match.group(1).strip() if a_match else ""
        explanation = e_match.group(1).strip() if e_match else ""

        # Fallback if structure was slightly altered by model
        if not question and "?" in text:
            parts = text.split("?")
            question = parts[0].strip() + "?"
            if len(parts) > 1:
                answer = parts[1].strip()

        return {
            "question": question,
            "answer": answer,
            "explanation": explanation
        }

    def _resolve_topic(
        self,
        chunk_text: str,
        source_ref: str,
        topics_data: Optional[Dict[str, Any]],
        candidate_terms: List[str]
    ) -> str:
        """
        Integrates Phase 3 Topic Detection:
        1. Checks for topic segment alignment (page number / timestamp match).
        2. Checks for cluster sentence overlap.
        3. Falls back to candidate domain terms or informative unigrams.
        """
        if topics_data and isinstance(topics_data, dict):
            topics_list = topics_data.get("topics", [])
            # Priority 1: Match source segment
            for t in topics_list:
                source_segments = t.get("source_segments", [])
                if source_ref and any(s == source_ref or source_ref in s for s in source_segments):
                    return t.get("title", "Lecture Concepts")

            # Priority 2: Match sentence content overlap
            chunk_lower = chunk_text.lower()
            for t in topics_list:
                for sent in t.get("sentences", []):
                    if len(sent) > 20 and sent.lower()[:35] in chunk_lower:
                        return t.get("title", "Lecture Concepts")

        # Priority 3: Use candidate domain keywords if available
        if candidate_terms:
            for term in candidate_terms:
                if term.lower() in chunk_text.lower():
                    return term.title()

        # Priority 4: Sensible linguistic fallback
        words = [w.capitalize() for w in chunk_text.split() if len(w) > 4 and w.isalpha()]
        return " ".join(words[:2]) if words else "Educational Concepts"

