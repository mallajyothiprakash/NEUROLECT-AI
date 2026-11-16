"""
NEUROLECT AI - Interactive Quiz Engine
Controls quiz session lifecycle, item navigation, scoring, semantic evaluation,
and persistence into the SQLite database.
Strictly RAG-free.
"""

import uuid
from typing import List, Dict, Any, Optional
from modules.answer_evaluator import SemanticAnswerEvaluator
from database.db_manager import DatabaseManager
import config
import logging

logger = logging.getLogger("QuizEngine")

class QuizEngine:
    """Manages interactive assessment sessions and evaluates student submissions."""

    def __init__(self):
        self.evaluator = SemanticAnswerEvaluator()
        self.db = DatabaseManager()

    def filter_questions(
        self,
        pool: List[Dict[str, Any]],
        num_questions: int,
        selected_types: Optional[List[str]] = None,
        selected_difficulties: Optional[List[str]] = None,
        selected_blooms: Optional[List[str]] = None,
        selected_topics: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Filters question pool based on user preferences including topic, type, difficulty, and Bloom level."""
        filtered = list(pool)

        if selected_types:
            filtered = [q for q in filtered if q.get("type") in selected_types]

        if selected_difficulties:
            filtered = [q for q in filtered if q.get("difficulty") in selected_difficulties]

        if selected_blooms:
            filtered = [q for q in filtered if q.get("bloom_level") in selected_blooms]

        if selected_topics:
            filtered = [q for q in filtered if q.get("topic") in selected_topics]

        if not filtered:
            # Fall back to unfiltered pool if criteria were too restrictive
            filtered = list(pool)

        return filtered[:num_questions]

    def evaluate_quiz_submission(
        self,
        lecture_id: Optional[str],
        questions: List[Dict[str, Any]],
        student_answers: Dict[str, str], # Maps question index or ID to submitted answer
        difficulty_label: str = "Mixed"
    ) -> Dict[str, Any]:
        """
        Grades all submitted answers:
        - Deterministic exact match for MCQs (with letter and option text extraction)
        - Semantic embedding evaluation for Short Answer / Descriptive / Conceptual / Analytical questions
        - Computes partial-credit marks, concept coverage, and human-review flags
        - Saves attempt and itemized answers to SQLite database.
        """
        attempt_id = f"atm_{uuid.uuid4().hex[:10]}"
        evaluated_items: List[Dict[str, Any]] = []
        
        correct_count = 0

        # Track Bloom's distribution
        bloom_summary = {lvl: {"count": 0, "correct": 0} for lvl in config.BLOOMS_TAXONOMY_LEVELS}

        for idx, q in enumerate(questions):
            q_id = q.get("id", str(idx))
            q_type = q.get("type", "Short Answer")
            q_text = q.get("question", "")
            correct_ans = q.get("correct_answer", "")
            bloom_lvl = q.get("bloom_level", "Understand")
            difficulty = q.get("difficulty", "Medium")
            topic = q.get("topic", "General Concepts")

            submitted_val = student_answers.get(q_id, "").strip()

            bloom_summary[bloom_lvl]["count"] += 1

            if "MCQ" in q_type:
                # Grade MCQ deterministically
                correct_letter = q.get("correct_letter", "A").upper()
                options = q.get("options", {})
                
                # Normalize student submission: extract leading letter if provided as "A: Option text"
                clean_sub = submitted_val.strip()
                selected_letter = clean_sub[:1].upper() if clean_sub and len(clean_sub) > 1 and clean_sub[1:2] in [":", ")", ".", " "] else clean_sub.upper()
                
                # Check match against letter, option value, or correct answer string
                is_correct = False
                if selected_letter == correct_letter:
                    is_correct = True
                elif clean_sub.lower() == correct_ans.lower():
                    is_correct = True
                elif options and correct_letter in options and clean_sub.lower() == options[correct_letter].lower():
                    is_correct = True

                mcq_max_marks = 1.0
                score = 100.0 if is_correct else 0.0
                awarded_marks = 1.0 if is_correct else 0.0

                if is_correct:
                    correct_count += 1
                    bloom_summary[bloom_lvl]["correct"] += 1
                    feedback = "Correct! Well done."
                else:
                    opt_str = f"option {correct_letter}: '{options.get(correct_letter, correct_ans)}'" if options else f"'{correct_ans}'"
                    feedback = f"Incorrect. The correct choice is {opt_str}."

                item_result = {
                    "question_id": q_id,
                    "question": q_text,
                    "type": q_type,
                    "student_answer": submitted_val,
                    "correct_answer": f"{correct_letter}: {options.get(correct_letter, correct_ans)}" if options else correct_ans,
                    "is_correct": is_correct,
                    "score": score,
                    "awarded_marks": awarded_marks,
                    "max_marks": mcq_max_marks,
                    "evaluation_method": "Deterministic Exact Option Match",
                    "semantic_similarity": 1.0 if is_correct else 0.0,
                    "concept_coverage": 1.0 if is_correct else 0.0,
                    "completeness_score": 1.0 if submitted_val else 0.0,
                    "human_review_recommended": False,
                    "review_reason": None,
                    "feedback": feedback,
                    "suggestions": "None" if is_correct else "Review the lecture section regarding this specific factual concept.",
                    "missing_concepts": [] if is_correct else ["Correct option not selected"],
                    "covered_concepts": ["Selected correct option"] if is_correct else [],
                    "explanation": q.get("explanation", ""),
                    "difficulty": difficulty,
                    "bloom_level": bloom_lvl,
                    "topic": topic,
                    "disclaimer": "MCQs are graded deterministically against the verified correct option."
                }
            else:
                # Free-text evaluation via SemanticAnswerEvaluator with partial-credit scoring
                max_marks_map = {
                    "Short Answer": 2.0,
                    "Conceptual": 5.0,
                    "Application-Based": 5.0,
                    "Descriptive": 10.0,
                    "Analytical": 10.0
                }
                assigned_max = max_marks_map.get(q_type, 5.0)

                eval_res = self.evaluator.evaluate_answer(
                    student_answer=submitted_val,
                    reference_answer=correct_ans,
                    question_text=q_text,
                    question_type=q_type,
                    max_marks=assigned_max
                )
                score = eval_res["score_percentage"]
                is_correct = eval_res["is_correct"]

                if is_correct:
                    correct_count += 1
                    bloom_summary[bloom_lvl]["correct"] += 1

                item_result = {
                    "question_id": q_id,
                    "question": q_text,
                    "type": q_type,
                    "student_answer": submitted_val,
                    "correct_answer": correct_ans,
                    "is_correct": is_correct,
                    "score": score,
                    "awarded_marks": eval_res.get("awarded_marks", round((score / 100.0) * assigned_max, 1)),
                    "max_marks": assigned_max,
                    "evaluation_method": eval_res.get("evaluation_method", "Semantic Embedding & Concept Coverage"),
                    "semantic_similarity": eval_res.get("semantic_similarity", 0.0),
                    "concept_coverage": eval_res.get("concept_coverage", 0.0),
                    "completeness_score": eval_res.get("completeness_score", 0.0),
                    "human_review_recommended": eval_res.get("human_review_recommended", False),
                    "review_reason": eval_res.get("review_reason"),
                    "feedback": eval_res["feedback"],
                    "suggestions": eval_res["suggestions"],
                    "missing_concepts": eval_res.get("missing_concepts", []),
                    "covered_concepts": eval_res.get("covered_concepts", []),
                    "explanation": q.get("explanation", ""),
                    "difficulty": difficulty,
                    "bloom_level": bloom_lvl,
                    "topic": topic,
                    "disclaimer": eval_res.get("disclaimer", "Automated NLP evaluations are algorithmic estimates.")
                }

            evaluated_items.append(item_result)

        total_questions = len(questions)
        total_awarded = round(sum(item.get("awarded_marks", 0.0) for item in evaluated_items), 1)
        total_possible = round(sum(item.get("max_marks", 1.0) for item in evaluated_items), 1)
        overall_pct = round((total_awarded / max(0.1, total_possible)) * 100, 1)

        # Persist into database
        self.db.save_quiz_attempt(
            attempt_id=attempt_id,
            lecture_id=lecture_id,
            total_questions=total_questions,
            correct_answers=correct_count,
            total_score=total_awarded,
            percentage=overall_pct,
            difficulty_selected=difficulty_label,
            bloom_summary=bloom_summary,
            answers=evaluated_items
        )

        return {
            "attempt_id": attempt_id,
            "total_questions": total_questions,
            "correct_answers": correct_count,
            "total_awarded_marks": total_awarded,
            "total_possible_marks": total_possible,
            "overall_percentage": overall_pct,
            "items": evaluated_items,
            "bloom_summary": bloom_summary
        }
