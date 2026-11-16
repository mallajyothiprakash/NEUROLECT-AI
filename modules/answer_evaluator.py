"""
NEUROLECT AI - Semantic Answer Evaluation Module
Evaluates student free-text responses against model reference answers using
SentenceTransformer embeddings (all-MiniLM-L6-v2), technical concept coverage,
completeness scoring, negation contrast detection, and partial-credit grading.
Strictly semantic evaluation, NOT document retrieval or RAG.
"""

import re
import string
from typing import Dict, Any, List, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity
from models.model_loader import model_loader
import logging

logger = logging.getLogger("AnswerEvaluator")

# Common function words to filter out when extracting reference concepts
CONCEPT_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "to", "from", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "don", "should", "now", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "up", "down", "in", "out", "on", "off",
    "this", "that", "these", "those", "it", "its", "they", "them", "their"
}

# Negation and contrast tokens that invert semantic meaning
NEGATION_TOKENS = {
    "not", "never", "no", "cannot", "neither", "nor", "fails", "failed",
    "without", "incorrect", "false", "opposite", "contradicts", "unlike", "instead"
}

# Default maximum marks mapped by pedagogical question type
DEFAULT_MAX_MARKS = {
    "Multiple Choice (MCQ)": 1.0,
    "Short Answer": 2.0,
    "Conceptual": 5.0,
    "Application-Based": 5.0,
    "Descriptive": 10.0,
    "Analytical": 10.0
}

def extract_technical_concepts(text: str) -> Set[str]:
    """Extracts distinctive content words (length >= 3, excluding stopwords)."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in CONCEPT_STOP_WORDS}

def detect_polarity_inversion(student_text: str, reference_text: str) -> bool:
    """
    Detects if student response introduces negation tokens absent in the reference answer,
    which often inverts semantic meaning despite high lexical/cosine similarity.
    """
    s_words = set(re.findall(r'\b[a-zA-Z]+\b', student_text.lower()))
    r_words = set(re.findall(r'\b[a-zA-Z]+\b', reference_text.lower()))
    
    student_negations = s_words.intersection(NEGATION_TOKENS)
    reference_negations = r_words.intersection(NEGATION_TOKENS)
    
    # If student has strong negation where reference has none
    return len(student_negations) > 0 and len(reference_negations) == 0

class SemanticAnswerEvaluator:
    """Evaluates student answers against reference answers with semantic similarity and concept coverage."""

    def __init__(self):
        self.loader = model_loader

    def evaluate_answer(
        self,
        student_answer: str,
        reference_answer: str,
        question_text: str = "",
        question_type: str = "Short Answer",
        max_marks: Optional[float] = None,
        key_terms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates a student's answer using:
        1. Contextual Semantic Similarity (all-MiniLM-L6-v2 SentenceTransformer cosine similarity)
        2. Technical Concept Coverage (reference keywords & key terms)
        3. Answer Completeness & Length Expectation
        4. Polarity / Negation Anomaly Detection
        5. Partial-Credit Scoring & Instructor Review Recommendation
        """
        s_clean = student_answer.strip() if student_answer else ""
        r_clean = reference_answer.strip() if reference_answer else ""

        assigned_max_marks = max_marks or DEFAULT_MAX_MARKS.get(question_type, 5.0)

        # Handle missing reference answer
        if not r_clean:
            return {
                "score_percentage": 0.0,
                "awarded_marks": 0.0,
                "max_marks": assigned_max_marks,
                "semantic_similarity": 0.0,
                "concept_coverage": 0.0,
                "completeness_score": 0.0,
                "grade": "Unrated",
                "is_correct": False,
                "evaluation_method": "Unavailable (Missing Reference)",
                "feedback": "Evaluation unavailable: No reference answer provided for this question.",
                "missing_concepts": [],
                "covered_concepts": [],
                "suggestions": "Contact the instructor for manual evaluation.",
                "reference_answer": "",
                "human_review_recommended": True,
                "review_reason": "Missing reference answer in question metadata.",
                "disclaimer": "Automated evaluation requires an official reference answer."
            }

        # Handle trivial or empty submissions
        if not s_clean or len(s_clean.split()) < 2:
            return {
                "score_percentage": 0.0,
                "awarded_marks": 0.0,
                "max_marks": assigned_max_marks,
                "semantic_similarity": 0.0,
                "concept_coverage": 0.0,
                "completeness_score": 0.0,
                "grade": "Unsatisfactory",
                "is_correct": False,
                "evaluation_method": "Deterministic (Zero Submission Check)",
                "feedback": "No answer or insufficient text provided.",
                "missing_concepts": ["Complete explanation missing"],
                "covered_concepts": [],
                "suggestions": "Provide a complete explanation addressing the core principles.",
                "reference_answer": r_clean,
                "human_review_recommended": False,
                "review_reason": None,
                "disclaimer": "Automated NLP evaluations are algorithmic estimates and benefit from instructor review."
            }

        # 1. Semantic Cosine Similarity (SentenceTransformer)
        similarity = 0.0
        eval_method = "SentenceTransformer Semantic Embedding (all-MiniLM-L6-v2) + Concept Analysis"
        try:
            st_model = self.loader.get_sentence_transformer()
            embeddings = st_model.encode([s_clean, r_clean], convert_to_numpy=True)
            cos_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            similarity = max(0.0, min(1.0, cos_sim))
        except Exception as e:
            logger.warning(f"SentenceTransformer evaluation error, using lexical fallback: {e}")
            eval_method = "Lexical Overlap & Keyword Fallback (Transparent Fallback)"
            s_words = extract_technical_concepts(s_clean)
            r_words = extract_technical_concepts(r_clean)
            similarity = len(s_words.intersection(r_words)) / max(1, len(r_words))

        # 2. Key Concept & Term Coverage
        ref_concepts = extract_technical_concepts(r_clean)
        if key_terms:
            ref_concepts.update({t.lower() for t in key_terms if len(t) >= 3})

        student_concepts = extract_technical_concepts(s_clean)
        
        covered_concepts = sorted(list(ref_concepts.intersection(student_concepts)))
        missing_concepts = sorted(list(ref_concepts - student_concepts))
        
        coverage_ratio = len(covered_concepts) / max(1, len(ref_concepts))

        # 3. Completeness & Length Expectation
        ref_word_count = len(r_clean.split())
        student_word_count = len(s_clean.split())
        
        # Desired length ratio (student should ideally write at least 60% of reference word count)
        expected_min_words = max(5, int(ref_word_count * 0.6))
        completeness_ratio = min(1.0, student_word_count / expected_min_words)

        # 4. Polarity Inversion & Contradiction Check
        has_negation_inversion = detect_polarity_inversion(s_clean, r_clean)
        
        # 5. Composite Scoring Formula:
        # 60% Semantic Similarity + 30% Concept Coverage + 10% Completeness
        raw_score = (0.60 * similarity) + (0.30 * coverage_ratio) + (0.10 * completeness_ratio)
        
        # If polarity inversion detected (e.g. "is NOT an optimization algorithm"), apply penalty
        if has_negation_inversion:
            raw_score *= 0.60
            logger.info("Applied polarity inversion penalty to student answer.")

        score_pct = round(max(0.0, min(100.0, raw_score * 100)), 1)
        awarded_marks = round((score_pct / 100.0) * assigned_max_marks, 1)

        # 6. Pedagogical Feedback & Categorization
        human_review = False
        review_reason = None

        if score_pct >= 85.0:
            grade = "Excellent"
            is_correct = True
            feedback = "Outstanding response! Demonstrates thorough conceptual understanding and accurate technical terminology."
            suggestions = "Great job. Your response accurately captures the core principles."
        elif score_pct >= 65.0:
            grade = "Good"
            is_correct = True
            feedback = "Good response! Captures the main concept, though some secondary details could be articulated further."
            suggestions = f"Consider including technical terms such as: {', '.join(missing_concepts[:3])}." if missing_concepts else "Elaborate slightly on the underlying mechanisms."
        elif score_pct >= 40.0:
            grade = "Partially Correct"
            is_correct = False
            feedback = "Partially correct. The answer touches on relevant ideas but misses critical theoretical elements."
            suggestions = f"Try addressing core concepts: {', '.join(missing_concepts[:4])}." if missing_concepts else "Review the lecture material for this section."
            human_review = True
            review_reason = "Borderline score (partial credit awarded); student demonstrates partial comprehension."
        else:
            grade = "Needs Improvement"
            is_correct = False
            feedback = "The answer does not sufficiently address the core requirements of the question."
            suggestions = "Review the provided reference answer and re-read the lecture notes for this section."

        # Flag human review for high-value descriptive answers with notable divergence
        if "Descriptive" in question_type or "Analytical" in question_type:
            if abs(similarity - coverage_ratio) > 0.45:
                human_review = True
                review_reason = "High divergence between semantic embedding similarity and technical concept coverage."
            elif has_negation_inversion:
                human_review = True
                review_reason = "Potential negation or contradiction detected in descriptive response."
            elif score_pct < 50.0:
                human_review = True
                review_reason = "Low score on high-weight descriptive item; instructor review recommended."

        return {
            "score_percentage": score_pct,
            "awarded_marks": awarded_marks,
            "max_marks": assigned_max_marks,
            "semantic_similarity": round(similarity, 3),
            "concept_coverage": round(coverage_ratio, 3),
            "completeness_score": round(completeness_ratio, 3),
            "grade": grade,
            "is_correct": is_correct,
            "evaluation_method": eval_method,
            "feedback": feedback,
            "missing_concepts": missing_concepts[:5],
            "covered_concepts": covered_concepts,
            "suggestions": suggestions,
            "reference_answer": r_clean,
            "human_review_recommended": human_review,
            "review_reason": review_reason,
            "disclaimer": "Note: Automated NLP evaluations are algorithmic estimates based on semantic embeddings and benefit from instructor review."
        }

