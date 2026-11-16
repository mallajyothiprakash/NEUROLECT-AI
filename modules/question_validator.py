"""
NEUROLECT AI - Question Quality Validation Module
Validates generated assessment questions for structural integrity, semantic groundedness,
MCQ option uniqueness, non-empty answers, and chunk-level deduplication without RAG.
"""

from typing import List, Dict, Any, Tuple, Set
import re
import string

# Generic interrogative and auxiliary tokens to ignore when computing lexical similarity
QUESTION_STOP_WORDS = {
    "what", "which", "where", "when", "why", "how", "who", "whom", "whose",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would",
    "should", "will", "the", "a", "an", "in", "on", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "of", "and", "or",
    "primary", "main", "explain", "describe", "define", "difference", "compare",
    "contrast", "state", "list", "identify", "following", "given", "based", "according"
}

def normalize_question_text(text: str) -> str:
    """Normalizes question string for exact duplicate matching."""
    cleaned = text.lower().strip()
    cleaned = re.sub(r'^(question|task|generate\s*(?:a\s*)?question):\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', cleaned)
    return " ".join(cleaned.split())

def extract_content_keywords(text: str) -> Set[str]:
    """Extracts distinctive content words (length >= 3, excluding meta-question words)."""
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return {w for w in words if w not in QUESTION_STOP_WORDS}

def calculate_jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Word-Set Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection) / float(union) if union > 0 else 0.0

class QuestionValidator:
    """Performs rigorous quality audits on generated educational assessment questions."""

    def validate_question_batch(
        self,
        questions: List[Dict[str, Any]],
        source_text: str
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Filters and audits questions. Returns:
        - valid_questions: list of cleaned, verified question dictionaries
        - warnings: list of pedagogical quality warnings
        """
        valid_questions: List[Dict[str, Any]] = []
        warnings: List[str] = []
        
        seen_normalized_questions: Set[str] = set()
        accepted_content_sets: List[Set[str]] = []

        source_content_words = extract_content_keywords(source_text)

        for idx, q in enumerate(questions):
            q_text = q.get("question", "").strip()
            ans_text = q.get("correct_answer", "").strip()
            q_type = q.get("type", "Short Answer")

            # Check 1: Non-empty question and answer
            if not q_text or len(q_text) < 10:
                warnings.append(f"Question #{idx+1} rejected: question text is too brief or empty.")
                continue

            if not ans_text or len(ans_text) < 2:
                warnings.append(f"Question #{idx+1} rejected: missing or trivial answer.")
                continue

            # Check 2: Clean malformed prefixes
            if re.match(r'^(question|task|generate\s*(?:a\s*)?question):\s*', q_text, flags=re.IGNORECASE):
                q_text = re.sub(r'^(question|task|generate\s*(?:a\s*)?question):\s*', '', q_text, flags=re.IGNORECASE).strip()

            if not q_text.endswith("?"):
                q_text += "?"
            q["question"] = q_text

            # Check 3: Full-Question Duplicate and Similarity Detection
            norm_q = normalize_question_text(q_text)
            if norm_q in seen_normalized_questions:
                warnings.append(f"Question #{idx+1} omitted: exact duplicate question detected.")
                continue

            q_content_words = extract_content_keywords(q_text)
            
            # Check pairwise Jaccard similarity against already accepted questions
            is_semantic_duplicate = False
            for accepted_words in accepted_content_sets:
                sim = calculate_jaccard_similarity(q_content_words, accepted_words)
                if sim >= 0.80 and len(q_content_words) >= 3:
                    is_semantic_duplicate = True
                    break
            
            if is_semantic_duplicate:
                warnings.append(f"Question #{idx+1} omitted: near-duplicate question detected (high content overlap).")
                continue

            # Check 4: MCQ specific integrity
            if "MCQ" in q_type:
                options = q.get("options", {})
                if not options or len(options) != 4:
                    warnings.append(f"MCQ #{idx+1} rejected: does not contain exactly 4 options.")
                    continue
                
                # Check for duplicate options (normalized)
                normalized_opts = []
                for v in options.values():
                    cleaned_opt = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', v.lower()).strip()
                    normalized_opts.append(" ".join(cleaned_opt.split()))

                if len(set(normalized_opts)) != len(normalized_opts):
                    warnings.append(f"MCQ #{idx+1} rejected: duplicate options found among choices.")
                    continue

                # Ensure correct letter points to existing option
                correct_letter = q.get("correct_letter", "")
                if correct_letter not in options:
                    warnings.append(f"MCQ #{idx+1} rejected: correct option letter '{correct_letter}' is invalid.")
                    continue

            # Check 5: Groundedness check (overlap with source text content keywords)
            overlap = q_content_words.intersection(source_content_words)
            
            # Strict rejection if zero content overlap on an elaborate question (>= 4 content keywords)
            if len(overlap) == 0 and len(q_content_words) >= 4:
                warnings.append(f"Question #{idx+1} rejected: question lacks factual grounding in lecture content.")
                continue
            elif len(overlap) < 2 and len(q_content_words) >= 6:
                warnings.append(f"Question #{idx+1} warning: low lexical overlap with source lecture text.")

            # Passed all checks
            seen_normalized_questions.add(norm_q)
            accepted_content_sets.append(q_content_words)
            valid_questions.append(q)

        return valid_questions, warnings

