"""
NEUROLECT AI - MCQ Distractor Generation Module
Generates plausible, domain-relevant, non-duplicate distractors for multiple-choice questions
using pretrained Flan-T5 model prompting, domain terminology sampling, and context keyphrases.
Strictly RAG-free.
"""

import random
import re
import string
from typing import List, Dict, Any, Optional
from models.model_loader import model_loader
import logging

logger = logging.getLogger("DistractorGenerator")

def normalize_text_for_comparison(text: str) -> str:
    """Normalizes text for duplicate and equality checking by stripping punctuation, whitespace, and casing."""
    if not text:
        return ""
    # Strip punctuation and lower
    cleaned = text.lower().strip()
    cleaned = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', cleaned)
    return " ".join(cleaned.split())

class DistractorGenerator:
    """Generates 3 plausible incorrect choices alongside the 1 correct answer for MCQs."""

    def __init__(self):
        self.loader = model_loader

    def generate_options(
        self,
        question: str,
        correct_answer: str,
        context_text: str,
        candidate_terms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Creates exactly 4 options (A, B, C, D) with 1 correct answer and 3 plausible distractors.
        Deduplicates options using normalized comparison.
        If 3 credible distractors cannot be generated, returns success=False without manufacturing
        misleading or synthetic template options.
        """
        clean_correct = correct_answer.strip()
        norm_correct = normalize_text_for_comparison(clean_correct)
        
        if not clean_correct or not norm_correct:
            return {
                "success": False,
                "options": {},
                "correct_letter": "",
                "correct_answer": "",
                "distractors": [],
                "reason": "Missing or empty correct answer."
            }

        distractors: List[str] = []
        seen_normalized = {norm_correct}

        # Attempt 1: Model-Assisted Generation via Flan-T5
        try:
            tokenizer, model = self.loader.get_question_generator()
            import torch
            
            prompt = (
                f"Context: {context_text[:500]}\n"
                f"Question: {question}\n"
                f"Correct Answer: {clean_correct}\n"
                f"Task: Generate 3 false but plausible incorrect choices for this question based on the subject. "
                f"Separate them with commas:\n"
                f"Incorrect Choices:"
            )

            inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    num_beams=2,
                    early_stopping=True
                )
            raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # Split model output on commas, semicolons, or newlines
            candidates = [c.strip() for c in re.split(r'[,;\n|]', raw_output) if c.strip()]
            for c in candidates:
                # Strip leading enumeration (1., a), -, *, Option:)
                clean_c = re.sub(r'^(?:\d+[\.\)]|[a-dA-D][\.\)]|[-*•]|option\s*[a-d\d]*:\s*|choice:\s*)', '', c, flags=re.IGNORECASE).strip()
                clean_c = clean_c.strip('"\'. ')
                norm_c = normalize_text_for_comparison(clean_c)
                
                if norm_c and norm_c not in seen_normalized:
                    if len(clean_c) > 1 and len(clean_c.split()) <= 10:
                        distractors.append(clean_c)
                        seen_normalized.add(norm_c)
                        if len(distractors) >= 3:
                            break
        except Exception as e:
            logger.warning(f"Model-based distractor generation error: {e}")

        # Attempt 2: Domain Terminology Extraction Fallback
        if len(distractors) < 3 and candidate_terms:
            for term in candidate_terms:
                clean_term = term.strip().title()
                norm_term = normalize_text_for_comparison(clean_term)
                if norm_term and norm_term not in seen_normalized and len(clean_term) > 2:
                    distractors.append(clean_term)
                    seen_normalized.add(norm_term)
                if len(distractors) >= 3:
                    break

        # Attempt 3: Context-Derived Keyphrase Sampling
        if len(distractors) < 3 and context_text:
            # Extract technical multi-word terms or capitalized entities from context
            context_candidates = re.findall(r'\b[A-Z][a-zA-Z]{3,}(?:\s+[A-Z][a-zA-Z]{3,})*\b', context_text)
            for phrase in context_candidates:
                clean_phrase = phrase.strip()
                norm_phrase = normalize_text_for_comparison(clean_phrase)
                if norm_phrase and norm_phrase not in seen_normalized and len(clean_phrase.split()) <= 4:
                    distractors.append(clean_phrase)
                    seen_normalized.add(norm_phrase)
                if len(distractors) >= 3:
                    break

        # Rigorous quality gate: Do not fabricate generic syntactic filler distractors
        if len(distractors) < 3:
            logger.warning(f"Could not generate 3 credible distractors for question: '{question[:60]}...'")
            return {
                "success": False,
                "options": {},
                "correct_letter": "",
                "correct_answer": clean_correct,
                "distractors": distractors,
                "reason": "Insufficient credible distractors found in lecture context."
            }

        # Take exactly 3 distractors
        final_distractors = distractors[:3]

        # Combine correct answer and distractors, shuffle them
        all_choices = [clean_correct] + final_distractors
        random.shuffle(all_choices)

        # Label choices A, B, C, D
        letters = ["A", "B", "C", "D"]
        options_dict = {}
        correct_letter = "A"

        for idx, choice in enumerate(all_choices[:4]):
            letter = letters[idx]
            options_dict[letter] = choice
            if normalize_text_for_comparison(choice) == norm_correct:
                correct_letter = letter

        return {
            "success": True,
            "options": options_dict,
            "correct_letter": correct_letter,
            "correct_answer": clean_correct,
            "distractors": final_distractors
        }

