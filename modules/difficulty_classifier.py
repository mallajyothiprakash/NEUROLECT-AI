"""
NEUROLECT AI - Question Difficulty Classification Module
Assesses questions and classifies them into Easy, Medium, or Hard based on
lexical complexity, cognitive demand (Bloom level), and syntactic depth.
"""

from typing import Dict, Any
import config

class DifficultyClassifier:
    """Classifies educational assessment items into Easy, Medium, and Hard tiers."""

    def classify_difficulty(
        self,
        question_text: str,
        bloom_level: str,
        question_type: str = "Short Answer"
    ) -> Dict[str, Any]:
        """
        Determines difficulty by combining:
        1. Bloom's Taxonomy Cognitive Tier (Remember/Understand -> Easy/Med; Analyze/Create -> Hard)
        2. Syntactic & Lexical Complexity (word count, multi-clause structure)
        3. Question Type Depth (Descriptive/Analytical -> higher difficulty)
        """
        words = question_text.split()
        word_count = len(words)
        
        # Difficulty points accumulation
        points = 0

        # 1. Cognitive level weighting
        bloom_weights = {
            "Remember": 1,
            "Understand": 2,
            "Apply": 3,
            "Analyze": 4,
            "Evaluate": 5,
            "Create": 5
        }
        points += bloom_weights.get(bloom_level, 2)

        # 2. Length and syntactic complexity
        if word_count > 25:
            points += 3
        elif word_count > 14:
            points += 2
        else:
            points += 1

        # 3. Question type demands
        if "Descriptive" in question_type or "Analytical" in question_type:
            points += 2
        elif "Application" in question_type:
            points += 1

        # Threshold classification
        if points <= 4:
            difficulty = "Easy"
            reason = "Focuses on foundational recall with straightforward syntax."
        elif points <= 7:
            difficulty = "Medium"
            reason = "Requires conceptual understanding or practical application."
        else:
            difficulty = "Hard"
            reason = "Demands multi-step analytical reasoning or comprehensive evaluation."

        return {
            "difficulty": difficulty,
            "complexity_score": points,
            "justification": reason
        }
