"""
NEUROLECT AI - Bloom's Taxonomy Classification Module
Classifies generated questions into the six cognitive levels:
1. Remember  2. Understand  3. Apply  4. Analyze  5. Evaluate  6. Create
Transparently uses pedagogical action verb hierarchies, syntactic framing, and cognitive complexity.
"""

import re
from typing import Dict, Any, Tuple
import config

class BloomsTaxonomyClassifier:
    """Classifies educational questions into Bloom's Revised Taxonomy levels."""

    # Cognitive Action Verb and Key Phrase Mapping
    BLOOM_TAXONOMY_MAP = {
        "Remember": {
            "verbs": ["define", "identify", "list", "name", "recall", "state", "what is", "which of", "who", "when", "where", "mention", "recognize"],
            "description": "Retrieving relevant knowledge from memory; factual recall."
        },
        "Understand": {
            "verbs": ["explain", "describe", "summarize", "paraphrase", "interpret", "clarify", "illustrate", "discuss", "why does", "what does it mean", "classify"],
            "description": "Constructing meaning from educational content; comprehension."
        },
        "Apply": {
            "verbs": ["apply", "demonstrate", "implement", "calculate", "solve", "execute", "use", "how would you", "given the scenario", "utilize", "operate"],
            "description": "Carrying out or using a procedure in a given situation."
        },
        "Analyze": {
            "verbs": ["analyze", "distinguish", "differentiate", "compare", "contrast", "deconstruct", "examine", "investigate", "what is the difference", "relate", "break down"],
            "description": "Breaking material into constituent parts and determining relationships."
        },
        "Evaluate": {
            "verbs": ["evaluate", "judge", "critique", "justify", "defend", "assess", "which is better", "prioritize", "validate", "rate", "determine the best"],
            "description": "Making judgments based on criteria and standards."
        },
        "Create": {
            "verbs": ["create", "design", "formulate", "construct", "develop", "devise", "propose", "synthesize", "generate", "how would you improve", "compose"],
            "description": "Putting elements together to form a novel, coherent whole."
        }
    }

    def classify_question(self, question_text: str, question_type: str = "Short Answer") -> Dict[str, Any]:
        """
        Classifies a question string into one of the 6 Bloom's Taxonomy cognitive tiers.
        Returns the level, confidence score, and pedagogical justification.
        """
        q_lower = question_text.lower().strip()
        
        # Scoring weights for each level
        scores = {lvl: 0 for lvl in config.BLOOMS_TAXONOMY_LEVELS}

        # Check verb matches
        for level, data in self.BLOOM_TAXONOMY_MAP.items():
            for verb in data["verbs"]:
                pattern = r'\b' + re.escape(verb) + r'\b'
                if re.search(pattern, q_lower):
                    # Higher weight if verb appears near the beginning
                    if q_lower.startswith(verb) or f"how {verb}" in q_lower or f"can you {verb}" in q_lower:
                        scores[level] += 3
                    else:
                        scores[level] += 1

        # Question type prior adjustments
        if "Analytical" in question_type:
            scores["Analyze"] += 2
        elif "Application" in question_type:
            scores["Apply"] += 2
        elif "Conceptual" in question_type:
            scores["Understand"] += 1
        elif "MCQ" in question_type:
            scores["Remember"] += 1

        # Select the level with the highest score
        best_level = max(scores, key=scores.get)
        if scores[best_level] == 0:
            # Default heuristics based on interrogative words
            if q_lower.startswith("why"):
                best_level = "Understand"
            elif q_lower.startswith("how"):
                best_level = "Apply"
            elif "compare" in q_lower or "difference" in q_lower:
                best_level = "Analyze"
            else:
                best_level = "Remember"

        return {
            "bloom_level": best_level,
            "description": self.BLOOM_TAXONOMY_MAP[best_level]["description"],
            "classifier_type": "Rule-Based Cognitive Verb & Syntactic Frame Analysis"
        }
