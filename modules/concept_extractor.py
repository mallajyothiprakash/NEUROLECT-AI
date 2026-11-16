"""
NEUROLECT AI - Important Concept Extraction Module
Identifies definitions, core explanations, key theorems, and factual statements
from lecture content grounded in source pages or timestamps.
Strictly RAG-free.
"""

import re
from typing import List, Dict, Any
from modules.content_models import LectureContent
import config

class ConceptExtractor:
    """Extracts grounded educational concepts and formal definitions from lecture text."""

    DEFINITION_PATTERNS = [
        r'\b([A-Z][a-zA-Z0-9\s_-]{2,30})\s+(?:is defined as|refers to|can be described as|is termed as)\s+([^.?!]+[.?!])',
        r'\b([A-Z][a-zA-Z0-9\s_-]{2,30})\s+(?:is a type of|is an essential|is a mechanism|is a framework|is a technique|is an architecture)\s+([^.?!]+[.?!])',
        r'\b([A-Z][a-zA-Z0-9\s_-]{2,30})\s*:\s*([^.?!]+[.?!])'
    ]

    EXPLANATION_PATTERNS = [
        r'\b(?:The key difference|The main advantage|The primary purpose|The reason why|In order to)\s+([^.?!]+[.?!])',
        r'\b(?:It is important to note that|Essentially|Specifically|Consequently|As a result)\s+([^.?!]+[.?!])',
        r'\b(?:The fundamental concept|The core principle|A critical property)\s+(?:of|is)\s+([^.?!]+[.?!])'
    ]

    def extract_concepts(
        self,
        lecture_content: LectureContent,
        max_concepts: int = config.MAX_CONCEPTS_EXTRACTED
    ) -> List[Dict[str, Any]]:
        """
        Scans sentences for definitions, core principles, and factual relationships,
        tagging each with its exact transcript timestamp or PDF page number.
        """
        concepts: List[Dict[str, Any]] = []
        seen_terms = set()

        # Step 1: Detect formal definitions
        for chunk in lecture_content.chunks:
            chunk_text = chunk["text"]
            source_ref = chunk.get("source_ref", "General")

            for pattern in self.DEFINITION_PATTERNS:
                matches = re.finditer(pattern, chunk_text, re.IGNORECASE)
                for m in matches:
                    term = m.group(1).strip().title()
                    definition = m.group(0).strip()
                    
                    if term.lower() not in seen_terms and len(term.split()) <= 5:
                        seen_terms.add(term.lower())
                        concepts.append({
                            "type": "Definition",
                            "term": term,
                            "statement": definition,
                            "source_ref": source_ref,
                            "importance": "High"
                        })
                        if len(concepts) >= max_concepts:
                            return concepts

        # Step 2: Extract key educational principles & explanations
        for chunk in lecture_content.chunks:
            chunk_text = chunk["text"]
            source_ref = chunk.get("source_ref", "General")

            for pattern in self.EXPLANATION_PATTERNS:
                matches = re.finditer(pattern, chunk_text, re.IGNORECASE)
                for m in matches:
                    statement = m.group(0).strip()
                    first_words = " ".join(statement.split()[:4])
                    if first_words.lower() not in seen_terms and len(statement.split()) > 7:
                        seen_terms.add(first_words.lower())
                        concepts.append({
                            "type": "Key Principle / Explanation",
                            "term": first_words,
                            "statement": statement,
                            "source_ref": source_ref,
                            "importance": "Medium"
                        })
                        if len(concepts) >= max_concepts:
                            return concepts

        # Step 3: High-salience educational sentence extraction fallback
        if len(concepts) < 3:
            for s in lecture_content.sentences[:8]:
                if len(s.split()) >= 10 and not any(c["statement"] == s for c in concepts):
                    concepts.append({
                        "type": "Important Statement",
                        "term": " ".join(s.split()[:3]),
                        "statement": s,
                        "source_ref": "Lecture Core",
                        "importance": "Standard"
                    })

        return concepts[:max_concepts]
