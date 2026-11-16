"""
NEUROLECT AI - Named Entity Recognition Module
Extracts educational entities (People, Organizations, Locations, Technical Entities)
using pretrained BERT NER (dslim/bert-base-NER).
Preserves source pages and timestamps.
Strictly RAG-free.
"""

from typing import List, Dict, Any, Optional
from models.model_loader import model_loader
from modules.content_models import LectureContent
import config
import logging

logger = logging.getLogger("NERProcessor")

class NamedEntityRecognizer:
    """Extracts and categorizes named entities from lecture chunks using transformer models."""

    def __init__(self):
        self.loader = model_loader

    def extract_entities(
        self,
        lecture_content: LectureContent,
        max_entities: int = config.MAX_ENTITIES_DISPLAYED
    ) -> Dict[str, Any]:
        """
        Runs transformer NER across lecture chunks and aggregates entities
        into People, Organizations, Locations, and Technical & Miscellaneous categories.
        Preserves source references (e.g. Page number or audio timestamp).
        """
        ner_pipeline = self.loader.get_ner_pipeline()
        chunks = lecture_content.chunks

        raw_entities = []
        logger.info(f"Extracting named entities across {len(chunks)} lecture chunks...")

        # Process all chunks sequentially without arbitrary truncation
        for chunk in chunks:
            text = chunk["text"]
            if len(text.strip()) < 10:
                continue
            try:
                # Pipeline with aggregation_strategy='simple' groups subwords into single entities
                chunk_entities = ner_pipeline(text)
                for ent in chunk_entities:
                    ent["source_ref"] = chunk.get("source_ref", "General")
                    raw_entities.append(ent)
            except Exception as e:
                logger.warning(f"NER failed on chunk {chunk.get('chunk_id')}: {e}")

        # Category mapping for dslim/bert-base-NER
        categorized = {
            "People": {},
            "Organizations": {},
            "Locations": {},
            "Technical & Miscellaneous": {}
        }

        category_map = {
            "PER": "People",
            "ORG": "Organizations",
            "LOC": "Locations",
            "MISC": "Technical & Miscellaneous"
        }

        for ent in raw_entities:
            entity_group = ent.get("entity_group", ent.get("entity", "MISC"))
            category = category_map.get(entity_group, "Technical & Miscellaneous")
            word = ent.get("word", "").strip()

            # Clean subword marks if present
            word = word.replace("##", "").strip()
            if len(word) < 2 or not any(c.isalnum() for c in word):
                continue

            word_key = word.title()
            score = float(ent.get("score", 0.0))
            source_ref = ent.get("source_ref", "General")

            if word_key not in categorized[category]:
                categorized[category][word_key] = {
                    "text": word_key,
                    "confidence": round(score, 3),
                    "count": 1,
                    "source_refs": [source_ref]
                }
            else:
                categorized[category][word_key]["count"] += 1
                categorized[category][word_key]["confidence"] = max(
                    categorized[category][word_key]["confidence"],
                    round(score, 3)
                )
                if source_ref not in categorized[category][word_key]["source_refs"]:
                    categorized[category][word_key]["source_refs"].append(source_ref)

        # Convert to sorted lists
        result = {}
        total_found = 0
        for cat, items in categorized.items():
            sorted_items = sorted(
                items.values(),
                key=lambda x: (x["count"], x["confidence"]),
                reverse=True
            )
            # Format primary source_ref for display
            for itm in sorted_items:
                itm["source_ref"] = ", ".join(itm["source_refs"][:2])

            result[cat] = sorted_items[:max_entities]
            total_found += len(result[cat])

        return {
            "model_used": config.MODELS_CONFIG["ner"]["model_id"],
            "total_entities_found": total_found,
            "categories": result
        }
