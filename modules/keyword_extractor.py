"""
NEUROLECT AI - KeyBERT-Style Keyword Extraction Module
Extracts key technical concepts and terminology using candidate n-gram generation,
SentenceTransformer contextual embeddings, Maximal Marginal Relevance (MMR) deduplication,
and topic association.
Strictly RAG-free.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.model_loader import model_loader
from modules.content_models import LectureContent
import config
import logging

logger = logging.getLogger("KeywordExtractor")

class KeywordExtractor:
    """Extracts, ranks, and deduplicates educational keywords using embedding similarity."""

    def __init__(self):
        self.loader = model_loader

    def extract_keywords(
        self,
        lecture_content: LectureContent,
        top_n: int = config.MAX_KEYWORDS_EXTRACTED,
        diversity: float = 0.6,
        topics_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extracts key educational terms using candidate phrase extraction and
        semantic cosine similarity against the lecture context.
        Associates each term with its most relevant detected topic.
        """
        text = lecture_content.cleaned_text
        if not text or len(text.split()) < 5:
            return []

        try:
            # 1. Candidate phrase extraction (unigrams and bigrams)
            vectorizer = CountVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b"
            )
            count_matrix = vectorizer.fit_transform([text])
            candidates = vectorizer.get_feature_names_out()

            if len(candidates) == 0:
                return []

            # 2. Compute embeddings using SentenceTransformer
            st_model = self.loader.get_sentence_transformer()
            doc_embedding = st_model.encode([text[:1500]], convert_to_numpy=True)
            candidate_embeddings = st_model.encode(list(candidates), convert_to_numpy=True)

            # 3. Maximal Marginal Relevance (MMR) for diversity and deduplication
            selected_keywords = self._mmr(
                doc_embedding=doc_embedding,
                candidate_embeddings=candidate_embeddings,
                words=candidates,
                top_n=top_n,
                diversity=diversity
            )

            results: List[Dict[str, Any]] = []
            for word, score in selected_keywords:
                count = text.lower().count(word.lower())
                associated_topic = self._associate_topic(word, topics_data)

                results.append({
                    "keyword": word.title(),
                    "relevance_score": round(float(score), 4),
                    "frequency": max(1, count),
                    "type": "Key Term" if len(word.split()) == 1 else "Technical Concept",
                    "associated_topic": associated_topic
                })

            return results

        except Exception as e:
            logger.warning(f"Embedding keyword extraction error, falling back to TF-IDF: {e}")
            return self._fallback_tfidf_keywords(text, top_n, topics_data)

    def _associate_topic(self, keyword: str, topics_data: Optional[Dict[str, Any]]) -> str:
        """Associates a keyword with the most contextually relevant detected topic."""
        if not topics_data or "topics" not in topics_data:
            return "General Lecture"

        kw_lower = keyword.lower()
        for topic in topics_data.get("topics", []):
            topic_title = topic.get("title", "")
            topic_preview = topic.get("content_preview", "").lower()
            topic_sentences = " ".join(topic.get("sentences", [])).lower()

            if kw_lower in topic_title.lower() or kw_lower in topic_preview or kw_lower in topic_sentences:
                return topic_title

        return topics_data["topics"][0].get("title", "Foundations") if topics_data.get("topics") else "General"

    def _mmr(
        self,
        doc_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        words: np.ndarray,
        top_n: int,
        diversity: float
    ) -> List[Tuple[str, float]]:
        """Maximal Marginal Relevance to diversify selected keywords and eliminate redundancy."""
        word_doc_sim = cosine_similarity(candidate_embeddings, doc_embedding)
        word_word_sim = cosine_similarity(candidate_embeddings, candidate_embeddings)

        keywords_idx = [int(np.argmax(word_doc_sim))]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        for _ in range(min(top_n - 1, len(candidates_idx))):
            candidate_similarities = word_doc_sim[candidates_idx, :]
            target_similarities = np.max(word_word_sim[candidates_idx][:, keywords_idx], axis=1)

            mmr = (1 - diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[int(np.argmax(mmr))]

            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        return [(words[idx], float(word_doc_sim[idx][0])) for idx in keywords_idx]

    def _fallback_tfidf_keywords(
        self,
        text: str,
        top_n: int,
        topics_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Fallback keyword extraction using TF-IDF."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=top_n)
            matrix = vec.fit_transform([text])
            feature_names = vec.get_feature_names_out()
            scores = matrix.toarray()[0]
            
            sorted_indices = scores.argsort()[::-1]
            results = []
            for idx in sorted_indices[:top_n]:
                term = feature_names[idx]
                results.append({
                    "keyword": term.title(),
                    "relevance_score": round(float(scores[idx]), 3),
                    "frequency": text.lower().count(term.lower()),
                    "type": "Key Term" if len(term.split()) == 1 else "Technical Concept",
                    "associated_topic": self._associate_topic(term, topics_data)
                })
            return results
        except Exception:
            return []
