"""
NEUROLECT AI - Topic and Subtopic Detection Module
Performs hierarchical topic discovery using semantic sentence embeddings,
centroid clustering, and n-gram salience analysis.
Associates detected topics with source segments (page numbers or timestamps).
Transparently documents the methodology.
Strictly RAG-free.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from modules.content_models import LectureContent
from models.model_loader import model_loader
import logging

logger = logging.getLogger("TopicDetector")

# Common educational meta-words to filter out from topic labels
EDUCATIONAL_META_WORDS = {
    "lecture", "chapter", "slide", "slides", "audio", "video", "page", "pages",
    "section", "sections", "overview", "introduction", "conclusion", "summary",
    "discuss", "discussed", "discusses", "important", "study", "notes",
    "concept", "concepts", "topic", "topics", "student", "students", "class"
}

class TopicDetector:
    """Discovers major topics, hierarchical subtopics, outlines, and source segment links."""

    def __init__(self):
        self.loader = model_loader

    def detect_topics(
        self,
        lecture_content: LectureContent,
        num_topics: int = 4
    ) -> Dict[str, Any]:
        """
        Discovers thematic topic clusters across the lecture:
        1. Contextual Sentence Embeddings via all-MiniLM-L6-v2
        2. K-Means / Centroid Clustering
        3. Topic title & subtopic derivation from keyphrase frequencies
        4. Cross-referencing detected topics with source segments (pages/timestamps)
        """
        sentences = [s.strip() for s in lecture_content.sentences if len(s.strip().split()) > 3]
        segments = lecture_content.segments or []

        if not sentences:
            sentences = [lecture_content.cleaned_text] if lecture_content.cleaned_text else ["General Lecture Content"]

        # Graceful handling of very short documents
        if len(sentences) < 4:
            single_title = self._derive_topic_label(lecture_content.cleaned_text)
            matched_sources = list({seg.get("source_ref", "General") for seg in segments}) or ["Lecture Start"]
            return {
                "methodology": "Semantic Salience Extraction (Short Document)",
                "total_topics": 1,
                "topics": [{
                    "topic_id": 1,
                    "title": single_title,
                    "subtopics": ["Foundational Principles", "Practical Applications"],
                    "sentences": sentences,
                    "content_preview": lecture_content.cleaned_text[:300] + "...",
                    "source_segments": matched_sources
                }],
                "outline": [
                    f"1. {single_title} ({', '.join(matched_sources[:2])})",
                    "   • Foundational Principles",
                    "   • Practical Applications"
                ]
            }

        # Dynamic cluster count bound by sentence count
        k = min(num_topics, max(2, len(sentences) // 3))

        try:
            # 1. Compute contextual embeddings with SentenceTransformer
            st_model = self.loader.get_sentence_transformer()
            embeddings = st_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            methodology_name = "Transformer Sentence Embedding Clustering (all-MiniLM-L6-v2 + KMeans)"
        except Exception as e:
            logger.warning(f"SentenceTransformer topic detection fallback to TF-IDF: {e}")
            vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
            X = vectorizer.fit_transform(sentences)
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X.toarray())
            methodology_name = "TF-IDF Vector Space Clustering (Transparent Fallback)"

        # Group sentences by topic cluster
        cluster_sentences: Dict[int, List[str]] = {i: [] for i in range(k)}
        for idx, lbl in enumerate(cluster_labels):
            cluster_sentences[lbl].append(sentences[idx])

        topics: List[Dict[str, Any]] = []
        outline: List[str] = []

        for c_id, sents in cluster_sentences.items():
            if not sents:
                continue
            combined_cluster_text = " ".join(sents)
            topic_title = self._derive_topic_label(combined_cluster_text)
            subtopics = self._derive_subtopics(sents, exclude_words=topic_title.lower().split())

            # Map sentences to corresponding source segments (pages/timestamps)
            matched_sources = self._find_matching_segments(sents, segments)

            topic_data = {
                "topic_id": c_id + 1,
                "title": topic_title,
                "subtopics": subtopics,
                "sentence_count": len(sents),
                "content_preview": " ".join(sents[:3])[:250] + "...",
                "source_segments": matched_sources,
                "sentences": sents
            }
            topics.append(topic_data)
            
            # Format hierarchical outline
            sources_label = f" ({', '.join(matched_sources[:2])})" if matched_sources else ""
            outline.append(f"{c_id + 1}. {topic_title}{sources_label}")
            for sub in subtopics:
                outline.append(f"   • {sub}")

        return {
            "methodology": methodology_name,
            "total_topics": len(topics),
            "topics": topics,
            "outline": outline
        }

    def _find_matching_segments(self, cluster_sentences: List[str], segments: List[Dict[str, Any]]) -> List[str]:
        """Finds source pages or audio timestamps associated with sentences in this topic."""
        if not segments:
            return ["Section 1"]

        matched_refs = set()
        cluster_snippet = " ".join(cluster_sentences).lower()

        for seg in segments:
            seg_text = seg.get("text", "").lower()
            if seg_text and (seg_text[:40] in cluster_snippet or any(s.lower()[:30] in seg_text for s in cluster_sentences)):
                matched_refs.add(seg.get("source_ref", "Segment"))

        return sorted(list(matched_refs)) if matched_refs else [segments[0].get("source_ref", "General")]

    def _derive_topic_label(self, text: str) -> str:
        """Derives a concise, representative topic label using n-gram frequency analysis with stopword filtering."""
        try:
            tfidf = TfidfVectorizer(ngram_range=(2, 3), stop_words="english", max_features=25)
            matrix = tfidf.fit_transform([text])
            feature_names = tfidf.get_feature_names_out()
            
            for phrase in feature_names:
                words = phrase.lower().split()
                # Ensure phrase contains no generic educational meta-words
                if not any(w in EDUCATIONAL_META_WORDS for w in words) and len(phrase) > 5:
                    return phrase.title()
        except Exception:
            pass
        
        # Fallback to top informative unigrams
        words = [w.capitalize() for w in text.split() if len(w) > 4 and w.isalpha() and w.lower() not in EDUCATIONAL_META_WORDS]
        return " ".join(words[:2]) if words else "Educational Concepts"

    def _derive_subtopics(self, sentences: List[str], exclude_words: Optional[List[str]] = None) -> List[str]:
        """Extracts informative subtopic phrases from cluster sentences."""
        subtopics = []
        exclude = set(exclude_words or []).union(EDUCATIONAL_META_WORDS)
        try:
            tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=15)
            tfidf.fit(sentences)
            features = tfidf.get_feature_names_out()
            for feat in features:
                feat_words = set(feat.lower().split())
                if not feat_words.intersection(exclude) and len(feat.strip()) > 3:
                    subtopics.append(feat.title())
                if len(subtopics) >= 3:
                    break
        except Exception:
            pass
        
        if not subtopics:
            subtopics = ["Core Principles", "Practical Implementations"]
        return subtopics[:3]
