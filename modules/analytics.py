"""
NEUROLECT AI - Learning Analytics & Pedagogical Diagnostic Module
Computes holistic educational performance metrics, topic mastery diagnostics,
cognitive Bloom distributions, difficulty analyses, question-type evaluations,
and personalized study recommendations grounded in persisted assessment data.
Strictly RAG-free.
"""

from typing import Dict, Any, List, Optional
from database.db_manager import DatabaseManager
import logging

logger = logging.getLogger("LearningAnalytics")

class LearningAnalytics:
    """Computes comprehensive educational assessment metrics from SQLite database records."""

    # Transparent and configurable mastery thresholds
    MASTERY_STRONG_THRESHOLD = 75.0      # Score percentage >= 75.0% for strong performance
    MASTERY_MODERATE_THRESHOLD = 50.0    # Score percentage between 50.0% and 74.9% for moderate
    MIN_SAMPLE_FOR_CONFIRMATION = 2      # Minimum questions attempted to confirm mastery

    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager()

    def generate_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Retrieves aggregated database analytics, evaluates topic mastery status,
        tracks score progression across attempts, and synthesizes data-backed recommendations.
        """
        data = self.db.get_overall_analytics()
        summary = data.get("summary", {})
        raw_topics = data.get("topic_stats", [])
        raw_bloom = data.get("bloom_stats", [])
        raw_diff = data.get("difficulty_stats", [])
        raw_q_types = data.get("question_type_stats", [])
        historical_attempts = data.get("historical_trend", [])

        total_attempts = summary.get("total_attempts", 0)

        # Handle empty database gracefully
        if total_attempts == 0:
            return self._build_empty_metrics()

        # 1. Process Topic Mastery Analytics
        topic_analysis = self._process_topic_mastery(raw_topics)
        
        # 2. Process Cognitive Bloom's Taxonomy Analytics
        bloom_analysis = self._process_bloom_analytics(raw_bloom)

        # 3. Process Difficulty Level Breakdown
        difficulty_analysis = self._process_difficulty_analytics(raw_diff)

        # 4. Process Question-Type Analytics
        question_type_analysis = self._process_question_type_analytics(raw_q_types)

        # 5. Process Historical Score Trends and Progress Tracking
        progress_tracking = self._process_progress_tracking(historical_attempts)

        # 6. Generate Data-Backed Personalized Recommendations
        recommendations = self._generate_personalized_recommendations(
            topic_analysis=topic_analysis,
            bloom_analysis=bloom_analysis,
            difficulty_analysis=difficulty_analysis,
            question_type_analysis=question_type_analysis,
            progress_tracking=progress_tracking
        )

        return {
            "summary": summary,
            "topic_stats": topic_analysis["enriched_topics"],
            "strong_topics": topic_analysis["strong_topics"],
            "moderate_topics": topic_analysis["moderate_topics"],
            "weak_topics": topic_analysis["weak_topics"],
            "bloom_stats": bloom_analysis["enriched_bloom"],
            "bloom_diagnosis": bloom_analysis["diagnosis"],
            "difficulty_stats": difficulty_analysis["enriched_diff"],
            "difficulty_diagnosis": difficulty_analysis["diagnosis"],
            "question_type_stats": question_type_analysis["enriched_types"],
            "question_type_diagnosis": question_type_analysis["diagnosis"],
            "historical_trend": historical_attempts,
            "progress_tracking": progress_tracking,
            "recommendations": recommendations,
            "pedagogical_caveat": (
                "Automated cognitive and difficulty classifications are heuristic approximations based on "
                "pedagogical action verbs and syntactic complexity. Metrics reflect student interaction data "
                "and provide developmental guidance rather than standardized psychological ratings."
            )
        }

    def _build_empty_metrics(self) -> Dict[str, Any]:
        """Returns clean zeroed structure when no assessment records exist."""
        return {
            "summary": {
                "total_attempts": 0,
                "total_questions": 0,
                "total_correct": 0,
                "average_score": 0.0,
                "highest_score": 0.0,
                "lowest_score": 0.0
            },
            "topic_stats": [],
            "strong_topics": [],
            "moderate_topics": [],
            "weak_topics": [],
            "bloom_stats": [],
            "bloom_diagnosis": "No assessment data available to evaluate cognitive Bloom levels.",
            "difficulty_stats": [],
            "difficulty_diagnosis": "No assessment data available to evaluate difficulty handling.",
            "question_type_stats": [],
            "question_type_diagnosis": "No question type records available.",
            "historical_trend": [],
            "progress_tracking": {
                "total_attempts": 0,
                "score_delta": 0.0,
                "status": "No Attempts Recorded",
                "latest_score": 0.0,
                "previous_score": 0.0,
                "summary": "Take an interactive assessment to start tracking learning progress."
            },
            "recommendations": [
                "No assessment attempts recorded yet. Launch an assessment from the Quiz page to generate diagnostics."
            ],
            "pedagogical_caveat": "No assessment records available."
        }

    def _process_topic_mastery(self, raw_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enriches topic records with mastery status, sample-size safeguards, and categorization.
        Avoids prematurely declaring a topic mastered on a sample of 1 question.
        """
        enriched = []
        strong_topics = []
        moderate_topics = []
        weak_topics = []

        for t in raw_topics:
            topic_name = t.get("topic", "General")
            count = t.get("count", 0)
            correct = t.get("correct_count", 0)
            incorrect = t.get("incorrect_count", max(0, count - correct))
            perf_pct = t.get("performance_pct", t.get("accuracy_pct", 0.0))
            avg_m = t.get("avg_marks", round(perf_pct / 10.0, 1))
            max_m = t.get("max_marks", 10.0)

            # Sample size safeguard and transparent mastery evaluation
            if count < self.MIN_SAMPLE_FOR_CONFIRMATION:
                if perf_pct >= self.MASTERY_STRONG_THRESHOLD:
                    status = "Mastery Emerging (1 question - further verification recommended)"
                elif perf_pct >= self.MASTERY_MODERATE_THRESHOLD:
                    status = "Moderate (1 question)"
                else:
                    status = "Needs Revision (1 question)"
            else:
                if perf_pct >= self.MASTERY_STRONG_THRESHOLD:
                    status = "Mastered"
                elif perf_pct >= self.MASTERY_MODERATE_THRESHOLD:
                    status = "Moderate Mastery"
                else:
                    status = "Needs Revision"

            item = {
                "topic": topic_name,
                "count": count,
                "correct_count": correct,
                "incorrect_count": incorrect,
                "performance_pct": perf_pct,
                "accuracy_pct": t.get("accuracy_pct", perf_pct),
                "avg_marks": avg_m,
                "max_marks": max_m,
                "mastery_status": status
            }
            enriched.append(item)

            # Categorize lists (maintaining >= 70% for strong_topics to preserve backward compatibility)
            if perf_pct >= 70.0:
                strong_topics.append(topic_name)
            elif perf_pct >= 50.0:
                moderate_topics.append(topic_name)
            else:
                weak_topics.append(topic_name)

        return {
            "enriched_topics": enriched,
            "strong_topics": strong_topics,
            "moderate_topics": moderate_topics,
            "weak_topics": weak_topics
        }

    def _process_bloom_analytics(self, raw_bloom: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes cognitive Bloom's Taxonomy mastery and derives pedagogical diagnosis."""
        enriched = []
        lower_order_scores = []
        higher_order_scores = []

        for b in raw_bloom:
            lvl = b.get("bloom_level", "Understand")
            count = b.get("count", 0)
            correct = b.get("correct_count", 0)
            incorrect = b.get("incorrect_count", max(0, count - correct))
            perf_pct = b.get("performance_pct", b.get("accuracy_pct", 0.0))

            if lvl in ["Remember", "Understand"]:
                lower_order_scores.append(perf_pct)
            else:
                higher_order_scores.append(perf_pct)

            enriched.append({
                "bloom_level": lvl,
                "count": count,
                "correct_count": correct,
                "incorrect_count": incorrect,
                "accuracy_pct": b.get("accuracy_pct", perf_pct),
                "performance_pct": perf_pct
            })

        # Derive diagnostic summary
        avg_lower = sum(lower_order_scores) / max(1, len(lower_order_scores)) if lower_order_scores else 0.0
        avg_higher = sum(higher_order_scores) / max(1, len(higher_order_scores)) if higher_order_scores else 0.0

        if higher_order_scores and avg_higher < 60.0 and avg_lower >= 70.0:
            diagnosis = (
                f"Strong foundational recall (Lower-order Bloom: {avg_lower:.1f}%), but struggles with "
                f"higher-order cognitive synthesis, analysis, or evaluation ({avg_higher:.1f}%)."
            )
        elif lower_order_scores and avg_lower < 60.0:
            diagnosis = f"Foundational recall and comprehension need reinforcement (Lower-order Bloom: {avg_lower:.1f}%)."
        else:
            diagnosis = "Balanced cognitive performance across foundational recall and analytical evaluation."

        return {
            "enriched_bloom": enriched,
            "avg_lower_order": round(avg_lower, 1),
            "avg_higher_order": round(avg_higher, 1),
            "diagnosis": diagnosis
        }

    def _process_difficulty_analytics(self, raw_diff: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes performance across Easy, Medium, and Hard tiers."""
        diff_map = {d.get("difficulty"): d.get("performance_pct", d.get("accuracy_pct", 0.0)) for d in raw_diff}
        easy_pct = diff_map.get("Easy")
        med_pct = diff_map.get("Medium")
        hard_pct = diff_map.get("Hard")

        if hard_pct is not None and hard_pct < 50.0 and (easy_pct or 0.0) >= 70.0:
            diagnosis = "Demonstrates solid mastery of basic/intermediate concepts, but struggles with advanced multi-step items."
        elif easy_pct is not None and easy_pct < 60.0:
            diagnosis = "Struggles with foundational questions; review of core lecture definitions is recommended."
        elif hard_pct is not None and hard_pct >= 75.0:
            diagnosis = "Exceptional performance across all difficulty tiers, including high-complexity analytical questions."
        else:
            diagnosis = "Performance is relatively consistent across the attempted difficulty tiers."

        return {
            "enriched_diff": raw_diff,
            "diagnosis": diagnosis,
            "easy_pct": easy_pct,
            "medium_pct": med_pct,
            "hard_pct": hard_pct
        }

    def _process_question_type_analytics(self, raw_types: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyzes comparative performance across objective MCQs and free-text question formats."""
        type_map = {t.get("question_type"): t.get("performance_pct", t.get("accuracy_pct", 0.0)) for t in raw_types}
        mcq_pct = type_map.get("Multiple Choice (MCQ)")
        desc_pct = type_map.get("Descriptive", type_map.get("Short Answer"))

        if mcq_pct is not None and desc_pct is not None and (mcq_pct - desc_pct) >= 25.0:
            diagnosis = (
                f"Substantial discrepancy between recognition and generation: Strong on objective MCQs ({mcq_pct:.1f}%), "
                f"but significantly lower on free-text descriptive responses ({desc_pct:.1f}%)."
            )
        elif desc_pct is not None and desc_pct >= 75.0:
            diagnosis = "Strong capability to articulate concepts and generate comprehensive free-text answers."
        else:
            diagnosis = "Performance is well-balanced across multiple-choice and open-ended question formats."

        return {
            "enriched_types": raw_types,
            "diagnosis": diagnosis
        }

    def _process_progress_tracking(self, historical_attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Computes score progression, improvement delta, and trend status across chronological attempts."""
        total = len(historical_attempts)
        if total == 0:
            return {
                "total_attempts": 0,
                "score_delta": 0.0,
                "status": "No Attempts",
                "latest_score": 0.0,
                "previous_score": 0.0,
                "summary": "No historical attempts."
            }

        scores = [float(a.get("percentage", 0.0)) for a in historical_attempts]
        earliest_score = scores[0]
        latest_score = scores[-1]
        previous_score = scores[-2] if total > 1 else earliest_score
        
        # Improvement delta between latest and earliest attempt
        score_delta = round(latest_score - earliest_score, 1)
        recent_delta = round(latest_score - previous_score, 1)

        # Progression status evaluation
        if score_delta >= 8.0:
            status = "Significant Improvement"
            summary = f"Overall score has improved by +{score_delta}% since the initial assessment."
        elif score_delta >= 2.0:
            status = "Steady Progress"
            summary = f"Steady positive trajectory (+{score_delta}% compared to initial attempt)."
        elif score_delta <= -8.0:
            status = "Performance Decline"
            summary = f"Recent scores have declined ({score_delta}% lower than initial baseline). Systematic revision recommended."
        else:
            status = "Consistent Performance"
            summary = f"Performance remains consistent across attempts (Delta: {score_delta}%)."

        return {
            "total_attempts": total,
            "score_delta": score_delta,
            "recent_delta": recent_delta,
            "status": status,
            "earliest_score": earliest_score,
            "latest_score": latest_score,
            "previous_score": previous_score,
            "summary": summary
        }

    def _generate_personalized_recommendations(
        self,
        topic_analysis: Dict[str, Any],
        bloom_analysis: Dict[str, Any],
        difficulty_analysis: Dict[str, Any],
        question_type_analysis: Dict[str, Any],
        progress_tracking: Dict[str, Any]
    ) -> List[str]:
        """
        Synthesizes transparent, data-grounded study recommendations.
        Every recommendation includes an explicit empirical reason derived from student history.
        """
        recs = []
        weak_topics = topic_analysis.get("weak_topics", [])
        moderate_topics = topic_analysis.get("moderate_topics", [])
        strong_topics = topic_analysis.get("strong_topics", [])

        # Recommendation 1: Topic Revision based on weak topics
        if weak_topics:
            topics_str = ", ".join(weak_topics[:3])
            recs.append(
                f"Priority Topic Revision: Focus immediate study on {topics_str}. "
                f"Historical performance on these topics is under 50%. Review the lecture notes and definitions."
            )
        elif moderate_topics:
            topics_str = ", ".join(moderate_topics[:2])
            recs.append(
                f"Reinforce Moderate Topics: Dedicate practice to {topics_str} "
                f"to transition performance from moderate (50–74%) to mastery."
            )
        elif strong_topics:
            recs.append(
                "Topic Mastery Confirmed: Strong performance across all attempted topics. "
                "Challenge yourself with advanced descriptive and application problems."
            )

        # Recommendation 2: Cognitive Bloom level guidance
        for b in bloom_analysis.get("enriched_bloom", []):
            lvl = b.get("bloom_level")
            perf = b.get("performance_pct", b.get("accuracy_pct", 0.0))
            if lvl in ["Analyze", "Evaluate", "Create"] and perf < 60.0:
                recs.append(
                    f"Higher-Order Reasoning Practice: Accuracy on {lvl} questions is currently {perf}%. "
                    f"Practice questions requiring comparison, contrast, and trade-off evaluation."
                )
                break

        # Recommendation 3: Difficulty guidance
        hard_pct = difficulty_analysis.get("hard_pct")
        easy_pct = difficulty_analysis.get("easy_pct")
        if hard_pct is not None and hard_pct < 50.0:
            recs.append(
                f"Analytical Complexity Strategy: Performance on Hard questions is {hard_pct}%. "
                f"Break complex scenario questions into constituent steps before formulating answers."
            )
        elif easy_pct is not None and easy_pct >= 85.0 and (hard_pct is None or hard_pct >= 70.0):
            recs.append(
                f"Advance Assessment Level: Foundational score is high ({easy_pct}%). "
                f"Configure quizzes with 'Hard' difficulty and 'Application-Based' items to build mastery."
            )

        # Recommendation 4: Question type guidance (generation vs recognition)
        for qt in question_type_analysis.get("enriched_types", []):
            q_name = qt.get("question_type", "")
            perf = qt.get("performance_pct", qt.get("accuracy_pct", 0.0))
            if "Descriptive" in q_name and perf < 60.0:
                recs.append(
                    f"Improve Free-Text Articulation: Descriptive answer score is {perf}%. "
                    f"Focus on incorporating key technical terminology and thorough explanations."
                )
                break

        # Recommendation 5: Spaced practice reminder
        recs.append(
            "Spaced Practice: Retake short quizzes on previously attempted topics after 48–72 hours "
            "to maximize long-term educational retention."
        )

        return recs
