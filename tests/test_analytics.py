"""
NEUROLECT AI - Dedicated Phase 6 Learning Analytics Tests
Tests overall performance metrics, topic mastery status, sample size safeguards,
difficulty analyses, all 6 Bloom levels, question-type evaluations, historical trend tracking,
empty database handling, and data-backed personalized recommendations.
Strictly RAG-free.
"""

import unittest
from pathlib import Path
from database.db_manager import DatabaseManager
from modules.analytics import LearningAnalytics
import config

class TestLearningAnalytics(unittest.TestCase):
    def setUp(self):
        self.test_db_path = config.TEMP_DIR / "test_analytics_phase6.db"
        self.db = DatabaseManager(db_path=self.test_db_path)
        self.analytics = LearningAnalytics(db=self.db)

    def tearDown(self):
        if self.test_db_path.exists():
            self.test_db_path.unlink()

    # ------------------------------------------------------------------------
    # 1. Empty Database Handling
    # ------------------------------------------------------------------------
    def test_empty_database_handling(self):
        """Verifies graceful handling and clean zeroed metrics when no attempts exist."""
        metrics = self.analytics.generate_dashboard_metrics()
        
        self.assertEqual(metrics["summary"]["total_attempts"], 0)
        self.assertEqual(metrics["summary"]["total_questions"], 0)
        self.assertEqual(metrics["summary"]["average_score"], 0.0)
        self.assertEqual(len(metrics["topic_stats"]), 0)
        self.assertEqual(len(metrics["strong_topics"]), 0)
        self.assertEqual(len(metrics["weak_topics"]), 0)
        self.assertEqual(metrics["progress_tracking"]["total_attempts"], 0)
        self.assertTrue(len(metrics["recommendations"]) > 0)
        self.assertIn("No assessment attempts", metrics["recommendations"][0])

    # ------------------------------------------------------------------------
    # 2. Overall Score & Summary Metrics Calculations
    # ------------------------------------------------------------------------
    def test_overall_score_and_summary_calculations(self):
        """Verifies summary metrics, average, highest, and lowest score calculations."""
        self.db.save_lecture("lec_1", "Deep Learning Foundations", "text", 400)

        # Attempt 1: 60%
        self.db.save_quiz_attempt(
            attempt_id="atm_1",
            lecture_id="lec_1",
            total_questions=2,
            correct_answers=1,
            total_score=6.0,
            percentage=60.0,
            difficulty_selected="Easy",
            bloom_summary={},
            answers=[
                {"question": "Q1", "type": "MCQ", "score": 100.0, "is_correct": True, "awarded_marks": 1.0, "max_marks": 1.0, "topic": "CNN"},
                {"question": "Q2", "type": "Short Answer", "score": 20.0, "is_correct": False, "awarded_marks": 0.4, "max_marks": 2.0, "topic": "RNN"}
            ]
        )

        # Attempt 2: 90%
        self.db.save_quiz_attempt(
            attempt_id="atm_2",
            lecture_id="lec_1",
            total_questions=2,
            correct_answers=2,
            total_score=9.0,
            percentage=90.0,
            difficulty_selected="Medium",
            bloom_summary={},
            answers=[
                {"question": "Q3", "type": "MCQ", "score": 100.0, "is_correct": True, "awarded_marks": 1.0, "max_marks": 1.0, "topic": "CNN"},
                {"question": "Q4", "type": "Short Answer", "score": 80.0, "is_correct": True, "awarded_marks": 1.6, "max_marks": 2.0, "topic": "RNN"}
            ]
        )

        metrics = self.analytics.generate_dashboard_metrics()
        summary = metrics["summary"]

        self.assertEqual(summary["total_attempts"], 2)
        self.assertEqual(summary["total_questions"], 4)
        self.assertEqual(summary["total_correct"], 3)
        self.assertEqual(summary["average_score"], 75.0)
        self.assertEqual(summary["highest_score"], 90.0)
        self.assertEqual(summary["lowest_score"], 60.0)

    # ------------------------------------------------------------------------
    # 3. Topic-Wise Performance & Minimum Sample Size Safeguards
    # ------------------------------------------------------------------------
    def test_topic_wise_performance_and_sample_size_safeguards(self):
        """Verifies mastery status calculation and sample size confirmation safeguards."""
        self.db.save_lecture("lec_topics", "Optimization Methods", "text", 500)

        # CNN has only 1 question (100% score) -> sample size safeguard applies
        # Gradient Descent has 2 questions (100% score) -> confirmed Mastered
        # Regularization has 2 questions (30% score) -> Needs Revision
        answers = [
            {"question": "Q_cnn", "type": "MCQ", "score": 100.0, "is_correct": True, "awarded_marks": 1.0, "max_marks": 1.0, "topic": "CNN"},
            {"question": "Q_gd1", "type": "Short Answer", "score": 90.0, "is_correct": True, "awarded_marks": 1.8, "max_marks": 2.0, "topic": "Gradient Descent"},
            {"question": "Q_gd2", "type": "MCQ", "score": 100.0, "is_correct": True, "awarded_marks": 1.0, "max_marks": 1.0, "topic": "Gradient Descent"},
            {"question": "Q_reg1", "type": "Short Answer", "score": 30.0, "is_correct": False, "awarded_marks": 0.6, "max_marks": 2.0, "topic": "Regularization"},
            {"question": "Q_reg2", "type": "MCQ", "score": 0.0, "is_correct": False, "awarded_marks": 0.0, "max_marks": 1.0, "topic": "Regularization"}
        ]

        self.db.save_quiz_attempt(
            attempt_id="atm_topics",
            lecture_id="lec_topics",
            total_questions=5,
            correct_answers=3,
            total_score=4.4,
            percentage=64.0,
            difficulty_selected="Mixed",
            bloom_summary={},
            answers=answers
        )

        metrics = self.analytics.generate_dashboard_metrics()
        topic_map = {t["topic"]: t for t in metrics["topic_stats"]}

        # Gradient Descent: 2 attempts, >= 75% -> Mastered
        self.assertIn("Gradient Descent", topic_map)
        self.assertEqual(topic_map["Gradient Descent"]["mastery_status"], "Mastered")
        self.assertEqual(topic_map["Gradient Descent"]["count"], 2)

        # CNN: only 1 attempt -> sample size safeguard flags as emerging/preliminary
        self.assertIn("CNN", topic_map)
        self.assertIn("Emerging", topic_map["CNN"]["mastery_status"])
        self.assertEqual(topic_map["CNN"]["count"], 1)

        # Regularization: < 50% -> Needs Revision
        self.assertIn("Regularization", topic_map)
        self.assertIn("Needs Revision", topic_map["Regularization"]["mastery_status"])

        # Categorization lists
        self.assertIn("Gradient Descent", metrics["strong_topics"])
        self.assertIn("Regularization", metrics["weak_topics"])

    # ------------------------------------------------------------------------
    # 4. Difficulty-Wise Analysis
    # ------------------------------------------------------------------------
    def test_difficulty_wise_analysis(self):
        """Verifies performance categorization and diagnostic generation across difficulty tiers."""
        self.db.save_lecture("lec_diff", "AI Architectures", "text", 300)

        answers = [
            {"question": "Q1", "type": "MCQ", "score": 100.0, "is_correct": True, "difficulty": "Easy", "topic": "AI"},
            {"question": "Q2", "type": "MCQ", "score": 80.0, "is_correct": True, "difficulty": "Medium", "topic": "AI"},
            {"question": "Q3", "type": "Analytical", "score": 30.0, "is_correct": False, "difficulty": "Hard", "topic": "AI"}
        ]

        self.db.save_quiz_attempt("atm_diff", "lec_diff", 3, 2, 7.0, 70.0, "Mixed", {}, answers)

        metrics = self.analytics.generate_dashboard_metrics()
        diff_stats = {d["difficulty"]: d for d in metrics["difficulty_stats"]}

        self.assertIn("Easy", diff_stats)
        self.assertIn("Medium", diff_stats)
        self.assertIn("Hard", diff_stats)
        self.assertEqual(diff_stats["Easy"]["accuracy_pct"], 100.0)
        self.assertEqual(diff_stats["Hard"]["accuracy_pct"], 0.0)

        # Diagnosis should reflect difficulty divergence
        self.assertIn("struggles with advanced", metrics["difficulty_diagnosis"].lower())

    # ------------------------------------------------------------------------
    # 5. All Six Bloom's Taxonomy Levels Analytics
    # ------------------------------------------------------------------------
    def test_all_six_blooms_taxonomy_analytics(self):
        """Verifies tracking and cognitive diagnosis across all 6 Bloom levels."""
        self.db.save_lecture("lec_bloom", "Cognitive NLP", "text", 300)

        answers = [
            {"question": "Q1", "type": "MCQ", "score": 100.0, "is_correct": True, "bloom_level": "Remember", "topic": "NLP"},
            {"question": "Q2", "type": "Short Answer", "score": 85.0, "is_correct": True, "bloom_level": "Understand", "topic": "NLP"},
            {"question": "Q3", "type": "Application-Based", "score": 75.0, "is_correct": True, "bloom_level": "Apply", "topic": "NLP"},
            {"question": "Q4", "type": "Analytical", "score": 40.0, "is_correct": False, "bloom_level": "Analyze", "topic": "NLP"},
            {"question": "Q5", "type": "Descriptive", "score": 35.0, "is_correct": False, "bloom_level": "Evaluate", "topic": "NLP"},
            {"question": "Q6", "type": "Descriptive", "score": 50.0, "is_correct": False, "bloom_level": "Create", "topic": "NLP"}
        ]

        self.db.save_quiz_attempt("atm_bloom", "lec_bloom", 6, 3, 6.0, 64.2, "Mixed", {}, answers)

        metrics = self.analytics.generate_dashboard_metrics()
        bloom_levels_tracked = {b["bloom_level"] for b in metrics["bloom_stats"]}
        
        expected_levels = {"Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"}
        self.assertEqual(bloom_levels_tracked, expected_levels)
        self.assertIn("pedagogical_caveat", metrics)

    # ------------------------------------------------------------------------
    # 6. Question-Type Comparative Analytics
    # ------------------------------------------------------------------------
    def test_question_type_comparative_analytics(self):
        """Verifies performance comparison between recognition (MCQ) and synthesis (Descriptive)."""
        self.db.save_lecture("lec_qtypes", "Assessment Paradigms", "text", 300)

        answers = [
            {"question": "Q1", "type": "Multiple Choice (MCQ)", "score": 100.0, "is_correct": True, "topic": "AI"},
            {"question": "Q2", "type": "Multiple Choice (MCQ)", "score": 100.0, "is_correct": True, "topic": "AI"},
            {"question": "Q3", "type": "Descriptive", "score": 35.0, "is_correct": False, "topic": "AI"}
        ]

        self.db.save_quiz_attempt("atm_qt", "lec_qtypes", 3, 2, 7.8, 78.3, "Mixed", {}, answers)

        metrics = self.analytics.generate_dashboard_metrics()
        qtype_stats = {qt["question_type"]: qt for qt in metrics["question_type_stats"]}

        self.assertIn("Multiple Choice (MCQ)", qtype_stats)
        self.assertIn("Descriptive", qtype_stats)
        self.assertEqual(qtype_stats["Multiple Choice (MCQ)"]["performance_pct"], 100.0)
        self.assertEqual(qtype_stats["Descriptive"]["performance_pct"], 35.0)
        self.assertIn("recognition and generation", metrics["question_type_diagnosis"].lower())

    # ------------------------------------------------------------------------
    # 7. Historical Score Trends and Progress Tracking
    # ------------------------------------------------------------------------
    def test_historical_score_trends_and_progression(self):
        """Verifies score improvement delta and chronological trajectory calculations."""
        self.db.save_lecture("lec_trend", "Trend Study", "text", 250)

        # Attempt 1: 50.0%
        self.db.save_quiz_attempt("a1", "lec_trend", 2, 1, 5.0, 50.0, "Easy", {}, [])
        # Attempt 2: 70.0%
        self.db.save_quiz_attempt("a2", "lec_trend", 2, 1, 7.0, 70.0, "Medium", {}, [])
        # Attempt 3: 85.0%
        self.db.save_quiz_attempt("a3", "lec_trend", 2, 2, 8.5, 85.0, "Hard", {}, [])

        metrics = self.analytics.generate_dashboard_metrics()
        prog = metrics["progress_tracking"]

        self.assertEqual(prog["total_attempts"], 3)
        self.assertEqual(prog["earliest_score"], 50.0)
        self.assertEqual(prog["latest_score"], 85.0)
        self.assertEqual(prog["score_delta"], 35.0)
        self.assertEqual(prog["status"], "Significant Improvement")

    # ------------------------------------------------------------------------
    # 8. Personalized Learning Recommendations & Empirical Reasons
    # ------------------------------------------------------------------------
    def test_personalized_recommendations_and_reasons(self):
        """Verifies that recommendations cite empirical performance reasons and topic names."""
        self.db.save_lecture("lec_recs", "Personalized Diagnostics", "text", 400)

        answers = [
            {"question": "Q1", "type": "MCQ", "score": 25.0, "is_correct": False, "topic": "Attention Mechanisms"},
            {"question": "Q2", "type": "Analytical", "score": 30.0, "is_correct": False, "bloom_level": "Analyze", "difficulty": "Hard", "topic": "Attention Mechanisms"}
        ]

        self.db.save_quiz_attempt("atm_recs", "lec_recs", 2, 0, 2.7, 27.5, "Hard", {}, answers)

        metrics = self.analytics.generate_dashboard_metrics()
        recs = metrics["recommendations"]

        self.assertTrue(len(recs) >= 3)
        # Recommendation must reference the weak topic
        self.assertTrue(any("Attention Mechanisms" in r for r in recs))
        # Recommendation must cite empirical reason (e.g. under 50%)
        self.assertTrue(any("under 50%" in r or "Accuracy" in r for r in recs))

if __name__ == "__main__":
    unittest.main()
