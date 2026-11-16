"""
NEUROLECT AI - SQLite Database Manager
Stores quiz sessions, student attempts, detailed answer evaluations, and analytics history.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import config

class DatabaseManager:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DATABASE_PATH
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=config.DB_TIMEOUT_SECONDS)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables for lectures, quiz attempts, and itemized responses with schema migration."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS lectures (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    word_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS quiz_attempts (
                    id TEXT PRIMARY KEY,
                    lecture_id TEXT,
                    total_questions INTEGER NOT NULL,
                    correct_answers INTEGER NOT NULL,
                    total_score REAL NOT NULL,
                    percentage REAL NOT NULL,
                    difficulty_selected TEXT,
                    bloom_summary_json TEXT,
                    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (lecture_id) REFERENCES lectures(id) ON DELETE SET NULL
                );
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS quiz_answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id TEXT NOT NULL,
                    question_id TEXT,
                    question_text TEXT NOT NULL,
                    question_type TEXT NOT NULL,
                    student_answer TEXT,
                    correct_answer TEXT,
                    is_correct INTEGER,
                    score REAL DEFAULT 0.0,
                    awarded_marks REAL DEFAULT 0.0,
                    max_marks REAL DEFAULT 1.0,
                    evaluation_method TEXT,
                    semantic_similarity REAL DEFAULT 0.0,
                    concept_coverage REAL DEFAULT 0.0,
                    human_review_recommended INTEGER DEFAULT 0,
                    feedback TEXT,
                    difficulty TEXT,
                    bloom_level TEXT,
                    topic TEXT,
                    FOREIGN KEY (attempt_id) REFERENCES quiz_attempts(id) ON DELETE CASCADE
                );
                """)

                # Backward-compatible column migration for existing databases
                cursor.execute("PRAGMA table_info(quiz_answers)")
                existing_cols = {row["name"] for row in cursor.fetchall()}
                new_columns = [
                    ("question_id", "TEXT"),
                    ("awarded_marks", "REAL DEFAULT 0.0"),
                    ("max_marks", "REAL DEFAULT 1.0"),
                    ("evaluation_method", "TEXT"),
                    ("semantic_similarity", "REAL DEFAULT 0.0"),
                    ("concept_coverage", "REAL DEFAULT 0.0"),
                    ("human_review_recommended", "INTEGER DEFAULT 0")
                ]
                for col_name, col_type in new_columns:
                    if col_name not in existing_cols:
                        cursor.execute(f"ALTER TABLE quiz_answers ADD COLUMN {col_name} {col_type}")
        finally:
            conn.close()

    def save_lecture(self, lecture_id: str, title: str, source_type: str, word_count: int) -> bool:
        """Saves or updates lecture metadata."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO lectures (id, title, source_type, word_count, created_at)
                VALUES (?, ?, ?, ?, ?)
                """, (lecture_id, title, source_type, word_count, datetime.now().isoformat()))
                return True
        except Exception as e:
            print(f"[DB Error] Failed to save lecture: {e}")
            return False
        finally:
            conn.close()

    def save_quiz_attempt(
        self,
        attempt_id: str,
        lecture_id: Optional[str],
        total_questions: int,
        correct_answers: int,
        total_score: float,
        percentage: float,
        difficulty_selected: str,
        bloom_summary: Dict[str, Any],
        answers: List[Dict[str, Any]]
    ) -> bool:
        """Saves a completed quiz attempt with all individual question responses."""
        conn = self._get_connection()
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO quiz_attempts (
                    id, lecture_id, total_questions, correct_answers,
                    total_score, percentage, difficulty_selected,
                    bloom_summary_json, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    attempt_id,
                    lecture_id,
                    total_questions,
                    correct_answers,
                    total_score,
                    percentage,
                    difficulty_selected,
                    json.dumps(bloom_summary),
                    datetime.now().isoformat()
                ))

                for ans in answers:
                    cursor.execute("""
                    INSERT INTO quiz_answers (
                        attempt_id, question_id, question_text, question_type,
                        student_answer, correct_answer, is_correct,
                        score, awarded_marks, max_marks, evaluation_method,
                        semantic_similarity, concept_coverage, human_review_recommended,
                        feedback, difficulty, bloom_level, topic
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        attempt_id,
                        ans.get("question_id", ans.get("id", "")),
                        ans.get("question", ""),
                        ans.get("type", "MCQ"),
                        ans.get("student_answer", ""),
                        ans.get("correct_answer", ""),
                        1 if ans.get("is_correct") else 0,
                        ans.get("score", 0.0),
                        ans.get("awarded_marks", ans.get("score", 0.0)),
                        ans.get("max_marks", 10.0),
                        ans.get("evaluation_method", "Standard Evaluation"),
                        ans.get("semantic_similarity", 0.0),
                        ans.get("concept_coverage", 0.0),
                        1 if ans.get("human_review_recommended") else 0,
                        ans.get("feedback", ""),
                        ans.get("difficulty", "Medium"),
                        ans.get("bloom_level", "Understand"),
                        ans.get("topic", "General")
                    ))
                return True
        except Exception as e:
            print(f"[DB Error] Failed to save quiz attempt: {e}")
            return False
        finally:
            conn.close()

    def get_quiz_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves past quiz attempts with lecture title and scores."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT q.id, q.lecture_id, l.title as lecture_title, l.source_type,
                   q.total_questions, q.correct_answers, q.total_score,
                   q.percentage, q.difficulty_selected, q.completed_at
            FROM quiz_attempts q
            LEFT JOIN lectures l ON q.lecture_id = l.id
            ORDER BY q.completed_at DESC
            LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_attempt_details(self, attempt_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full details and answer breakdowns for a given attempt."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM quiz_attempts WHERE id = ?", (attempt_id,))
            attempt = cursor.fetchone()
            if not attempt:
                return None
            
            attempt_dict = dict(attempt)
            cursor.execute("SELECT * FROM quiz_answers WHERE attempt_id = ?", (attempt_id,))
            attempt_dict["answers"] = [dict(row) for row in cursor.fetchall()]
            return attempt_dict
        finally:
            conn.close()

    def get_overall_analytics(self) -> Dict[str, Any]:
        """Calculates comprehensive learning analytics across all assessment attempts."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # 1. High-Level Summary Metrics
            cursor.execute("""
            SELECT 
                COUNT(*) as total_attempts,
                COALESCE(SUM(total_questions), 0) as total_questions,
                COALESCE(SUM(correct_answers), 0) as total_correct,
                ROUND(COALESCE(AVG(percentage), 0.0), 1) as average_score,
                ROUND(COALESCE(MAX(percentage), 0.0), 1) as highest_score,
                ROUND(COALESCE(MIN(percentage), 0.0), 1) as lowest_score
            FROM quiz_attempts
            """)
            summary = dict(cursor.fetchone() or {})

            # 2. Topic-wise Performance
            cursor.execute("""
            SELECT 
                topic,
                COUNT(*) as count,
                SUM(is_correct) as correct_count,
                (COUNT(*) - SUM(is_correct)) as incorrect_count,
                ROUND(AVG(is_correct) * 100, 1) as accuracy_pct,
                ROUND(AVG(COALESCE(awarded_marks, score)), 2) as avg_marks,
                ROUND(AVG(COALESCE(max_marks, 10.0)), 2) as max_marks,
                ROUND(AVG(score), 1) as performance_pct
            FROM quiz_answers
            WHERE topic IS NOT NULL AND topic != ''
            GROUP BY topic
            ORDER BY performance_pct DESC, count DESC
            """)
            topic_stats = [dict(row) for row in cursor.fetchall()]

            # 3. Cognitive Bloom's Taxonomy Performance
            cursor.execute("""
            SELECT 
                bloom_level,
                COUNT(*) as count,
                SUM(is_correct) as correct_count,
                (COUNT(*) - SUM(is_correct)) as incorrect_count,
                ROUND(AVG(is_correct) * 100, 1) as accuracy_pct,
                ROUND(AVG(score), 1) as performance_pct
            FROM quiz_answers
            WHERE bloom_level IS NOT NULL AND bloom_level != ''
            GROUP BY bloom_level
            ORDER BY count DESC
            """)
            bloom_stats = [dict(row) for row in cursor.fetchall()]

            # 4. Difficulty-wise Performance
            cursor.execute("""
            SELECT 
                difficulty,
                COUNT(*) as count,
                SUM(is_correct) as correct_count,
                (COUNT(*) - SUM(is_correct)) as incorrect_count,
                ROUND(AVG(is_correct) * 100, 1) as accuracy_pct,
                ROUND(AVG(score), 1) as performance_pct
            FROM quiz_answers
            WHERE difficulty IS NOT NULL AND difficulty != ''
            GROUP BY difficulty
            ORDER BY count DESC
            """)
            difficulty_stats = [dict(row) for row in cursor.fetchall()]

            # 5. Pedagogical Question-Type Performance
            cursor.execute("""
            SELECT 
                question_type,
                COUNT(*) as count,
                SUM(is_correct) as correct_count,
                (COUNT(*) - SUM(is_correct)) as incorrect_count,
                ROUND(AVG(is_correct) * 100, 1) as accuracy_pct,
                ROUND(AVG(score), 1) as performance_pct,
                ROUND(AVG(COALESCE(awarded_marks, score)), 2) as avg_marks,
                ROUND(AVG(COALESCE(max_marks, 10.0)), 2) as max_marks
            FROM quiz_answers
            WHERE question_type IS NOT NULL AND question_type != ''
            GROUP BY question_type
            ORDER BY count DESC
            """)
            question_type_stats = [dict(row) for row in cursor.fetchall()]

            # 6. Chronological Assessment Trend
            cursor.execute("""
            SELECT 
                q.id as attempt_id,
                q.completed_at,
                ROUND(q.percentage, 1) as percentage,
                q.total_score,
                q.total_questions,
                q.correct_answers,
                q.difficulty_selected,
                COALESCE(l.title, 'General Lecture') as lecture_title
            FROM quiz_attempts q
            LEFT JOIN lectures l ON q.lecture_id = l.id
            ORDER BY q.completed_at ASC
            """)
            historical_trend = [dict(row) for row in cursor.fetchall()]

            return {
                "summary": summary,
                "topic_stats": topic_stats,
                "bloom_stats": bloom_stats,
                "difficulty_stats": difficulty_stats,
                "question_type_stats": question_type_stats,
                "historical_trend": historical_trend
            }
        finally:
            conn.close()
