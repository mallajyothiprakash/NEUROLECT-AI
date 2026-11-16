# NEUROLECT AI — Phase 5 Implementation Report
**System:** Automated Assessment Engine  
**Author:** Senior NLP Engineer, Python Backend Architect & Educational Assessment Developer  
**Date:** November 16, 2026  
**Target Hardware:** Apple M5 (ARM64, 10 CPU cores, 16 GB Unified Memory, macOS 27.2.1)  
**System Architecture:** Strictly RAG-free (No vector databases, no document retrieval, no LangChain/LlamaIndex)  

---

## 1. Executive Summary

Phase 5 of **NEUROLECT AI** implements a comprehensive, pedagogically sound, and mathematically robust **Automated Assessment Engine**. It replaces simplistic single-metric string matching with a multimodal evaluation pipeline that combines:
1. **Deterministic MCQ Scoring:** Exact match on option letters, full option texts, or normalized strings.
2. **Contextual Semantic Similarity:** PyTorch SentenceTransformer embeddings (`sentence-transformers/all-MiniLM-L6-v2`) via MPS on Apple M5 with graceful lexical overlap fallback.
3. **Technical Concept Coverage Analysis:** Set-based keyword extraction filtering function words to measure domain terminology retention.
4. **Answer Completeness & Length Expectation:** Adaptive word-count ratio evaluation preventing one-line answers from gaming semantic similarity.
5. **Polarity Inversion & Negation Detection:** Identifies contrastive negation tokens (e.g., *"is NOT an optimization algorithm"*) to prevent inverted answers from receiving falsely inflated semantic scores.
6. **Partial-Credit Marks Calculation:** Computes awarded marks against question-type-specific maximum marks (1.0 for MCQ, 2.0 for Short Answer, 5.0 for Conceptual/Application, 10.0 for Descriptive/Analytical).
7. **Human-Review Recommendation Flag:** Automatically flags borderline responses, negation anomalies, and high-value descriptive divergences for instructor review.
8. **Topic, Bloom, Type & Difficulty Quiz Filtering:** Filterable quiz configuration with SQLite attempt persistence and historical tracking.

The entire test suite now comprises **66 tests across 17 test files, passing 100% in 0.608 seconds**.

---

## 2. Existing Functionality Discovered

Before implementation, an inspection of the codebase revealed:
- `modules/answer_evaluator.py` had a basic prototype combining `0.70 * similarity + 0.30 * coverage_ratio`, but lacked question-type weighting, marks scaling, negation detection, and completeness bounds. All 4-letter words (including stopwords like "with" and "that") were treated as concepts.
- `modules/quiz_engine.py` supported filtering by question type, difficulty, and Bloom level, but **lacked topic filtering**. MCQ grading was purely case-sensitive without parsing option prefixes like `"A: Option text"`.
- `database/db_manager.py` stored quiz attempts and basic answer rows, but omitted awarded marks, max marks, evaluation method, semantic similarity, concept coverage, and human-review flags.
- `pages/quiz_page.py` lacked topic selection in the setup screen and did not display evaluation methodologies, semantic metrics, or instructor review alerts.

---

## 3. Files Created and Modified

### Created:
- **`tests/test_assessment.py`**: Dedicated test suite with 10 unit test cases verifying deterministic MCQ grading, exact matching, paraphrasing, partial credit, off-topic rejection, concept coverage, polarity inversion, missing reference handling, multi-criteria filtering, and database persistence.
- **`PHASE5_IMPLEMENTATION_REPORT.md`**: This comprehensive implementation and verification report.

### Modified:
- **`modules/answer_evaluator.py`**:
  - Implemented `extract_technical_concepts()` using stopword filtering.
  - Implemented `detect_polarity_inversion()` for negation contrast detection.
  - Integrated question-type-specific `max_marks` assignment (`DEFAULT_MAX_MARKS`).
  - Added completeness ratio evaluation.
  - Implemented multi-factor composite scoring:
    $$\text{Raw Score} = 0.60 \times \text{Similarity} + 0.30 \times \text{Concept Coverage} + 0.10 \times \text{Completeness}$$
  - Applied polarity discount factor ($0.60\times$) when student response contradicts the reference answer.
  - Added `human_review_recommended` flag with explanatory review reasons.
  - Added disclaimers distinguishing automated NLP estimates from verified academic marks.
- **`modules/quiz_engine.py`**:
  - Extended `filter_questions()` to support `selected_topics` alongside question types, difficulties, and Bloom levels.
  - Enhanced MCQ evaluation to handle raw letters, prefixed option strings (`"A: ..."`), and full option text matches.
  - Added marks accumulation (`total_awarded_marks`, `total_possible_marks`, `overall_percentage`).
  - Enriched answer payloads with evaluation method, concept breakdown, marks, and review flags.
- **`database/db_manager.py`**:
  - Upgraded `quiz_answers` schema with backward-compatible migration via `PRAGMA table_info` and `ALTER TABLE`.
  - Added columns: `question_id`, `awarded_marks`, `max_marks`, `evaluation_method`, `semantic_similarity`, `concept_coverage`, `human_review_recommended`.
  - Updated `save_quiz_attempt()` to store all itemized metrics within single ACID transactions.
- **`pages/quiz_page.py`**:
  - Added topic multi-selection filter derived from the current question pool.
  - Enhanced quiz report view with marks breakdowns, evaluation method badges, metric columns (Semantic Similarity, Concept Coverage, Completeness), covered/missing concept pills, and instructor review alerts.
  - Added an expandable past assessment history widget on the setup screen.

---

## 4. Assessment Algorithms & Scoring Methodology

### 4.1 Multiple Choice Question (MCQ) Evaluation
- **Method:** `Deterministic Exact Option Match`
- **Scoring:** Binary (100% / 0%).
- **Marks:** 1.0 / 1.0 (or 0.0 / 1.0).
- **Matching Rule:** Matches student selection against:
  1. Correct letter (`A`, `B`, `C`, `D`).
  2. Prefixed option string (`"A: Option text"`, `"A) Option text"`).
  3. Option dictionary value (`options[correct_letter]`).
  4. Normalized correct answer string.

### 4.2 Free-Text Answer Evaluation (Short Answer, Conceptual, Application, Descriptive, Analytical)
- **Method:** `SentenceTransformer Semantic Embedding (all-MiniLM-L6-v2) + Concept Analysis`
- **Max Marks Map:**
  - Short Answer: **2.0 marks**
  - Conceptual: **5.0 marks**
  - Application-Based: **5.0 marks**
  - Descriptive: **10.0 marks**
  - Analytical: **10.0 marks**
- **Evaluation Factors:**
  1. *Contextual Cosine Similarity ($S_{\text{sim}}$):* Cosine similarity of 384-dimensional dense vectors from `all-MiniLM-L6-v2`.
  2. *Concept Coverage ($C_{\text{cov}}$):* Ratio of reference technical terms covered in the student answer:
     $$C_{\text{cov}} = \frac{|\text{Concepts}_{\text{ref}} \cap \text{Concepts}_{\text{student}}|}{\max(1, |\text{Concepts}_{\text{ref}}|)}$$
  3. *Answer Completeness ($L_{\text{comp}}$):* Ratio of student word count to minimum expected length ($60\%$ of reference length):
     $$L_{\text{comp}} = \min\left(1.0, \frac{\text{WordCount}_{\text{student}}}{\max(5, \lfloor 0.6 \times \text{WordCount}_{\text{ref}}\rfloor)}\right)$$
  4. *Negation Inversion Penalty:* If student response introduces negation words (`not`, `never`, `cannot`, `fails`, etc.) absent in the reference answer, $S_{\text{raw}}$ is multiplied by $0.60$ and flagged for instructor review.
  5. *Composite Score Calculation:*
     $$\text{Score \%} = \min\left(100.0, \left(0.60 S_{\text{sim}} + 0.30 C_{\text{cov}} + 0.10 L_{\text{comp}}\right) \times 100\right)$$
     $$\text{Awarded Marks} = \text{round}\left(\frac{\text{Score \%}}{100} \times \text{Max Marks}, 1\right)$$

### 4.3 Human Review Recommendation Logic
The flag `human_review_recommended = True` is triggered when:
- **Borderline Score:** Score percentage is between 40.0% and 60.0% (partial credit awarded).
- **Metric Divergence:** $|S_{\text{sim}} - C_{\text{cov}}| > 0.45$ on descriptive or analytical items (indicates keyword stuffing or generic fluff).
- **Polarity Inversion:** Negation contradiction detected.
- **Missing Reference Answer:** Reference answer unavailable in question metadata.
- **Low Descriptive Score:** High-weight item (10 marks) scoring below 50.0%.

---

## 5. Database Schema & Migration Details

The SQLite database (`data/neurolect.db`) schema was upgraded cleanly without deleting existing tables or data.

### Upgraded Table: `quiz_answers`
```sql
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
```

### Automatic Migration Routine:
`DatabaseManager._init_db()` queries `PRAGMA table_info(quiz_answers)` and dynamically executes `ALTER TABLE quiz_answers ADD COLUMN ...` for any missing columns if connecting to an older database file.

---

## 6. Testing & Verification

### 6.1 Dedicated Assessment Test Suite (`tests/test_assessment.py`)
| Test Case | Objective | Result |
|---|---|---|
| `test_mcq_evaluation_correct_and_incorrect` | Exact letter, lowercase letter, option prefix, incorrect selection | **PASS** |
| `test_empty_and_trivial_answers` | Empty string `""` and single-word submission grading | **PASS** |
| `test_exact_and_semantically_equivalent_answers` | Exact match and high-similarity paraphrase evaluation | **PASS** |
| `test_partially_correct_answers` | Partial credit scoring and borderline human review flag | **PASS** |
| `test_incorrect_and_off_topic_answers` | Off-topic answer rejection and low mark assignment | **PASS** |
| `test_concept_coverage_analysis` | Extraction of covered and missing technical terminology | **PASS** |
| `test_negation_inversion_detection` | Polarity contradiction detection and score discounting | **PASS** |
| `test_missing_reference_answer` | Graceful zero-score handling when reference answer is empty | **PASS** |
| `test_quiz_question_filtering` | Multi-criteria filtering by topic, type, difficulty, Bloom level | **PASS** |
| `test_database_persistence_and_attempt_details` | Full attempt and itemized answer persistence and retrieval | **PASS** |

### 6.2 Full Test Suite Results
```bash
source .venv/bin/activate && python -m unittest discover -s tests -p "test_*.py"
```
- **Total Tests:** 66
- **Failures:** 0
- **Errors:** 0
- **Pass Rate:** **100%**
- **Execution Time:** **0.608 seconds**

---

## 7. Model Inference Status vs Mocked Tests

- **Mocked Unit Tests:** Fast, deterministic mocked vectors were used in unit tests to verify mathematical grading logic, cosine thresholds, concept extraction, and database persistence in milliseconds without requiring active network calls to Hugging Face.
- **Transparent Lexical Fallback:** When SentenceTransformers is offline or unavailable, `SemanticAnswerEvaluator` automatically falls back to set-based lexical Jaccard overlap, ensuring the application remains functional even in strictly disconnected environments.
- **Pretrained Sentence Transformer (`sentence-transformers/all-MiniLM-L6-v2`):** Loaded lazily through `model_loader.get_sentence_transformer()` and cached as a shared singleton for KeyBERT, Topic Detection, and Answer Evaluation.

---

## 8. Apple M5 Compatibility

1. **MPS Device Mapping:** `model_loader` routes `all-MiniLM-L6-v2` to Apple Silicon MPS (`torch.backends.mps.is_available()`).
2. **Unified Memory Footprint:** `all-MiniLM-L6-v2` requires ~90 MB RAM, representing less than 0.6% of the 16 GB Unified Memory.
3. **Batch Vector Encoding:** Both student and reference strings are encoded simultaneously in a single call (`st_model.encode([student, reference], convert_to_numpy=True)`), minimizing GPU dispatch overhead.
4. **Safe Unloading:** Can be unloaded via `model_loader.unload_model("sentence_similarity")` if needed.

---

## 9. Known Limitations

1. **Short Answer Lexical Sensitivity:** For very short answers (< 4 words), semantic embeddings can be sensitive to phrasing; concept coverage provides a stabilizing safeguard.
2. **Offline Sentence Transformer Download:** If `all-MiniLM-L6-v2` is not cached locally, the engine uses transparent lexical fallback until weights are pre-downloaded.

---

## 10. Conclusion & Readiness

Phase 5 Automated Assessment Engine is fully implemented, verified, and operational.
