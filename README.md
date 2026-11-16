# NEUROLECT AI — Intelligent Lecture Understanding, Question Generation & Automated Assessment System

[![Python Version](https://img.shields.io/badge/Python-3.10%20--%203.14-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Frontend-Streamlit-red.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/DL-PyTorch%20%7C%20Transformers-orange.svg)](https://huggingface.co/)
[![Architecture](https://img.shields.io/badge/Architecture-Strictly%20RAG--Free-green.svg)](#strict-rag-free-architecture)

**NEUROLECT AI** is a genuine transformer-based Natural Language Processing college-level application designed to ingest multi-modal educational lectures (Audio recordings, Video recordings, and PDF/Text documents) and transform them into structured study notes, hierarchical summaries, multi-type assessment questions, interactive quizzes, and personalized learning analytics.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Strict RAG-Free Architecture](#strict-rag-free-architecture)
3. [Three Equal-Priority Input Modalities](#three-equal-priority-input-modalities)
4. [NLP Intelligence Engine](#nlp-intelligence-engine)
5. [Question Intelligence & Assessment](#question-intelligence--assessment)
6. [Learning Analytics Dashboard](#learning-analytics-dashboard)
7. [Pretrained Model Specifications](#pretrained-model-specifications)
8. [Project Structure](#project-structure)
9. [Installation & Setup](#installation--setup)
10. [Running the Application](#running-the-application)
11. [Running Unit & Integration Tests](#running-unit--integration-tests)

---

## Key Features

- **Multi-Modal Lecture Ingestion:**
  - **Audio:** MP3, WAV, M4A with OpenAI Whisper transcription and segment timestamp tracking.
  - **Video:** MP4, MOV, MKV with FFmpeg 16kHz mono audio extraction, Whisper transcription, and automatic temporary file cleanup.
  - **PDF & Text:** PyMuPDF page-preserving extraction, TXT file reading, and direct text paste.
- **Common Content Representation:** All inputs feed into a unified `ProcessedLectureContent` schema.
- **Hierarchical Transformer Summarization:** DistilBART / BART-large-CNN for concise summaries, detailed summaries, section-wise breakdowns, and structured lecture notes.
- **Topic Discovery & KeyBERT Terminology:** Semantic sentence clustering and Maximal Marginal Relevance (MMR) deduplication.
- **BERT Named Entity Recognition:** Categorizes People (PER), Organizations (ORG), Locations (LOC), and Technical Entities (MISC).
- **Grounded Concept Mining:** Extracts definitions and core principles anchored to exact page numbers or audio timestamps.
- **Multi-Type Question Generation:** Instruction-tuned Flan-T5 generates MCQs, Short-Answer, Descriptive, Conceptual, Application-based, and Analytical questions.
- **Pedagogical Classifications:**
  - Bloom's Revised Taxonomy (Remember, Understand, Apply, Analyze, Evaluate, Create).
  - Difficulty Levels (Easy, Medium, Hard).
- **Automated Distractor Synthesis:** Generates exactly 4 non-duplicate, plausible MCQ choices (1 correct, 3 distractors).
- **Question Quality Auditing:** Validates groundedness, eliminates chunk duplicates, and filters malformed outputs.
- **Interactive Quiz Stepper:** Timed/self-paced quiz interface with instant scoring.
- **SentenceTransformer Semantic Evaluation:** Algorithmic grading of free-text student answers against reference solutions using contextual cosine similarity and concept coverage.
- **SQLite Analytics Engine:** Tracks quiz attempts, topic-wise accuracy, Bloom distribution, and generates automated revision recommendations.

---

## Strict RAG-Free Architecture

NEUROLECT AI is **completely RAG-free**. It does not use:
- Vector databases (No FAISS, ChromaDB, Pinecone, or Weaviate).
- Embedding-based document retrieval.
- Similarity-based document search.
- LangChain or LlamaIndex RAG workflows.

Instead, the system utilizes direct text extraction, sentence segmentation, token/word chunking to respect transformer context windows, and direct sequential inference through pretrained models (`Whisper`, `BART`, `Flan-T5`, `BERT-NER`, `all-MiniLM-L6-v2`).

Sentence Transformers are utilized solely for **semantic answer evaluation** and **topic grouping**, never for document retrieval.

---

## Three Equal-Priority Input Modalities

Every input format is a first-class citizen and connects directly to the same common intelligence engine:

```
[Audio: MP3 / WAV / M4A] ────► [Whisper ASR] ───────────────┐
                                                            │
[Video: MP4 / MOV / MKV] ────► [FFmpeg] ──► [Whisper ASR] ──┼──► [Unified Content Engine] ──► [NLP Intelligence Engine]
                                                            │
[PDF / TXT / Pasted Text] ───► [PyMuPDF / Normalizer] ──────┘
```

---

## Pretrained Model Specifications

| Task | Checkpoint | Device / Execution | Role |
| :--- | :--- | :--- | :--- |
| **Speech Recognition** | `openai/whisper-tiny` | CPU / MPS / CUDA | Converts audio lectures and video sound tracks into timestamped transcripts. |
| **Summarization** | `sshleifer/distilbart-cnn-12-6` | CPU / MPS / CUDA | Hierarchical chunk-based lecture summarization. |
| **Question Generation** | `google/flan-t5-base` | CPU / MPS / CUDA | Instruction-tuned multi-type question and distractor generation. |
| **Named Entity Recognition**| `dslim/bert-base-NER` | CPU / MPS / CUDA | Extracts People, Organizations, Locations, and Technical concepts. |
| **Semantic Answer Evaluation** | `sentence-transformers/all-MiniLM-L6-v2` | CPU / MPS / CUDA | Cosine similarity grading of student written answers against reference solutions. |

*Models are loaded lazily upon request and cached in-memory.*

---

## Project Structure

```
NEUROLECT_AI/
│
├── app.py                      # Main Streamlit application controller
├── config.py                   # Central system configuration, paths & constants
├── requirements.txt            # Dependency specifications
├── README.md                   # Complete architectural and operational documentation
│
├── modules/                    # Core NLP & processing engines
│   ├── content_engine.py       # Unified Content Processing Engine (dataclasses)
│   ├── audio_processor.py      # Audio upload & Whisper transcription
│   ├── video_processor.py      # Video upload, FFmpeg extraction & cleanup
│   ├── pdf_processor.py        # PyMuPDF page-preserving extraction
│   ├── text_processor.py       # TXT & direct text normalization
│   ├── summarizer.py           # Hierarchical transformer summarization
│   ├── topic_detector.py       # Semantic sentence clustering & outlines
│   ├── keyword_extractor.py    # KeyBERT-style MMR keyword extraction
│   ├── ner_processor.py        # BERT Named Entity Recognition
│   ├── concept_extractor.py    # Grounded definition & principle mining
│   ├── question_generator.py   # Multi-type Flan-T5 question generation
│   ├── distractor_generator.py # MCQ distractor synthesizer (4 options)
│   ├── bloom_classifier.py     # Bloom's Revised Taxonomy classification
│   ├── difficulty_classifier.py# Easy, Medium, Hard classification
│   ├── question_validator.py   # Structural & groundedness quality auditor
│   ├── answer_evaluator.py     # SentenceTransformer semantic answer grader
│   ├── quiz_engine.py          # Interactive quiz controller & scoring
│   └── analytics.py            # Diagnostic learning analytics & recommendations
│
├── models/
│   └── model_loader.py         # Lazy singleton model loader & cache
│
├── database/
│   └── db_manager.py           # SQLite manager (lectures, attempts, answers)
│
├── utils/
│   ├── validators.py           # Format, size, and content validation
│   ├── file_manager.py         # Safe file storage & temp cleanup
│   └── helpers.py              # Text cleaning, chunking, timestamps
│
├── pages/                      # Streamlit UI page controllers
│   ├── home_page.py            # Hero dashboard & modality portal
│   ├── audio_page.py           # Audio processing studio
│   ├── video_page.py           # Video processing studio
│   ├── document_page.py        # PDF & Text document studio
│   ├── summary_page.py         # Multi-tier summary viewer
│   ├── analysis_page.py        # Topic outline & keyword viewer
│   ├── ner_concept_page.py     # NER & concept extraction viewer
│   ├── question_page.py        # Question generation & export studio
│   ├── quiz_page.py            # Interactive assessment interface
│   ├── analytics_page.py       # Learning analytics dashboard
│   └── history_page.py         # Quiz attempt history & detailed review
│
├── assets/
│   └── style.css               # Modern educational dark theme
│
├── tests/                      # Automated unit and integration test suite (87 tests)
│   ├── test_analytics.py       # Diagnostic analytics, mastery & recommendation tests
│   ├── test_assessment.py      # MCQ, short/descriptive, negation & partial credit tests
│   ├── test_assessment_analytics.py # Evaluation & SQLite persistence baseline tests
│   ├── test_audio.py           # Audio validation & Whisper timestamp tests
│   ├── test_audio_video.py     # Audio/video validator & cleanup tests
│   ├── test_concept_extractor.py # Definition & principle extraction pattern tests
│   ├── test_end_to_end.py      # Pipeline integration baseline test
│   ├── test_end_to_end_integration.py # Phase 7 full 8-step lifecycle cross-phase integration
│   ├── test_helpers.py         # Paragraph splitting, text cleaning & chunking tests
│   ├── test_keyword_extractor.py # KeyBERT & TF-IDF keyword extraction tests
│   ├── test_model_loader.py    # Singleton, memory unloading & cache tests
│   ├── test_ner_processor.py   # BERT-NER categorization & source preservation tests
│   ├── test_nlp_modules.py     # Bloom, difficulty, concept & validation tests
│   ├── test_pdf.py             # PyMuPDF extraction & page tracking tests
│   ├── test_questions.py       # Flan-T5 multi-type question & distractor tests
│   ├── test_summarizer.py      # Multi-chunk completeness (9 chunks) & summary tests
│   ├── test_text.py            # Text normalization & encoding fallback tests
│   ├── test_topic_detector.py  # Topic clustering & TF-IDF fallback tests
│   ├── test_video.py           # Video validation, FFmpeg extraction & cleanup tests
│   └── test_video_question_integration.py # Video transcript to FLAN-T5 question integration
│
└── data/
    ├── neurolect.db            # SQLite database file
    ├── uploads/                # File uploads directory
    └── temp/                   # Temporary processing directory
```

---

## Installation & Setup

### 1. Prerequisites
- **Python:** 3.10 to 3.14 recommended.
- **FFmpeg:** Required for video audio track extraction.
  - **macOS:** `brew install ffmpeg`
  - **Windows (PowerShell):** `winget install Gyan.FFmpeg`
  - **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install -y ffmpeg`

### 2. Setup Virtual Environment

#### On Windows (PowerShell):
```powershell
# Navigate to project root
cd D:\projects\NEUROLECT_AI

# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

#### On macOS / Linux (Terminal / zsh):
```bash
# Navigate to project root
cd /path/to/NEUROLECT_AI

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Running Unit & Integration Tests

Execute the automated test suite covering all modules across Phases 1 through 7:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected result:
```
----------------------------------------------------------------------
Ran 87 tests in 11.429s
OK
```

All 87 tests verify 100% functionality of multimodal ingestion, hierarchical summarization, topic clustering, KeyBERT extraction, BERT-NER, concept mining, Flan-T5 multi-type question generation, Bloom's and difficulty classification, distractor synthesis, semantic answer evaluation, SQLite persistence, learning analytics diagnostics, and video-to-FLAN-T5 live question generation.
