# 🧠 NEUROLECT AI — Intelligent Lecture Understanding & Assessment Engine

[![Python 3.10 - 3.14](https://img.shields.io/badge/Python-3.10%20--%203.14-blue.svg)](https://www.python.org/)
[![UI Framework](https://img.shields.io/badge/Frontend-Streamlit%201.66-red.svg)](https://streamlit.io/)
[![Deep Learning](https://img.shields.io/badge/Backend-PyTorch%20%7C%20Transformers-orange.svg)](https://huggingface.co/)
[![Architecture](https://img.shields.io/badge/Architecture-Strictly%20RAG--Free-green.svg)](#architecture-overview)
[![Test Suite](https://img.shields.io/badge/Tests-91%20%2F%2091%20Passing-brightgreen.svg)](#testing--verification)

> **NEUROLECT AI** is a genuine transformer-based Natural Language Processing college-level application designed to ingest multi-modal educational lectures (Audio recordings, Video recordings, and PDF/Text documents) and transform them into structured study notes, hierarchical summaries, multi-type assessment questions, interactive quizzes, and personalized learning analytics.

📖 **For the complete, in-depth technical reference, architectural diagrams, model specifications, and module documentation, see [NEUROLECT_AI_DOCUMENTATION.md](NEUROLECT_AI_DOCUMENTATION.md).**

---

## Key Features

- **Multimodal Lecture Ingestion:**
  - **Audio (`.mp3`, `.wav`, `.m4a`):** OpenAI Whisper transcription with segment timestamps.
  - **Video (`.mp4`, `.mov`, `.mkv`):** Automatic FFmpeg 16kHz audio extraction and transcription with temp file cleanup.
  - **PDF & Text (`.pdf`, `.txt`):** PyMuPDF page-preserving extraction and pasted notes.
- **Hierarchical Transformer Summarization:** Multi-tier summaries (Concise, Detailed, Section-wise, Notes, Topic-wise) using `DistilBART`.
- **Topic Discovery & KeyBERT Keywords:** Unsupervised semantic sentence clustering and Maximal Marginal Relevance (MMR) terminology extraction.
- **BERT Named Entity Recognition:** Categorizes People, Organizations, Locations, and Technical concepts.
- **Grounded Concept Mining:** Linguistic pattern mining extracting definitions and core principles tagged to exact page numbers or timestamps.
- **Multi-Type Question Generation:** Instruction-tuned `FLAN-T5` two-pass synthesis across 6 pedagogical types:
  - *Multiple Choice (MCQ)*, *Short Answer*, *Descriptive*, *Conceptual*, *Application-Based*, *Analytical*.
- **Pedagogical Taxonomy Mapping:** Automatic classification into the 6 Revised Bloom's Taxonomy levels and 3 difficulty tiers (Easy, Medium, Hard).
- **Tri-Tier Distractor Synthesis:** Rigorously generates 4 distinct MCQ options (1 correct, 3 plausible distractors) with shuffling and letter tracking.
- **Automated Free-Text Evaluation:** Semantic answer grading using `all-MiniLM-L6-v2` cosine similarity ($60\%$), technical concept coverage ($30\%$), and answer completeness ($10\%$).
- **Polarity Inversion Detection:** Flags contradictory negation tokens and applies a $40\%$ discount with human review alerts.
- **Diagnostic Learning Analytics:** Longitudinal score tracking, topic mastery matrix ($\ge 2$ attempt safeguards), Bloom tier diagnostics, and data-backed study plans.

---

## Architecture Overview

NEUROLECT AI operates on a **strictly RAG-free architecture**:
- **Zero Vector Databases:** No FAISS, ChromaDB, Pinecone, or Weaviate.
- **Zero Document Retrieval:** No similarity search across fragmented embeddings.
- **Zero External API Egress:** No calls to proprietary cloud APIs (OpenAI GPT, Anthropic Claude).
- **Sequential Context Budgeting:** Direct sequential chunking with 50-word overlaps preserving temporal narrative context.
- **Dedicated Task Specialization:** Pretrained models are loaded lazily and cached in-memory.

```
[Audio / Video / PDF / Text] ──► [UnifiedContentEngine] ──► [Standardized LectureContent]
                                                                    │
      ┌─────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┐
      ▼                                                             ▼                                                             ▼
[Lecture Understanding]                                   [Question Intelligence]                                       [Automated Assessment]
• DistilBART Summaries                                    • FLAN-T5 Two-Pass Synthesis                                  • Deterministic MCQ Grading
• Topic K-Means Discovery                                 • Tri-Tier Distractor Engine                                  • MiniLM Semantic Cosine Grading
• KeyBERT MMR Extraction                                  • Bloom's & Difficulty Classifiers                           • Concept Coverage Analysis
• BERT Named Entities                                     • Jaccard Deduplication Audit                                 • Negation Inversion Detection
• Grounded Concept Mining                                 • Exact Requested Count Capping                               • SQLite Persistence & Analytics
```

---

## Technology Stack Summary

| Layer | Primary Technologies | Role in NEUROLECT AI |
| :--- | :--- | :--- |
| **Frontend** | Streamlit 1.66.1, HTML5/CSS3 | Single-page reactive UI, dark theme, audio/video players, quiz stepper |
| **Backend** | Python 3.14.6 (3.10–3.14 compatible) | Modular processing pipelines, configuration engine, session state |
| **Deep Learning** | PyTorch 2.15.0, Transformers 4.57.6, Accelerate | Native Apple Silicon MPS / CUDA / CPU tensor computation |
| **ASR** | `openai/whisper-tiny` | Speech-to-text audio and video lecture transcription |
| **Summarization** | `sshleifer/distilbart-cnn-12-6` | Hierarchical chunk-based lecture summarization |
| **Question Gen** | `google/flan-t5-base` | Two-pass seq2seq question and distractor generation |
| **NER** | `dslim/bert-base-NER` | Educational entity classification (PER, ORG, LOC, MISC) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Semantic answer evaluation and topic clustering |
| **Classical ML** | `scikit-learn` 1.9.1, `scipy`, `numpy` | TF-IDF, K-Means clustering, MMR diversification, cosine distance |
| **Documents** | `pymupdf` (fitz) 1.28.2 | High-performance PDF extraction and page tracking |
| **Media Utility**| System FFmpeg 6.0+ | Demuxing video containers and 16kHz mono audio extraction |
| **Database** | SQLite 3.43+ | ACID persistence for lectures, quiz attempts, and itemized answers |

---

## Quick Start & Installation

### 1. Prerequisites
- **Python:** 3.10 to 3.14 recommended (Tested on Python 3.14.6 ARM64).
- **FFmpeg:** Required for video processing (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux, `winget install Gyan.FFmpeg` on Windows).

### 2. Setup Virtual Environment

#### On macOS / Linux:
```bash
cd /path/to/NEUROLECT_AI
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### On Windows (PowerShell):
```powershell
cd D:\projects\NEUROLECT_AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Running the Application

Launch the Streamlit web application:

```bash
source .venv/bin/activate
streamlit run app.py
```

Access the application in your browser at:
```
http://localhost:8501
```

---

## Testing & Verification

Run the comprehensive automated unit and integration test suite (91 tests across 20 test files):

```bash
source .venv/bin/activate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m unittest discover -s tests -p "test_*.py" -v
```

**Verified Test Output:**
```
----------------------------------------------------------------------
Ran 91 tests in 15.427s

OK (100% Pass Rate)
```

---

## Complete Documentation Reference

For comprehensive technical explanations, implementation details, algorithm formulas, and database schemas:

👉 **Read the [Complete Master Technical Documentation (NEUROLECT_AI_DOCUMENTATION.md)](NEUROLECT_AI_DOCUMENTATION.md)**

---

## License & Academic Attribution
Developed as an advanced academic and research platform for Intelligent Lecture Understanding and Automated Educational Assessment. Released under the MIT License.
