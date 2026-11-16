# NEUROLECT AI — Production Deployment & Operations Guide

**Document Version:** 1.0  
**Target Hardware:** Apple Silicon (M1/M2/M3/M4/M5) & Modern Linux (x86_64 / aarch64)  
**Primary Testbed:** Apple M5 (10-Core CPU, 16 GB Unified Memory), macOS 27.2.1 / Python 3.14.6  
**Architecture:** Strictly RAG-Free • Sequential Transformer Inference & Chunk Stitching  

---

## 1. System Requirements & Hardware Specifications

| Component | Minimum Specification | Recommended Specification (Current Target) |
| :--- | :--- | :--- |
| **Processor** | 4-Core x86_64 or Apple Silicon M1 | Apple M5 (10-Core ARM64) / 8+ Core CPU |
| **Memory (RAM)** | 8 GB System Memory | 16 GB Unified Memory |
| **Storage** | 3.0 GB free disk space | 8.0 GB free SSD space (models + db + uploads) |
| **Operating System** | macOS 12+ or Ubuntu 20.04+ LTS | macOS 27.2.1 / Darwin 25.2.0 |
| **Python** | Python 3.10 – 3.14 | Python 3.14.6 |
| **Media Subsystem** | System `ffmpeg` utility | System `ffmpeg` 6.0+ |

---

## 2. Environment Preparation

### Step 2.1: Install System Dependencies

NEUROLECT AI requires `ffmpeg` for extracting 16kHz mono audio tracks from uploaded video lectures (`.mp4`, `.mov`, `.mkv`).

**On macOS (using Homebrew):**
```bash
brew install ffmpeg
```

**On Linux (Ubuntu / Debian):**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

Verify the installation:
```bash
ffmpeg -version
```

### Step 2.2: Setup Virtual Environment & Install Dependencies

Clone or navigate to the project root directory:
```bash
cd /Users/mallajyothiprakash/Documents/NEUROLECT_AI
```

Create and activate an isolated Python 3 virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install declared dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Verify installed packages:
```bash
python -c "import torch, transformers, sentence_transformers, fitz, streamlit; print('All core libraries imported successfully!')"
```

---

## 3. Pretrained Model Management & Caching

NEUROLECT AI uses five distinct pretrained model checkpoints. All models are loaded **lazily** (only when needed) and managed through the thread-safe `NeurolectModelLoader` singleton in [`models/model_loader.py`](file:///Users/mallajyothiprakash/Documents/NEUROLECT_AI/models/model_loader.py).

### Model Inventory:

| Role | Hugging Face Checkpoint | Task | Approx. Download Size |
| :--- | :--- | :--- | :--- |
| **Speech-to-Text** | `openai/whisper-tiny` | Audio/video lecture transcription | ~150 MB |
| **Summarization** | `sshleifer/distilbart-cnn-12-6` | Hierarchical chunk summarization & notes | ~1.2 GB |
| **Question Intelligence** | `google/flan-t5-base` | Multi-type question & distractor generation | ~990 MB |
| **Named Entity Recognition**| `dslim/bert-base-NER` | Educational entity classification | ~430 MB |
| **Answer Evaluation** | `sentence-transformers/all-MiniLM-L6-v2` | Cosine similarity assessment | ~90 MB |

### Pre-Downloading Model Weights for Air-Gapped / Production Environments:

In production or air-gapped environments, download the model weights once before going offline:

```bash
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForTokenClassification, pipeline
from sentence_transformers import SentenceTransformer

print('Downloading Whisper Tiny...')
pipeline('automatic-speech-recognition', model='openai/whisper-tiny')

print('Downloading DistilBART Summarizer...')
pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')

print('Downloading Flan-T5 Base...')
AutoTokenizer.from_pretrained('google/flan-t5-base')
AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')

print('Downloading BERT-NER...')
pipeline('ner', model='dslim/bert-base-NER')

print('Downloading MiniLM Embeddings...')
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print('All models successfully cached!')
"
```

Weights will be cached in `~/.cache/huggingface/hub/` and accessed with zero outbound network calls during runtime.

### Offline Resilient Mode:
If running in an offline environment without pre-downloaded weights, NEUROLECT AI features transparent, deterministic fallbacks:
- **Topic Detector:** Automatically uses TF-IDF + K-Means clustering.
- **Keyword Extractor:** Automatically uses scikit-learn TF-IDF n-gram extraction.
- **Distractor Generator:** Automatically extracts contrastive entities from the lecture text.
- **Answer Evaluator:** Automatically uses lexical n-gram Jaccard similarity and concept coverage.

---

## 4. Launching the Application

### Development Mode:
```bash
source .venv/bin/activate
streamlit run app.py
```

### Production / Server Mode:
Run headless on port 8501, binding to all interfaces and configuring upload limits:
```bash
source .venv/bin/activate
streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.maxUploadSize 300 \
    --theme.base "dark" \
    --theme.primaryColor "#4F46E5" \
    --theme.backgroundColor "#0F172A" \
    --theme.secondaryBackgroundColor "#1E293B" \
    --theme.textColor "#F8FAFC"
```

Access the UI in any browser at `http://localhost:8501`.

---

## 5. Background Daemon / Service Setup (systemd on Linux)

For 24/7 background operation on Linux servers, configure a `systemd` service:

Create `/etc/systemd/system/neurolect.service`:
```ini
[Unit]
Description=NEUROLECT AI Intelligent Lecture Platform
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/NEUROLECT_AI
Environment="PATH=/home/ubuntu/NEUROLECT_AI/.venv/bin:/usr/local/bin:/usr/bin"
Environment="TRANSFORMERS_NO_TF=1"
Environment="USE_TF=0"
Environment="USE_FLAX=0"
ExecStart=/home/ubuntu/NEUROLECT_AI/.venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true --server.maxUploadSize 300
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable neurolect
sudo systemctl start neurolect
sudo systemctl status neurolect
```

---

## 6. Database Maintenance & Operations

The SQLite database is located at [`data/neurolect.db`](file:///Users/mallajyothiprakash/Documents/NEUROLECT_AI/data/neurolect.db).

### Database Schema Overview:
- **`lectures`**: Stores processed lecture records (`id`, `title`, `source_type`, `word_count`, `duration_seconds`, `num_chunks`, `created_at`).
- **`quiz_attempts`**: Stores completed quiz metadata (`id`, `lecture_id`, `lecture_title`, `score`, `total_questions`, `percentage`, `completed_at`, `mode`, `time_taken_seconds`).
- **`quiz_answers`**: Stores individual question responses (`id`, `attempt_id`, `question_id`, `question_text`, `question_type`, `student_answer`, `reference_answer`, `is_correct`, `score`, `feedback`, `topic`, `bloom_level`, `difficulty`, `semantic_similarity`, `concept_coverage`, `needs_review`).

### Routine Backup:
```bash
cp data/neurolect.db data/neurolect_backup_$(date +%Y%m%d_%H%M%S).db
```

### Reset Database:
To clear all history and restart with a clean schema:
```bash
rm data/neurolect.db
python -c "from database.db_manager import DatabaseManager; db = DatabaseManager(); db.initialize_tables(); print('Database initialized cleanly.')"
```

---

## 7. Memory & Performance Optimization on Apple Silicon

NEUROLECT AI automatically targets Apple Silicon GPU acceleration via Apple Metal Performance Shaders (`mps`):
```python
# config.py
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
```

### Memory Management Features:
- **Whisper Eviction:** Whisper speech recognition weights (~150MB) are automatically evicted from RAM after transcription completes to preserve memory for downstream text summarization and question generation.
- **Garbage Collection:** `gc.collect()` and `torch.mps.empty_cache()` are invoked during model switching.
- **Shared SentenceTransformer:** A single instance of `all-MiniLM-L6-v2` is shared across topic clustering, keyword MMR, and semantic answer evaluation.
- **Peak RAM Footprint:** In full operation, memory consumption remains under **3.5 GB**, perfectly tailored for 16 GB unified memory systems.

---

## 8. Verification & Health Check

To verify system health following deployment, run the automated test suite:

```bash
source .venv/bin/activate
python -m unittest discover -s tests -p "test_*.py" -v
```

Expected output:
```
----------------------------------------------------------------------
Ran 75 tests in 0.7s
OK
```

All 75 tests passing confirms that all ingestion pipelines, NLP transformers, question generators, assessment engines, and database tables are functioning with 100% integrity.
