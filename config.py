"""
NEUROLECT AI - Configuration Module
Intelligent Lecture Understanding, Question Generation & Automated Assessment System
"""

import os
from pathlib import Path

# Enforce clean PyTorch backend and prevent TensorFlow/Keras conflicts
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_PATH = DATA_DIR / "neurolect.db"
TEMP_DIR = DATA_DIR / "temp"
UPLOADS_DIR = DATA_DIR / "uploads"
ASSETS_DIR = PROJECT_ROOT / "assets"
MODELS_CACHE_DIR = DATA_DIR / "models_cache"

# Ensure runtime directories exist
for directory in [DATA_DIR, TEMP_DIR, UPLOADS_DIR, ASSETS_DIR, MODELS_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hardware and Compute Configuration
# Automatically select device: MPS for Apple Silicon, CUDA for Nvidia, CPU fallback
import torch

if torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
else:
    DEFAULT_DEVICE = "cpu"

# Supported File Formats
SUPPORTED_AUDIO_EXTENSIONS = [".mp3", ".wav", ".m4a"]
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".mkv"]
SUPPORTED_DOCUMENT_EXTENSIONS = [".pdf", ".txt"]

# File Size Limits (in Bytes)
MAX_AUDIO_SIZE_BYTES = 150 * 1024 * 1024       # 150 MB
MAX_VIDEO_SIZE_BYTES = 300 * 1024 * 1024       # 300 MB
MAX_DOCUMENT_SIZE_BYTES = 50 * 1024 * 1024     # 50 MB

# Pretrained Hugging Face Models Configuration (Strictly CPU-friendly checkpoints)
MODELS_CONFIG = {
    "whisper": {
        "model_id": "openai/whisper-tiny",
        "description": "Lightweight OpenAI Whisper model for fast CPU speech transcription",
        "task": "automatic-speech-recognition"
    },
    "summarizer": {
        "model_id": "sshleifer/distilbart-cnn-12-6",
        "fallback_id": "facebook/bart-large-cnn",
        "description": "DistilBART model fine-tuned on CNN/DailyMail for lecture summarization",
        "task": "summarization",
        "max_input_tokens": 1024,
        "max_summary_tokens": 200,
        "min_summary_tokens": 40
    },
    "question_generator": {
        "model_id": "google/flan-t5-base",
        "fallback_id": "google/flan-t5-small",
        "description": "Instruction-tuned FLAN-T5 for multi-type educational question and distractor generation",
        "task": "text2text-generation",
        "max_input_tokens": 512,
        "max_output_tokens": 256
    },
    "ner": {
        "model_id": "dslim/bert-base-NER",
        "description": "BERT fine-tuned for Named Entity Recognition (PER, ORG, LOC, MISC)",
        "task": "ner",
        "aggregation_strategy": "simple"
    },
    "sentence_similarity": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "High-efficiency Sentence Transformer for semantic similarity & answer evaluation",
        "dimension": 384
    }
}

# Educational & Pedagogical Taxonomies
BLOOMS_TAXONOMY_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

DIFFICULTY_LEVELS = [
    "Easy",
    "Medium",
    "Hard"
]

QUESTION_TYPES = [
    "Multiple Choice (MCQ)",
    "Short Answer",
    "Descriptive",
    "Conceptual",
    "Application-Based",
    "Analytical"
]

# Chunking & NLP Limits
CHUNK_SIZE_WORDS = 350
CHUNK_OVERLAP_WORDS = 50
MAX_KEYWORDS_EXTRACTED = 15
MAX_ENTITIES_DISPLAYED = 30
MAX_CONCEPTS_EXTRACTED = 12

# Database Settings
DB_TIMEOUT_SECONDS = 15.0
