"""
NEUROLECT AI - Pretrained Model Loader & Lazy Singleton Cache
Manages lazy-loading, caching, memory management (unloading/garbage collection),
device mapping (CPU/MPS/CUDA), and error handling for Whisper, Summarizer,
Flan-T5, BERT-NER, and SentenceTransformers.
"""

import os
import gc
import logging
from typing import Optional, Any, Dict, List
import config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeurolectModelLoader")

class ModelLoader:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._models = {}
            cls._instance.device = config.DEFAULT_DEVICE
        return cls._instance

    def _get_device_str(self) -> str:
        return self.device

    def is_model_loaded(self, model_key: str) -> bool:
        """Returns True if the specified model is currently resident in memory."""
        return model_key in self._models

    def unload_model(self, model_key: str) -> bool:
        """
        Safely unloads a model from memory and triggers garbage collection.
        Frees CPU RAM / GPU VRAM.
        """
        if model_key in self._models:
            logger.info(f"Unloading model '{model_key}' from memory...")
            del self._models[model_key]
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass
            return True
        return False

    def unload_transcription_models(self):
        """
        Unloads speech-to-text models (Whisper) after transcription completes,
        freeing ~150-500MB RAM for downstream NLP analysis without affecting text models.
        """
        self.unload_model("whisper")

    def get_loaded_models_info(self) -> Dict[str, Any]:
        """Returns summary of all currently cached models and their device assignment."""
        model_sizes_approx = {
            "whisper": "approx. 150 MB (whisper-tiny)",
            "summarizer": "approx. 300 MB - 1.2 GB (DistilBART/T5)",
            "question_generator": "approx. 300 MB - 990 MB (Flan-T5)",
            "ner": "approx. 430 MB (bert-base-NER)",
            "sentence_similarity": "approx. 90 MB (all-MiniLM-L6-v2)"
        }
        loaded = {}
        for key in self._models.keys():
            loaded[key] = {
                "resident": True,
                "device": self.device,
                "memory_footprint": model_sizes_approx.get(key, "Unknown")
            }
        return {
            "device": self.device,
            "total_loaded_models": len(self._models),
            "loaded_models": loaded
        }

    def get_whisper_pipeline(self) -> Any:
        """Lazily loads and caches the Whisper speech-to-text pipeline."""
        model_key = "whisper"
        if model_key in self._models:
            return self._models[model_key]

        model_cfg = config.MODELS_CONFIG["whisper"]
        model_id = model_cfg["model_id"]
        logger.info(f"Loading Whisper pipeline: {model_id} on device {self.device}...")

        try:
            from transformers import pipeline
            device = "cpu" if self.device == "mps" else self.device
            device_id = 0 if device == "cuda" else -1
            
            asr_pipeline = pipeline(
                task="automatic-speech-recognition",
                model=model_id,
                chunk_length_s=30,
                return_timestamps=True,
                device=device_id
            )
            self._models[model_key] = asr_pipeline
            return asr_pipeline
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_id}': {e}")
            raise RuntimeError(
                f"Speech Recognition Model Unavailable: Could not load '{model_id}'. "
                f"Error: {e}. Please ensure network access or valid cached weights."
            )

    def get_summarizer_pipeline(self) -> Any:
        """Lazily loads and caches the transformer summarization pipeline."""
        model_key = "summarizer"
        if model_key in self._models:
            return self._models[model_key]

        model_cfg = config.MODELS_CONFIG["summarizer"]
        model_id = model_cfg["model_id"]
        logger.info(f"Loading Summarizer pipeline: {model_id}...")

        try:
            from transformers import pipeline
            device_id = 0 if self.device == "cuda" else -1
            summarizer = pipeline(
                task=model_cfg.get("task", "summarization"),
                model=model_id,
                device=device_id
            )
            self._models[model_key] = summarizer
            return summarizer
        except Exception as e:
            fallback_id = model_cfg.get("fallback_id")
            if fallback_id and fallback_id != model_id:
                try:
                    logger.warning(f"Attempting fallback summarization model '{fallback_id}'...")
                    from transformers import pipeline
                    device_id = 0 if self.device == "cuda" else -1
                    summarizer = pipeline(task="summarization", model=fallback_id, device=device_id)
                    self._models[model_key] = summarizer
                    return summarizer
                except Exception as fb_err:
                    logger.error(f"Fallback summarizer failed: {fb_err}")
            raise RuntimeError(f"Summarizer Model Unavailable: {e}")

    def get_question_generator(self) -> tuple:
        """Lazily loads Flan-T5 model and tokenizer for question and distractor generation."""
        model_key = "question_generator"
        if model_key in self._models:
            return self._models[model_key]

        model_cfg = config.MODELS_CONFIG["question_generator"]
        model_id = model_cfg["model_id"]
        logger.info(f"Loading Question Generator: {model_id}...")

        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Prefer local cached weights to avoid hanging on network timeouts
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id, local_files_only=True)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
            
            if self.device == "cuda":
                model = model.to("cuda")
            elif self.device == "mps":
                model = model.to("mps")
            else:
                model = model.to("cpu")
                
            model.eval()
            self._models[model_key] = (tokenizer, model)
            return (tokenizer, model)
        except Exception as e:
            fallback_id = model_cfg.get("fallback_id")
            if fallback_id and fallback_id != model_id:
                try:
                    logger.warning(f"Attempting fallback QG model '{fallback_id}'...")
                    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(fallback_id, local_files_only=True)
                        model = AutoModelForSeq2SeqLM.from_pretrained(fallback_id, local_files_only=True)
                    except Exception:
                        tokenizer = AutoTokenizer.from_pretrained(fallback_id)
                        model = AutoModelForSeq2SeqLM.from_pretrained(fallback_id)
                    model.eval()
                    self._models[model_key] = (tokenizer, model)
                    return (tokenizer, model)
                except Exception as fb_err:
                    logger.error(f"Fallback QG failed: {fb_err}")
            raise RuntimeError(f"Question Generator Model Unavailable: {e}")

    def get_ner_pipeline(self) -> Any:
        """Lazily loads and caches the BERT Named Entity Recognition pipeline."""
        model_key = "ner"
        if model_key in self._models:
            return self._models[model_key]

        model_cfg = config.MODELS_CONFIG["ner"]
        model_id = model_cfg["model_id"]
        logger.info(f"Loading NER pipeline: {model_id}...")

        try:
            from transformers import pipeline
            device_id = 0 if self.device == "cuda" else -1
            ner_pipe = pipeline(
                task="ner",
                model=model_id,
                aggregation_strategy=model_cfg.get("aggregation_strategy", "simple"),
                device=device_id
            )
            self._models[model_key] = ner_pipe
            return ner_pipe
        except Exception as e:
            logger.error(f"Failed to load NER pipeline: {e}")
            raise RuntimeError(f"NER Model Unavailable: {e}")

    def get_sentence_transformer(self) -> Any:
        """
        Lazily loads and caches the SentenceTransformer model.
        Crucial: Shared between KeyBERT, Topic Detection, and Semantic Evaluation!
        """
        model_key = "sentence_similarity"
        if model_key in self._models:
            return self._models[model_key]

        model_cfg = config.MODELS_CONFIG["sentence_similarity"]
        model_id = model_cfg["model_id"]
        logger.info(f"Loading SentenceTransformer: {model_id}...")

        try:
            from sentence_transformers import SentenceTransformer
            st_model = SentenceTransformer(model_id, device=self.device)
            self._models[model_key] = st_model
            return st_model
        except Exception as e:
            logger.error(f"Failed to load SentenceTransformer: {e}")
            raise RuntimeError(f"SentenceTransformer Model Unavailable: {e}")

# Global singleton
model_loader = ModelLoader()
