"""
Unit Tests for Model Loader & Memory Management
Tests lazy loading, singleton caching, model unloading, memory cleanup,
and verified shared cache usage between KeyBERT and Topic Detection.
"""

import unittest
from unittest.mock import MagicMock, patch
from models.model_loader import ModelLoader
from modules.keyword_extractor import KeywordExtractor
from modules.topic_detector import TopicDetector

class TestModelLoaderMemory(unittest.TestCase):
    def setUp(self):
        self.loader = ModelLoader()

    def test_singleton_pattern(self):
        loader2 = ModelLoader()
        self.assertIs(self.loader, loader2)

    def test_model_unloading_and_garbage_collection(self):
        # Inject mock model into cache
        self.loader._models["mock_test_model"] = MagicMock()
        self.assertTrue(self.loader.is_model_loaded("mock_test_model"))

        # Unload model
        unloaded = self.loader.unload_model("mock_test_model")
        self.assertTrue(unloaded)
        self.assertFalse(self.loader.is_model_loaded("mock_test_model"))

        # Unloading non-existent model returns False safely
        self.assertFalse(self.loader.unload_model("non_existent_key"))

    def test_unload_transcription_models(self):
        self.loader._models["whisper"] = MagicMock()
        self.assertTrue(self.loader.is_model_loaded("whisper"))

        self.loader.unload_transcription_models()
        self.assertFalse(self.loader.is_model_loaded("whisper"))

    def test_shared_model_between_keyword_and_topic_detector(self):
        """
        Verifies that KeywordExtractor and TopicDetector share the same
        SentenceTransformer model instance from model_loader cache.
        """
        kw_extractor = KeywordExtractor()
        topic_detector = TopicDetector()

        mock_st = MagicMock()
        self.loader._models["sentence_similarity"] = mock_st

        st1 = kw_extractor.loader.get_sentence_transformer()
        st2 = topic_detector.loader.get_sentence_transformer()

        self.assertIs(st1, st2, "SentenceTransformer instance is not shared between KeyBERT and TopicDetector!")
        self.assertIs(st1, mock_st)

    def test_loaded_models_info(self):
        self.loader._models["sentence_similarity"] = MagicMock()
        info = self.loader.get_loaded_models_info()
        self.assertIn("device", info)
        self.assertIn("loaded_models", info)
        self.assertIn("sentence_similarity", info["loaded_models"])

if __name__ == "__main__":
    unittest.main()
