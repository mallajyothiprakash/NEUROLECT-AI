"""
Unit Tests for Text Processor Module (Phase 2)
Tests raw educational text processing, TXT file reading with multi-encoding fallback,
input length validation, chunking, and LectureContent standardization.
"""

import unittest
from pathlib import Path
from modules.text_processor import TextProcessor
from modules.content_models import LectureContent
import config

SAMPLE_TEXT = """
Machine learning algorithms build a mathematical model based on sample training data.
Supervised learning uses labeled inputs and targets to train predictive mappings.
Unsupervised learning discovers hidden patterns or clusters in unlabeled data.
Reinforcement learning agents interact with dynamic environments to maximize cumulative rewards.
Deep learning uses artificial neural networks with multiple representation layers.
"""

class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()
        self.test_dir = config.TEMP_DIR / "test_text_artifacts"
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # 1. UTF-8 TXT file
        self.utf8_txt = self.test_dir / "lecture_utf8.txt"
        with open(self.utf8_txt, "w", encoding="utf-8") as f:
            f.write(SAMPLE_TEXT)

        # 2. Latin-1 TXT file
        self.latin1_txt = self.test_dir / "lecture_latin1.txt"
        with open(self.latin1_txt, "w", encoding="latin-1") as f:
            f.write("Machine learning étude: un résumé en français avec des caractères accentués comme é, à, ç, et ü.")

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_process_raw_text(self):
        content = self.processor.process_raw_text(
            text=SAMPLE_TEXT,
            lecture_title="ML Paradigms Lecture"
        )

        self.assertIsInstance(content, LectureContent)
        self.assertEqual(content.source_type, "text")
        self.assertEqual(content.source_name, "ML Paradigms Lecture")
        self.assertGreater(content.word_count, 30)
        self.assertGreater(len(content.sentences), 3)
        self.assertGreaterEqual(len(content.chunks), 1)
        self.assertIn("Reinforcement learning", content.cleaned_text)

    def test_process_txt_file_utf8(self):
        content = self.processor.process_txt_file(self.utf8_txt)
        self.assertIsInstance(content, LectureContent)
        self.assertEqual(content.source_type, "text")
        self.assertEqual(content.metadata["encoding"], "utf-8")
        self.assertIn("Supervised learning", content.cleaned_text)

    def test_process_txt_file_latin1_fallback(self):
        content = self.processor.process_txt_file(self.latin1_txt)
        self.assertIsInstance(content, LectureContent)
        self.assertEqual(content.source_type, "text")
        self.assertIn("Machine learning", content.cleaned_text)

    def test_text_validation_rejection(self):
        # Empty string
        with self.assertRaises(ValueError):
            self.processor.process_raw_text("")

        # Whitespace only
        with self.assertRaises(ValueError):
            self.processor.process_raw_text("   \n\t  ")

        # Too brief text (fewer than 15 words)
        with self.assertRaises(ValueError):
            self.processor.process_raw_text("Too short to analyze.")

if __name__ == "__main__":
    unittest.main()
