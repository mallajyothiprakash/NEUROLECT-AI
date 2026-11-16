"""
Unit Tests for Helper Functions & Boundary Processing
Tests clean_text, sentence segmentation with oversized paragraphs,
word chunking without trailing text loss, and timestamp formatting.
"""

import unittest
from utils.helpers import clean_text, split_into_sentences, chunk_text, format_timestamp, calculate_reading_time

class TestHelpers(unittest.TestCase):
    def test_clean_text(self):
        dirty = "  Hello   world!  \r\n\r\nThis has\xa0non-breaking spaces   and\ttabs.   \n\n\n\nExtra lines."
        cleaned = clean_text(dirty)
        self.assertNotIn("\xa0", cleaned)
        self.assertNotIn("\t", cleaned)
        self.assertNotIn("\n\n\n", cleaned)
        self.assertIn("Hello world!", cleaned)

    def test_oversized_unpunctuated_paragraph_splitting(self):
        """Tests that a long paragraph without periods is decomposed safely at clause boundaries."""
        long_paragraph = (
            "This is a very long run-on educational paragraph discussing supervised machine learning, "
            "where labeled input pairs are processed sequentially by optimization engines, "
            "and gradient calculations update the weight matrices iteratively, "
            "which in turn reduces the error metric across all training batches, "
            "and finally yields a trained classifier capable of generalizing to unseen validation samples."
        )
        sentences = split_into_sentences(long_paragraph, max_sentence_words=15)
        self.assertGreater(len(sentences), 1)
        for s in sentences:
            self.assertLessEqual(len(s.split()), 20)

    def test_chunk_text_preserves_final_portion(self):
        """Verifies that no text is silently discarded at the end of chunking."""
        words_sample = [f"word_{i}" for i in range(1, 351)] # 350 words
        full_text = ". ".join([" ".join(words_sample[i:i+25]) for i in range(0, len(words_sample), 25)]) + ". FINAL_UNIQUE_TOKEN."

        chunks = chunk_text(full_text, chunk_size_words=100, overlap_words=20)
        self.assertGreaterEqual(len(chunks), 3)

        # Verify the final chunk contains the final unique token
        last_chunk_text = chunks[-1]["text"]
        self.assertIn("FINAL_UNIQUE_TOKEN", last_chunk_text)

    def test_format_timestamp(self):
        self.assertEqual(format_timestamp(0), "00:00")
        self.assertEqual(format_timestamp(65), "01:05")
        self.assertEqual(format_timestamp(3665), "01:01:05")

    def test_calculate_reading_time(self):
        self.assertEqual(calculate_reading_time(100), "1 min read")
        self.assertEqual(calculate_reading_time(1000), "5 min read")

if __name__ == "__main__":
    unittest.main()
