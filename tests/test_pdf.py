"""
Unit Tests for PDF Processor Module (Phase 2)
Tests PDF validation, PyMuPDF text extraction, page number preservation,
progress callback reporting, and corrupted/blank document rejection.
"""

import unittest
import pymupdf
from pathlib import Path
from modules.pdf_processor import PDFProcessor
from modules.content_models import LectureContent
import config

class TestPDFProcessing(unittest.TestCase):
    def setUp(self):
        self.processor = PDFProcessor()
        self.test_dir = config.TEMP_DIR / "test_docs"
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create a valid 2-page test PDF
        self.valid_pdf_path = self.test_dir / "sample_lecture.pdf"
        doc = pymupdf.open()
        
        page1 = doc.new_page()
        page1.insert_text((50, 72), "Lecture Chapter 1: Introduction to Natural Language Processing.")
        page1.insert_text((50, 100), "NLP is a subfield of artificial intelligence focused on human-computer communication.")
        
        page2 = doc.new_page()
        page2.insert_text((50, 72), "Lecture Chapter 2: Deep Learning Transformers.")
        page2.insert_text((50, 100), "Transformers utilize self-attention mechanisms to process sequential tokens in parallel.")
        
        doc.save(str(self.valid_pdf_path))
        doc.close()

        # 2. Create a blank PDF (page exists, but zero readable text)
        self.blank_pdf_path = self.test_dir / "blank.pdf"
        blank_doc = pymupdf.open()
        blank_doc.new_page() # Blank page without text
        blank_doc.save(str(self.blank_pdf_path))
        blank_doc.close()

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_pdf_validation(self):
        # Valid PDF
        valid, msg = self.processor.validate_pdf(self.valid_pdf_path)
        self.assertTrue(valid, msg)

        # Missing PDF
        missing = self.test_dir / "does_not_exist.pdf"
        valid, msg = self.processor.validate_pdf(missing)
        self.assertFalse(valid)
        self.assertIn("does not exist", msg)

        # Bad extension
        wrong_ext = self.test_dir / "doc.docx"
        wrong_ext.touch()
        valid, msg = self.processor.validate_pdf(wrong_ext)
        self.assertFalse(valid)
        self.assertIn("Unsupported", msg)

    def test_valid_pdf_extraction_with_progress(self):
        progress_calls = []
        def track_progress(pct, msg):
            progress_calls.append((pct, msg))

        content = self.processor.extract_text_from_pdf(
            self.valid_pdf_path,
            source_name="SampleLecture.pdf",
            progress_callback=track_progress
        )

        self.assertIsInstance(content, LectureContent)
        self.assertEqual(content.source_type, "pdf")
        self.assertEqual(content.source_name, "SampleLecture.pdf")
        self.assertGreater(content.word_count, 15)
        self.assertEqual(len(content.segments), 2)
        self.assertIn("Natural Language Processing", content.cleaned_text)
        self.assertEqual(content.segments[0]["source_ref"], "Page 1")
        self.assertEqual(content.segments[1]["source_ref"], "Page 2")
        self.assertEqual(content.metadata["total_pages"], 2)

        # Verify progress tracking
        self.assertGreater(len(progress_calls), 0)
        self.assertEqual(progress_calls[-1][0], 1.0)

    def test_blank_pdf_rejection(self):
        with self.assertRaises(ValueError) as ctx:
            self.processor.extract_text_from_pdf(self.blank_pdf_path)
        self.assertIn("no readable text", str(ctx.exception).lower())

    def test_corrupted_pdf_rejection(self):
        corrupt_path = self.test_dir / "corrupted.pdf"
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT A REAL PDF HEADER")
        with self.assertRaises(ValueError):
            self.processor.extract_text_from_pdf(corrupt_path)

if __name__ == "__main__":
    unittest.main()
