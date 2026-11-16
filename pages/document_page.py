"""
NEUROLECT AI - PDF & Text Document Processing Page
Handles PDF extraction with PyMuPDF page tracking, TXT file uploads,
and direct lecture text pasting into the common NLP engine.
"""

import streamlit as st
from pathlib import Path
from utils.validators import validate_document_file, validate_text_content
from utils.file_manager import save_uploaded_file, cleanup_file
from modules.pdf_processor import PDFProcessor
from modules.text_processor import TextProcessor
import config

def render_document_page():
    st.header("📄 PDF and Text Processing Studio")
    st.markdown("Extract and structure text from PDF slides, lecture notes, or direct text input.")

    tab_pdf, tab_txt, tab_paste = st.tabs(["📑 PDF Document Upload", "📁 TXT File Upload", "✍️ Direct Text Paste"])

    # 1. PDF Upload Tab
    with tab_pdf:
        st.subheader("Upload Educational PDF")
        uploaded_pdf = st.file_uploader(
            "Upload PDF Lecture Document",
            type=["pdf"],
            help=f"Maximum allowed size: {config.MAX_DOCUMENT_SIZE_BYTES // (1024*1024)} MB",
            key="uploader_pdf"
        )
        if uploaded_pdf:
            is_valid, msg = validate_document_file(uploaded_pdf)
            if not is_valid:
                st.error(f"❌ {msg}")
            else:
                st.caption(f"PDF File: `{uploaded_pdf.name}` | Size: `{uploaded_pdf.size / (1024*1024):.2f} MB`")
                if st.button("Extract Text from PDF with PyMuPDF 🚀", key="btn_extract_pdf", type="primary"):
                    saved_pdf = None
                    try:
                        with st.spinner("Extracting text and tracking page boundaries..."):
                            saved_pdf = save_uploaded_file(uploaded_pdf)
                            processor = PDFProcessor()
                            lecture_content = processor.extract_text_from_pdf(
                                pdf_path=saved_pdf,
                                source_name=uploaded_pdf.name
                            )

                        _update_active_lecture(lecture_content)
                        st.success(f"✅ Extracted {lecture_content.metadata.get('total_pages', 0)} pages ({lecture_content.word_count:,} words) successfully!")
                    except Exception as e:
                        st.error(f"⚠️ PDF Extraction Error: {e}")
                    finally:
                        cleanup_file(saved_pdf)

    # 2. TXT File Tab
    with tab_txt:
        st.subheader("Upload Plain Text File")
        uploaded_txt = st.file_uploader(
            "Upload Plain Text Lecture (.txt)",
            type=["txt"],
            key="uploader_txt"
        )
        if uploaded_txt:
            st.caption(f"File: `{uploaded_txt.name}` | Size: `{uploaded_txt.size / 1024:.1f} KB`")
            if st.button("Process TXT File 🚀", key="btn_process_txt", type="primary"):
                saved_txt = None
                try:
                    with st.spinner("Reading and structuring text file..."):
                        saved_txt = save_uploaded_file(uploaded_txt)
                        processor = TextProcessor()
                        lecture_content = processor.process_txt_file(
                            txt_path=saved_txt,
                            source_name=uploaded_txt.name
                        )
                    _update_active_lecture(lecture_content)
                    st.success(f"✅ Processed {lecture_content.word_count:,} words from text file!")
                except Exception as e:
                    st.error(f"⚠️ Text File Error: {e}")
                finally:
                    cleanup_file(saved_txt)

    # 3. Direct Paste Tab
    with tab_paste:
        st.subheader("Direct Educational Content Input")
        lecture_title = st.text_input("Lecture / Chapter Title", value="Introduction to Machine Learning")
        pasted_text = st.text_area(
            "Paste Lecture Notes or Transcript",
            height=250,
            placeholder="Paste educational material, research notes, or lecture transcripts here (minimum 15 words)..."
        )
        if st.button("Process Pasted Content 🚀", key="btn_process_paste", type="primary"):
            try:
                with st.spinner("Normalizing text and building chunks..."):
                    processor = TextProcessor()
                    lecture_content = processor.process_raw_text(
                        text=pasted_text,
                        lecture_title=lecture_title
                    )
                _update_active_lecture(lecture_content)
                st.success(f"✅ Processed {lecture_content.word_count:,} words of educational text!")
            except Exception as e:
                st.error(f"⚠️ Input Validation Error: {e}")

    # Display active document content
    current = st.session_state.get("current_lecture")
    if current and current.source_type in ["pdf", "text"]:
        st.markdown("---")
        st.subheader(f"📝 Active Document: {current.source_name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Word Count", f"{current.word_count:,}")
        col2.metric("Reading Time", current.reading_time)
        col3.metric("Chunks Generated", len(current.chunks))

        if current.segments and current.source_type == "pdf":
            st.markdown("#### 📑 Extracted Pages")
            for seg in current.segments:
                with st.expander(f"📄 {seg.get('source_ref', 'Page')} ({len(seg['text'].split())} words)"):
                    st.write(seg["text"])

        with st.expander("Full Extracted Document", expanded=not bool(current.segments)):
            st.text_area("Cleaned Text", current.cleaned_text, height=250, disabled=True)

        st.download_button(
            label="Download Extracted Text (.txt) 💾",
            data=current.cleaned_text,
            file_name=f"{Path(current.source_name).stem}_extracted.txt",
            mime="text/plain"
        )

def _update_active_lecture(lecture_content):
    from database.db_manager import DatabaseManager
    try:
        db = DatabaseManager()
        db.save_lecture(
            lecture_id=lecture_content.content_id,
            title=lecture_content.source_name,
            source_type=lecture_content.source_type,
            word_count=lecture_content.word_count
        )
    except Exception:
        pass

    st.session_state["current_lecture"] = lecture_content
    st.session_state["lecture_summaries"] = None
    st.session_state["lecture_topics"] = None
    st.session_state["lecture_keywords"] = None
    st.session_state["lecture_ner"] = None
    st.session_state["lecture_concepts"] = None
    st.session_state["generated_questions"] = None
    st.session_state["quiz_active"] = False
    st.session_state["quiz_questions"] = []
    st.session_state["student_answers"] = {}
    st.session_state["quiz_result"] = None
    st.session_state["quiz_index"] = 0
