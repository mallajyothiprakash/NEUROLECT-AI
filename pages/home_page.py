"""
NEUROLECT AI - Home Dashboard Page
Presents the unified educational platform with equal priority for Audio, Video, and PDF/Text.
"""

import streamlit as st

def render_home_page():
    # Hero Section
    st.markdown("""
    <div class="hero-container">
        <div class="rag-free-badge">🛡️ Strictly RAG-Free • Direct Pretrained Transformer Inference</div>
        <h1 class="hero-title">NEUROLECT AI</h1>
        <p class="hero-subtitle">
            Intelligent Lecture Understanding, Question Generation & Automated Assessment System.
            Transform educational audio, video, and PDF lecture notes into structured study materials,
            multi-type transformer questions, interactive quizzes, and cognitive learning analytics.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Active Lecture Status
    current_lecture = st.session_state.get("current_lecture")
    if current_lecture:
        st.success(f"**Active Lecture Loaded:** `{current_lecture.source_name}` ({current_lecture.source_type.upper()}) — {current_lecture.word_count:,} words | {current_lecture.reading_time}")
    else:
        st.info("💡 **Getting Started:** Select any input module below to upload a lecture recording or document.")

    st.subheader("📚 Three Equal-Priority Input Modalities")
    st.markdown("Regardless of which format you choose, every lecture feeds into the **same Common NLP Intelligence Engine**.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="input-card">
            <span class="badge-tag badge-audio">🎙️ Audio Processing</span>
            <h3 style="margin-top:0; color:#f8fafc;">Audio Lectures</h3>
            <p style="color:#94a3b8; font-size:0.92rem; line-height:1.5;">
                Upload lecture recordings in <b>MP3, WAV, or M4A</b>. Transcribed using OpenAI Whisper on CPU/GPU with segment-level timestamp preservation.
            </p>
            <div style="margin-top:auto; padding-top:1rem;">
                <span style="font-size:0.8rem; color:#60a5fa;">✓ Whisper Speech-to-Text<br>✓ Timestamps Preserved</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Audio Studio 🎙️", use_container_width=True, key="btn_goto_audio"):
            st.session_state["nav_page"] = "Audio Lecture Processing"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="input-card">
            <span class="badge-tag badge-video">🎥 Video Processing</span>
            <h3 style="margin-top:0; color:#f8fafc;">Video Lectures</h3>
            <p style="color:#94a3b8; font-size:0.92rem; line-height:1.5;">
                First-class video pipeline for <b>MP4, MOV, and MKV</b>. Automatic FFmpeg audio track extraction, Whisper transcription, and clean temporary file management.
            </p>
            <div style="margin-top:auto; padding-top:1rem;">
                <span style="font-size:0.8rem; color:#60a5fa;">✓ FFmpeg 16kHz Extraction<br>✓ Safe Temp Cleanup</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Video Studio 🎥", use_container_width=True, key="btn_goto_video"):
            st.session_state["nav_page"] = "Video Lecture Processing"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="input-card">
            <span class="badge-tag badge-document">📄 Documents & Text</span>
            <h3 style="margin-top:0; color:#f8fafc;">PDF & Notes</h3>
            <p style="color:#94a3b8; font-size:0.92rem; line-height:1.5;">
                Extract structured text from <b>PDFs</b> via PyMuPDF with page tracking, upload plain <b>TXT</b> files, or directly paste lecture transcripts.
            </p>
            <div style="margin-top:auto; padding-top:1rem;">
                <span style="font-size:0.8rem; color:#60a5fa;">✓ PyMuPDF Page Tracking<br>✓ Context Normalization</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Document Studio 📄", use_container_width=True, key="btn_goto_doc"):
            st.session_state["nav_page"] = "PDF and Text Processing"
            st.rerun()

    st.markdown("---")
    st.subheader("⚙️ System Architecture & NLP Intelligence Pipeline")
    
    col_arch1, col_arch2 = st.columns([3, 2])
    with col_arch1:
        st.markdown("""
        1. **Unified Common Representation:** All inputs convert into standard text chunks while preserving source page/timestamp tags.
        2. **Hierarchical Summarization:** Powered by pretrained `DistilBART-CNN-12-6` / `BART-large-CNN` with long-text chunk stitching.
        3. **Topic & KeyBERT Terminology:** Semantic sentence clustering (`all-MiniLM-L6-v2`) and Maximal Marginal Relevance (MMR).
        4. **Pretrained Transformer Question Generation:** Multi-type generation (`Flan-T5`) supporting MCQ, Short Answer, Descriptive, Conceptual, Application, and Analytical items.
        5. **Assessment & Semantic Grading:** Sentence Transformer cosine similarity and keyword coverage evaluation for student responses.
        """)
    with col_arch2:
        st.info("""
        **Pedagogical Taxonomies Built-in:**
        - **Bloom's Revised Taxonomy:** Remember, Understand, Apply, Analyze, Evaluate, Create
        - **Difficulty Classification:** Easy, Medium, Hard
        - **Question Validation:** Zero-duplicate MCQ distractors, groundedness checks
        """)
