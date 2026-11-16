"""
NEUROLECT AI - Audio Processing Page
Handles MP3, WAV, and M4A audio uploads, Whisper speech-to-text,
timestamped segment viewing, and handoff to the common NLP engine.
"""

import streamlit as st
from pathlib import Path
from utils.validators import validate_audio_file
from utils.file_manager import save_uploaded_file, cleanup_file
from modules.audio_processor import AudioProcessor
import config

def render_audio_page():
    st.header("🎙️ Audio Lecture Processing Studio")
    st.markdown("Transcribe audio lectures using pretrained **OpenAI Whisper** models with timestamp alignment.")

    uploaded_file = st.file_uploader(
        "Upload Audio Recording (MP3, WAV, M4A)",
        type=["mp3", "wav", "m4a"],
        help=f"Maximum allowed size: {config.MAX_AUDIO_SIZE_BYTES // (1024*1024)} MB"
    )

    if uploaded_file:
        is_valid, msg = validate_audio_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ {msg}")
            return

        st.audio(uploaded_file)
        st.caption(f"File Name: `{uploaded_file.name}` | Size: `{uploaded_file.size / (1024*1024):.2f} MB`")

        if st.button("Transcribe Audio Recording 🚀", type="primary"):
            saved_path = None
            try:
                with st.spinner("Saving audio and initializing Whisper model..."):
                    saved_path = save_uploaded_file(uploaded_file)
                    processor = AudioProcessor()

                with st.spinner("Transcribing speech with timestamp alignment (this may take a moment on CPU)..."):
                    lecture_content = processor.transcribe_audio(
                        audio_path=saved_path,
                        source_name=uploaded_file.name
                    )

                # Persist lecture into SQLite database
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

                # Store into session state and reset downstream caches
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

                st.success("✅ Audio transcription complete! Content successfully delivered to Common NLP Engine.")
            except Exception as e:
                st.error(f"⚠️ Speech Recognition Error: {e}")
            finally:
                cleanup_file(saved_path)

    # Display active audio transcript if available
    current = st.session_state.get("current_lecture")
    if current and current.source_type == "audio":
        st.markdown("---")
        st.subheader("📝 Generated Lecture Transcript")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Word Count", f"{current.word_count:,}")
        col2.metric("Reading Time", current.reading_time)
        col3.metric("Duration", current.metadata.get("duration_formatted", "N/A"))

        # Segments table with timestamps
        if current.segments:
            st.markdown("#### ⏱️ Timestamped Segments")
            for seg in current.segments:
                with st.expander(f"📍 {seg.get('source_ref', 'Segment')} - {seg['text'][:60]}..."):
                    st.write(seg["text"])

        with st.expander("Full Text Transcript", expanded=not bool(current.segments)):
            st.text_area("Full Transcript", current.cleaned_text, height=250, disabled=True)

        st.download_button(
            label="Download Full Transcript (.txt) 💾",
            data=current.cleaned_text,
            file_name=f"{Path(current.source_name).stem}_transcript.txt",
            mime="text/plain"
        )
