"""
NEUROLECT AI - Video Processing Page
Handles MP4, MOV, and MKV video uploads, FFmpeg 16kHz mono audio extraction,
Whisper speech-to-text, timestamped viewing, and automatic temp cleanup.
"""

import streamlit as st
from pathlib import Path
from utils.validators import validate_video_file
from utils.file_manager import save_uploaded_file, cleanup_file
from modules.video_processor import VideoProcessor
import config

def render_video_page():
    st.header("🎥 Video Lecture Processing Studio")
    st.markdown("First-class video processing: extracts audio tracks via **FFmpeg** and transcribes speech via **Whisper**.")

    # Check FFmpeg readiness
    ffmpeg_ready = VideoProcessor.is_ffmpeg_available()
    if not ffmpeg_ready:
        st.warning(
            "⚠️ **FFmpeg was not detected on your system PATH.**\n\n"
            "To process video files, please install FFmpeg:\n"
            "- **macOS:** `brew install ffmpeg`\n"
            "- **Windows:** `winget install Gyan.FFmpeg`\n"
            "- **Linux:** `sudo apt install ffmpeg`"
        )

    uploaded_file = st.file_uploader(
        "Upload Video Lecture (MP4, MOV, MKV)",
        type=["mp4", "mov", "mkv"],
        help=f"Maximum allowed size: {config.MAX_VIDEO_SIZE_BYTES // (1024*1024)} MB"
    )

    if uploaded_file:
        is_valid, msg = validate_video_file(uploaded_file)
        if not is_valid:
            st.error(f"❌ {msg}")
            return

        st.video(uploaded_file)
        st.caption(f"Video File: `{uploaded_file.name}` | Size: `{uploaded_file.size / (1024*1024):.2f} MB`")

        if st.button("Extract Audio & Transcribe Video 🚀", type="primary", disabled=not ffmpeg_ready):
            saved_video_path = None
            try:
                with st.spinner("Saving video file securely..."):
                    saved_video_path = save_uploaded_file(uploaded_file)
                    processor = VideoProcessor()

                with st.spinner("Step 1/2: Extracting 16kHz audio stream using FFmpeg..."):
                    pass # Handled inside processor

                with st.spinner("Step 2/2: Transcribing speech with Whisper & cleaning temp files..."):
                    lecture_content = processor.process_video(
                        video_path=saved_video_path,
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

                st.success("✅ Video audio extracted, transcribed, and temp files cleanly removed!")
            except Exception as e:
                st.error(f"⚠️ Video Processing Error: {e}")
            finally:
                cleanup_file(saved_video_path)

    # Display active video transcript if available
    current = st.session_state.get("current_lecture")
    if current and current.source_type == "video":
        st.markdown("---")
        st.subheader("📝 Video Lecture Transcript")

        col1, col2, col3 = st.columns(3)
        col1.metric("Word Count", f"{current.word_count:,}")
        col2.metric("Reading Time", current.reading_time)
        col3.metric("Video Duration", current.metadata.get("duration_formatted", "N/A"))

        if current.segments:
            st.markdown("#### ⏱️ Timestamped Transcript Segments")
            for seg in current.segments:
                with st.expander(f"📍 {seg.get('source_ref', 'Timestamp')} - {seg['text'][:60]}..."):
                    st.write(seg["text"])

        with st.expander("Full Text Transcript", expanded=not bool(current.segments)):
            st.text_area("Full Transcript", current.cleaned_text, height=250, disabled=True)

        st.download_button(
            label="Download Video Transcript (.txt) 💾",
            data=current.cleaned_text,
            file_name=f"{Path(current.source_name).stem}_video_transcript.txt",
            mime="text/plain"
        )
