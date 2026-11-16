"""
NEUROLECT AI — Intelligent Lecture Understanding, Question Generation & Automated Assessment System
Main Streamlit Application Controller.
Strictly RAG-Free • Pretrained Transformer Architecture.
"""

import streamlit as st
from pathlib import Path
import config

# Set page configuration
st.set_page_config(
    page_title="NEUROLECT AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = config.ASSETS_DIR / "style.css"
    if css_file.exists():
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Initialize session state variables
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Home Dashboard"
if "current_lecture" not in st.session_state:
    st.session_state["current_lecture"] = None
if "lecture_summaries" not in st.session_state:
    st.session_state["lecture_summaries"] = None
if "lecture_topics" not in st.session_state:
    st.session_state["lecture_topics"] = None
if "lecture_keywords" not in st.session_state:
    st.session_state["lecture_keywords"] = None
if "lecture_ner" not in st.session_state:
    st.session_state["lecture_ner"] = None
if "lecture_concepts" not in st.session_state:
    st.session_state["lecture_concepts"] = None
if "generated_questions" not in st.session_state:
    st.session_state["generated_questions"] = None

# Import pages lazily
from pages.home_page import render_home_page
from pages.audio_page import render_audio_page
from pages.video_page import render_video_page
from pages.document_page import render_document_page
from pages.summary_page import render_summary_page
from pages.analysis_page import render_analysis_page
from pages.ner_concept_page import render_ner_concept_page
from pages.question_page import render_question_page
from pages.quiz_page import render_quiz_page
from pages.analytics_page import render_analytics_page
from pages.history_page import render_history_page

# Sidebar Navigation
PAGES = {
    "Home Dashboard": render_home_page,
    "Audio Lecture Processing": render_audio_page,
    "Video Lecture Processing": render_video_page,
    "PDF and Text Processing": render_document_page,
    "Lecture Summary": render_summary_page,
    "Topic and Keyword Analysis": render_analysis_page,
    "Named Entity and Concept Extraction": render_ner_concept_page,
    "Question Generation": render_question_page,
    "Interactive Quiz": render_quiz_page,
    "Learning Analytics": render_analytics_page,
    "Quiz History": render_history_page
}

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 0.5rem 0;">
        <h2 style="margin-bottom:0; color:#3b82f6;">🧠 NEUROLECT AI</h2>
        <p style="font-size:0.75rem; color:#94a3b8; margin-top:2px;">Intelligent Lecture & Assessment Engine</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Active Lecture Widget
    current = st.session_state.get("current_lecture")
    if current:
        source_icon = {
            "audio": "🎙️",
            "video": "🎥",
            "pdf": "📑",
            "text": "📄"
        }.get(current.source_type, "📚")
        
        st.markdown(f"""
        <div style="background:#1e293b; padding:0.75rem; border-radius:8px; border:1px solid #334155; margin-bottom:1rem;">
            <div style="font-size:0.75rem; color:#94a3b8; text-transform:uppercase;">Active Lecture</div>
            <div style="font-weight:600; font-size:0.9rem; color:#f8fafc; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                {source_icon} {current.source_name}
            </div>
            <div style="font-size:0.75rem; color:#60a5fa; margin-top:3px;">
                {current.word_count:,} words • {current.reading_time}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.caption("No lecture currently loaded.")

    # Page Selection
    current_nav = st.session_state["nav_page"]
    page_keys = list(PAGES.keys())
    nav_index = page_keys.index(current_nav) if current_nav in page_keys else 0

    selected_page = st.radio(
        "Navigation Menu",
        options=page_keys,
        index=nav_index,
        label_visibility="collapsed"
    )
    st.session_state["nav_page"] = selected_page

    st.markdown("---")
    st.caption("🛡️ **Strictly RAG-Free Architecture**\nDirect transformer inference with sequence-level chunking.")

# Render Selected View
render_fn = PAGES.get(st.session_state["nav_page"], render_home_page)
render_fn()
