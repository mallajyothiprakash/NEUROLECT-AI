"""
NEUROLECT AI - Lecture Summary Page
Displays concise, detailed, section-wise, and structured lecture notes generated
using pretrained transformer summarization (DistilBART / BART-large-CNN).
"""

import streamlit as st
from modules.summarizer import LectureSummarizer

def render_summary_page():
    st.header("📋 Lecture Summarization Studio")
    st.markdown("Transformer-based multi-tier summaries generated without RAG, using hierarchical chunk synthesis.")

    current = st.session_state.get("current_lecture")
    if not current:
        st.warning("⚠️ No active lecture loaded. Please upload Audio, Video, or PDF/Text first.")
        return

    st.info(f"**Target Lecture:** `{current.source_name}` ({current.source_type.upper()}) — {current.word_count:,} words")

    # Generate button or view existing
    summaries = st.session_state.get("lecture_summaries")

    if not summaries:
        if st.button("Generate Lecture Summaries with Transformers 🚀", type="primary"):
            try:
                with st.spinner("Summarizing text chunks using transformer model (this may take a few moments)..."):
                    summarizer = LectureSummarizer()
                    summaries = summarizer.generate_all_summaries(current)
                    st.session_state["lecture_summaries"] = summaries
                st.success("✅ Summaries generated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ Summarization Error: {e}")
                return
    else:
        if st.button("Regenerate Summaries 🔄"):
            st.session_state["lecture_summaries"] = None
            st.rerun()

    if summaries:
        st.markdown("---")
        # Summary Overview Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Original Length", f"{summaries.get('original_word_count', 0):,} words")
        m2.metric("Concise Summary", f"{summaries.get('concise_word_count', 0):,} words")
        m3.metric("Compression Ratio", summaries.get("compression_ratio", "N/A"))

        # Tabs for different summary formats
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Concise Summary",
            "📖 Detailed Summary",
            "📑 Section-wise Summaries",
            "📌 Structured Lecture Notes",
            "🏛️ Topic-wise Summaries"
        ])

        with tab1:
            st.markdown("### 🎯 Executive Concise Summary")
            st.markdown(f"> {summaries.get('concise_summary', '')}")
            st.download_button(
                "Download Concise Summary (.txt)",
                data=summaries.get("concise_summary", ""),
                file_name=f"{current.source_name}_concise_summary.txt",
                mime="text/plain"
            )

        with tab2:
            st.markdown("### 📖 Detailed Comprehensive Summary")
            st.write(summaries.get("detailed_summary", ""))
            st.download_button(
                "Download Detailed Summary (.txt)",
                data=summaries.get("detailed_summary", ""),
                file_name=f"{current.source_name}_detailed_summary.txt",
                mime="text/plain"
            )

        with tab3:
            st.markdown("### 📑 Section-by-Section Breakdowns")
            st.caption("Each chunk summary is anchored to its original page number or audio timestamp.")
            for sec in summaries.get("section_summaries", []):
                with st.expander(f"📍 {sec.get('source_ref', 'Section')} ({sec.get('summary_word_count', 0)} words)"):
                    st.write(sec.get("summary", ""))

        with tab4:
            st.markdown("### 📌 Structured Key Takeaways")
            for note in summaries.get("structured_notes", []):
                st.markdown(f"#### {note.get('header', '')}")
                for pt in note.get("key_points", []):
                    st.markdown(f"- {pt}")
                st.markdown("")

        with tab5:
            st.markdown("### 🏛️ Topic-wise Summaries")
            topic_summaries = summaries.get("topic_summaries", [])
            if topic_summaries:
                for ts in topic_summaries:
                    with st.expander(f"🔹 {ts.get('title', 'Topic')} ({ts.get('sentence_count', 0)} sentences)", expanded=True):
                        st.write(ts.get("summary", ""))
                        if ts.get("source_segments"):
                            st.caption(f"Sources: `{', '.join(ts.get('source_segments'))}`")
            else:
                st.info("Run **Topic and Keyword Analysis** to view topic-clustered summaries, or generate them directly here.")
