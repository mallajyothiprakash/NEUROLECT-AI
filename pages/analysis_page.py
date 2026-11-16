"""
NEUROLECT AI - Topic and Keyword Analysis Page
Visualizes automatically detected topics, subtopic hierarchies, lecture outlines,
and KeyBERT-style keyword relevance rankings with MMR deduplication.
"""

import streamlit as st
from modules.topic_detector import TopicDetector
from modules.keyword_extractor import KeywordExtractor

def render_analysis_page():
    st.header("🧠 Topic Discovery & Keyword Intelligence")
    st.markdown("Unsupervised semantic clustering and contextual embedding relevance extraction.")

    current = st.session_state.get("current_lecture")
    if not current:
        st.warning("⚠️ No active lecture loaded. Please upload Audio, Video, or PDF/Text first.")
        return

    st.info(f"**Target Lecture:** `{current.source_name}` — {current.word_count:,} words")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_analysis = st.button("Run Topic & Keyword Analysis 🚀", type="primary")
    with col_btn2:
        if st.session_state.get("lecture_topics"):
            if st.button("Re-run Analysis 🔄"):
                st.session_state["lecture_topics"] = None
                st.session_state["lecture_keywords"] = None
                st.rerun()

    if run_analysis:
        try:
            with st.spinner("Analyzing semantic sentence clusters and extracting outline..."):
                t_detector = TopicDetector()
                topics_data = t_detector.detect_topics(current)
                st.session_state["lecture_topics"] = topics_data

            with st.spinner("Computing candidate keywords with SentenceTransformers & MMR..."):
                kw_extractor = KeywordExtractor()
                keywords = kw_extractor.extract_keywords(current, topics_data=topics_data)
                st.session_state["lecture_keywords"] = keywords

            st.success("✅ Analysis completed!")
            st.rerun()
        except Exception as e:
            st.error(f"⚠️ Analysis Error: {e}")
            return

    topics_data = st.session_state.get("lecture_topics")
    keywords = st.session_state.get("lecture_keywords")

    if topics_data or keywords:
        st.markdown("---")
        tab_topics, tab_keywords = st.tabs(["📚 Topics & Structured Outline", "🔑 Key Educational Terminology"])

        with tab_topics:
            if topics_data:
                st.caption(f"Methodology: **{topics_data.get('methodology', 'Semantic Clustering')}**")
                
                col_out, col_top = st.columns([1, 2])
                with col_out:
                    st.subheader("📑 Lecture Outline")
                    for line in topics_data.get("outline", []):
                        st.markdown(line)

                with col_top:
                    st.subheader("🏛️ Topic Clusters & Subtopics")
                    for t in topics_data.get("topics", []):
                        with st.expander(f"🔹 {t.get('title', 'Topic')} ({t.get('sentence_count', 0)} sentences)", expanded=True):
                            st.markdown(f"**Subtopics:** {', '.join(t.get('subtopics', []))}")
                            st.markdown(f"**Preview:** _{t.get('content_preview', '')}_")

        with tab_keywords:
            if keywords:
                st.subheader("🔑 Ranked Key Terms & Technical Concepts")
                st.caption("Extracted via SentenceTransformer cosine similarity with Maximal Marginal Relevance.")
                
                # Render as responsive metric cards or table
                kw_cols = st.columns(4)
                for idx, kw in enumerate(keywords[:12]):
                    col = kw_cols[idx % 4]
                    col.metric(
                        label=kw["type"],
                        value=kw["keyword"],
                        delta=f"Sim: {kw['relevance_score']:.2f}"
                    )

                st.markdown("#### Full Terminology Breakdown")
                st.dataframe(keywords, use_container_width=True)
