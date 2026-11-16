"""
NEUROLECT AI - Learning Analytics & Pedagogical Dashboard
Visualizes assessment trends, topic mastery matrices, cognitive Bloom distributions,
difficulty adaptations, question-type evaluations, and personalized study recommendations.
Strictly RAG-free.
"""

import streamlit as st
import pandas as pd
from modules.analytics import LearningAnalytics

def render_analytics_page():
    st.header("📈 Learning Analytics & Diagnostic Dashboard")
    st.markdown("Holistic assessment tracking and pedagogical mastery breakdown across Bloom's Taxonomy.")

    analytics_engine = LearningAnalytics()
    data = analytics_engine.generate_dashboard_metrics()

    summary = data.get("summary", {})
    total_attempts = summary.get("total_attempts", 0)

    if total_attempts == 0:
        st.info("ℹ️ No assessment attempts recorded yet. Take an interactive quiz in the **Interactive Quiz** tab to populate your personalized analytics dashboard!")
        return

    # 1. High-Level Summary Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Quizzes Attempted", total_attempts)
    col2.metric("Questions Attempted", summary.get("total_questions", 0))
    col3.metric("Correct Responses", summary.get("total_correct", 0))
    avg_score = summary.get("average_score", 0.0)
    col4.metric("Average Score", f"{avg_score:.1f}%")
    
    prog = data.get("progress_tracking", {})
    delta_val = prog.get("score_delta", 0.0)
    col5.metric(
        "Score Progression",
        f"{prog.get('latest_score', 0.0):.1f}%",
        delta=f"{delta_val:+.1f}%" if total_attempts > 1 else None
    )

    st.markdown("---")

    # 2. Historical Score Trend Chart
    historical = data.get("historical_trend", [])
    if historical:
        st.subheader("📈 Longitudinal Assessment Score Progression")
        st.caption(f"Status: **{prog.get('status', 'Consistent')}** — {prog.get('summary', '')}")
        
        df_hist = pd.DataFrame(historical)
        if not df_hist.empty and "percentage" in df_hist.columns:
            # Prepare plotting dataframe with clean chronological indexing
            df_hist["Attempt"] = [f"Attempt {i+1}" for i in range(len(df_hist))]
            st.line_chart(
                data=df_hist,
                x="Attempt",
                y="percentage",
                color="#3b82f6"
            )

    st.markdown("---")

    # 3. Topic Mastery Matrix & Breakdown
    topic_stats = data.get("topic_stats", [])
    if topic_stats:
        st.subheader("📚 Topic-Wise Performance & Mastery Matrix")
        
        c_top1, c_top2 = st.columns([1, 2])
        with c_top1:
            strong = data.get("strong_topics", [])
            moderate = data.get("moderate_topics", [])
            weak = data.get("weak_topics", [])

            if strong:
                st.success(f"**Strong Topics (≥ 70%):**\n" + "\n".join([f"- ✅ {t}" for t in strong]))
            if moderate:
                st.info(f"**Moderate Topics (50–69%):**\n" + "\n".join([f"- ⚖️ {t}" for t in moderate]))
            if weak:
                st.error(f"**Topics Needing Revision (< 50%):**\n" + "\n".join([f"- ⚠️ {t}" for t in weak]))

        with c_top2:
            df_topic = pd.DataFrame(topic_stats)
            if not df_topic.empty and "topic" in df_topic.columns:
                st.bar_chart(
                    data=df_topic,
                    x="topic",
                    y="performance_pct",
                    color="#2563eb"
                )

        # Formatted Data Table
        display_cols = ["topic", "count", "correct_count", "incorrect_count", "avg_marks", "max_marks", "performance_pct", "mastery_status"]
        col_rename = {
            "topic": "Topic",
            "count": "Attempted",
            "correct_count": "Correct",
            "incorrect_count": "Incorrect",
            "avg_marks": "Avg Marks",
            "max_marks": "Max Marks",
            "performance_pct": "Score %",
            "mastery_status": "Mastery Status"
        }
        df_topic_clean = df_topic[[c for c in display_cols if c in df_topic.columns]].rename(columns=col_rename)
        st.dataframe(df_topic_clean, use_container_width=True)

    st.markdown("---")

    # 4. Cognitive Bloom's Taxonomy & Difficulty Breakdown
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.subheader("🧠 Cognitive Bloom's Taxonomy Breakdown")
        st.caption(data.get("bloom_diagnosis", ""))
        bloom_stats = data.get("bloom_stats", [])
        if bloom_stats:
            df_bloom = pd.DataFrame(bloom_stats)
            if not df_bloom.empty and "bloom_level" in df_bloom.columns:
                st.bar_chart(data=df_bloom, x="bloom_level", y="performance_pct", color="#8b5cf6")
                st.dataframe(df_bloom, use_container_width=True)

    with col_b2:
        st.subheader("⚖️ Difficulty Level Breakdown")
        st.caption(data.get("difficulty_diagnosis", ""))
        diff_stats = data.get("difficulty_stats", [])
        if diff_stats:
            df_diff = pd.DataFrame(diff_stats)
            if not df_diff.empty and "difficulty" in df_diff.columns:
                st.bar_chart(data=df_diff, x="difficulty", y="performance_pct", color="#f59e0b")
                st.dataframe(df_diff, use_container_width=True)

    st.markdown("---")

    # 5. Question-Type Comparison
    q_type_stats = data.get("question_type_stats", [])
    if q_type_stats:
        st.subheader("📝 Question-Type Performance Breakdown")
        st.caption(data.get("question_type_diagnosis", ""))
        df_qtypes = pd.DataFrame(q_type_stats)
        if not df_qtypes.empty and "question_type" in df_qtypes.columns:
            st.bar_chart(data=df_qtypes, x="question_type", y="performance_pct", color="#10b981")
            st.dataframe(df_qtypes, use_container_width=True)

    st.markdown("---")

    # 6. Data-Backed Personalized Recommendations
    st.subheader("💡 Personalized Pedagogical Recommendations")
    recs = data.get("recommendations", [])
    for idx, r in enumerate(recs):
        st.info(f"📌 **Action Item {idx+1}:** {r}")

    # Pedagogical disclaimer
    st.caption(f"*{data.get('pedagogical_caveat', '')}*")
