"""
NEUROLECT AI - Quiz History Page
Lists past assessment attempts stored in SQLite with full itemized review capabilities.
"""

import streamlit as st
import pandas as pd
from database.db_manager import DatabaseManager

def render_history_page():
    st.header("📜 Assessment History & Attempt Archive")
    st.markdown("Review past quiz sessions, performance trends, and previous responses stored in SQLite.")

    db = DatabaseManager()
    history = db.get_quiz_history(limit=50)

    if not history:
        st.info("ℹ️ No quiz history found in database. Complete an assessment to record your attempts!")
        return

    # Convert to DataFrame
    df = pd.DataFrame(history)
    st.subheader(f"Recent Quiz Sessions ({len(history)} attempts)")
    st.dataframe(
        df[["completed_at", "lecture_title", "source_type", "total_questions", "correct_answers", "percentage", "difficulty_selected"]],
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("🔍 Inspect Detailed Attempt")
    
    attempt_options = [f"{h['completed_at'][:19]} — {h['lecture_title'] or 'Lecture'} ({h['percentage']}%)" for h in history]
    selected_idx = st.selectbox("Select Attempt to Inspect", range(len(history)), format_func=lambda i: attempt_options[i])

    selected_attempt_id = history[selected_idx]["id"]
    details = db.get_attempt_details(selected_attempt_id)

    if details and "answers" in details:
        st.markdown(f"#### Score: `{details.get('percentage')}%` | Date: `{details.get('completed_at')}`")
        for idx, ans in enumerate(details["answers"]):
            is_corr = bool(ans.get("is_correct"))
            status = "✅ Correct" if is_corr else "❌ Incorrect"
            with st.expander(f"{status} | Q{idx+1}: {ans.get('question_text')[:70]}..."):
                st.markdown(f"**Full Question:** {ans.get('question_text')}")
                st.markdown(f"**Your Answer:** `{ans.get('student_answer') or '[No Answer]'}`")
                st.markdown(f"**Correct Reference Answer:** `{ans.get('correct_answer')}`")
                st.markdown(f"**Evaluator Feedback:** {ans.get('feedback')}")
                st.caption(f"Score Awarded: `{ans.get('score')}` | Bloom Level: `{ans.get('bloom_level')}` | Difficulty: `{ans.get('difficulty')}`")
