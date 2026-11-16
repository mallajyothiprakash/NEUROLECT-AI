"""
NEUROLECT AI - Interactive Assessment & Quiz Page
Administers timed or self-paced quizzes, conducts instant grading,
executes SentenceTransformer semantic answer evaluation, and stores results to SQLite.
"""

import streamlit as st
from modules.quiz_engine import QuizEngine
from database.db_manager import DatabaseManager
import config

def render_quiz_page():
    st.header("🎯 Interactive Assessment & Quiz Engine")
    st.markdown("Take interactive quizzes with automated semantic scoring and granular cognitive feedback.")

    questions_data = st.session_state.get("generated_questions", {})
    available_pool = questions_data.get("questions", []) if questions_data else []

    if not available_pool:
        st.warning("⚠️ No questions available to start a quiz. Please generate questions in the **Question Generation** tab first.")
        return

    # Check quiz session status in st.session_state
    if "quiz_active" not in st.session_state:
        st.session_state["quiz_active"] = False
        st.session_state["quiz_questions"] = []
        st.session_state["student_answers"] = {}
        st.session_state["quiz_result"] = None
        st.session_state["quiz_index"] = 0

    # 1. Setup Screen
    if not st.session_state["quiz_active"] and not st.session_state["quiz_result"]:
        st.subheader("⚙️ Configure Your Assessment")
        
        c1, c2 = st.columns(2)
        with c1:
            q_count = st.slider("Number of Questions to Attempt", min_value=1, max_value=len(available_pool), value=min(5, len(available_pool)))
            selected_diffs = st.multiselect("Filter by Difficulty", options=config.DIFFICULTY_LEVELS, default=config.DIFFICULTY_LEVELS)
            
            # Topic selection from question pool
            available_topics = sorted(list({q.get("topic") for q in available_pool if q.get("topic")}))
            selected_topics = st.multiselect("Filter by Topic", options=available_topics, default=available_topics) if available_topics else None

        with c2:
            selected_types = st.multiselect("Filter by Question Types", options=config.QUESTION_TYPES, default=config.QUESTION_TYPES)
            selected_blooms = st.multiselect("Filter by Bloom's Taxonomy", options=config.BLOOMS_TAXONOMY_LEVELS, default=config.BLOOMS_TAXONOMY_LEVELS)

        if st.button("Start Assessment 🚀", type="primary", use_container_width=True):
            engine = QuizEngine()
            selected_q = engine.filter_questions(
                pool=available_pool,
                num_questions=q_count,
                selected_types=selected_types,
                selected_difficulties=selected_diffs,
                selected_blooms=selected_blooms,
                selected_topics=selected_topics
            )
            if not selected_q:
                st.error("No questions matched the selected filters. Please broaden your selection.")
                return

            st.session_state["quiz_questions"] = selected_q
            st.session_state["student_answers"] = {}
            st.session_state["quiz_index"] = 0
            st.session_state["quiz_active"] = True
            st.session_state["quiz_result"] = None
            st.rerun()

        # Past Quiz History Expander
        db = DatabaseManager()
        history = db.get_quiz_history(limit=5)
        if history:
            st.markdown("---")
            with st.expander("📜 Recent Assessment History", expanded=False):
                for h in history:
                    st.markdown(
                        f"**{h.get('lecture_title', 'General Lecture')}** • "
                        f"Score: `{h.get('percentage')}%` ({h.get('correct_answers')}/{h.get('total_questions')} correct) • "
                        f"Completed: `{h.get('completed_at')[:16]}`"
                    )

    # 2. Active Quiz Stepper
    elif st.session_state["quiz_active"]:
        questions = st.session_state["quiz_questions"]
        curr_idx = st.session_state["quiz_index"]
        total_q = len(questions)
        q = questions[curr_idx]
        q_id = q.get("id", str(curr_idx))

        # Progress bar
        progress = (curr_idx + 1) / total_q
        st.progress(progress)
        st.caption(f"Question {curr_idx + 1} of {total_q} • Progress: {int(progress * 100)}%")

        # Question header
        st.markdown(f"""
        <div class="question-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem; align-items:center;">
                <span style="font-weight:700; color:#93c5fd;">Question {curr_idx + 1} • {q['type']}</span>
                <div>
                    <span class="taxonomy-pill pill-{q['bloom_level'].lower()}">{q['bloom_level']}</span>
                    <span class="difficulty-{q['difficulty'].lower()}">{q['difficulty']}</span>
                </div>
            </div>
            <div style="font-size:1.15rem; font-weight:600; color:#f8fafc;">
                {q['question']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Input response
        if "MCQ" in q.get("type", "") and q.get("options"):
            st.markdown("##### Select your answer:")
            options = q["options"]
            option_keys = list(options.keys())
            formatted_options = [f"{k}: {options[k]}" for k in option_keys]

            prev_answer = st.session_state["student_answers"].get(q_id, "")
            default_index = 0
            for idx, k in enumerate(option_keys):
                if k.upper() == prev_answer.upper():
                    default_index = idx

            choice = st.radio(
                "Options",
                options=formatted_options,
                index=default_index,
                label_visibility="collapsed"
            )
            # Store selected letter
            st.session_state["student_answers"][q_id] = choice.split(":")[0].strip()

        else:
            st.markdown("##### Write your response:")
            prev_text = st.session_state["student_answers"].get(q_id, "")
            user_text = st.text_area(
                "Your Answer",
                value=prev_text,
                height=150,
                placeholder="Write your explanation or answer in your own words...",
                label_visibility="collapsed"
            )
            st.session_state["student_answers"][q_id] = user_text

        # Navigation Controls
        st.markdown("")
        col_prev, col_next, col_sub = st.columns([1, 1, 2])
        
        with col_prev:
            if curr_idx > 0:
                if st.button("⬅️ Previous"):
                    st.session_state["quiz_index"] -= 1
                    st.rerun()

        with col_next:
            if curr_idx < total_q - 1:
                if st.button("Next ➡️"):
                    st.session_state["quiz_index"] += 1
                    st.rerun()

        with col_sub:
            if st.button("Submit Assessment & Grade 🏁", type="primary", use_container_width=True):
                with st.spinner("Grading responses with SentenceTransformer semantic evaluator..."):
                    engine = QuizEngine()
                    current_lec = st.session_state.get("current_lecture")
                    lec_id = current_lec.content_id if current_lec else None
                    
                    results = engine.evaluate_quiz_submission(
                        lecture_id=lec_id,
                        questions=questions,
                        student_answers=st.session_state["student_answers"]
                    )
                    st.session_state["quiz_result"] = results
                    st.session_state["quiz_active"] = False
                    st.rerun()

    # 3. Quiz Report Screen
    elif st.session_state["quiz_result"]:
        res = st.session_state["quiz_result"]
        st.subheader("📊 Assessment Performance Report")

        # Score banner
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Overall Score", f"{res['overall_percentage']}%")
        col_r2.metric("Questions Correct", f"{res['correct_answers']} / {res['total_questions']}")
        status_label = "Passed 🎉" if res['overall_percentage'] >= 60 else "Review Recommended 📚"
        col_r3.metric("Assessment Status", status_label)

        st.markdown("---")
        st.subheader("📝 Granular Question-by-Question Breakdown")

        for idx, item in enumerate(res["items"]):
            is_corr = item.get("is_correct", False)
            status_emoji = "✅" if is_corr else "❌"
            awarded = item.get("awarded_marks", item.get("score", 0.0))
            max_m = item.get("max_marks", 10.0)
            
            with st.expander(f"{status_emoji} Q{idx+1} ({item.get('type')}) — Marks: {awarded} / {max_m} ({item.get('score')}%)", expanded=True):
                st.markdown(f"**Question:** {item.get('question')}")
                st.markdown(f"**Your Answer:** `{item.get('student_answer') or '[No Answer Provided]'}`")
                st.markdown(f"**Correct Reference Answer:** `{item.get('correct_answer')}`")
                
                # Evaluation Method & Marks
                st.caption(f"Evaluation Method: `{item.get('evaluation_method')}`")
                
                # Feedback & Suggestions
                st.markdown(f"**Feedback:** {item.get('feedback')}")
                if item.get("suggestions") and item.get("suggestions") != "None":
                    st.info(f"💡 **Suggestion for Improvement:** {item.get('suggestions')}")
                
                # Semantic Similarity & Concept Coverage Breakdown for Free-Text
                if "MCQ" not in item.get("type", ""):
                    c_sim1, c_sim2, c_sim3 = st.columns(3)
                    c_sim1.metric("Semantic Similarity", f"{item.get('semantic_similarity', 0.0):.2f}")
                    c_sim2.metric("Concept Coverage", f"{item.get('concept_coverage', 0.0):.2f}")
                    c_sim3.metric("Completeness", f"{item.get('completeness_score', 0.0):.2f}")

                    if item.get("missing_concepts"):
                        st.warning(f"Key missing concepts: `{', '.join(item.get('missing_concepts'))}`")
                    if item.get("covered_concepts"):
                        st.success(f"Concepts identified: `{', '.join(item.get('covered_concepts'))}`")

                # Human Review Recommendation Alert
                if item.get("human_review_recommended"):
                    st.warning(f"⚠️ **Instructor Review Recommended:** {item.get('review_reason')}")

                st.caption(f"Topic: `{item.get('topic')}` | Bloom Level: `{item.get('bloom_level')}` | Difficulty: `{item.get('difficulty')}`")
                st.caption(f"*{item.get('disclaimer')}*")

        if st.button("Take Another Assessment 🔄", type="primary"):
            st.session_state["quiz_active"] = False
            st.session_state["quiz_result"] = None
            st.session_state["student_answers"] = {}
            st.rerun()
