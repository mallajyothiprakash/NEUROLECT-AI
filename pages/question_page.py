"""
NEUROLECT AI - Pretrained Question Generation Page
Generates multi-type educational questions using instruction-tuned FLAN-T5.
Features MCQ distractors, Bloom's Taxonomy, difficulty scoring, and quality auditing.
"""

import streamlit as st
import json
from modules.question_generator import QuestionGenerator
import config

def render_question_page():
    st.header("❓ AI Question Intelligence Studio")
    st.markdown("Transformer-based question generation supporting 6 distinct pedagogical categories.")

    current = st.session_state.get("current_lecture")
    if not current:
        st.warning("⚠️ No active lecture loaded. Please upload Audio, Video, or PDF/Text first.")
        return

    st.info(f"**Target Lecture:** `{current.source_name}` — {current.word_count:,} words")

    with st.expander("⚙️ Question Generation Settings", expanded=True):
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            selected_types = st.multiselect(
                "Select Question Types to Generate",
                options=config.QUESTION_TYPES,
                default=config.QUESTION_TYPES
            )
        with col_s2:
            num_q = st.slider("Number of Questions", min_value=2, max_value=20, value=6)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        generate_btn = st.button("Generate Questions with Flan-T5 🚀", type="primary")
    with col_btn2:
        if st.session_state.get("generated_questions"):
            if st.button("Clear / Regenerate 🔄"):
                st.session_state["generated_questions"] = None
                st.rerun()

    if generate_btn:
        if not selected_types:
            st.error("Please select at least one question type.")
            return

        try:
            with st.spinner("Generating questions with Flan-T5 & synthesizing distractors..."):
                q_generator = QuestionGenerator()
                # Use extracted keywords and topics if available
                keywords_data = st.session_state.get("lecture_keywords", [])
                candidate_terms = [k["keyword"] for k in keywords_data] if keywords_data else None
                topics_data = st.session_state.get("lecture_topics")

                results = q_generator.generate_questions(
                    lecture_content=current,
                    question_types=selected_types,
                    num_questions=num_q,
                    candidate_keywords=candidate_terms,
                    topics_data=topics_data
                )
                st.session_state["generated_questions"] = results
            
            req_cnt = results.get("requested_count", num_q)
            fin_cnt = len(results.get("questions", []))
            if fin_cnt == req_cnt:
                st.success(f"✅ Successfully generated and validated all {fin_cnt} requested questions!")
            elif fin_cnt > 0:
                st.warning(f"⚠️ Requested: {req_cnt} questions. Successfully generated: {fin_cnt} valid questions. {req_cnt - fin_cnt} additional questions could not pass validation.")
            else:
                cat = results.get("failure_category", "EMPTY_GENERATION")
                st.warning(f"⚠️ Question Generation Audit: {cat}. See detailed diagnostics below.")
            st.rerun()
        except Exception as e:
            st.error(f"⚠️ Question Generation Error: {e}")
            return

    results = st.session_state.get("generated_questions")
    if results:
        if results.get("questions"):
            questions = results["questions"]
            warnings = results.get("warnings", [])

            st.markdown("---")
            
            # Validation Audits
            if warnings:
                with st.expander(f"⚠️ Quality Validation Audit ({len(warnings)} notifications)", expanded=False):
                    for w in warnings:
                        st.caption(f"• {w}")

            st.subheader(f"📝 Generated Assessment Questions ({len(questions)} items)")
            
            for idx, q in enumerate(questions):
                with st.container():
                    st.markdown(f"""
                    <div class="question-card">
                        <div style="display:flex; justify-content:space-between; margin-bottom:0.5rem; align-items:center;">
                            <span style="font-weight:700; color:#93c5fd;">Question {idx+1} • {q['type']}</span>
                            <div>
                                <span class="taxonomy-pill pill-{q['bloom_level'].lower()}">{q['bloom_level']}</span>
                                <span class="difficulty-{q['difficulty'].lower()}">{q['difficulty']}</span>
                            </div>
                        </div>
                        <div style="font-size:1.05rem; font-weight:600; color:#f8fafc; margin-bottom:0.75rem;">
                            {q['question']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # MCQ Options Display
                    if "MCQ" in q.get("type", "") and q.get("options"):
                        st.markdown("**Options:**")
                        for letter, opt in q["options"].items():
                            is_correct = (letter == q.get("correct_letter"))
                            badge = "✅ *(Correct Answer)*" if is_correct else ""
                            st.markdown(f"- **{letter}:** {opt} {badge}")

                    with st.expander("Reveal Reference Answer & Rationale"):
                        st.markdown(f"**Reference Answer:** {q.get('correct_answer')}")
                        st.markdown(f"**Pedagogical Explanation:** {q.get('explanation')}")
                        st.caption(f"Topic: `{q.get('topic')}` | Anchor: `{q.get('source_ref')}`")

                    st.markdown("")

            # Export controls
            c_exp1, c_exp2 = st.columns(2)
            with c_exp1:
                st.download_button(
                    "Export Questions as JSON 💾",
                    data=json.dumps(questions, indent=2),
                    file_name=f"{current.source_name}_questions.json",
                    mime="application/json",
                    use_container_width=True
                )
            with c_exp2:
                # Format comprehensive text export preserving all fields
                txt_lines = []
                txt_lines.append(f"NEUROLECT AI - ASSESSMENT QUESTION EXPORT")
                txt_lines.append(f"Lecture: {current.source_name}")
                txt_lines.append(f"Total Questions: {len(questions)}\n")
                txt_lines.append("=" * 60)
                
                for i, q in enumerate(questions):
                    txt_lines.append(f"Question #{i+1} [ID: {q.get('id', 'N/A')}]")
                    txt_lines.append(f"Type: {q.get('type')}")
                    txt_lines.append(f"Topic: {q.get('topic', 'General')}")
                    txt_lines.append(f"Bloom's Taxonomy: {q.get('bloom_level')} | Difficulty: {q.get('difficulty')}")
                    txt_lines.append(f"Source Reference: {q.get('source_ref', 'General')}")
                    txt_lines.append("-" * 40)
                    txt_lines.append(f"Question: {q.get('question')}\n")
                    if "MCQ" in q.get("type", "") and q.get("options"):
                        txt_lines.append("Options:")
                        for l, o in q["options"].items():
                            is_correct = (l == q.get("correct_letter"))
                            marker = " (Correct Answer)" if is_correct else ""
                            txt_lines.append(f"  [{l}] {o}{marker}")
                        txt_lines.append("")
                    txt_lines.append(f"Correct Answer: {q.get('correct_answer')}")
                    txt_lines.append(f"Explanation: {q.get('explanation', 'N/A')}\n")
                    txt_lines.append("=" * 60)
                
                st.download_button(
                    "Export Questions as Text (.txt) 💾",
                    data="\n".join(txt_lines),
                    file_name=f"{current.source_name}_questions.txt",
                    mime="text/plain",
                    use_container_width=True
                )
        else:
            # Step 5: Informative Empty-State Diagnostic Display
            st.markdown("---")
            cat = results.get("failure_category", "EMPTY_GENERATION")
            warnings = results.get("warnings", [])
            counts = results.get("diagnostic_counts", {})
            
            CATEGORY_INFO = {
                "MODEL_UNAVAILABLE": {
                    "title": "FLAN-T5 Model Unavailable",
                    "icon": "🤖",
                    "reason": "The question-generation model could not be loaded into memory.",
                    "action": "Ensure model weights for 'google/flan-t5-base' are cached on disk."
                },
                "EMPTY_TRANSCRIPT": {
                    "title": "No Lecture Content Received",
                    "icon": "📄",
                    "reason": "The question generator received an empty lecture transcript or zero content chunks.",
                    "action": "Upload an audio/video recording with clear speech or a document containing at least 15 words."
                },
                "EMPTY_GENERATION": {
                    "title": "FLAN-T5 Returned No Valid Candidates",
                    "icon": "❓",
                    "reason": "The model was unable to formulate valid question-and-answer pairs from the text chunks.",
                    "action": "Try selecting additional question types (e.g. Short Answer, Descriptive) or adjust question count."
                },
                "VALIDATION_FAILURE": {
                    "title": "Generated Items Failed Quality Audits",
                    "icon": "🛡️",
                    "reason": "Questions were synthesized by FLAN-T5, but all candidates were rejected by factual grounding or deduplication checks.",
                    "action": "Provide richer educational material with distinct technical terminology."
                },
                "DISTRACTOR_FAILURE": {
                    "title": "MCQ Distractor Generation Failed",
                    "icon": "🎯",
                    "reason": "Could not synthesize 3 credible, non-duplicate distractors from the lecture context for MCQs.",
                    "action": "Select open-ended question types (Short Answer, Descriptive, Conceptual) which do not require distractors."
                }
            }
            
            info = CATEGORY_INFO.get(cat, {
                "title": "Question Generation Incomplete",
                "icon": "⚠️",
                "reason": "The generation pipeline could not produce valid assessment questions.",
                "action": "Check the quality notifications below and retry."
            })
            
            st.markdown(f"""
            <div style="background:#1e293b; border:1px solid #475569; border-radius:10px; padding:1.25rem; margin-top:1rem;">
                <h4 style="color:#f8fafc; margin-top:0;">{info['icon']} {info['title']}</h4>
                <p style="color:#94a3b8; font-size:0.95rem; margin-bottom:0.5rem;">
                    <strong>Diagnosis:</strong> {info['reason']}
                </p>
                <p style="color:#60a5fa; font-size:0.95rem; margin-bottom:0.5rem;">
                    <strong>Recommended Action:</strong> {info['action']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if warnings:
                with st.expander(f"📋 Generation Audit Log ({len(warnings)} notifications)", expanded=True):
                    for w in warnings:
                        st.caption(f"• {w}")
