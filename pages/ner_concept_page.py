"""
NEUROLECT AI - Named Entity & Concept Extraction Page
Categorizes People, Organizations, Locations, and Technical entities via BERT-NER
and extracts formal definitions and principles grounded in page numbers / timestamps.
"""

import streamlit as st
from modules.ner_processor import NamedEntityRecognizer
from modules.concept_extractor import ConceptExtractor

def render_ner_concept_page():
    st.header("🏷️ Named Entity & Concept Extraction Studio")
    st.markdown("Pretrained BERT NER categorization and grounded educational definitions.")

    current = st.session_state.get("current_lecture")
    if not current:
        st.warning("⚠️ No active lecture loaded. Please upload Audio, Video, or PDF/Text first.")
        return

    st.info(f"**Target Lecture:** `{current.source_name}` — {current.word_count:,} words")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        run_extract = st.button("Extract Entities & Concepts 🚀", type="primary")
    with col_b2:
        if st.session_state.get("lecture_ner"):
            if st.button("Re-extract 🔄"):
                st.session_state["lecture_ner"] = None
                st.session_state["lecture_concepts"] = None
                st.rerun()

    if run_extract:
        try:
            with st.spinner("Extracting named entities with dslim/bert-base-NER..."):
                ner_engine = NamedEntityRecognizer()
                ner_data = ner_engine.extract_entities(current)
                st.session_state["lecture_ner"] = ner_data

            with st.spinner("Mining formal definitions and core principles..."):
                c_extractor = ConceptExtractor()
                concepts = c_extractor.extract_concepts(current)
                st.session_state["lecture_concepts"] = concepts

            st.success("✅ Entity and Concept extraction completed!")
            st.rerun()
        except Exception as e:
            st.error(f"⚠️ Extraction Error: {e}")
            return

    ner_data = st.session_state.get("lecture_ner")
    concepts = st.session_state.get("lecture_concepts")

    if ner_data or concepts:
        st.markdown("---")
        tab_concepts, tab_ner = st.tabs(["💡 Grounded Educational Concepts", "🏷️ Pretrained BERT Named Entities"])

        with tab_concepts:
            if concepts:
                st.subheader(f"💡 Extracted Key Concepts & Definitions ({len(concepts)} found)")
                st.caption("Each statement is grounded in original PDF pages or transcript timestamps.")
                
                for idx, c in enumerate(concepts):
                    with st.expander(f"{idx+1}. [{c.get('type')}] {c.get('term')} — ({c.get('source_ref')})", expanded=True):
                        st.markdown(f"**Statement:** {c.get('statement')}")
                        st.caption(f"Importance: `{c.get('importance')}` | Source: `{c.get('source_ref')}`")

        with tab_ner:
            if ner_data:
                st.subheader(f"🏷️ Named Entities Categorization ({ner_data.get('total_entities_found', 0)} entities)")
                st.caption(f"Model: `{ner_data.get('model_used')}`")
                
                cat_cols = st.columns(4)
                categories = ner_data.get("categories", {})

                for idx, (cat_name, items) in enumerate(categories.items()):
                    with cat_cols[idx]:
                        st.markdown(f"**{cat_name}** ({len(items)})")
                        if items:
                            for item in items[:6]:
                                st.markdown(f"- **{item['text']}** _(x{item['count']})_")
                        else:
                            st.caption("None detected")
