"""Main Streamlit application for the QA System."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for candidate in (str(PROJECT_ROOT), str(PROJECT_ROOT / "backend")):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from backend.qa_service import (
    load_qa_model,
    run_prediction,
    validate_question_input,
)
from frontend.components import (
    render_examples,
    render_how_it_works,
    render_sidebar,
    render_technology_stack,
)
from frontend.styles import render_css, render_footer, render_header


def setup_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Intelligent QA System – WHEELTRIX",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_resource(show_spinner=False)
def load_model():
    """Load and cache the QA model once per session."""
    return load_qa_model()


def main() -> None:
    """Main application entry point."""
    setup_page()
    render_css()
    render_header()

    max_len, show_confidence, show_highlight = render_sidebar()

    # Load model
    with st.spinner("Loading AI model … (first load may take ~30 seconds)"):
        try:
            model = load_model()
            st.success("✅ Model loaded successfully.")
        except Exception as exc:
            st.error(
                f"❌ Failed to load model: {exc}\n\n"
                "Please ensure you have internet access and transformers installed:\n"
                "`pip install transformers torch`"
            )
            st.stop()

    # Layout: two columns
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown(
            '<div class="section-header">📄 Input</div>', unsafe_allow_html=True
        )

        # Session state initialization
        if "context_input" not in st.session_state:
            st.session_state["context_input"] = ""
        if "question_input" not in st.session_state:
            st.session_state["question_input"] = ""

        context = st.text_area(
            "Context / Passage",
            value=st.session_state["context_input"],
            height=220,
            placeholder=(
                "Paste or type a paragraph here. "
                "The system will extract the answer from this text."
            ),
            key="context_input",
        )

        question = st.text_input(
            "Your Question",
            value=st.session_state["question_input"],
            placeholder="Type your question about the context above …",
            key="question_input",
        )

        btn_col1, btn_col2 = st.columns([1, 1])
        get_answer_btn = btn_col1.button(
            "🔍 Get Answer", type="primary", use_container_width=True
        )
        clear_btn = btn_col2.button("🗑️ Clear", use_container_width=True)

        if clear_btn:
            st.session_state["context_input"] = ""
            st.session_state["question_input"] = ""
            st.rerun()

        # Validation & Prediction
        if get_answer_btn:
            errors = validate_question_input(context, question)
            if errors:
                for err in errors:
                    st.warning(err)
            else:
                with st.spinner("Extracting answer …"):
                    result = run_prediction(model, question, context, max_len)

                answer = result["answer"]
                confidence = result["score"]
                elapsed = result["elapsed_time"]

                if not answer or answer.strip() == "":
                    st.warning(
                        "The model could not find a specific answer span. "
                        "Try rephrasing your question or providing more context."
                    )
                else:
                    st.markdown("#### ✅ Predicted Answer")
                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True,
                    )

                    if show_confidence:
                        conf_pct = int(confidence * 100)
                        st.progress(conf_pct)
                        st.markdown(
                            f'<div class="conf-label">Confidence: {conf_pct}%  '
                            f'&nbsp;|&nbsp; Inference time: {elapsed:.2f}s</div>',
                            unsafe_allow_html=True,
                        )

                    if show_highlight and answer in context:
                        highlighted = context.replace(
                            answer, f"**:green[{answer}]**", 1
                        )
                        st.markdown("#### 📌 Answer in Context")
                        st.markdown(highlighted)

    with col_right:
        render_examples()
        render_technology_stack()

    render_how_it_works()
    render_footer()


if __name__ == "__main__":
    main()
