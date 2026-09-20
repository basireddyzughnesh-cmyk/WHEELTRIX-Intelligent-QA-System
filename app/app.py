"""
=============================================================================
app.py
-----------------------------------------------------------------------------
Intelligent User Question Answering System
Streamlit Application

Dataset  : Stanford Question Answering Dataset (SQuAD) 1.1
Model    : distilbert-base-uncased-distilled-squad

Run:
    streamlit run app/app.py
=============================================================================
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Allow src/ imports when running from the app/ directory
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from model import QAModel  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration – must be the very first Streamlit command
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent QA System – WHEELTRIX",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS – professional, clean layout
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Main background */
    .main { background-color: #f8f9fb; }

    /* Title banner */
    .title-banner {
        background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #3949ab 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: #ffffff;
    }
    .title-banner h1 { font-size: 2rem; font-weight: 700; margin: 0; color: #ffffff; }
    .title-banner p  { font-size: 1rem; margin: 0.4rem 0 0 0; color: #c5cae9; }

    /* Card containers */
    .card {
        background: #ffffff;
        border: 1px solid #e8eaf6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Answer box */
    .answer-box {
        background: #e8f5e9;
        border-left: 5px solid #43a047;
        border-radius: 6px;
        padding: 1rem 1.5rem;
        font-size: 1.15rem;
        font-weight: 600;
        color: #1b5e20;
        margin-top: 0.5rem;
    }

    /* Confidence bar */
    .conf-label { font-size: 0.85rem; color: #546e7a; margin-top: 0.5rem; }

    /* Example chip */
    .chip {
        display: inline-block;
        background: #e3f2fd;
        border: 1px solid #90caf9;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.8rem;
        color: #1565c0;
        margin: 2px;
    }

    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #283593;
        border-bottom: 2px solid #e8eaf6;
        padding-bottom: 4px;
        margin-bottom: 12px;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.75rem;
        color: #90a4ae;
        margin-top: 2rem;
        border-top: 1px solid #e0e0e0;
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model() -> QAModel:
    """Load and cache the QA model (runs once per session)."""
    qa = QAModel()
    qa.load_pretrained()
    return qa

# ---------------------------------------------------------------------------
# Example Q&A pairs
# ---------------------------------------------------------------------------
EXAMPLES = [
    {
        "label": "🌐 Technology",
        "question": "What is the Transformer architecture?",
        "context": (
            "The Transformer architecture was introduced by Vaswani et al. in the "
            "2017 paper 'Attention Is All You Need'. It relies entirely on a "
            "self-attention mechanism to draw global dependencies between input "
            "and output, dispensing with recurrence and convolutions entirely. "
            "The Transformer has become the dominant architecture for natural "
            "language processing tasks."
        ),
    },
    {
        "label": "🤖 AI Model",
        "question": "How much faster is DistilBERT than BERT?",
        "context": (
            "DistilBERT is a smaller, faster, cheaper and lighter version of BERT. "
            "It has 40% fewer parameters than bert-base-uncased, runs 60% faster "
            "while preserving over 95% of BERT's performance as measured on the "
            "GLUE language understanding benchmark."
        ),
    },
    {
        "label": "📚 Dataset",
        "question": "How many question-answer pairs does SQuAD 1.1 contain?",
        "context": (
            "The Stanford Question Answering Dataset (SQuAD) is a reading "
            "comprehension dataset consisting of questions posed by crowdworkers "
            "on a set of Wikipedia articles. SQuAD 1.1 contains 100,000+ "
            "question-answer pairs on 500+ articles."
        ),
    },
    {
        "label": "🐍 Programming",
        "question": "Who created Python?",
        "context": (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability with significant "
            "indentation. Python was created by Guido van Rossum and first "
            "released in 1991."
        ),
    },
]

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.markdown("## 🔧 About the Project")
        st.markdown(
            """
            **Project Title**  
            Intelligent User Question Answering System Using NLP and Transformer-Based ML

            ---
            **Dataset**  
            SQuAD 1.1 (Stanford QA Dataset)  
            87,599 train / 10,570 dev QA pairs  
            Wikipedia-sourced passages

            ---
            **Model**  
            `distilbert-base-uncased-distilled-squad`  
            Fine-tuned on SQuAD 1.1 by Hugging Face

            ---
            **Technologies**
            - Python 3.10+
            - Hugging Face Transformers
            - PyTorch
            - Streamlit

            ---
            **Submitted to**  
            WHEELTRIX – HR Department

            ---
            **Internship Domain**  
            Data Science + Machine Learning + NLP
            """
        )

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        max_len = st.slider(
            "Max token length", min_value=128, max_value=512, value=384, step=64
        )
        show_confidence = st.checkbox("Show confidence score", value=True)
        show_highlight = st.checkbox("Highlight answer in context", value=True)

    return max_len, show_confidence, show_highlight

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    # Header
    st.markdown(
        """
        <div class="title-banner">
            <h1>🤖 Intelligent User Question Answering System</h1>
            <p>
                Powered by DistilBERT · Trained on SQuAD 1.1 · 
                Submitted to WHEELTRIX HR Department
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    max_len, show_confidence, show_highlight = render_sidebar()

    # ── Load model ────────────────────────────────────────────────────────
    with st.spinner("Loading AI model … (first load may take ~30 seconds)"):
        try:
            model = load_model()
            st.success("✅ Model loaded successfully.")
        except Exception as e:
            st.error(
                f"❌ Failed to load model: {e}\n\n"
                "Please ensure you have internet access and transformers installed:\n"
                "`pip install transformers torch`"
            )
            st.stop()

    # ── Layout: two columns ───────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown('<div class="section-header">📄 Input</div>', unsafe_allow_html=True)

        # Session-state initialisation
        if "context_val" not in st.session_state:
            st.session_state.context_val = ""
        if "question_val" not in st.session_state:
            st.session_state.question_val = ""

        context = st.text_area(
            "Context / Passage",
            value=st.session_state.context_val,
            height=220,
            placeholder=(
                "Paste or type a paragraph here. "
                "The system will extract the answer from this text."
            ),
            key="context_input",
        )

        question = st.text_input(
            "Your Question",
            value=st.session_state.question_val,
            placeholder="Type your question about the context above …",
            key="question_input",
        )

        btn_col1, btn_col2 = st.columns([1, 1])
        get_answer_btn = btn_col1.button("🔍 Get Answer", type="primary", use_container_width=True)
        clear_btn = btn_col2.button("🗑️ Clear", use_container_width=True)

        if clear_btn:
            st.session_state.context_val = ""
            st.session_state.question_val = ""
            st.rerun()

        # ── Validation & Prediction ───────────────────────────────────────
        if get_answer_btn:
            errors = []
            if not context.strip():
                errors.append("⚠️ Please provide a context/passage.")
            elif len(context.strip()) < 20:
                errors.append("⚠️ Context is too short. Please provide a meaningful passage.")
            elif len(context) > 10_000:
                errors.append("⚠️ Context exceeds 10 000 characters. Please shorten it.")

            if not question.strip():
                errors.append("⚠️ Please enter a question.")
            elif len(question.strip()) < 5:
                errors.append("⚠️ Question is too short.")
            elif len(question) > 500:
                errors.append("⚠️ Question exceeds 500 characters.")

            if errors:
                for err in errors:
                    st.warning(err)
            else:
                with st.spinner("Extracting answer …"):
                    t0 = time.time()
                    result = model.predict(
                        question=question.strip(),
                        context=context.strip(),
                        max_length=max_len,
                    )
                    elapsed = time.time() - t0

                answer = result["answer"]
                confidence = result["score"]

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
                            answer,
                            f"**:green[{answer}]**",
                            1,
                        )
                        st.markdown("#### 📌 Answer in Context")
                        st.markdown(highlighted)

    # ── Right column: Examples ────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-header">💡 Try an Example</div>', unsafe_allow_html=True)
        st.markdown(
            "Click any example below to populate the input fields."
        )

        for ex in EXAMPLES:
            with st.expander(ex["label"]):
                st.markdown(f"**Q:** {ex['question']}")
                st.markdown(
                    f"**Context snippet:** {ex['context'][:120]}…"
                    if len(ex["context"]) > 120
                    else f"**Context:** {ex['context']}"
                )
                if st.button(f"Use this example", key=f"btn_{ex['label']}"):
                    st.session_state.context_val = ex["context"]
                    st.session_state.question_val = ex["question"]
                    st.rerun()

        # Technology stack card
        st.markdown("---")
        st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)
        tech_data = {
            "Component": ["Language", "Framework", "Model", "Dataset", "UI"],
            "Technology": ["Python 3.10+", "Hugging Face Transformers", "DistilBERT", "SQuAD 1.1", "Streamlit"],
        }
        import pandas as pd
        st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    # ── How it works ──────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("ℹ️ How does this system work?"):
        st.markdown(
            """
            **Extractive Question Answering Pipeline:**

            1. **Input** – User provides a passage (context) and a question.
            2. **Tokenisation** – The text is converted to token IDs using the DistilBERT tokeniser.
            3. **Model** – `distilbert-base-uncased-distilled-squad` processes the combined  
               `[CLS] question [SEP] context [SEP]` sequence.
            4. **Span Prediction** – The model outputs a start-token logit and an end-token logit  
               for every position; the span with the highest combined score is selected.
            5. **Answer Extraction** – The character span corresponding to the predicted token  
               positions is extracted from the original context.
            6. **Confidence** – A normalised score derived from the logit magnitudes is displayed.

            **Model:** `distilbert-base-uncased-distilled-squad`  
            - 66 M parameters (vs 110 M for BERT-Base)  
            - 60% faster inference than BERT  
            - Pre-trained on SQuAD 1.1 by Hugging Face  
            - Exact Match ≈ 79–80%  |  F1 Score ≈ 86–87% (on full SQuAD 1.1 dev set)
            """
        )

    # Footer
    st.markdown(
        '<div class="footer">'
        "Intelligent User Question Answering System · "
        "Internship Project · WHEELTRIX HR Submission · "
        "Powered by Hugging Face Transformers &amp; Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
