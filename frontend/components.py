"""Components for the QA application sidebar and examples."""

from __future__ import annotations

import pandas as pd
import streamlit as st

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


def render_sidebar() -> tuple[int, bool, bool]:
    """Render the sidebar with settings and project info."""
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


def render_examples() -> None:
    """Render example Q&A pairs in the sidebar."""
    st.markdown('<div class="section-header">💡 Try an Example</div>', unsafe_allow_html=True)
    st.markdown("Click any example below to populate the input fields.")

    for ex in EXAMPLES:
        with st.expander(ex["label"]):
            st.markdown(f"**Q:** {ex['question']}")
            st.markdown(
                f"**Context snippet:** {ex['context'][:120]}…"
                if len(ex["context"]) > 120
                else f"**Context:** {ex['context']}"
            )
            if st.button("Use this example", key=f"btn_{ex['label']}"):
                st.session_state["context_input"] = ex["context"]
                st.session_state["question_input"] = ex["question"]
                st.rerun()


def render_technology_stack() -> None:
    """Render the technology stack table."""
    st.markdown("---")
    st.markdown(
        '<div class="section-header">🛠️ Technology Stack</div>',
        unsafe_allow_html=True,
    )
    tech_data = {
        "Component": ["Language", "Framework", "Model", "Dataset", "UI"],
        "Technology": [
            "Python 3.10+",
            "Hugging Face Transformers",
            "DistilBERT",
            "SQuAD 1.1",
            "Streamlit",
        ],
    }
    st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)


def render_how_it_works() -> None:
    """Render the 'How it works' explanation."""
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
