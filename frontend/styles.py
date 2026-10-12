"""Streamlit UI components and styling for the QA application."""

from __future__ import annotations

import streamlit as st


def render_css() -> None:
    """Apply custom CSS styling to the Streamlit app."""
    st.markdown(
        """
        <style>
        .main { background-color: #f8f9fb; }
        .title-banner {
            background: linear-gradient(135deg, #1a237e 0%, #283593 60%, #3949ab 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            color: #ffffff;
        }
        .title-banner h1 { font-size: 2rem; font-weight: 700; margin: 0; color: #ffffff; }
        .title-banner p { font-size: 1rem; margin: 0.4rem 0 0 0; color: #c5cae9; }
        .card {
            background: #ffffff;
            border: 1px solid #e8eaf6;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
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
        .conf-label { font-size: 0.85rem; color: #546e7a; margin-top: 0.5rem; }
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
        .section-header {
            font-size: 1rem;
            font-weight: 600;
            color: #283593;
            border-bottom: 2px solid #e8eaf6;
            padding-bottom: 4px;
            margin-bottom: 12px;
        }
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


def render_header() -> None:
    """Render the application header."""
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


def render_footer() -> None:
    """Render the application footer."""
    st.markdown(
        '<div class="footer">'
        "Intelligent User Question Answering System · "
        "Internship Project · WHEELTRIX HR Submission · "
        "Powered by Hugging Face Transformers &amp; Streamlit"
        "</div>",
        unsafe_allow_html=True,
    )
