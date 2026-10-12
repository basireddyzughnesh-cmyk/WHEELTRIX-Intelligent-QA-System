"""Business logic for the QA application.

Handles validation, prediction orchestration, and model management.
"""

from __future__ import annotations

import time
from typing import Any

from backend.model import QAModel


def validate_question_input(context: str, question: str) -> list[str]:
    """Return a list of validation errors."""
    errors: list[str] = []

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

    return errors


def load_qa_model() -> QAModel:
    """Load and initialize the QA model."""
    qa = QAModel()
    qa.load_pretrained()
    return qa


def run_prediction(
    model: QAModel, question: str, context: str, max_length: int
) -> dict[str, Any]:
    """Run a prediction and return model output plus elapsed time."""
    start = time.time()
    result = model.predict(
        question=question.strip(),
        context=context.strip(),
        max_length=max_length,
    )
    elapsed = time.time() - start
    result["elapsed_time"] = elapsed
    return result
