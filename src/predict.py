"""
=============================================================================
predict.py
-----------------------------------------------------------------------------
Intelligent User Question Answering System
Dataset: Stanford Question Answering Dataset (SQuAD) 1.1

Provides:
  - Single-question prediction (interactive CLI)
  - Batch prediction over a CSV file
  - Demo mode: run predictions over curated example questions
=============================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import QAModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Curated demo examples (context + question pairs based on SQuAD-like passages)
# ---------------------------------------------------------------------------
DEMO_EXAMPLES = [
    {
        "context": (
            "The Stanford Question Answering Dataset (SQuAD) is a reading "
            "comprehension dataset consisting of questions posed by crowdworkers "
            "on a set of Wikipedia articles, where the answer to every question "
            "is a segment of text, or span, from the corresponding reading passage, "
            "or the question might be unanswerable. SQuAD 1.1 contains 100,000+ "
            "question-answer pairs on 500+ articles."
        ),
        "question": "What is SQuAD?",
    },
    {
        "context": (
            "DistilBERT is a smaller, faster, cheaper and lighter version of BERT. "
            "It has 40% fewer parameters than bert-base-uncased, runs 60% faster "
            "while preserving over 95% of BERT's performance as measured on the "
            "GLUE language understanding benchmark. DistilBERT was introduced by "
            "Hugging Face in 2019."
        ),
        "question": "How many fewer parameters does DistilBERT have compared to BERT?",
    },
    {
        "context": (
            "The Transformer architecture was introduced by Vaswani et al. in the "
            "2017 paper 'Attention Is All You Need'. It relies entirely on a "
            "self-attention mechanism to draw global dependencies between input "
            "and output, dispensing with recurrence and convolutions entirely. "
            "The Transformer has become the dominant architecture for natural "
            "language processing tasks."
        ),
        "question": "What paper introduced the Transformer architecture?",
    },
    {
        "context": (
            "Python is a high-level, general-purpose programming language. Its "
            "design philosophy emphasizes code readability with the use of "
            "significant indentation. Python is dynamically typed and garbage-"
            "collected. It supports multiple programming paradigms, including "
            "structured, object-oriented and functional programming. "
            "Python was created by Guido van Rossum and first released in 1991."
        ),
        "question": "Who created Python?",
    },
    {
        "context": (
            "Natural Language Processing (NLP) is a subfield of linguistics, "
            "computer science, and artificial intelligence concerned with the "
            "interactions between computers and human language, in particular "
            "how to program computers to process and analyse large amounts of "
            "natural language data. Key NLP tasks include text classification, "
            "named entity recognition, machine translation, sentiment analysis, "
            "and question answering."
        ),
        "question": "What are some key NLP tasks?",
    },
]


# ===========================================================================
# Single prediction
# ===========================================================================

def predict_single(
    model: QAModel,
    question: str,
    context: str,
    verbose: bool = True,
) -> dict:
    """
    Run inference for a single (question, context) pair.

    Returns
    -------
    dict  with keys: question, context, answer, score
    """
    result = model.predict(question, context)

    if verbose:
        print("\n" + "═" * 60)
        print("  Question  :", question)
        print("  Answer    :", result["answer"])
        print("  Confidence:", f"{result['score']:.4f}")
        print("═" * 60)

    return {
        "question": question,
        "context": context,
        "answer": result["answer"],
        "confidence": result["score"],
    }


# ===========================================================================
# Batch prediction from CSV
# ===========================================================================

def predict_batch(
    model: QAModel,
    input_csv: str,
    output_csv: str | None = None,
) -> pd.DataFrame:
    """
    Run predictions over a CSV file containing 'question' and 'context' columns.

    Parameters
    ----------
    model      : QAModel
    input_csv  : str  path to input CSV
    output_csv : str or None  – if given, save results here

    Returns
    -------
    pd.DataFrame  with original columns plus 'predicted_answer' and 'confidence'
    """
    df = pd.read_csv(input_csv)
    required = {"question", "context"}
    assert required.issubset(df.columns), (
        f"Input CSV must contain columns: {required}"
    )

    logger.info("Running batch prediction on %d rows …", len(df))

    answers = []
    confidences = []

    for _, row in df.iterrows():
        result = model.predict(row["question"], row["context"])
        answers.append(result["answer"])
        confidences.append(result["score"])

    df["predicted_answer"] = answers
    df["confidence"] = confidences

    if output_csv:
        df.to_csv(output_csv, index=False)
        logger.info("Batch results saved → %s", output_csv)

    return df


# ===========================================================================
# Demo mode
# ===========================================================================

def run_demo(model: QAModel) -> list[dict]:
    """
    Run predictions on the built-in DEMO_EXAMPLES and save to JSON.
    """
    logger.info("Running demo predictions …")
    demo_results = []

    for i, ex in enumerate(DEMO_EXAMPLES, start=1):
        print(f"\n--- Demo {i}/{len(DEMO_EXAMPLES)} ---")
        result = predict_single(model, ex["question"], ex["context"], verbose=True)
        demo_results.append(result)

    # Save demo output
    demo_path = RESULTS_DIR / "demo_predictions.json"
    with open(demo_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "question": r["question"],
                    "answer": r["answer"],
                    "confidence": r["confidence"],
                }
                for r in demo_results
            ],
            f,
            indent=2,
        )
    logger.info("Demo predictions saved → %s", demo_path)
    return demo_results


# ===========================================================================
# Interactive CLI
# ===========================================================================

def interactive_mode(model: QAModel) -> None:
    """
    Simple REPL: user pastes a context, then types a question.
    Type 'quit' to exit.
    """
    print("\n" + "=" * 60)
    print("  Intelligent Question Answering System – Interactive Mode")
    print("  Type 'quit' to exit.")
    print("=" * 60)

    while True:
        print("\nPaste your context/passage (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        context = " ".join(lines).strip()

        if context.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not context:
            print("Context cannot be empty. Try again.")
            continue

        question = input("\nEnter your question: ").strip()
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if not question:
            print("Question cannot be empty. Try again.")
            continue

        predict_single(model, question, context, verbose=True)


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run QA predictions using DistilBERT on SQuAD 1.1"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "interactive", "batch"],
        help=(
            "demo        – Run built-in example predictions\n"
            "interactive – REPL for custom context + question\n"
            "batch       – Predict over a CSV file (requires --input_csv)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pretrained",
        choices=["pretrained", "local"],
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default=None,
        help="Path to input CSV for batch mode.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(RESULTS_DIR / "batch_predictions.csv"),
        help="Path to save batch predictions.",
    )
    args = parser.parse_args()

    # Load model
    qa = QAModel()
    if args.model == "pretrained":
        qa.load_pretrained()
    else:
        qa.load_local()

    if args.mode == "demo":
        run_demo(qa)
    elif args.mode == "interactive":
        interactive_mode(qa)
    elif args.mode == "batch":
        assert args.input_csv, "--input_csv is required for batch mode."
        predict_batch(qa, args.input_csv, args.output_csv)
