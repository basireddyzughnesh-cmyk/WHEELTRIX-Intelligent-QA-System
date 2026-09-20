"""
=============================================================================
evaluate.py
-----------------------------------------------------------------------------
Intelligent User Question Answering System
Dataset: Stanford Question Answering Dataset (SQuAD) 1.1

Official SQuAD 1.1 evaluation metrics:
  1. Exact Match (EM) – 1 if the predicted span exactly matches any gold answer
  2. F1 Score         – token-level overlap between prediction and best gold answer

The metric implementations follow the official SQuAD evaluation script
(Rajpurkar et al., 2016):
    https://rajpurkar.github.io/SQuAD-explorer/

Usage
-----
# Evaluate on dev split (default 500 examples – adjust with --sample):
python src/evaluate.py --sample 500

# Evaluate on the full dev split (takes ~20–40 min on CPU):
python src/evaluate.py
=============================================================================
"""

import argparse
import json
import logging
import re
import string
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import (
    download_squad,
    parse_squad_json,
    normalize_dataframe,
    RAW_DIR,
)
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


# ===========================================================================
# Official SQuAD metric helpers
# ===========================================================================

def normalize_answer(s: str) -> str:
    """
    Lower-case, remove punctuation, remove articles, and collapse whitespace.
    This is the exact normalisation used in the official SQuAD evaluation script.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list[str]:
    """Return normalised token list for F1 calculation."""
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(gold_answer: str, predicted: str) -> int:
    """Return 1 if prediction matches gold after normalisation, else 0."""
    return int(normalize_answer(gold_answer) == normalize_answer(predicted))


def compute_f1(gold_answer: str, predicted: str) -> float:
    """
    Token-level F1 score between the gold answer and the prediction.
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    gold_tokens = get_tokens(gold_answer)
    pred_tokens = get_tokens(predicted)

    common = Counter(gold_tokens) & Counter(pred_tokens)
    num_same = sum(common.values())

    if not gold_tokens or not pred_tokens:
        return int(gold_tokens == pred_tokens)

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def best_exact_and_f1(gold_answers: list[str], predicted: str) -> tuple[int, float]:
    """
    Return the maximum EM and F1 over all gold answer strings.
    SQuAD provides one canonical answer per QA in v1.1, but the
    original evaluation script treats all annotator answers; this
    function supports both cases.
    """
    exact = max(compute_exact(gold, predicted) for gold in gold_answers)
    f1 = max(compute_f1(gold, predicted) for gold in gold_answers)
    return exact, f1


# ===========================================================================
# Evaluation runner
# ===========================================================================

def evaluate_model(
    model: QAModel,
    dev_df: pd.DataFrame,
    sample_size: int | None = None,
) -> dict:
    """
    Run model predictions over the dev split and compute aggregate metrics.

    Parameters
    ----------
    model : QAModel  (must have load_pretrained() or load_local() already called)
    dev_df : pd.DataFrame  (from parse_squad_json for split='dev')
    sample_size : int or None  – evaluate only first N rows

    Returns
    -------
    dict with keys:
        exact_match   – float (0–100)
        f1            – float (0–100)
        num_examples  – int
        predictions   – list[dict]  (per-sample results)
    """
    subset = dev_df.head(sample_size) if sample_size else dev_df
    logger.info("Evaluating on %d examples …", len(subset))

    exact_scores = []
    f1_scores = []
    predictions = []

    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Evaluating"):
        question = row["question"]
        context = row["context"]
        gold_answer = row["answer_text"]

        result = model.predict(question, context)
        predicted = result["answer"]

        exact, f1 = best_exact_and_f1([gold_answer], predicted)
        exact_scores.append(exact)
        f1_scores.append(f1)

        predictions.append(
            {
                "question_id": row.get("question_id", ""),
                "question": question,
                "context_snippet": context[:200] + "…" if len(context) > 200 else context,
                "gold_answer": gold_answer,
                "predicted_answer": predicted,
                "exact_match": exact,
                "f1_score": round(f1, 4),
                "confidence": result["score"],
            }
        )

    aggregate = {
        "exact_match": round(100.0 * sum(exact_scores) / len(exact_scores), 2),
        "f1": round(100.0 * sum(f1_scores) / len(f1_scores), 2),
        "num_examples": len(subset),
        "predictions": predictions,
    }

    logger.info(
        "--- Evaluation Results ---\n"
        "  Examples    : %d\n"
        "  Exact Match : %.2f%%\n"
        "  F1 Score    : %.2f%%",
        aggregate["num_examples"],
        aggregate["exact_match"],
        aggregate["f1"],
    )
    return aggregate


# ===========================================================================
# Save results
# ===========================================================================

def save_results(results: dict, filename_stem: str = "evaluation_results") -> None:
    """Save aggregate metrics and top-20 sample predictions to disk."""
    # JSON – full predictions
    json_path = RESULTS_DIR / f"{filename_stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "exact_match": results["exact_match"],
                "f1": results["f1"],
                "num_examples": results["num_examples"],
                "sample_predictions": results["predictions"][:20],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Results saved → %s", json_path)

    # CSV – full predictions for inspection
    csv_path = RESULTS_DIR / f"{filename_stem}_predictions.csv"
    pd.DataFrame(results["predictions"]).to_csv(csv_path, index=False)
    logger.info("Predictions CSV saved → %s", csv_path)

    # Metrics summary text
    txt_path = RESULTS_DIR / f"{filename_stem}_summary.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("   SQuAD 1.1 Evaluation Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model         : distilbert-base-uncased-distilled-squad\n")
        f.write(f"Split         : dev\n")
        f.write(f"Examples      : {results['num_examples']}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Exact Match   : {results['exact_match']:.2f}%\n")
        f.write(f"F1 Score      : {results['f1']:.2f}%\n")
        f.write("=" * 50 + "\n")
    logger.info("Summary saved → %s", txt_path)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate QA model on SQuAD 1.1 dev split"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=500,
        help=(
            "Number of dev examples to evaluate. "
            "Use None/0 for the full dev set (~10 570 examples)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pretrained",
        choices=["pretrained", "local"],
        help="'pretrained' = HuggingFace checkpoint; 'local' = models/finetuned_squad_model",
    )
    args = parser.parse_args()
    sample_size = args.sample if args.sample and args.sample > 0 else None

    # Load model
    qa_model = QAModel()
    if args.model == "pretrained":
        qa_model.load_pretrained()
    else:
        qa_model.load_local()

    # Load dev data
    dev_json = download_squad("dev")
    dev_df = parse_squad_json(dev_json, "dev")
    dev_df = normalize_dataframe(dev_df)

    # Evaluate
    results = evaluate_model(qa_model, dev_df, sample_size=sample_size)

    # Save
    save_results(results)

    # Print sample predictions to console
    print("\n--- Sample Predictions (first 5) ---")
    for pred in results["predictions"][:5]:
        print(f"  Q  : {pred['question']}")
        print(f"  Gold : {pred['gold_answer']}")
        print(f"  Pred : {pred['predicted_answer']}")
        print(f"  EM={pred['exact_match']}  F1={pred['f1_score']:.4f}")
        print()
