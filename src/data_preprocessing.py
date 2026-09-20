"""
=============================================================================
data_preprocessing.py
-----------------------------------------------------------------------------
Intelligent User Question Answering System
Dataset: Stanford Question Answering Dataset (SQuAD) 1.1

Responsibilities
----------------
1. Download / load raw SQuAD 1.1 JSON files
2. Parse the deeply-nested JSON structure into a flat Pandas DataFrame
3. Basic data-quality checks (missing values, duplicates)
4. Light text normalisation
5. Tokenise contexts and questions with the DistilBERT tokeniser
6. Map answer character positions → token start / end positions
7. Save processed artefacts to  data/processed/
=============================================================================
"""

import os
import re
import json
import logging
import requests
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import DistilBertTokenizerFast

# Force UTF-8 output on Windows so Unicode characters print correctly
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# SQuAD 1.1 download URLs (official Rajpurkar et al. releases)
# ---------------------------------------------------------------------------
SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}

MODEL_NAME = "distilbert-base-uncased-distilled-squad"
MAX_LENGTH = 384          # maximum token length per sample
DOC_STRIDE = 128          # stride when context is split into windows


# ===========================================================================
# 1. Download
# ===========================================================================

def download_squad(split: str = "train") -> Path:
    """
    Download the SQuAD 1.1 JSON file for the given split if not already present.

    Parameters
    ----------
    split : str
        Either "train" or "dev".

    Returns
    -------
    Path
        Local file path of the downloaded JSON.
    """
    assert split in SQUAD_URLS, f"split must be 'train' or 'dev', got '{split}'"
    url = SQUAD_URLS[split]
    dest = RAW_DIR / f"{split}-v1.1.json"

    if dest.exists():
        logger.info("Found cached file: %s", dest)
        return dest

    logger.info("Downloading SQuAD 1.1 %s split …", split)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    logger.info("Saved → %s  (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return dest


# ===========================================================================
# 2. Parse JSON → flat DataFrame
# ===========================================================================

def parse_squad_json(filepath: Path, split_name: str = "train") -> pd.DataFrame:
    """
    Parse the SQuAD 1.1 JSON file into a flat DataFrame.

    SQuAD JSON hierarchy
    --------------------
    data                          ← list of topics / articles
      └─ title                    ← article title (e.g. "Beyoncé")
      └─ paragraphs               ← list of paragraphs for this article
           └─ context             ← the passage text
           └─ qas                 ← list of question-answer pairs
                └─ id             ← unique QA identifier
                └─ question       ← question string
                └─ answers        ← list of answer dicts
                     └─ text      ← answer string
                     └─ answer_start  ← character offset in context

    Returns
    -------
    pd.DataFrame with columns:
        split, title, context, question_id, question,
        answer_text, answer_start
    """
    logger.info("Parsing %s …", filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for article in tqdm(raw["data"], desc=f"Parsing {split_name}"):
        title = article.get("title", "")
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                q_id = qa["id"]
                question = qa["question"]
                answers = qa.get("answers", [])
                if answers:
                    # SQuAD 1.1 provides one canonical answer per QA pair
                    # (multiple annotations are present; we keep the first)
                    answer_text = answers[0]["text"]
                    answer_start = answers[0]["answer_start"]
                else:
                    answer_text = ""
                    answer_start = -1

                records.append(
                    {
                        "split": split_name,
                        "title": title,
                        "context": context,
                        "question_id": q_id,
                        "question": question,
                        "answer_text": answer_text,
                        "answer_start": answer_start,
                    }
                )

    df = pd.DataFrame(records)
    logger.info(
        "  %s split → %d QA pairs across %d unique contexts",
        split_name,
        len(df),
        df["context"].nunique(),
    )
    return df


# ===========================================================================
# 3. Data quality checks
# ===========================================================================

def check_data_quality(df: pd.DataFrame, split_name: str) -> None:
    """
    Log missing-value counts and duplicate counts for the given DataFrame.
    No rows are silently dropped; findings are reported to the caller.
    """
    logger.info("--- Data quality report: %s ---", split_name)

    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values found:\n%s", missing[missing > 0])
    else:
        logger.info("  No missing values found.")

    # Duplicate question IDs
    dup_ids = df.duplicated(subset=["question_id"]).sum()
    logger.info("  Duplicate question IDs : %d", dup_ids)

    # Empty answers
    empty_answers = (df["answer_text"] == "").sum()
    logger.info("  Empty answer texts     : %d", empty_answers)

    logger.info(
        "  Rows: %d  |  Unique contexts: %d  |  Unique questions: %d",
        len(df),
        df["context"].nunique(),
        df["question"].nunique(),
    )


# ===========================================================================
# 4. Compute derived statistics columns
# ===========================================================================

def add_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add word-count columns used in EDA and evaluation.
    """
    df = df.copy()
    df["question_word_count"] = df["question"].str.split().str.len()
    df["context_word_count"] = df["context"].str.split().str.len()
    df["answer_word_count"] = df["answer_text"].str.split().str.len().fillna(0).astype(int)
    df["question_char_count"] = df["question"].str.len()
    df["context_char_count"] = df["context"].str.len()
    df["answer_char_count"] = df["answer_text"].str.len()
    return df


# ===========================================================================
# 5. Light text normalisation
# ===========================================================================

def normalize_text(text: str) -> str:
    """
    Apply minimal normalisation:
      - strip leading / trailing whitespace
      - collapse multiple internal whitespace characters to a single space
      - no lowercasing (the tokeniser handles that)
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply normalize_text to question and context columns.
    answer_text is NOT normalised so character positions stay valid.
    """
    df = df.copy()
    df["question"] = df["question"].map(normalize_text)
    df["context"] = df["context"].map(normalize_text)
    return df


# ===========================================================================
# 6. Tokenisation + answer token-position mapping
# ===========================================================================

def get_tokenizer() -> DistilBertTokenizerFast:
    """
    Load the DistilBERT fast tokeniser (required for offset_mapping support).
    """
    logger.info("Loading tokeniser: %s", MODEL_NAME)
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    return tokenizer


def encode_sample(
    tokenizer: DistilBertTokenizerFast,
    question: str,
    context: str,
    answer_text: str,
    answer_start: int,
    max_length: int = MAX_LENGTH,
    doc_stride: int = DOC_STRIDE,
) -> dict:
    """
    Tokenise a single (question, context) pair and map the answer character
    span to token start / end positions.

    Returns a dict with:
        input_ids, attention_mask, start_positions, end_positions
    """
    encoding = tokenizer(
        question,
        context,
        max_length=max_length,
        truncation="only_second",   # truncate context, never the question
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # answer character span
    answer_end = answer_start + len(answer_text)

    # offset_mapping maps each token → (char_start, char_end) in the original string
    offset_mapping = encoding["offset_mapping"]
    sequence_ids = encoding.sequence_ids(0)       # 0=question, 1=context, None=special

    start_position = 0
    end_position = 0

    # Walk through tokens to find start / end positions inside context tokens
    for idx, (offset, seq_id) in enumerate(zip(offset_mapping[0], sequence_ids)):
        if seq_id != 1:
            continue  # skip question tokens and special tokens
        token_start, token_end = offset
        if token_start <= answer_start < token_end:
            start_position = idx
        if token_start < answer_end <= token_end:
            end_position = idx

    return {
        "input_ids": encoding["input_ids"][0],
        "attention_mask": encoding["attention_mask"][0],
        "start_positions": start_position,
        "end_positions": end_position,
    }


def tokenize_dataframe(
    df: pd.DataFrame,
    tokenizer: DistilBertTokenizerFast,
    sample_size: int | None = None,
) -> list[dict]:
    """
    Tokenise every row in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: question, context, answer_text, answer_start.
    tokenizer : DistilBertTokenizerFast
    sample_size : int or None
        If provided, tokenise only the first *sample_size* rows (useful for
        quick experiments on a regular laptop).

    Returns
    -------
    list[dict]  – one dict per example with keys:
        input_ids, attention_mask, start_positions, end_positions
    """
    subset = df.head(sample_size) if sample_size else df
    logger.info("Tokenising %d examples …", len(subset))

    encoded_samples = []
    for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Tokenising"):
        sample = encode_sample(
            tokenizer=tokenizer,
            question=row["question"],
            context=row["context"],
            answer_text=row["answer_text"],
            answer_start=int(row["answer_start"]),
        )
        encoded_samples.append(sample)

    logger.info("Tokenisation complete → %d samples", len(encoded_samples))
    return encoded_samples


# ===========================================================================
# 7. Save / load helpers
# ===========================================================================

def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    logger.info("Saved DataFrame → %s", path)


def load_dataframe(filename: str) -> pd.DataFrame:
    path = PROCESSED_DIR / filename
    df = pd.read_csv(path)
    logger.info("Loaded DataFrame ← %s  (%d rows)", path, len(df))
    return df


# ===========================================================================
# 8. Orchestration – run full pipeline
# ===========================================================================

def run_preprocessing_pipeline(
    download: bool = True,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the complete preprocessing pipeline.

    Steps
    -----
    1. Download (or load) SQuAD 1.1 JSON files
    2. Parse JSON → DataFrames
    3. Data-quality checks
    4. Normalisation
    5. Add length columns
    6. Save processed CSVs
    7. (Optionally) Tokenise a sample

    Parameters
    ----------
    download : bool
        If True, download JSON files when not already present.
    sample_size : int or None
        Tokenise only this many rows (None = all rows).

    Returns
    -------
    (train_df, dev_df)
    """
    # -- Step 1 : Load JSON --------------------------------------------------
    if download:
        train_path = download_squad("train")
        dev_path = download_squad("dev")
    else:
        train_path = RAW_DIR / "train-v1.1.json"
        dev_path = RAW_DIR / "dev-v1.1.json"
        assert train_path.exists(), f"Not found: {train_path}"
        assert dev_path.exists(), f"Not found: {dev_path}"

    # -- Step 2 : Parse ------------------------------------------------------
    train_df = parse_squad_json(train_path, "train")
    dev_df = parse_squad_json(dev_path, "dev")

    # -- Step 3 : Quality checks ---------------------------------------------
    check_data_quality(train_df, "train")
    check_data_quality(dev_df, "dev")

    # -- Step 4 : Normalise --------------------------------------------------
    train_df = normalize_dataframe(train_df)
    dev_df = normalize_dataframe(dev_df)

    # -- Step 5 : Length columns ---------------------------------------------
    train_df = add_length_columns(train_df)
    dev_df = add_length_columns(dev_df)

    # -- Step 6 : Save -------------------------------------------------------
    save_dataframe(train_df, "squad_train.csv")
    save_dataframe(dev_df, "squad_dev.csv")

    logger.info("Preprocessing complete.")
    logger.info("  Train rows : %d", len(train_df))
    logger.info("  Dev rows   : %d", len(dev_df))

    return train_df, dev_df


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    train_df, dev_df = run_preprocessing_pipeline(download=True, sample_size=None)
    print("\n--- Train DataFrame head ---")
    print(train_df[["question", "answer_text", "context"]].head(3).to_string(index=False))
    print("\n--- Dev DataFrame head ---")
    print(dev_df[["question", "answer_text", "context"]].head(3).to_string(index=False))
