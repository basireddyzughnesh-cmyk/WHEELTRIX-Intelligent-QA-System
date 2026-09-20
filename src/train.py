"""
=============================================================================
train.py
-----------------------------------------------------------------------------
Intelligent User Question Answering System
Dataset: Stanford Question Answering Dataset (SQuAD) 1.1

Fine-tunes (or continues training) the DistilBERT QA model on SQuAD 1.1.

HARDWARE NOTE
-------------
Full fine-tuning of DistilBERT on SQuAD 1.1 (~87 000 examples, 3 epochs)
typically requires:
  - GPU: ~20–30 minutes on a modern NVIDIA GPU
  - CPU: several hours on a laptop CPU

For reproducible results on a regular laptop WITHOUT a GPU this script
supports a *lightweight mode* (--sample flag) that fine-tunes on a small
subset (default: 2 000 training / 500 validation examples).

For demonstration / submission purposes the pretrained
distilbert-base-uncased-distilled-squad checkpoint is already fine-tuned on
SQuAD 1.1 by Hugging Face and produces strong results without any additional
training.  The evaluate.py and predict.py modules use this pretrained
checkpoint directly.  This train.py module is provided for completeness and
for users who wish to reproduce fine-tuning.

Usage
-----
# Full fine-tuning (requires GPU or significant time):
python src/train.py

# Lightweight fine-tuning on 2000 samples (runs on laptop CPU):
python src/train.py --sample 2000 --epochs 1
=============================================================================
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_preprocessing import (
    download_squad,
    parse_squad_json,
    normalize_dataframe,
    add_length_columns,
    tokenize_dataframe,
    RAW_DIR,
    PROCESSED_DIR,
)
from model import SQuADDataset, get_device, QAModel, MODELS_DIR

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
# Default hyper-parameters
# ---------------------------------------------------------------------------
DEFAULT_MODEL_NAME = "distilbert-base-uncased-distilled-squad"
DEFAULT_BATCH_SIZE = 8          # safe for 8 GB RAM on CPU
DEFAULT_EPOCHS = 2
DEFAULT_LEARNING_RATE = 3e-5
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_SAVE_NAME = "finetuned_squad_model"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Training loop
# ===========================================================================

def train(
    sample_size: int | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    save_name: str = DEFAULT_SAVE_NAME,
) -> None:
    """
    Fine-tune DistilBERT on SQuAD 1.1.

    Parameters
    ----------
    sample_size : int or None
        Number of training examples to use.  None = full dataset.
    epochs : int
    batch_size : int
    learning_rate : float
    save_name : str
        Name of the subdirectory under models/ where the model is saved.
    """
    device = get_device()

    # ── 1. Load / download data ──────────────────────────────────────────
    train_json = download_squad("train")
    dev_json = download_squad("dev")

    train_raw = parse_squad_json(train_json, "train")
    dev_raw = parse_squad_json(dev_json, "dev")

    train_raw = normalize_dataframe(train_raw)
    dev_raw = normalize_dataframe(dev_raw)

    # ── 2. Tokeniser ─────────────────────────────────────────────────────
    tokenizer = DistilBertTokenizerFast.from_pretrained(DEFAULT_MODEL_NAME)

    # Lightweight mode: use a subset
    if sample_size:
        logger.info(
            "Lightweight mode: using first %d train / %d dev examples.",
            sample_size,
            max(sample_size // 4, 100),
        )
        train_raw = train_raw.head(sample_size)
        dev_raw = dev_raw.head(max(sample_size // 4, 100))

    train_encoded = tokenize_dataframe(train_raw, tokenizer)
    dev_encoded = tokenize_dataframe(dev_raw, tokenizer)

    train_dataset = SQuADDataset(train_encoded)
    dev_dataset = SQuADDataset(dev_encoded)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # ── 3. Model ─────────────────────────────────────────────────────────
    logger.info("Loading model: %s", DEFAULT_MODEL_NAME)
    model = DistilBertForQuestionAnswering.from_pretrained(DEFAULT_MODEL_NAME)
    model.to(device)

    # ── 4. Optimiser + scheduler ─────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * DEFAULT_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info("Starting fine-tuning …")
    logger.info("  Epochs        : %d", epochs)
    logger.info("  Batch size    : %d", batch_size)
    logger.info("  Learning rate : %.2e", learning_rate)
    logger.info("  Total steps   : %d", total_steps)
    logger.info("  Device        : %s", device)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")

        for batch in progress:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)
        logger.info("Epoch %d  train_loss=%.4f", epoch, avg_train_loss)

        # ── Validation ───────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(dev_loader)
        logger.info("Epoch %d  val_loss=%.4f", epoch, avg_val_loss)

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = MODELS_DIR / save_name
            save_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(save_path))
            tokenizer.save_pretrained(str(save_path))
            logger.info("  ✓ Best model saved → %s", save_path)

    # Save training summary
    summary_path = RESULTS_DIR / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model       : {DEFAULT_MODEL_NAME}\n")
        f.write(f"Epochs      : {epochs}\n")
        f.write(f"Batch size  : {batch_size}\n")
        f.write(f"LR          : {learning_rate}\n")
        f.write(f"Train size  : {len(train_dataset)}\n")
        f.write(f"Dev size    : {len(dev_dataset)}\n")
        f.write(f"Best val loss: {best_val_loss:.4f}\n")
        f.write(f"Saved to    : {MODELS_DIR / save_name}\n")
    logger.info("Training summary saved → %s", summary_path)
    logger.info("Fine-tuning complete.")


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT on SQuAD 1.1"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help=(
            "Use only this many training examples. "
            "Omit to use the full SQuAD 1.1 training set."
        ),
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--save_name", type=str, default=DEFAULT_SAVE_NAME)
    args = parser.parse_args()

    train(
        sample_size=args.sample,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_name=args.save_name,
    )
