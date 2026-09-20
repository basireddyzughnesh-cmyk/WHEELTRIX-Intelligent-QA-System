"""
=============================================================================
model.py
-----------------------------------------------------------------------------
Intelligent User Question Answering System
Dataset: Stanford Question Answering Dataset (SQuAD) 1.1

This module provides:
  - DistilBERT QA model wrapper (loading, saving, inference)
  - PyTorch Dataset class for SQuAD-formatted samples
  - Helper to move model between CPU / GPU
=============================================================================
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast,
)

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
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert-base-uncased-distilled-squad"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Device helper
# ===========================================================================

def get_device() -> torch.device:
    """
    Return GPU device if CUDA is available, otherwise CPU.
    DistilBERT inference on CPU is fast enough for a demo; fine-tuning on
    CPU is slow but functional for small sample sizes.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA available – using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available – using CPU.")
    return device


# ===========================================================================
# PyTorch Dataset
# ===========================================================================

class SQuADDataset(Dataset):
    """
    PyTorch Dataset wrapper around the list of tokenised SQuAD samples
    produced by data_preprocessing.tokenize_dataframe().

    Each item in *encoded_samples* is a dict with keys:
        input_ids        – List[int]  length MAX_LENGTH
        attention_mask   – List[int]  length MAX_LENGTH
        start_positions  – int
        end_positions    – int
    """

    def __init__(self, encoded_samples: list[dict]):
        self.samples = encoded_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "input_ids": torch.tensor(sample["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(sample["attention_mask"], dtype=torch.long),
            "start_positions": torch.tensor(sample["start_positions"], dtype=torch.long),
            "end_positions": torch.tensor(sample["end_positions"], dtype=torch.long),
        }


# ===========================================================================
# Model wrapper
# ===========================================================================

class QAModel:
    """
    Thin wrapper around DistilBertForQuestionAnswering.

    Usage
    -----
    # Load the pretrained HuggingFace model (no fine-tuning needed for demo):
    qa = QAModel()
    qa.load_pretrained()

    # Run inference:
    answer = qa.predict(question="Who invented Python?",
                        context="Python was created by Guido van Rossum …")

    # Save fine-tuned model:
    qa.save("my_finetuned_squad_model")

    # Load a previously saved model:
    qa.load_local("my_finetuned_squad_model")
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.device = get_device()
        self.tokenizer: Optional[DistilBertTokenizerFast] = None
        self.model: Optional[DistilBertForQuestionAnswering] = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_pretrained(self) -> None:
        """
        Download and load the pretrained  distilbert-base-uncased-distilled-squad
        model and its tokeniser from the Hugging Face Hub.

        distilbert-base-uncased-distilled-squad is already fine-tuned on
        SQuAD 1.1 by Hugging Face, so it can be used directly for inference.
        """
        logger.info("Loading pretrained model: %s", self.model_name)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded and moved to %s.", self.device)

    def load_local(self, save_name: str = "finetuned_squad_model") -> None:
        """
        Load a locally saved fine-tuned model from models/<save_name>/.
        """
        model_path = MODELS_DIR / save_name
        assert model_path.exists(), f"Model directory not found: {model_path}"
        logger.info("Loading local model from %s …", model_path)
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
        self.model = DistilBertForQuestionAnswering.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        logger.info("Local model loaded.")

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self, save_name: str = "finetuned_squad_model") -> None:
        """
        Save the current model and tokeniser to models/<save_name>/.
        """
        assert self.model is not None, "No model loaded."
        save_path = MODELS_DIR / save_name
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        logger.info("Model saved → %s", save_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        question: str,
        context: str,
        max_length: int = 384,
        doc_stride: int = 128,
        return_confidence: bool = False,
    ) -> dict:
        """
        Extract the most likely answer span from *context* given *question*.

        The method handles contexts longer than MAX_LENGTH by using a
        sliding-window approach (doc_stride).

        Parameters
        ----------
        question : str
        context  : str
        max_length : int
            Maximum token length of the encoded sequence.
        doc_stride : int
            Overlap between consecutive context windows.
        return_confidence : bool
            If True, also return a normalised confidence score
            (softmax probability of start × end token).

        Returns
        -------
        dict with keys:
            answer        – str
            score         – float  (confidence, 0–1)
            start_index   – int    (character offset in context)
            end_index     – int    (character offset in context)
        """
        assert self.model is not None, "Call load_pretrained() or load_local() first."

        # Tokenise with sliding window
        inputs = self.tokenizer(
            question,
            context,
            max_length=max_length,
            truncation="only_second",
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        offset_mapping = inputs.pop("offset_mapping")
        overflow_to_sample = inputs.pop("overflow_to_sample_mapping", None)

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        best_answer = ""
        best_score = -float("inf")
        best_char_start = 0
        best_char_end = 0

        with torch.no_grad():
            outputs = self.model(**inputs)

        start_logits = outputs.start_logits  # (num_windows, seq_len)
        end_logits = outputs.end_logits      # (num_windows, seq_len)

        for window_idx in range(start_logits.shape[0]):
            s_logits = start_logits[window_idx]
            e_logits = end_logits[window_idx]
            offsets = offset_mapping[window_idx]
            seq_ids = inputs["attention_mask"][window_idx]  # 1 for real tokens

            # Get the sequence_ids to identify context tokens
            token_type_ids = self.tokenizer(
                question,
                context,
                max_length=max_length,
                truncation="only_second",
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=False,
                padding="max_length",
            ).sequence_ids(window_idx)

            # Find valid context token range
            ctx_start = next(
                (i for i, t in enumerate(token_type_ids) if t == 1), None
            )
            ctx_end = (
                len(token_type_ids)
                - 1
                - next(
                    (i for i, t in enumerate(reversed(token_type_ids)) if t == 1), None
                )
            )

            if ctx_start is None:
                continue

            # Zero out logits outside the context
            s_logits = s_logits.cpu().numpy().copy()
            e_logits = e_logits.cpu().numpy().copy()
            s_logits[:ctx_start] = -float("inf")
            e_logits[:ctx_start] = -float("inf")
            s_logits[ctx_end + 1 :] = -float("inf")
            e_logits[ctx_end + 1 :] = -float("inf")

            # Best valid (start ≤ end) pair
            import numpy as np  # local import to avoid top-level dep on train path
            start_idx = int(np.argmax(s_logits))
            # Scan end positions from start_idx onward
            end_candidates = np.argsort(e_logits[start_idx:])[::-1]
            if len(end_candidates) == 0:
                continue
            end_idx = start_idx + int(end_candidates[0])

            score = float(s_logits[start_idx]) + float(e_logits[end_idx])

            if score > best_score:
                best_score = score
                char_start = int(offsets[start_idx][0])
                char_end = int(offsets[end_idx][1])
                best_answer = context[char_start:char_end]
                best_char_start = char_start
                best_char_end = char_end

        # Normalised confidence via softmax-style transform
        import math
        confidence = 1 / (1 + math.exp(-best_score / 10))

        return {
            "answer": best_answer.strip(),
            "score": round(confidence, 4),
            "start_index": best_char_start,
            "end_index": best_char_end,
        }


# ===========================================================================
# CLI self-test
# ===========================================================================

if __name__ == "__main__":
    qa = QAModel()
    qa.load_pretrained()

    context = (
        "The Stanford Question Answering Dataset (SQuAD) is a reading comprehension "
        "dataset consisting of questions posed by crowdworkers on a set of Wikipedia "
        "articles, where the answer to every question is a segment of text, or span, "
        "from the corresponding reading passage."
    )
    question = "What is SQuAD?"
    result = qa.predict(question, context)
    print(f"\nQuestion : {question}")
    print(f"Answer   : {result['answer']}")
    print(f"Score    : {result['score']}")
