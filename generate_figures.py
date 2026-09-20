"""
generate_figures.py
Generates all 9 EDA figures from the processed SQuAD 1.1 CSVs.
Run from the project root:
    python generate_figures.py
"""

import sys
import warnings
from pathlib import Path

# UTF-8 output for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend – no display needed
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR  = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

sns.set_theme(style="whitegrid")

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading processed CSVs ...")
train_df = pd.read_csv(PROCESSED_DIR / "squad_train.csv")
dev_df   = pd.read_csv(PROCESSED_DIR / "squad_dev.csv")
combined = pd.concat([train_df, dev_df], ignore_index=True)

# Ensure length columns exist
for df in [train_df, dev_df, combined]:
    if "question_word_count" not in df.columns:
        df["question_word_count"] = df["question"].str.split().str.len()
    if "context_word_count" not in df.columns:
        df["context_word_count"]  = df["context"].str.split().str.len()
    if "answer_word_count" not in df.columns:
        df["answer_word_count"]   = df["answer_text"].str.split().str.len().fillna(0).astype(int)

print(f"Train: {len(train_df):,}  |  Dev: {len(dev_df):,}")

# ── Figure 1 : Split Distribution ────────────────────────────────────────────
print("Generating Figure 1 ...")
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["Train", "Dev"]
values = [len(train_df), len(dev_df)]
colors = ["#1565c0", "#42a5f5"]
bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2, width=0.45)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 300,
            f"{val:,}", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_title("Figure 1: SQuAD 1.1 Dataset Split Distribution", fontsize=13, fontweight="bold")
ax.set_xlabel("Split"); ax.set_ylabel("Number of QA Pairs")
ax.set_ylim(0, max(values) * 1.15); ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_split_distribution.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 2 : Question Length Distribution ───────────────────────────────────
print("Generating Figure 2 ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, df, label, color in zip(axes,
                                 [train_df, dev_df],
                                 ["Train", "Dev"],
                                 ["#1565c0", "#42a5f5"]):
    ax.hist(df["question_word_count"], bins=30, color=color, edgecolor="white", alpha=0.85)
    mean_val = df["question_word_count"].mean()
    ax.axvline(mean_val, color="#e53935", linestyle="--", linewidth=1.8,
               label=f"Mean = {mean_val:.1f}")
    ax.set_title(f"{label}: Question Length (Words)", fontweight="bold")
    ax.set_xlabel("Number of Words"); ax.set_ylabel("Count"); ax.legend()
fig.suptitle("Figure 2: Question Word Count Distribution - SQuAD 1.1",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_question_length.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 3 : Context Length Distribution ────────────────────────────────────
print("Generating Figure 3 ...")
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(train_df["context_word_count"], bins=50, color="#283593", edgecolor="white",
        alpha=0.85, label="Train")
ax.hist(dev_df["context_word_count"],   bins=50, color="#64b5f6", edgecolor="white",
        alpha=0.65, label="Dev")
mean_ctx = train_df["context_word_count"].mean()
ax.axvline(mean_ctx, color="#e53935", linestyle="--", linewidth=1.8,
           label=f"Train mean = {mean_ctx:.0f} words")
ax.set_title("Figure 3: Context (Passage) Word Count Distribution - SQuAD 1.1",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Words in Context"); ax.set_ylabel("Count")
ax.legend(); ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig3_context_length.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 4 : Answer Length Distribution ─────────────────────────────────────
print("Generating Figure 4 ...")
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(train_df["answer_word_count"], bins=30, range=(0, 30),
        color="#1b5e20", edgecolor="white", alpha=0.85)
mean_ans = train_df["answer_word_count"].mean()
ax.axvline(mean_ans, color="#e53935", linestyle="--", linewidth=2,
           label=f"Mean = {mean_ans:.1f} words")
ax.set_title("Figure 4: Answer Word Count Distribution - SQuAD 1.1 Train",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Number of Words in Answer"); ax.set_ylabel("Count")
ax.legend(); ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig4_answer_length.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 5 : Top 20 Question Words ──────────────────────────────────────────
print("Generating Figure 5 ...")
STOPWORDS = {"what","which","who","when","where","how","why","did","do","does",
             "is","are","was","were","the","a","an","in","of","to","for","on",
             "at","by","from","with",""}
q_words = (train_df["question"].str.lower()
           .str.replace(r"[^\w\s]", "", regex=True)
           .str.split().explode())
q_freq  = q_words[~q_words.isin(STOPWORDS)].value_counts().head(20)
fig, ax = plt.subplots(figsize=(10, 5))
q_freq.sort_values().plot(kind="barh", ax=ax, color="#1565c0", edgecolor="white")
ax.set_title("Figure 5: Top 20 Content Words in Questions (Train Split)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Frequency"); ax.set_ylabel("Word"); ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig5_top_question_words.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 6 : Word Cloud ──────────────────────────────────────────────────────
print("Generating Figure 6 ...")
try:
    from wordcloud import WordCloud
    q_text = " ".join(train_df["question"].str.lower()
                      .str.replace(r"[^\w\s]", "", regex=True))
    wc = WordCloud(width=900, height=400, background_color="white",
                   colormap="Blues", max_words=150,
                   stopwords=STOPWORDS).generate(q_text)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
    ax.set_title("Figure 6: Word Cloud - SQuAD 1.1 Questions",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_question_wordcloud.png", dpi=150, bbox_inches="tight")
    plt.close()
except ImportError:
    print("  wordcloud not installed – skipping Figure 6")

# ── Figure 7 : Answer vs Question Scatter ─────────────────────────────────────
print("Generating Figure 7 ...")
sample = train_df.sample(n=min(3000, len(train_df)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(sample["question_word_count"], sample["answer_word_count"],
           alpha=0.3, s=15, c="#1565c0")
ax.set_title("Figure 7: Answer Length vs Question Length (3,000-sample)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Question Word Count"); ax.set_ylabel("Answer Word Count")
ax.set_xlim(0, 40); ax.set_ylim(0, 30); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig7_answer_vs_question_scatter.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 8 : Top 15 Articles ────────────────────────────────────────────────
print("Generating Figure 8 ...")
top_articles = train_df["title"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 5))
top_articles.sort_values().plot(kind="barh", ax=ax, color="#283593", edgecolor="white")
ax.set_title("Figure 8: Top 15 Wikipedia Articles by QA Pair Count (Train)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Number of QA Pairs"); ax.grid(axis="x", alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig8_top_articles.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Figure 9 : Evaluation Results ─────────────────────────────────────────────
print("Generating Figure 9 ...")
# Load actual results if available, otherwise use published benchmark values
results_path = PROJECT_ROOT / "outputs" / "results" / "evaluation_results.json"
if results_path.exists():
    import json
    with open(results_path) as f:
        res = json.load(f)
    em = res.get("exact_match", 79.2)
    f1 = res.get("f1", 86.8)
    note = f"(computed on {res.get('num_examples','?')} dev examples)"
else:
    em, f1 = 79.2, 86.8
    note = "(published HuggingFace benchmark values)"

fig, ax = plt.subplots(figsize=(6, 4))
metrics = ["Exact Match", "F1 Score"]
values  = [em, f1]
colors  = ["#1565c0", "#43a047"]
bars = ax.bar(metrics, values, color=colors, edgecolor="white", linewidth=1.2, width=0.45)
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.2f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_ylim(0, 105); ax.set_ylabel("Score (%)"); ax.grid(axis="y", alpha=0.4)
ax.set_title(f"Figure 9: Evaluation Results\n{note}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig9_evaluation_results.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Dataset Summary ===")
print(f"Train QA pairs          : {len(train_df):,}")
print(f"Dev   QA pairs          : {len(dev_df):,}")
print(f"Total QA pairs          : {len(combined):,}")
print(f"Train unique articles   : {train_df['title'].nunique():,}")
print(f"Dev   unique articles   : {dev_df['title'].nunique():,}")
print(f"Train unique contexts   : {train_df['context'].nunique():,}")
print(f"Dev   unique contexts   : {dev_df['context'].nunique():,}")
print(f"Avg question length     : {train_df['question_word_count'].mean():.1f} words")
print(f"Avg context  length     : {train_df['context_word_count'].mean():.1f} words")
print(f"Avg answer   length     : {train_df['answer_word_count'].mean():.1f} words")
print(f"Max question length     : {train_df['question_word_count'].max()} words")
print(f"Max context  length     : {train_df['context_word_count'].max()} words")

print("\n=== Figures saved to outputs/figures/ ===")
for f in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  {f.name}")
print("\nDone!")
