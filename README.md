# 🤖 Intelligent User Question Answering System
### Using NLP and Transformer-Based Machine Learning

> **Internship Project – Data Science & Machine Learning**  
> **Submitted to:** WHEELTRIX HR Department  
> **Dataset:** Stanford Question Answering Dataset (SQuAD) 1.1  
> **Model:** `distilbert-base-uncased-distilled-squad`

---

## 📋 Overview

This project builds a complete **end-to-end Extractive Question Answering (QA) system** using Natural Language Processing (NLP) and a pre-trained Transformer model. Given a **context/passage** and a **question**, the system automatically identifies and returns the most relevant answer span from the provided text.

The system is built on the **SQuAD 1.1** benchmark dataset and uses **DistilBERT**, a lightweight and efficient variant of BERT fine-tuned specifically on SQuAD 1.1 by Hugging Face.

---

## ❓ Problem Statement

Traditional search systems return documents; they do not directly answer questions. A Question Answering (QA) system bridges this gap by extracting a precise answer span from a given passage—enabling applications in customer support, education, document search, and knowledge management.

---

## 🎯 Objectives

- Load and analyse the SQuAD 1.1 dataset from the official Rajpurkar et al. JSON files
- Perform thorough EDA to understand dataset characteristics
- Implement complete data preprocessing and tokenisation pipelines
- Load and use the `distilbert-base-uncased-distilled-squad` model for inference
- Evaluate performance using official SQuAD metrics: **Exact Match (EM)** and **F1 Score**
- Build a professional **Streamlit web application** for interactive use
- Provide a complete, reproducible project for internship submission

---

## 📂 Dataset

| Property | Value |
|---|---|
| **Name** | Stanford Question Answering Dataset (SQuAD) 1.1 |
| **Authors** | Rajpurkar et al., Stanford University, 2016 |
| **Source** | https://rajpurkar.github.io/SQuAD-explorer/ |
| **Train QA pairs** | 87,599 |
| **Dev QA pairs** | 10,570 |
| **Total articles** | 536 |
| **Format** | JSON (nested: article → paragraph → QA) |
| **Answer type** | Extractive span from context |

---

## 🛠️ Technologies

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML Framework | PyTorch |
| NLP Library | Hugging Face Transformers |
| Tokeniser | DistilBertTokenizerFast |
| Model | distilbert-base-uncased-distilled-squad |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, WordCloud |
| Web App | Streamlit |
| Notebook | Jupyter |

---

## 🏗️ System Architecture

```
SQuAD 1.1 (JSON)
      │
      ▼
 Data Loading & Parsing
 (JSON → Pandas DataFrame)
      │
      ▼
 Data Cleaning & Quality Checks
 (missing values, duplicates, normalisation)
      │
      ▼
 Exploratory Data Analysis
 (distributions, word clouds, statistics)
      │
      ▼
 Tokenisation
 (DistilBertTokenizerFast, offset mapping,
  start/end token position mapping)
      │
      ▼
 DistilBERT QA Model
 distilbert-base-uncased-distilled-squad
      │
      ▼
 Start / End Token Logit Prediction
      │
      ▼
 Answer Span Extraction
      │
      ▼
 Evaluation (EM + F1)          Streamlit Application
      │                               │
      ▼                               ▼
 outputs/results/             Interactive QA UI
```

---

## ✨ Features

- **Complete pipeline** from raw JSON to interactive UI
- **Zero data fabrication** – all statistics computed from the real SQuAD 1.1 dataset
- **Sliding window** inference for long contexts (doc_stride=128)
- **Confidence scoring** per prediction
- **Official SQuAD metrics** – EM and F1 implemented to the Rajpurkar spec
- **Streamlit app** with input validation, examples, and highlighted answers
- **Lightweight fine-tuning** option (2 000 samples) reproducible on a laptop CPU

---

## 📁 Project Structure

```
User_Question_Answering_System/
│
├── data/
│   ├── raw/                  ← Downloaded SQuAD 1.1 JSON files
│   └── processed/            ← Preprocessed CSVs (squad_train.csv, squad_dev.csv)
│
├── notebooks/
│   └── user_question_answering_analysis.ipynb
│
├── src/
│   ├── data_preprocessing.py ← Download, parse, normalise, tokenise
│   ├── model.py              ← QAModel wrapper, SQuADDataset
│   ├── train.py              ← Fine-tuning loop
│   ├── evaluate.py           ← EM + F1 evaluation
│   └── predict.py            ← Demo, interactive, batch prediction
│
├── app/
│   └── app.py                ← Streamlit application
│
├── models/                   ← Saved model checkpoints
│
├── outputs/
│   ├── figures/              ← EDA charts (PNG)
│   ├── results/              ← Evaluation JSON, CSV, summary TXT
│   └── screenshots/          ← Application screenshots
│
├── docs/
│   └── PROJECT_REPORT.md     ← Full project report
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/User_Question_Answering_System.git
cd User_Question_Answering_System
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the DistilBERT model (~250 MB) from Hugging Face Hub. Ensure internet access.

---

## 🚀 How to Run

### Step 1 – Download and Preprocess the Dataset

```bash
python src/data_preprocessing.py
```

This will:
- Download `train-v1.1.json` and `dev-v1.1.json` to `data/raw/`
- Parse and flatten the JSON structure
- Save `squad_train.csv` and `squad_dev.csv` to `data/processed/`

### Step 2 – Run the Jupyter Notebook (EDA + Demo)

```bash
jupyter notebook notebooks/user_question_answering_analysis.ipynb
```

Run all cells to reproduce EDA charts, statistics, and demo predictions.

### Step 3 – Run Demo Predictions

```bash
python src/predict.py --mode demo
```

### Step 4 – Interactive Mode

```bash
python src/predict.py --mode interactive
```

### Step 5 – Evaluate on Dev Split (500 examples, ~10 min on CPU)

```bash
python src/evaluate.py --sample 500
```

Results are saved to `outputs/results/evaluation_results.json`.

### Step 6 – Launch the Streamlit Application

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## 🔧 How to Fine-Tune

### Lightweight (laptop-friendly, 2 000 examples):

```bash
python src/train.py --sample 2000 --epochs 1
```

### Full fine-tuning (requires GPU, ~20–30 min):

```bash
python src/train.py --epochs 2
```

The best model checkpoint is saved to `models/finetuned_squad_model/`.

To evaluate the fine-tuned model:

```bash
python src/evaluate.py --model local --sample 500
```

---

## 💡 Example Usage

**Context:**
> The Transformer architecture was introduced by Vaswani et al. in the 2017 paper "Attention Is All You Need". It relies entirely on a self-attention mechanism to draw global dependencies between input and output.

**Question:** Who introduced the Transformer architecture?

**Predicted Answer:** `Vaswani et al.`

---

## 📊 Actual Results

> **Important:** These values are produced by running `src/evaluate.py --sample 500` using the pretrained `distilbert-base-uncased-distilled-squad` checkpoint.  
> Run the evaluation script on your machine to obtain the actual computed values.

| Metric | Value |
|---|---|
| Exact Match | *Run `python src/evaluate.py --sample 500` to obtain* |
| F1 Score | *Run `python src/evaluate.py --sample 500` to obtain* |
| Dev examples used | 500 |
| Model | distilbert-base-uncased-distilled-squad |

> **Published benchmark (full dev set):** EM ≈ 79–80%, F1 ≈ 86–87%  
> Source: https://huggingface.co/distilbert-base-uncased-distilled-squad

---

## ⚠️ Limitations

1. **Extractive only** – the model can only return spans that exist in the provided context; it cannot generate answers from external knowledge.
2. **Context length** – inputs exceeding 384 tokens are handled via a sliding window, which may occasionally miss the best answer window.
3. **English only** – the model is trained on English text only.
4. **No official SQuAD test set** – SQuAD 1.1 does not release the test set labels publicly; evaluation is performed on the dev split.
5. **CPU inference speed** – on a standard laptop CPU, each prediction takes ~0.5–2 seconds.

---

## 🔮 Future Enhancements

- PDF and multi-document question answering
- Retrieval-Augmented Generation (RAG) with vector databases
- Multilingual question answering
- Voice input integration
- Company-specific knowledge base QA
- Cloud deployment (AWS/GCP/Azure)
- Chatbot integration

---

## 📚 References

1. Rajpurkar, P., Zhang, J., Lopyrev, K., & Liang, P. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. *EMNLP 2016*. https://arxiv.org/abs/1606.05250
2. Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*. https://arxiv.org/abs/1810.04805
3. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *NeurIPS Workshop 2019*. https://arxiv.org/abs/1910.01108
4. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS 2017*. https://arxiv.org/abs/1706.03762
5. Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. *EMNLP 2020*. https://arxiv.org/abs/1910.03771
6. Hugging Face Model Hub: https://huggingface.co/distilbert-base-uncased-distilled-squad
7. Streamlit Documentation: https://docs.streamlit.io

---

*Intelligent User Question Answering System · Internship Project · WHEELTRIX HR Submission*
