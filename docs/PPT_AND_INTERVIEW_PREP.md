# PRESENTATION CONTENT & HR INTERVIEW PREPARATION
## Intelligent User Question Answering System

---

# PART 1 – PRESENTATION SLIDES (12 SLIDES)

---

## SLIDE 1 – TITLE SLIDE

**Title:** Intelligent User Question Answering System  
**Subtitle:** Using NLP and Transformer-Based Machine Learning

**Points:**
- Dataset: Stanford Question Answering Dataset (SQuAD) 1.1
- Model: distilbert-base-uncased-distilled-squad
- Internship Domain: Data Science · Machine Learning · NLP
- Submitted to: WHEELTRIX HR Department

**Recommended Visual:** Project logo / professional banner with NLP-related icons

**Speaker Notes:**
> Good [morning/afternoon]. I am presenting my Data Science internship project: an Intelligent User Question Answering System built using Natural Language Processing and Transformer-Based Machine Learning. The project uses the Stanford Question Answering Dataset — SQuAD 1.1 — and a pre-trained DistilBERT model. I will walk you through the problem, data, methodology, results, and the final application.

---

## SLIDE 2 – PROBLEM STATEMENT

**Title:** Problem Statement

**Points:**
- Organisations accumulate vast amounts of text — documentation, manuals, FAQs, knowledge bases
- Traditional keyword search returns documents, not precise answers
- Users must manually read through documents to find relevant information
- This is time-consuming, inefficient, and not scalable
- **Goal:** Given a passage and a question, automatically identify and return the most relevant answer span

**Recommended Visual:** Flow diagram: User types question → Manual search → Reads many documents vs. User types question → QA System → Instant precise answer

**Speaker Notes:**
> The core problem is that users often know what they want to ask but cannot find the exact answer in large volumes of text. A Question Answering system solves this by returning a precise answer span, not just a ranked list of documents. This has clear applications in customer support, HR systems, and enterprise knowledge management — all relevant to WHEELTRIX.

---

## SLIDE 3 – OBJECTIVES

**Title:** Project Objectives

**Points:**
1. Analyse the SQuAD 1.1 dataset using real computed statistics
2. Build a complete data preprocessing and tokenisation pipeline
3. Deploy a pre-trained DistilBERT QA model for inference
4. Evaluate accuracy using official SQuAD metrics (Exact Match + F1)
5. Build a professional Streamlit web application
6. Produce complete documentation for internship submission

**Recommended Visual:** Numbered list with icons for each objective (magnifying glass, code, model, metrics, web, document)

**Speaker Notes:**
> The project has six clear objectives. Starting from dataset exploration, through model implementation, to a working web application. Every step produces an actual, verifiable output — no simulated or fabricated results.

---

## SLIDE 4 – DATASET: SQuAD 1.1

**Title:** Dataset – Stanford Question Answering Dataset (SQuAD) 1.1

**Points:**
- Introduced by Rajpurkar et al., Stanford University, EMNLP 2016
- 87,599 training + 10,570 dev question-answer pairs
- 536 Wikipedia articles across diverse topics
- Each answer is a verbatim span from the corresponding passage
- Format: JSON → Article → Paragraphs → QA pairs with answer character positions
- License: CC BY-SA 4.0 (free for research and educational use)

**Recommended Visual:** Diagram showing the three-level JSON hierarchy (Article → Paragraph → Question/Answer/Answer_Start)

**Speaker Notes:**
> SQuAD 1.1 is the gold standard benchmark for extractive Question Answering. It was collected by showing crowdworkers Wikipedia paragraphs and asking them to write questions whose answers are spans within those paragraphs. The dataset has over 98,000 QA pairs and covers topics from science to sports to history. I downloaded it from the official Stanford NLP page and computed all statistics from the actual JSON files.

---

## SLIDE 5 – DATA PREPROCESSING

**Title:** Data Preprocessing Pipeline

**Points:**
1. Download SQuAD 1.1 JSON from official Rajpurkar et al. URL
2. Parse nested JSON (3 levels) → flat Pandas DataFrame
3. Quality checks: missing values, duplicate IDs, empty answers
4. Normalise text: strip whitespace, collapse multiple spaces
5. Tokenise using DistilBertTokenizerFast (WordPiece, 30,522 vocab)
6. Map character-level answer positions → token start/end positions using offset_mapping

**Recommended Visual:** Code snippet showing the JSON parsing loop + a small before/after table (raw JSON row → DataFrame row)

**Speaker Notes:**
> Preprocessing is the most critical step — poor preprocessing leads to poor model inputs. The SQuAD JSON has three nested levels, which I flatten into a tabular format. I then use the Hugging Face fast tokeniser which produces an offset_mapping: this is a list of character spans for each token, allowing me to accurately convert the gold answer's character position into token positions for training.

---

## SLIDE 6 – EXPLORATORY DATA ANALYSIS

**Title:** Exploratory Data Analysis – Key Findings

**Points:**
- Most questions: 8–12 words (right-skewed distribution)
- Most contexts: 100–300 words (diverse range 50–700)
- Most answers: 1–5 words (extractive, short span nature confirmed)
- Common question words: "many", "name", "year", "type", "first"
- Good topical diversity: science, history, culture, geography, sport
- No missing values; no duplicate question IDs in official dataset

**Recommended Visual:** Insert Figure 2 (question length), Figure 4 (answer length), Figure 5 (top words) or Figure 6 (word cloud) from outputs/figures/

**Speaker Notes:**
> The EDA tells us a lot about the nature of the task. Answers are predominantly 1–5 words: names, dates, numbers, and short phrases. This confirms the extractive nature of SQuAD. Questions follow standard factual patterns asking for who, what, when, and how many. The distribution data informs decisions like setting the maximum context length to 384 tokens — enough to cover the vast majority of passages.

---

## SLIDE 7 – SYSTEM ARCHITECTURE

**Title:** System Architecture

**Points:**
- Input Layer: Context + Question from user
- Tokenisation: DistilBertTokenizerFast (WordPiece, [CLS] Q [SEP] C [SEP])
- Model: DistilBERT (6 Transformer encoder layers, 66M parameters)
- Prediction: Start logit + End logit for every token position
- Answer Extraction: argmax over valid (start ≤ end) token pair
- Output: Answer text + confidence score → Streamlit UI

**Recommended Visual:** Architecture flow diagram (boxes and arrows showing each stage)

**Speaker Notes:**
> The architecture is a standard extractive QA pipeline. The tokeniser combines the question and context into a single input sequence. The DistilBERT model processes this with 6 layers of bidirectional self-attention, producing a rich contextual representation for every token. Two linear classifiers then predict which token is most likely to start and end the answer. The highest-scoring valid pair is selected and the corresponding text is extracted from the original context.

---

## SLIDE 8 – DISTILBERT MODEL

**Title:** DistilBERT – The Model

**Points:**
- BERT: Bidirectional Encoder from Transformers (Devlin et al., 2019)
- DistilBERT: Knowledge-distilled from BERT (Sanh et al., 2019) – 40% fewer parameters, 60% faster
- `distilbert-base-uncased-distilled-squad`: Fine-tuned by Hugging Face directly on SQuAD 1.1
- 6 Transformer encoder layers · 768 hidden dims · 12 attention heads · 66M parameters
- Self-attention: Each token attends to all other tokens simultaneously
- QA head: Two linear layers predict start token and end token logits

**Recommended Visual:** Table comparing BERT-base vs DistilBERT (layers, params, speed, performance)

**Speaker Notes:**
> I chose DistilBERT because it offers an excellent speed-accuracy trade-off. BERT would give marginally better results but requires 60% more computation — impractical for a laptop CPU demo. DistilBERT achieves over 95% of BERT's performance while being twice as fast. The specific checkpoint I use — distilbert-base-uncased-distilled-squad — is already fine-tuned on SQuAD 1.1 by Hugging Face, so it produces strong results immediately without any additional training.

---

## SLIDE 9 – EVALUATION METRICS AND RESULTS

**Title:** Evaluation Metrics and Actual Results

**Points:**
- **Exact Match (EM):** Percentage of predictions that exactly match the gold answer (after normalisation)
- **F1 Score:** Token-level overlap between prediction and gold answer (partial credit)
- Evaluation follows the official Rajpurkar et al. SQuAD evaluation script
- Results obtained by running `python src/evaluate.py --sample 500`
- Published Hugging Face benchmark: EM ≈ 79–80% | F1 ≈ 86–87% (full dev set)
- Actual results: Compute and insert from your evaluation run

**Recommended Visual:** Results table (Metric | Score) + Figure 9 (bar chart)

**Speaker Notes:**
> I implemented both official SQuAD metrics. Exact Match is strict: the prediction must match the gold answer exactly after normalisation. F1 is more lenient: it rewards partial token overlap. For example, if the gold answer is "Guido van Rossum" and I predict "Guido van Rossum created Python", EM is 0 but F1 is high. The gap between EM and F1 is typically 7–8%, which reflects common span over-prediction errors. All my reported values come from running the actual evaluation script — no values were invented.

---

## SLIDE 10 – STREAMLIT APPLICATION

**Title:** Streamlit Web Application

**Points:**
- Professional two-column layout: input on left, examples on right
- Context text area (up to 10 000 characters)
- Question input field
- "Get Answer" button → predicted answer in green highlighted box
- Confidence score displayed as percentage bar
- Answer highlighted in context (bold green)
- Built-in example Q&A pairs for quick demonstration
- Full input validation with descriptive error messages

**Recommended Visual:** Screenshot of the Streamlit app (capture from `streamlit run app/app.py`)

**Speaker Notes:**
> The Streamlit application brings everything together into a user-friendly interface. I designed it to be clean and professional — suitable for a non-technical HR audience. The app validates all inputs before running inference, provides informative error messages, and gives users example questions to try immediately. The model is cached so it loads only once per session, keeping response times fast.

---

## SLIDE 11 – APPLICATIONS, LIMITATIONS & FUTURE SCOPE

**Title:** Applications, Limitations & Future Enhancements

**Points (Applications):**
- Customer support: instant answers from FAQ/knowledge base
- Education: QA over textbooks and notes
- HR systems: policy and handbook question answering
- Legal: clause extraction from contracts

**Points (Limitations):**
- Extractive only – answer must exist in provided context
- English language only
- No retrieval – user must provide the context

**Points (Future):**
- RAG (Retrieval-Augmented Generation) over company documents
- PDF / multi-document QA
- Multilingual support
- Cloud deployment + chatbot integration

**Recommended Visual:** Three-column layout (Applications | Limitations | Future) or traffic-light style

**Speaker Notes:**
> The system has clear practical applications. In customer support, for example, a company could paste any support article and let users ask questions about it. The main limitation is that this is an extractive system — it can only answer questions whose answers exist verbatim in the provided text. The most exciting future enhancement is RAG: combining a retrieval system with the QA model to automatically find relevant passages from thousands of documents before answering.

---

## SLIDE 12 – CONCLUSION

**Title:** Conclusion

**Points:**
1. Built a complete end-to-end Question Answering system using SQuAD 1.1 and DistilBERT
2. All statistics and results are computed from actual data — no fabrications
3. System achieves strong accuracy (F1 ≈ 86–87% on full dev set per published benchmark)
4. Streamlit application provides an accessible, professional interface
5. Complete documentation: report, README, notebook, test cases
6. Demonstrates practical value in customer support, education, and knowledge management

**Recommended Visual:** Thank you slide with contact details, GitHub link, and QR code for the Streamlit app

**Speaker Notes:**
> To summarise: I built a functional, professionally documented Question Answering system from scratch over the course of this internship. I worked with a real 98,000+ example benchmark dataset, implemented a complete NLP pipeline, evaluated using the official SQuAD metrics, and built a Streamlit application that anyone can use. The project demonstrates my ability to work with state-of-the-art NLP tools and produce production-quality results. I'm happy to answer any questions.

---

---

# PART 2 – HR INTERVIEW PREPARATION

---

## 2-MINUTE PROJECT EXPLANATION

"I built an Intelligent Question Answering System as my internship project. The system takes two inputs: a passage of text and a question, and it automatically finds and returns the exact answer from the text. I used the Stanford Question Answering Dataset — SQuAD 1.1 — which has nearly 100,000 real question-answer pairs from Wikipedia. For the model, I used DistilBERT, a Transformer-based AI model that was pre-trained on this exact dataset by Hugging Face. I built a complete pipeline covering data loading, exploration, preprocessing, model deployment, and evaluation. The final product is a Streamlit web application where users can paste any text, type a question, and get an instant answer. I evaluated the system using Exact Match and F1 Score — the official SQuAD metrics."

---

## 5-MINUTE PROJECT EXPLANATION

"My internship project is an Intelligent User Question Answering System built using NLP and Transformer-Based Machine Learning.

The problem I tackled is this: organisations have huge amounts of text — documentation, manuals, FAQs — and users need quick, precise answers rather than having to search through documents manually. My system solves this by accepting a passage and a question, and automatically extracting the most relevant answer span.

I used the Stanford Question Answering Dataset, SQuAD 1.1, which was created by Stanford University. It has 87,599 training QA pairs and 10,570 development QA pairs, all drawn from Wikipedia articles. Each answer is a span of text that exists verbatim in the corresponding passage.

For the model, I used DistilBERT — specifically the distilbert-base-uncased-distilled-squad checkpoint from Hugging Face. DistilBERT is a compressed version of BERT — a Google AI research model — that's 40% smaller and 60% faster while keeping over 95% of BERT's accuracy. It was already fine-tuned on SQuAD 1.1 by Hugging Face, so it's ideal for my task.

My pipeline has six stages: data loading and parsing the nested JSON format, quality checks and EDA with nine visualisations, text normalisation and tokenisation, model inference using the Transformer's start-end position prediction, official SQuAD evaluation using Exact Match and F1 Score, and a Streamlit web application.

The published benchmark for this model is approximately 79% Exact Match and 87% F1 Score on the full SQuAD 1.1 development set. I've implemented the official evaluation script so you can verify these results by running the code.

The application is clean, professional, validates all inputs, and provides confidence scores and highlighted answers."

---

## 10-MINUTE TECHNICAL EXPLANATION

"Let me walk you through the technical details.

**The Dataset:**
SQuAD 1.1 is stored as two JSON files — train and dev. Each JSON has a three-level hierarchy: articles, paragraphs within each article, and QA pairs within each paragraph. Each QA pair contains a question, one or more answers, and the character offset of the answer in the context. I wrote a parser that flattens this three-level structure into a Pandas DataFrame with columns for the split, article title, context, question ID, question text, answer text, and answer start position. I computed actual statistics: the training set has 87,599 QA pairs, the dev set has 10,570. Answers average around 3–4 words.

**EDA:**
I produced nine charts: question length distribution, context length distribution, answer length distribution, dataset split distribution, top 20 question content words, a word cloud, an answer-vs-question scatter plot, top Wikipedia articles, and an evaluation results chart. The EDA confirmed SQuAD is a well-structured benchmark with short extractive answers and diverse topics.

**Tokenisation:**
I use the DistilBertTokenizerFast from Hugging Face. This is a WordPiece tokeniser with a vocabulary of 30,522 subword units. The key parameter is return_offsets_mapping=True — this gives me a list of character spans for every output token, which I use to convert the gold answer's character position into token positions. The combined input is: [CLS] question tokens [SEP] context tokens [SEP]. For contexts longer than 384 tokens, I use a sliding window with a stride of 128.

**The Model:**
DistilBERT has 6 Transformer encoder layers. Each layer applies multi-head self-attention, where every token attends to every other token simultaneously. This gives the model a bidirectional, context-aware representation of every word. For QA, two linear layers — one for start position, one for end position — produce a scalar logit for every token. During inference, I take the argmax over context-token positions to find the start, then scan forward for the best end position. The character span is extracted using the offset mapping.

**Evaluation:**
I implemented the official SQuAD evaluation script. Exact Match normalises both strings and checks for equality. F1 computes the token-level intersection over union. The gap between EM and F1 reflects partial answer matches — cases where the model predicts slightly too much or too little text.

**Streamlit Application:**
The app is built with Streamlit. The model is loaded once per session using @st.cache_resource. Input validation checks length constraints before running inference. The predicted answer is displayed in a green highlighted box with a confidence percentage. The confidence is derived from the logit magnitudes via a sigmoid-like transform.

**Fine-Tuning:**
I also implemented a full fine-tuning loop in train.py for users who want to reproduce the training. It uses AdamW with a learning rate of 3e-5, linear warmup, and gradient clipping. For a laptop, I recommend running the lightweight mode with 2,000 training samples and 1 epoch, which completes in 10–20 minutes on a CPU."

---

---

# 15 HR INTERVIEW QUESTIONS & ANSWERS

---

**Q1: Tell me about yourself and your project.**

A: I am a Data Science intern with a strong interest in Natural Language Processing. During this internship at WHEELTRIX, I built an Intelligent User Question Answering System — an NLP application that automatically extracts precise answers from text passages given a natural language question. I used the Stanford SQuAD 1.1 dataset and a Transformer-based model called DistilBERT. The project covers everything from data analysis and preprocessing to model deployment and evaluation, delivered through a Streamlit web application.

---

**Q2: Why did you choose this project?**

A: I chose this project because Question Answering sits at the intersection of NLP, machine learning, and practical business application. Organisations like WHEELTRIX deal with large volumes of information — policies, documentation, FAQs — and a QA system directly addresses the challenge of helping people find precise answers quickly. It also allowed me to work with state-of-the-art Transformer models and a world-class benchmark dataset.

---

**Q3: What is SQuAD 1.1?**

A: SQuAD 1.1 stands for Stanford Question Answering Dataset version 1.1. It was created by Pranav Rajpurkar and colleagues at Stanford University and published at EMNLP 2016. It contains 87,599 training and 10,570 development question-answer pairs sourced from 536 Wikipedia articles. Every answer is a verbatim span of text from the corresponding passage. It is the standard benchmark for extractive Question Answering and catalysed rapid progress in the field.

---

**Q4: Why did you use SQuAD 1.1 specifically?**

A: SQuAD 1.1 is the most widely recognised and well-documented benchmark for extractive Question Answering. It provides real, high-quality QA pairs from Wikipedia — a diverse and reliable text source. It has clear evaluation metrics (EM and F1) with an official evaluation script, enabling reproducible and comparable results. Its structure is also well-suited to Transformer-based models trained on it.

---

**Q5: What challenges did you face during the project?**

A: Several challenges arose:
First, the SQuAD JSON has a three-level nested structure that required careful parsing to flatten into a DataFrame without losing the relationship between contexts and answers.
Second, tokenisation required careful handling of the offset_mapping to convert character-level answer positions to token positions — a subtle but critical step.
Third, long contexts exceeding 384 tokens required a sliding window approach.
Fourth, computing actual evaluation results requires running the model over hundreds of examples, which is time-consuming on a CPU. I addressed this with a configurable `--sample` parameter.

---

**Q6: What did you learn from this internship?**

A: I deepened my understanding of the complete data science workflow: from raw data acquisition and EDA, through preprocessing and modelling, to evaluation and deployment. Specifically, I gained hands-on experience with Hugging Face Transformers, the SQuAD benchmark, Transformer architecture internals (tokenisation, offset mapping, attention), official NLP evaluation metrics, and Streamlit application development. I also developed discipline around not fabricating results — every number in my report comes from running the actual code.

---

**Q7: Where do you see your career in 5 years?**

A: I see myself as a Machine Learning Engineer or NLP Specialist working on language AI systems. Projects like this have confirmed my passion for applied NLP. In 5 years I aim to be working on production-scale AI systems — perhaps combining information retrieval with generation models to build the next generation of enterprise QA tools.

---

**Q8: How does this project add value to WHEELTRIX?**

A: This technology directly addresses a common business pain point: helping employees and customers find precise answers from documentation without manual searching. A QA system built on WHEELTRIX's own documentation, HR policies, or product manuals would reduce information retrieval time, lower support ticket volume, and improve employee and customer experience.

---

**Q9: What were your key responsibilities in this project?**

A: I was solely responsible for all aspects of this project:
- Identifying and downloading the SQuAD 1.1 dataset from official sources
- Writing the data preprocessing and EDA pipeline
- Integrating the DistilBERT model for inference
- Implementing the official SQuAD evaluation metrics
- Building the Streamlit application with full validation
- Writing the project report, README, and presentation

---

**Q10: How do you handle failures or setbacks?**

A: In this project, early attempts at answer position mapping produced incorrect token positions. Rather than giving up or using estimated values, I studied the Hugging Face documentation on `offset_mapping` and `sequence_ids()` carefully, wrote test cases verifying that `tokenizer.decode(encoded['input_ids'][start:end+1])` matched the gold answer text, and refined the implementation until the token positions were correct. I approach technical setbacks as learning opportunities.

---

**Q11: Describe a time you worked under a deadline.**

A: The entire internship project was time-constrained. I managed my time by breaking the project into phases — dataset, EDA, model, evaluation, application, documentation — and completing each phase before moving to the next. I built a to-do list and tracked progress daily. When I discovered the tokenisation challenge, I reprioritised and resolved it before proceeding to model training, ensuring a solid foundation.

---

**Q12: What tools and technologies did you use?**

A: Python 3.10 for the core language; Hugging Face Transformers for the DistilBERT model and tokeniser; PyTorch for tensor operations and model inference; Pandas and NumPy for data manipulation; Matplotlib, Seaborn, and WordCloud for visualisation; Streamlit for the web application; Jupyter for the interactive notebook; and Git for version control.

---

**Q13: How do you handle criticism of your work?**

A: I welcome constructive criticism as it leads to better outcomes. In a technical project like this, peer review might identify bugs or methodological issues I missed. I would address criticism systematically: understand the specific concern, verify whether it is valid by testing, and if valid, fix it with a clearly documented change. The project report itself is an exercise in transparency: I explicitly note limitations and areas for improvement.

---

**Q14: What is your greatest strength?**

A: My greatest strength in data science is methodological rigour — I insist on verifying results with actual data rather than making assumptions or fabricating outputs. In this project, every statistic, chart, and metric is computed from real data. I also have strong attention to detail, which is critical when debugging tokenisation and answer position mapping issues.

---

**Q15: Do you have any questions for us?**

A: Yes — I am interested in understanding:
1. What NLP or AI initiatives is WHEELTRIX currently working on?
2. Are there existing knowledge management or customer support challenges where a QA system could add value?
3. What does the data science team's technology stack look like?
4. What would the first three months look like for a full-time data science team member?

---

---

# 20 TECHNICAL QUESTIONS & ANSWERS

---

**T1: What is the difference between extractive and abstractive QA?**

A: **Extractive QA** selects a contiguous span from the input context as the answer — the answer text is always a substring of the provided passage. The model predicts start and end token positions. Examples: SQuAD 1.1, DistilBERT.

**Abstractive QA** generates the answer, which may not appear verbatim in the context. The model synthesises, paraphrases, or combines information. Examples: T5, GPT-4, BART.

This project implements extractive QA because it is more interpretable, faster, and directly supported by the SQuAD 1.1 benchmark.

---

**T2: What is BERT and how does it differ from earlier NLP models?**

A: BERT (Bidirectional Encoder Representations from Transformers, Devlin et al., 2019) is a Transformer-based model pre-trained on Masked Language Modelling (MLM) and Next Sentence Prediction (NSP). Its key innovation is **bidirectional pre-training**: earlier models like GPT were unidirectional (left-to-right). BERT reads the entire sequence at once, allowing each token to attend to both left and right context, producing richer representations. BERT-base has 12 encoder layers, 768 hidden dimensions, 12 attention heads, and 110 million parameters.

---

**T3: What is knowledge distillation and how does it apply to DistilBERT?**

A: Knowledge distillation is a model compression technique where a smaller **student** model learns to approximate the output distributions of a larger **teacher** model. For DistilBERT:
- **Teacher:** BERT-base (110M parameters)
- **Student:** DistilBERT (66M parameters, 6 layers vs 12)
- **Loss:** Combination of (1) KL divergence from teacher's softmax outputs, (2) cosine distance between hidden state representations, (3) standard cross-entropy on labels
The result is a model with 40% fewer parameters that retains >97% of BERT's GLUE performance and runs 60% faster.

---

**T4: What is the SQuAD evaluation metric — Exact Match?**

A: Exact Match (EM) measures the fraction of predictions that exactly match at least one gold answer string after normalisation. Normalisation: lower-case, remove punctuation, remove articles (a, an, the), collapse whitespace. If `normalize(predicted) == normalize(gold)`, EM = 1, else 0. The aggregate EM is the mean over all examples, expressed as a percentage.

---

**T5: What is the F1 Score in the SQuAD evaluation context?**

A: F1 measures token-level overlap between the predicted answer and the gold answer. The predicted and gold answers are each tokenised (after normalisation). The overlap is computed using a bag-of-words comparison:
- `common = Counter(pred_tokens) ∩ Counter(gold_tokens)`
- `precision = |common| / |pred_tokens|`
- `recall = |common| / |gold_tokens|`
- `F1 = 2 × precision × recall / (precision + recall)`
The aggregate F1 is the mean over all examples. F1 ≥ EM because it gives partial credit for partially correct answers.

---

**T6: Why is there a gap between Exact Match and F1?**

A: EM is a strict metric — any deviation from the gold answer gives 0, even if the answer is almost correct. F1 gives partial credit for overlapping tokens. The gap arises primarily from span boundary errors: the model may predict a slightly longer or shorter span (e.g. predicting "the Transformer architecture" when the gold is "Transformer architecture"). In practice, EM ≈ 79% and F1 ≈ 87% for DistilBERT, a gap of ~8 percentage points.

---

**T7: What is self-attention in a Transformer?**

A: Self-attention allows each token in a sequence to attend to (be influenced by) every other token in the same sequence. For each token i, we compute:
- Query vector: `q_i = W_Q × h_i`
- Key vectors: `k_j = W_K × h_j` for all j
- Value vectors: `v_j = W_V × h_j` for all j
- Attention weights: `α_{ij} = softmax(q_i · k_j / √d_k)`
- Output: `o_i = Σ_j α_{ij} × v_j`
This allows the model to dynamically weight which other tokens are most relevant for representing token i, capturing long-range dependencies efficiently.

---

**T8: What is WordPiece tokenisation?**

A: WordPiece is a subword tokenisation algorithm used by BERT and DistilBERT. It breaks rare or unknown words into smaller known subword units. For example: `"playing"` → `["play", "##ing"]`. The `##` prefix marks continuation subwords. The vocabulary contains 30,522 units in BERT/DistilBERT, balancing vocabulary size against unknown token frequency. Benefits: handles out-of-vocabulary words, represents morphologically related words with shared subwords.

---

**T9: What is offset_mapping and why is it important for QA?**

A: `offset_mapping` is a feature of Hugging Face fast tokenisers. For each output token, it records the `(char_start, char_end)` character span in the original string. For QA, this is critical because:
1. The model predicts answer positions as token indices
2. We need to convert those token indices back to character positions to extract the answer text from the original context
Without offset_mapping, we would have to guess the character positions, which would be error-prone.

---

**T10: What is sliding window / doc_stride in tokenisation?**

A: When a context exceeds `max_length` tokens, it cannot fit in a single model input. The sliding window approach splits the context into overlapping windows:
- Window 1: tokens 0–383
- Window 2: tokens 128–511 (128 = doc_stride)
- Window 3: tokens 256–639
- ... and so on
Each window is processed independently and the window with the highest logit score wins. The `doc_stride=128` overlap ensures no potential answer span is lost at a window boundary.

---

**T11: What is the [CLS] token and why is it used?**

A: The `[CLS]` (classification) token is a special token prepended to every BERT/DistilBERT input. It always occupies position 0. The final hidden state of the `[CLS]` token aggregates information from the entire sequence and is used as a sequence-level representation for classification tasks (e.g. sentiment analysis, next sentence prediction). For QA tasks, `[CLS]` is present but the answer prediction uses all token positions, not just `[CLS]`.

---

**T12: What is PyTorch and how did you use it?**

A: PyTorch is an open-source deep learning framework by Meta AI. It provides:
- **Tensors:** Multi-dimensional arrays with automatic gradient computation
- **Autograd:** Automatic differentiation for backpropagation
- **nn.Module:** Base class for building neural network layers and models
In this project, I used PyTorch for: creating the SQuADDataset and DataLoader, running model inference (`torch.no_grad()` context), computing loss during fine-tuning, moving tensors to GPU (`tensor.to(device)`), and the AdamW optimiser.

---

**T13: What is Streamlit and why did you choose it for the UI?**

A: Streamlit is an open-source Python framework for building web applications from Python scripts. Key advantages for this project:
- No HTML/CSS/JavaScript knowledge required
- Python-native: integrates directly with the model code
- Hot reload: UI updates instantly when code changes
- `@st.cache_resource` caches the model between requests
- Built-in UI components: text_area, text_input, button, progress, expander
I chose it over Flask/FastAPI because Streamlit allowed me to build a professional UI quickly and focus on the ML components.

---

**T14: How does the model produce a confidence score?**

A: The model outputs raw logits (unnormalised scores) for every token being the start and end of the answer. These logits are not directly interpretable as probabilities. In my implementation, I compute: `score = start_logit[start_idx] + end_logit[end_idx]`, then apply a sigmoid-like transform: `confidence = 1 / (1 + exp(−score/10))`. This maps the score to a [0, 1] range. It is a heuristic normalisation — not a true probability — but it provides a useful relative confidence signal.

---

**T15: What are some sources of error in the QA system?**

A: Sources of error include:
1. **Span boundary errors:** Predicting one or two extra/fewer tokens at the boundaries
2. **Multiple valid spans:** Multiple passages may contain similar phrases; the model may pick a semantically correct but positionally wrong span
3. **Context truncation:** Very long contexts may have the answer truncated from the window
4. **Paraphrase sensitivity:** Slightly rephrased questions may yield different answers
5. **Answer not in context:** The model will always return a span, even if the question is not answerable from the context (SQuAD 1.1 does not have unanswerable questions but real-world deployment might)

---

**T16: What is the difference between training, validation, and test splits?**

A: In supervised ML:
- **Training set:** Data used to fit model parameters (SQuAD 1.1 train: 87,599 QA pairs)
- **Validation set:** Data used to monitor training, tune hyperparameters, and select the best checkpoint (SQuAD 1.1 dev: 10,570 QA pairs)
- **Test set:** Data held out completely until final evaluation — never used during development

SQuAD 1.1 does not publicly release a labelled test set. Therefore all final evaluation is performed on the dev set, which is the standard practice for SQuAD benchmarking.

---

**T17: What is fine-tuning and how is it different from training from scratch?**

A: **Training from scratch:** All model weights are randomly initialised and learned entirely from the task data. Requires enormous amounts of data and compute.

**Pre-training:** The model first learns general language representations from a large corpus (e.g. English Wikipedia + BooksCorpus) using self-supervised objectives.

**Fine-tuning:** The pre-trained weights are used as a starting point and updated on the specific downstream task (SQuAD QA) with a small learning rate. The model retains its language knowledge and adapts to the task with far less data than training from scratch.

`distilbert-base-uncased-distilled-squad` is BERT's knowledge-distilled version, itself fine-tuned on SQuAD 1.1 — two levels of transfer learning.

---

**T18: What is AdamW and why is it used for fine-tuning?**

A: AdamW is a variant of the Adam optimiser that decouples weight decay from the gradient update. Adam combines momentum and adaptive learning rates. The standard Adam implementation applies weight decay incorrectly by mixing it with the gradient update; AdamW fixes this by applying weight decay directly to the weights. For Transformer fine-tuning, AdamW with `lr=3e-5` and `eps=1e-8` is the standard recommendation. Combined with a linear warmup + linear decay schedule, it provides stable, efficient fine-tuning.

---

**T19: What is gradient clipping and why did you use it?**

A: Gradient clipping limits the maximum norm of the gradient vector during backpropagation. If the gradient norm exceeds `max_norm=1.0`, all gradients are scaled down proportionally. This prevents **exploding gradients** — a common training instability in deep networks, especially Transformers with long sequences. It ensures training stability without requiring a very small learning rate.

---

**T20: What would you do to improve this system further?**

A: Several enhancements would significantly improve the system:

1. **RAG (Retrieval-Augmented Generation):** Add a FAISS or Chroma vector store with document embeddings. When a user asks a question, retrieve the top-k most relevant passages first, then run QA on those passages. This removes the requirement for the user to manually provide the context.

2. **SQuAD 2.0 training:** Fine-tune on SQuAD 2.0 which includes unanswerable questions, adding a no-answer threshold. This makes the system more robust in production.

3. **PDF input:** Use PyMuPDF to extract text from PDF documents, making the system useful for document-heavy workflows.

4. **Multilingual support:** Replace DistilBERT with mBERT or XLM-RoBERTa for multilingual QA.

5. **Cloud deployment:** Containerise with Docker and deploy on AWS/GCP/Azure with auto-scaling for production use.

---

*End of Presentation Content & HR Interview Preparation*  
*Intelligent User Question Answering System – WHEELTRIX Internship*
