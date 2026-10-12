# Project Architecture

## Overview

This project follows a **frontend/backend separation** architecture for clear concerns and maintainability.

```
User_Question_Answering_System/
├── frontend/                    ← UI Layer (Streamlit)
│   ├── __init__.py
│   ├── app.py                   ← Main entry point (streamlit run frontend/app.py)
│   ├── components.py            ← UI components (sidebar, examples, tech stack)
│   └── styles.py                ← CSS styling and layout
│
├── backend/                     ← Business Logic & ML Layer
│   ├── __init__.py
│   ├── model.py                 ← QAModel class (DistilBERT wrapper)
│   └── qa_service.py            ← Service layer (validation, prediction)
│
├── app/                         ← Legacy (can be removed after migration)
├── src/                         ← Original ML code (for reference)
│
├── data/                        ← Raw/processed datasets
├── models/                      ← Saved model checkpoints
├── notebooks/                   ← Jupyter notebooks for exploration
├── outputs/                     ← Results, figures, evaluations
│
├── requirements.txt
└── README.md
```

## Layer Architecture

### Frontend Layer (`frontend/`)

Handles all user-facing interactions and UI rendering.

**Modules:**
- **`app.py`** – Main Streamlit application entry point
  - Session management
  - Model loading and caching
  - User input handling
  - Result display
  - Run with: `streamlit run frontend/app.py`

- **`components.py`** – Reusable UI components
  - `render_sidebar()` – Settings and project info
  - `render_examples()` – Example Q&A pairs
  - `render_technology_stack()` – Tech stack table
  - `render_how_it_works()` – Pipeline explanation

- **`styles.py`** – Styling and layout
  - `render_css()` – Custom CSS
  - `render_header()` – Application header
  - `render_footer()` – Application footer

**Responsibilities:**
- Display UI components
- Collect user input (context, question)
- Show predictions and confidence scores
- Provide example interactions

### Backend Layer (`backend/`)

Encapsulates ML model logic and business rules.

**Modules:**
- **`model.py`** – ML model wrapper
  - `QAModel` class wraps DistilBERT
  - `load_pretrained()` – Load from Hugging Face
  - `load_local()` – Load saved checkpoint
  - `save()` – Save model and tokenizer
  - `predict()` – Run inference

- **`qa_service.py`** – Service layer
  - `validate_question_input()` – Input validation
  - `load_qa_model()` – Model initialization
  - `run_prediction()` – Orchestrate prediction + timing

**Responsibilities:**
- Model management
- Input validation
- Prediction orchestration
- Business logic (not presentation)

## Data Flow

```
User Input
    ↓
[Frontend: Input Fields]
    ↓
validate_question_input()
    ↓
[Frontend: Display Errors OR Continue]
    ↓
run_prediction()
    ├─→ QAModel.predict()
    │   ├─→ Tokenization
    │   ├─→ Model inference
    │   └─→ Answer extraction
    └─→ Measure elapsed time
    ↓
[Frontend: Display Answer + Confidence]
    ↓
User sees results
```

## Import Structure

### From Frontend
```python
from backend.qa_service import (
    load_qa_model,
    run_prediction,
    validate_question_input,
)
from frontend.components import render_examples, render_sidebar
from frontend.styles import render_css, render_header, render_footer
```

### From Backend
```python
from backend.model import QAModel
from backend.qa_service import validate_question_input, run_prediction
```

## Running the Application

### Streamlit App (New Structure)
```bash
streamlit run frontend/app.py
```

### Legacy App (Still Available)
```bash
streamlit run app/app.py
```

## Adding New Features

### Add a Backend Service
1. Create a new file in `backend/` (e.g., `backend/cache_service.py`)
2. Import and use in `qa_service.py` or directly in `frontend/app.py`

Example:
```python
# backend/cache_service.py
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_predict(question: str, context: str) -> dict:
    # Cached prediction logic
    pass
```

### Add a UI Component
1. Create a function in `frontend/components.py`
2. Call it from `frontend/app.py`

Example:
```python
# frontend/components.py
def render_history() -> None:
    st.markdown("### 📝 Previous Predictions")
    # Display prediction history
```

## Testing Strategy

### Backend Tests
```bash
pytest backend/tests/
```

Test files:
- `backend/tests/test_model.py` – Model loading and inference
- `backend/tests/test_qa_service.py` – Validation and service logic

### Frontend Tests
```bash
pytest frontend/tests/
```

Test files:
- `frontend/tests/test_components.py` – Component rendering
- `frontend/tests/test_app.py` – Integration tests

## Deployment

### Development
```bash
streamlit run frontend/app.py
```

### Production
```bash
streamlit run frontend/app.py --server.port 8501 --server.address 0.0.0.0
```

Or use Docker:
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "frontend/app.py"]
```

## Migration Notes

The new `frontend/` and `backend/` structure replaces the old `app/app.py` and `src/model.py`. The old files are retained for reference but deprecated:

- ✅ Use: `frontend/app.py`, `backend/model.py`, `backend/qa_service.py`
- ❌ Legacy: `app/app.py`, `src/model.py`, `app/ui.py`, `src/qa_service.py`

## Benefits

✅ **Clear Separation of Concerns** – UI and ML logic are separate  
✅ **Scalability** – Easy to add new features without touching existing code  
✅ **Testability** – Backend can be tested independently  
✅ **Reusability** – Backend services can be used by CLI, API, or other UIs  
✅ **Maintainability** – Easier to understand and modify  
