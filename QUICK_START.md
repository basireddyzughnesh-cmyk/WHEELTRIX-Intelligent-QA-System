# Quick Start Guide

## New Project Structure (Frontend/Backend)

```
frontend/          ← Streamlit UI Layer
backend/           ← ML Business Logic Layer
```

---

## Running the Application

### Option 1: Run New Modular Structure (Recommended)
```bash
streamlit run frontend/app.py
```

### Option 2: Run Legacy Monolithic App
```bash
streamlit run app/app.py
```

---

## Project Layout

### Frontend (User Interface)
```
frontend/
├── app.py              ← Main entry point (~130 lines)
├── components.py       ← UI components: sidebar, examples (~140 lines)
└── styles.py           ← CSS styling and layout (~90 lines)
```

### Backend (Business Logic)
```
backend/
├── model.py            ← QAModel wrapper, DistilBERT (~205 lines)
└── qa_service.py       ← Validation, prediction logic (~41 lines)
```

---

## File Responsibilities

| File | Purpose | Key Functions |
|------|---------|---|
| `frontend/app.py` | Streamlit entry point | `setup_page()`, `load_model()`, `main()` |
| `frontend/components.py` | UI components | `render_sidebar()`, `render_examples()`, `render_how_it_works()` |
| `frontend/styles.py` | Styling & layout | `render_css()`, `render_header()`, `render_footer()` |
| `backend/model.py` | ML model wrapper | `QAModel.load_pretrained()`, `QAModel.predict()` |
| `backend/qa_service.py` | Business logic | `validate_question_input()`, `run_prediction()` |

---

## Data Flow

```
User enters context + question
    ↓
frontend/app.py (UI collection)
    ↓
backend/qa_service.validate_question_input()
    ↓
backend/qa_service.run_prediction()
    ├→ backend/model.QAModel.predict()
    └→ Measure time
    ↓
Display answer + confidence + time
```

---

## Key Features

✅ **Clean Separation** – Frontend and backend are separate  
✅ **Modular** – Each file has single responsibility  
✅ **Testable** – Backend can be tested independently  
✅ **Reusable** – Backend services can be used by other UIs (API, CLI)  
✅ **Extensible** – Easy to add new features  
✅ **Professional** – Production-ready structure  

---

## Making Changes

### Add a New UI Component
1. Create function in `frontend/components.py`
2. Call from `frontend/app.py`

Example:
```python
# frontend/components.py
def render_statistics() -> None:
    st.markdown("## Stats")
    # ...

# frontend/app.py
render_statistics()
```

### Add a New Backend Service
1. Create file in `backend/` (e.g., `backend/cache.py`)
2. Import and use in `frontend/app.py` or `backend/qa_service.py`

Example:
```python
# backend/cache.py
@lru_cache(maxsize=100)
def cached_predict(q, c):
    return model.predict(q, c)

# frontend/app.py
from backend.cache import cached_predict
```

### Use Backend in Other Context
Backend services can be imported anywhere:

```python
# For REST API
from backend.qa_service import validate_question_input, run_prediction

# For CLI tool
from backend.model import QAModel

# For batch processing
from backend.qa_service import load_qa_model
```

---

## Documentation

- **ARCHITECTURE.md** – Detailed architecture and design patterns
- **PROJECT_STRUCTURE.txt** – Visual tree and dependencies
- **REFACTORING_SUMMARY.md** – Before/after comparison
- **README.md** – Project overview

---

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run frontend/app.py
```

---

## Troubleshooting

### Model fails to load
```
Error: Failed to load model

Solution:
- Ensure internet connection
- Run: pip install transformers torch
```

### Import errors
```
Error: ModuleNotFoundError: No module named 'backend'

Solution:
- Ensure you run from project root
- Check sys.path in frontend/app.py
```

---

## Best Practices

✅ **Do:**
- Keep UI logic in `frontend/`
- Keep business logic in `backend/`
- Reuse backend services
- Write tests for backend services
- Document new modules

❌ **Don't:**
- Mix UI and ML logic
- Put Streamlit code in backend
- Add new imports to app.py unnecessarily
- Modify old app/ and src/ files

---

## Next Steps

1. ✅ Run the app: `streamlit run frontend/app.py`
2. ✅ Test with examples
3. ✅ Read ARCHITECTURE.md for deep dive
4. ✅ (Optional) Add REST API with FastAPI
5. ✅ (Optional) Add unit tests

---

## Questions?

See detailed documentation:
- **Architecture:** `ARCHITECTURE.md`
- **Structure:** `PROJECT_STRUCTURE.txt`
- **Refactoring:** `REFACTORING_SUMMARY.md`

**Happy coding! 🚀**
