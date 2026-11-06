# 🔎 MatchWise AI — Explainable Resume & Job Description Analyzer

A small, practical **NLP project** that compares a resume and job description using **TF-IDF, cosine similarity, and interpretable skill extraction**. Built with Python, scikit-learn, Streamlit, and pypdf. **No API keys or paid services needed.**


## Features

- Upload a resume (`.pdf` or `.txt`) or paste resume text
- Paste a target job description
- Calculate TF-IDF-based **lexical** similarity (0–100%)
- Extract a curated list of technical skills and highlight overlap / gaps
- Explain how the percentages are computed
- Download a plain-text report
- Try a built-in example immediately

## Quick start

Requires **Python 3.10+**.

```bash
# Clone *your own* GitHub repository after publishing, or unzip the starter.
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Open the local URL displayed by Streamlit (usually `http://localhost:8501`). The example input is already filled in; click **Analyze match**.

### Run tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## How it works

1. `matcher/parsing.py` reads plain text or extracts selectable text from PDF resumes. Files are limited to 5 MB and PDFs to 25 pages.
2. `matcher/analysis.py` creates two TF-IDF spaces for the resume and job description: 70% **word uni-/bigrams** and 30% **character 3–5 grams**. It computes the cosine similarity for each and produces a weighted percentage.
3. The skill extractor uses a clearly defined `SKILL_ALIASES` vocabulary and phrase boundaries. It counts which skills from the job description are also explicitly present in the resume.
4. `app.py` presents metrics, matches, missing recognized skills, limitations, and a downloadable report.

**Important technical detail:** TF-IDF is a classical NLP / information-retrieval technique. This project does not use an LLM, transformer embeddings, or supervised learning, and the score is not a probability of being hired.

## Project structure

```text
matchwise-ai-resume-matcher/
├── app.py
├── matcher/
│   ├── __init__.py
│   ├── analysis.py
│   └── parsing.py
├── samples/
│   ├── resume.txt
│   └── job_description.txt
├── tests/
│   ├── test_analysis.py
│   └── test_parsing.py
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Limitations and sensible next improvements

- A scanned/image-only PDF needs OCR (not supported yet).
- TF-IDF measures overlapping words/phrases, not full semantic meaning. An embedding model could improve paraphrase handling.
- Skills not listed in `SKILL_ALIASES` are ignored; spelling and synonyms matter.
- A missing skill in the text does **not** show the candidate lacks that skill.
- The app does not persist uploaded files; do not upload sensitive documents to an untrusted public deployment.

**Extension ideas:** add sentence-transformers embeddings, multilingual support, more aliases, skill-category grouping, and user-requested job matching with an opt-in explanation dashboard.

## Ownership and attribution

If you use this starter, read and change the implementation, add your own contributions, test it, and truthfully describe it as a project you **built/adapted**. Never represent an unchanged third-party repository as solely your original work.

## License

Provided as a starter project for learning and adaptation. When publishing, choose a license appropriate for your own repository.
