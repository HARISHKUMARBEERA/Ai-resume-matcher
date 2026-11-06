"""Explainable resume-to-job matching using TF-IDF and rule-based skill extraction.

This is a *similarity prototype*, not a classifier of applicant quality or a
model trained on hiring outcomes. Never use its scores to make employment
or eligibility decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Explicit, auditable vocabulary. Match complete terms rather than substrings.
# For example, "go" should never match the "go" in "ongoing".
SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    "Python": ("python",),
    "Java": ("java",),
    "JavaScript": ("javascript", "js",),
    "TypeScript": ("typescript",),
    "React": ("react", "react.js", "reactjs"),
    "Angular": ("angular",),
    "Vue": ("vue", "vue.js", "vuejs"),
    "Node.js": ("node.js", "nodejs"),
    "FastAPI": ("fastapi",),
    "Flask": ("flask",),
    "Django": ("django",),
    "Spring Boot": ("spring boot",),
    "SQL": ("sql",),
    "PostgreSQL": ("postgresql", "postgres",),
    "MySQL": ("mysql",),
    "MongoDB": ("mongodb", "mongo db"),
    "Redis": ("redis",),
    "AWS": ("aws", "amazon web services"),
    "Azure": ("azure", "microsoft azure"),
    "Google Cloud": ("gcp", "google cloud", "google cloud platform"),
    "Docker": ("docker",),
    "Kubernetes": ("kubernetes", "k8s"),
    "Git": ("git",),
    "GitHub Actions": ("github actions",),
    "CI/CD": ("ci/cd", "continuous integration", "continuous deployment"),
    "REST APIs": ("rest api", "rest apis", "restful api", "restful apis"),
    "GraphQL": ("graphql",),
    "Machine Learning": ("machine learning",),
    "Deep Learning": ("deep learning",),
    "NLP": ("nlp", "natural language processing"),
    "Computer Vision": ("computer vision",),
    "PyTorch": ("pytorch",),
    "TensorFlow": ("tensorflow",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    "Pandas": ("pandas",),
    "NumPy": ("numpy",),
    "LLMs": ("llm", "llms", "large language model", "large language models"),
    "RAG": ("rag", "retrieval augmented generation", "retrieval-augmented generation"),
    "LangChain": ("langchain",),
    "LangGraph": ("langgraph",),
    "Data Analysis": ("data analysis",),
    "Data Visualization": ("data visualization",),
    "Linux": ("linux",),
    "HTML": ("html",),
    "CSS": ("css",),
}


@dataclass(frozen=True)
class MatchReport:
    semantic_similarity: float   # 0..100, not a hiring decision score
    skill_coverage: float         # percentage of JD-listed known skills in resume
    matched_skills: tuple[str, ...]
    missing_skills: tuple[str, ...]
    jd_skills: tuple[str, ...]
    resume_skills: tuple[str, ...]
    notes: tuple[str, ...]


def _contains_term(text: str, phrase: str) -> bool:
    """Case-insensitive phrase boundary matching including '+'/'#' punctuation."""
    return bool(re.search(r"(?<![\w])" + re.escape(phrase) + r"(?![\w])", text, re.I))


def extract_skills(text: str) -> tuple[str, ...]:
    """Extract a transparent shortlist of technical skills, alphabetically."""
    if not text or not text.strip():
        return ()
    return tuple(sorted(
        name for name, aliases in SKILL_ALIASES.items()
        if any(_contains_term(text, alias) for alias in aliases)
    ))


def _similarity(resume: str, job_description: str) -> float:
    """Blend word and character n-gram cosine similarity.

    Models are fitted *only* on the two supplied documents, so these numbers
    compare lexical overlap, rather than learned hiring outcomes or meaning.
    """
    vectors = []
    for analyzer, ngram_range, weight in [
        ("word", (1, 2), 0.7),
        ("char_wb", (3, 5), 0.3),
    ]:
        try:
            features = TfidfVectorizer(
                analyzer=analyzer,
                ngram_range=ngram_range,
                stop_words="english" if analyzer == "word" else None,
                sublinear_tf=True,
            ).fit_transform([resume, job_description])
            similarity = float(cosine_similarity(features[0:1], features[1:2])[0, 0])
        except ValueError:  # e.g. both inputs are only English stop words
            similarity = 0.0
        vectors.append(weight * similarity)
    return round(min(100.0, max(0.0, sum(vectors) * 100)), 1)


def analyze_match(resume: str, job_description: str) -> MatchReport:
    """Analyze nonempty text inputs. Raises ValueError for missing input."""
    resume = resume.strip()
    job_description = job_description.strip()
    if not resume:
        raise ValueError("Please provide some resume text.")
    if not job_description:
        raise ValueError("Please provide a job description.")

    resume_skills = extract_skills(resume)
    jd_skills = extract_skills(job_description)
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)
    matched = tuple(sorted(resume_set & jd_set))
    missing = tuple(sorted(jd_set - resume_set))
    coverage = round(100 * len(matched) / len(jd_skills), 1) if jd_skills else 0.0

    notes = [
        "Similarity measures text overlap, not candidate suitability or qualification.",
        "Skills use a limited, editable phrase list; missing skills may still be present under other wording.",
    ]
    if not jd_skills:
        notes.append("No skills in the built-in vocabulary were found in the job description; skill coverage is unavailable.")
    return MatchReport(
        semantic_similarity=_similarity(resume, job_description),
        skill_coverage=coverage,
        matched_skills=matched,
        missing_skills=missing,
        jd_skills=jd_skills,
        resume_skills=resume_skills,
        notes=tuple(notes),
    )
