"""Streamlit entry point for MatchWise AI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from matcher.analysis import analyze_match
from matcher.parsing import extract_resume_text

BASE = Path(__file__).resolve().parent

st.set_page_config(page_title="MatchWise | Resume Analyzer", page_icon="🔎", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1080px; padding-top: 2rem;}
.small-note {color:#788399; font-size:0.91rem;}
</style>
""", unsafe_allow_html=True)

st.title("🔎 MatchWise AI")
st.caption("Explainable NLP resume ↔ job-description comparison · no paid API required")

with st.sidebar:
    st.header("About this demo")
    st.write("Uses TF-IDF, cosine similarity, and rule-based technical skill extraction.")
    st.info("This is a learning prototype, **not** a hiring or candidate-screening tool.")
    st.write("Your resume is processed in the running app session. The app does not store or send it to an AI API.")
    st.divider()
    st.caption("Tip: The provided example resume and job description work immediately.")

sample_resume = (BASE / "samples" / "resume.txt").read_text(encoding="utf-8")
sample_jd = (BASE / "samples" / "job_description.txt").read_text(encoding="utf-8")

left, right = st.columns(2, gap="large")
with left:
    st.subheader("1. Your resume")
    resume_file = st.file_uploader("Upload a PDF or TXT (max 5 MB)", type=["pdf", "txt"])
    st.caption("Or paste resume text below; uploaded file takes priority.")
    resume_text = st.text_area("Resume text", sample_resume, height=265)
with right:
    st.subheader("2. Target job description")
    jd_text = st.text_area("Paste the job description", sample_jd, height=349)

if st.button("Analyze match", type="primary", use_container_width=True):
    try:
        source_text = extract_resume_text(resume_file.name, resume_file.getvalue()) if resume_file else resume_text
        result = analyze_match(source_text, jd_text)
    except ValueError as exc:
        st.error(str(exc))
    else:
        st.divider()
        st.subheader("3. Match report")
        a, b, c = st.columns(3)
        a.metric("Text similarity", f"{result.semantic_similarity:.1f}%")
        b.metric("Skill coverage", f"{result.skill_coverage:.1f}%" if result.jd_skills else "N/A")
        c.metric("Matched skills", f"{len(result.matched_skills)} / {len(result.jd_skills)}")
        st.progress(result.semantic_similarity / 100, text="TF-IDF lexical similarity")
        if result.jd_skills:
            st.progress(result.skill_coverage / 100, text="Recognized job skills found in resume")

        matched_col, missing_col = st.columns(2, gap="large")
        with matched_col:
            st.success("**Skills found in both texts**")
            if result.matched_skills:
                for skill in result.matched_skills:
                    st.write(f"✅ {skill}")
            else:
                st.write("No recognized overlapping skills.")
        with missing_col:
            st.warning("**Job skills not found in resume text**")
            if result.missing_skills:
                for skill in result.missing_skills:
                    st.write(f"◻️ {skill}")
            else:
                st.write("None among recognized job skills.")

        with st.expander("Methodology and limitations", expanded=True):
            st.write("**Text similarity:** weighted cosine similarity of word (70%) and character (30%) TF-IDF vectors. This measures lexical resemblance, not deep semantic understanding.")
            st.write("**Skill coverage:** matched recognized job skills divided by all recognized job skills, not a probabilistic recommendation.")
            for note in result.notes:
                st.caption(f"• {note}")

        report_text = "\n".join([
            "MatchWise AI report",
            f"Text similarity: {result.semantic_similarity:.1f}%",
            f"Skill coverage: {result.skill_coverage:.1f}%" if result.jd_skills else "Skill coverage: N/A",
            "Matched: " + (", ".join(result.matched_skills) or "None"),
            "Not found: " + (", ".join(result.missing_skills) or "None"),
            "Disclaimer: Text similarity is not a candidate fit or hiring score.",
        ])
        st.download_button("Download text report", data=report_text, file_name="matchwise_report.txt", mime="text/plain")
