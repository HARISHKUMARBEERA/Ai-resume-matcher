import pytest

from matcher.analysis import analyze_match, extract_skills


def test_skill_synonyms_and_boundaries():
    assert "React" in extract_skills("Built with React.js")
    assert "AWS" in extract_skills("Experience with Amazon Web Services")
    assert "Git" not in extract_skills("digital strategy")
    assert "Java" not in extract_skills("Strong JavaScript expertise")
    assert "RAG" not in extract_skills("storage solutions")


def test_skill_coverage_and_missing():
    result = analyze_match("Python Flask React Docker", "Python React Kubernetes")
    assert result.skill_coverage == 66.7
    assert result.matched_skills == ("Python", "React")
    assert result.missing_skills == ("Kubernetes",)


def test_identical_document_has_max_similarity():
    result = analyze_match("Python React Docker API", "Python React Docker API")
    assert result.semantic_similarity == pytest.approx(100.0)


def test_blank_document_rejected():
    with pytest.raises(ValueError, match="resume"):
        analyze_match("", "Python engineer")
    with pytest.raises(ValueError, match="job description"):
        analyze_match("Python developer", "  ")


def test_no_known_skills_is_not_100_percent_coverage():
    report = analyze_match("Worked with customers", "Excellent collaboration")
    assert report.skill_coverage == 0
    assert not report.jd_skills
    assert any("unavailable" in note for note in report.notes)
