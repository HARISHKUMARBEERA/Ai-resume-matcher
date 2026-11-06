import pytest
from pypdf import PdfWriter
from io import BytesIO

from matcher.parsing import extract_resume_text


def test_txt_parsing():
    assert extract_resume_text("resume.TXT", b"\xef\xbb\xbfPython engineer") == "Python engineer"


def test_unsupported_type():
    with pytest.raises(ValueError, match=".pdf or .txt"):
        extract_resume_text("resume.docx", b"data")


def test_empty_resume():
    with pytest.raises(ValueError, match="No selectable text"):
        extract_resume_text("resume.txt", b"  ")


def test_oversized_resume():
    with pytest.raises(ValueError, match="too large"):
        extract_resume_text("resume.txt", b"x" * (5 * 1024 * 1024 + 1))


def test_blank_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    pdf = BytesIO()
    writer.write(pdf)
    with pytest.raises(ValueError, match="No selectable text"):
        extract_resume_text("resume.pdf", pdf.getvalue())
