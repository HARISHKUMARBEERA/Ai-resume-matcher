"""Minimal .pdf and .txt resume parsing with safe user-facing errors."""

from __future__ import annotations

from io import BytesIO
from pypdf import PdfReader

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def extract_resume_text(name: str, data: bytes) -> str:
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File is too large (maximum 5 MB).")
    lower = name.lower()
    if lower.endswith(".txt"):
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError("Text file must be UTF-8 encoded.") from exc
    elif lower.endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("Password-protected PDF files are not supported.")
            if len(reader.pages) > 25:
                raise ValueError("PDF is too long (maximum 25 pages).")
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Could not read this PDF. Try a text-based PDF or .txt file.") from exc
    else:
        raise ValueError("Please upload a .pdf or .txt file.")
    text = text.strip()
    if not text:
        raise ValueError("No selectable text found. Scanned image PDFs need OCR; try .txt instead.")
    return text
