import os
import pdfplumber
from pathlib import Path
from typing import Union
import streamlit as st


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def save_uploaded_file(uploaded_file) -> str:
    """
    Saves a Streamlit uploaded file to the uploads/ directory.
    Returns the saved file path.
    """
    file_path = UPLOAD_DIR / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(file_path)


def extract_text_from_pdf(path: Union[str, Path]) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

    return "\n".join(text_parts)


def load_text(path: Union[str, Path]) -> str:
    """
    Load text content from a .txt file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")

    return path.read_text(encoding="utf-8")


def load_document(path: Union[str, Path]) -> str:
    """
    Load a document (PDF or TXT) and return its text content.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return extract_text_from_pdf(path)
    elif ext == ".txt":
        return load_text(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
