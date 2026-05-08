import io
from typing import List

import streamlit as st
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError


CHUNK_WORDS = 700
MIN_CHUNK_WORDS = 30


st.set_page_config(
    page_title="AI Research Paper Summarizer",
    page_icon="📄",
    layout="wide",
)


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise ValueError("PDF is password-protected and cannot be read.")
    parts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n".join(parts)


def chunk_text(text: str, words_per_chunk: int = CHUNK_WORDS) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]
    if len(chunks) > 1 and len(chunks[-1].split()) < MIN_CHUNK_WORDS:
        tail = chunks.pop()
        chunks[-1] = chunks[-1] + " " + tail
    return chunks


def main() -> None:
    st.title("📄 AI Research Paper Summarizer")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is None:
        st.info("Drop a PDF above to extract its text.")
        return

    try:
        text = extract_pdf_text(uploaded.read())
    except PdfReadError:
        st.error("This file does not appear to be a valid PDF, or it is corrupted.")
        return
    except ValueError as e:
        st.error(str(e))
        return

    if not text.strip():
        st.warning("No extractable text — likely a scanned/image PDF.")
        return

    st.success(f"Extracted {len(text.split()):,} words across {len(chunk_text(text))} chunk(s).")
    with st.expander("Preview extracted text"):
        st.text(text[:2000] + ("…" if len(text) > 2000 else ""))


if __name__ == "__main__":
    main()
