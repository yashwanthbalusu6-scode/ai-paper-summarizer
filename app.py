import io
from typing import List

import streamlit as st
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError
from transformers import pipeline


MODEL_NAME = "facebook/bart-large-cnn"
CHUNK_WORDS = 700
MIN_CHUNK_WORDS = 30


st.set_page_config(
    page_title="AI Research Paper Summarizer",
    page_icon="📄",
    layout="wide",
)


def load_summarizer():
    return pipeline("summarization", model=MODEL_NAME)


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


def summarize_text(summarizer, text: str, max_words: int) -> str:
    chunks = chunk_text(text)
    if not chunks:
        return ""
    max_tokens = max(40, int(max_words * 1.3))
    min_tokens = max(20, int(max_tokens * 0.4))
    pieces: List[str] = []
    for chunk in chunks:
        out = summarizer(
            chunk,
            max_length=max_tokens,
            min_length=min_tokens,
            do_sample=False,
            truncation=True,
        )
        pieces.append(out[0]["summary_text"].strip())
    return " ".join(pieces)


def main() -> None:
    st.title("📄 AI Research Paper Summarizer")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is None:
        st.info("Drop a PDF above to summarize it.")
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

    if st.button("Generate summary"):
        summarizer = load_summarizer()
        summary = summarize_text(summarizer, text, max_words=100)
        st.subheader("Summary")
        st.write(summary)


if __name__ == "__main__":
    main()
