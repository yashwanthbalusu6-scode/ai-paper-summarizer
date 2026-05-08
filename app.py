import io
import time
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
    initial_sidebar_state="expanded",
)


@st.cache_resource
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
    progress = st.progress(0.0, text="Summarizing…")
    pieces: List[str] = []
    for i, chunk in enumerate(chunks, start=1):
        out = summarizer(
            chunk,
            max_length=max_tokens,
            min_length=min_tokens,
            do_sample=False,
            truncation=True,
        )
        pieces.append(out[0]["summary_text"].strip())
        progress.progress(i / len(chunks), text=f"Summarizing chunk {i}/{len(chunks)}…")
    progress.empty()
    return " ".join(pieces)


def main() -> None:
    st.title("📄 AI Research Paper Summarizer")
    st.caption("Upload a PDF and get a concise summary powered by facebook/bart-large-cnn.")

    with st.sidebar:
        st.header("⚙️ Settings")
        max_words = st.slider("Summary length (words)", 50, 150, 100, step=10)
        st.caption("Longer summaries preserve more detail.")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is None:
        st.info("Drop a PDF above to get started.")
        return

    try:
        with st.spinner("Extracting text…"):
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

    with st.expander("Preview extracted text"):
        st.text(text[:2000] + ("…" if len(text) > 2000 else ""))

    if st.button("✨ Generate summary"):
        summarizer = load_summarizer()
        start = time.time()
        summary = summarize_text(summarizer, text, max_words)
        elapsed = time.time() - start
        st.subheader("📝 Summary")
        st.write(summary)
        st.caption(f"Generated in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
