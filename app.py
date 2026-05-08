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


st.markdown(
    """
    <style>
    .main { padding-top: 1rem; }
    .block-container { padding-top: 2rem; max-width: 1100px; }
    h1 { color: #1f2937; font-weight: 700; }
    .subtitle { color: #6b7280; font-size: 1.05rem; margin-top: -0.5rem; margin-bottom: 1.5rem; }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 10px;
        text-align: center;
    }
    .stat-card .label { font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; opacity: 0.9; }
    .stat-card .value { font-size: 1.6rem; font-weight: 700; margin-top: 0.25rem; }
    .summary-box {
        background: #f8fafc;
        border-left: 4px solid #667eea;
        padding: 1.25rem 1.5rem;
        border-radius: 6px;
        line-height: 1.65;
        color: #1f2937;
        white-space: pre-wrap;
    }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.55rem 1.25rem;
        border-radius: 8px;
    }
    .stButton > button:hover, .stDownloadButton > button:hover { opacity: 0.92; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading BART summarization model (first run downloads ~1.6GB)…")
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
    combined = " ".join(pieces)
    if len(chunks) > 1 and len(combined.split()) > max_words * 1.5:
        final = summarizer(
            combined,
            max_length=max_tokens,
            min_length=min_tokens,
            do_sample=False,
            truncation=True,
        )
        combined = final[0]["summary_text"].strip()
    return combined


def stat_card(label: str, value: str) -> str:
    return f'<div class="stat-card"><div class="label">{label}</div><div class="value">{value}</div></div>'


def main() -> None:
    st.title("📄 AI Research Paper Summarizer")
    st.markdown(
        '<div class="subtitle">Upload a PDF and get a concise summary powered by '
        'Hugging Face <code>facebook/bart-large-cnn</code>.</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("⚙️ Settings")
        max_words = st.slider("Summary length (words)", 50, 150, 100, step=10)
        st.caption("Longer summaries preserve more detail; shorter ones are punchier.")
        st.divider()
        st.markdown("**Model:** `facebook/bart-large-cnn`")
        st.markdown("**Chunking:** ~700 words per pass")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], accept_multiple_files=False)

    if uploaded is None:
        st.info("👆 Drop a research paper PDF above to get started.")
        return

    file_bytes = uploaded.read()
    try:
        with st.spinner("Extracting text from PDF…"):
            text = extract_pdf_text(file_bytes)
    except PdfReadError:
        st.error("This file does not appear to be a valid PDF, or it is corrupted.")
        return
    except ValueError as e:
        st.error(str(e))
        return
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        return

    if not text.strip():
        st.warning(
            "No extractable text found in this PDF — it may be a scanned/image-based document. "
            "OCR support (pytesseract) can be added as a fallback."
        )
        return

    word_count = len(text.split())
    char_count = len(text)
    cols = st.columns(3)
    cols[0].markdown(stat_card("Words extracted", f"{word_count:,}"), unsafe_allow_html=True)
    cols[1].markdown(stat_card("Characters", f"{char_count:,}"), unsafe_allow_html=True)
    cols[2].markdown(stat_card("Target length", f"{max_words} words"), unsafe_allow_html=True)

    with st.expander("Preview extracted text"):
        st.text(text[:2000] + ("…" if len(text) > 2000 else ""))

    if st.button("✨ Generate summary"):
        try:
            summarizer = load_summarizer()
        except Exception as e:
            st.error(f"Failed to load summarization model: {e}")
            return

        start = time.time()
        try:
            summary = summarize_text(summarizer, text, max_words)
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            return
        elapsed = time.time() - start

        st.subheader("📝 Summary")
        st.markdown(f'<div class="summary-box">{summary}</div>', unsafe_allow_html=True)

        summary_words = len(summary.split())
        cols = st.columns(3)
        cols[0].markdown(stat_card("Summary words", str(summary_words)), unsafe_allow_html=True)
        cols[1].markdown(
            stat_card("Compression", f"{(1 - summary_words / max(word_count, 1)) * 100:.1f}%"),
            unsafe_allow_html=True,
        )
        cols[2].markdown(stat_card("Time", f"{elapsed:.1f}s"), unsafe_allow_html=True)

        base_name = uploaded.name.rsplit(".", 1)[0]
        st.download_button(
            label="⬇️ Download summary (.txt)",
            data=summary,
            file_name=f"{base_name}_summary.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
