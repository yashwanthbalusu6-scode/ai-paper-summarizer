# AI Research Paper Summarizer

AI-powered research paper summarizer using Hugging Face Transformers and Streamlit. Drop in a PDF and get a concise, length-controlled summary in seconds.

> **Live demo:** _add your Streamlit Cloud URL here after deploying._

## Features

- Clean Streamlit UI with custom styling and sidebar controls
- PDF text extraction via `PyPDF2` with graceful handling of corrupted/encrypted files
- Automatic chunking so long papers fit inside BART's 1024-token context
- Summarization with `facebook/bart-large-cnn`
- Adjustable summary length (50–150 words)
- Compression and timing statistics
- One-click download of the summary as a `.txt` file
- Model is cached with `@st.cache_resource` so it loads only once per session

## Stack

| Layer | Tool |
| --- | --- |
| UI | Streamlit 1.28 |
| Model | `facebook/bart-large-cnn` via `transformers` 4.35 |
| Inference | PyTorch 2.1 |
| PDF parsing | PyPDF2 3.0 |

## Quickstart

```bash
git clone https://github.com/<your-username>/ai-paper-summarizer.git
cd ai-paper-summarizer

python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>. The first summarization downloads the BART model (~1.6 GB) into the Hugging Face cache; subsequent runs are fast.

## Docker

```bash
docker build -t ai-paper-summarizer .
docker run --rm -p 8501:8501 ai-paper-summarizer
```

Mount a host directory at `/app/.cache/huggingface` to persist the model between runs:

```bash
docker run --rm -p 8501:8501 -v $HOME/.cache/huggingface:/app/.cache/huggingface ai-paper-summarizer
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub (public).
2. Go to <https://streamlit.io/cloud> and sign in with GitHub.
3. **New app** → pick this repo, branch `main`, main file `app.py`.
4. Deploy. The first build pulls dependencies (~5 min) and the first run pulls the model.

## Project layout

```
ai-paper-summarizer/
├── app.py              # Streamlit entry point
├── requirements.txt    # Pinned dependencies
├── Dockerfile          # Container build
├── .gitignore
└── README.md
```

## Notes & limits

- BART's input window is 1024 tokens. Long papers are chunked, summarized per chunk, then optionally re-summarized to hit the target length.
- Scanned/image-based PDFs have no extractable text. Add `pytesseract` + `pdf2image` for OCR fallback if needed.
- First run downloads ~1.6 GB of model weights — keep that in mind on Streamlit Cloud's free tier.

## License

MIT
