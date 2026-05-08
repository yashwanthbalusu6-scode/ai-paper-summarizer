import streamlit as st


st.set_page_config(
    page_title="AI Research Paper Summarizer",
    page_icon="📄",
    layout="wide",
)


def main() -> None:
    st.title("📄 AI Research Paper Summarizer")
    st.write("Coming soon — PDF upload and BART summarization.")


if __name__ == "__main__":
    main()
