import tempfile
from pathlib import Path

import streamlit as st

from docmind.config import get_settings
from docmind.embeddings import BgeEmbedder
from docmind.index import Index
from docmind.llm import GeminiProvider
from docmind.parsing import ParseError
from docmind.qa import answer_question
from docmind.vector_store import PineconeVectorStore

st.set_page_config(page_title="DocMind AI", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading models and connecting to Pinecone...")
def get_index() -> Index:
    settings = get_settings()
    vs = PineconeVectorStore(api_key=settings.pinecone_api_key, index_name=settings.pinecone_index)
    return Index(BgeEmbedder(), vs)


@st.cache_resource
def get_llm() -> GeminiProvider:
    settings = get_settings()
    return GeminiProvider(settings.gemini_api_key, settings.gemini_model)


index = get_index()
llm = get_llm()

st.title("📄 DocMind AI")
st.caption("Multimodal document intelligence + grounded RAG — local embeddings, managed vector search, Gemini generation.")

with st.sidebar:
    st.subheader("Documents")
    uploaded = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
    if uploaded is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / uploaded.name
            path.write_bytes(uploaded.getvalue())
            try:
                with st.spinner(f"Processing {uploaded.name}..."):
                    doc_id, n, status = index.add_file(path)
                if status == "indexed":
                    st.success(f"Indexed {n} chunks from {uploaded.name}")
                elif status == "duplicate":
                    st.info(f"{uploaded.name} is already in the index.")
                else:  # no_text
                    st.warning(
                        f"{uploaded.name} appears to be a scanned/image-based document. "
                        "OCR extraction is on the roadmap but not yet enabled in this build."
                    )
            except ParseError as e:
                st.error(str(e))

    st.divider()
    docs = index.documents()
    if not docs:
        st.caption("No documents indexed yet.")
    for d in docs:
        col1, col2 = st.columns([4, 1])
        col1.write(f"**{d['doc_name']}**  \n{d['n_chunks']} chunks")
        if col2.button("🗑️", key=f"del-{d['doc_id']}"):
            index.delete_document(d["doc_id"])
            st.rerun()

st.subheader("Structured extraction")
docs = index.documents()
if docs:
    col1, col2, col3 = st.columns([3, 2, 1])
    doc_choice = col1.selectbox(
        "Document", options=[d["doc_id"] for d in docs],
        format_func=lambda did: next(d["doc_name"] for d in docs if d["doc_id"] == did),
    )
    doc_type = col2.selectbox("Type", ["invoice", "resume", "contract"])
    if col3.button("Extract"):
        from docmind.extraction import SCHEMAS

        with st.spinner("Extracting structured data..."):
            text = index.document_text(doc_choice)
            try:
                result = llm.extract_structured(text, SCHEMAS[doc_type])
                data = result.model_dump()
                line_items = data.pop("line_items", None)

                cols = st.columns(2)
                for i, (field, value) in enumerate(data.items()):
                    label = field.replace("_", " ").title()
                    cols[i % 2].metric(label, value if value not in (None, "") else "—")

                if line_items:
                    st.write("**Line items**")
                    headers = list(line_items[0].keys())
                    header_row = "| " + " | ".join(h.replace("_", " ").title() for h in headers) + " |"
                    sep_row = "|" + "---|" * len(headers)
                    rows = [
                        "| " + " | ".join(str(item.get(h, "")) for h in headers) + " |"
                        for item in line_items
                    ]
                    st.markdown("\n".join([header_row, sep_row, *rows]))

                with st.expander("Raw JSON (validated Pydantic output)"):
                    st.json(data | ({"line_items": line_items} if line_items else {}))
            except Exception as e:
                st.error(f"Extraction failed: {e}")
else:
    st.caption("Upload a document above to try structured extraction.")

st.divider()
st.subheader("Ask a question")
query = st.text_input("Your question", placeholder="e.g. When is payment due?")

if st.button("Ask", type="primary") and query:
    with st.spinner("Retrieving evidence and generating answer..."):
        result = answer_question(index, llm, query)

    from docmind.qa import is_no_answer

    if is_no_answer(result):
        st.warning(result.answer)
    else:
        st.write(result.answer)
        with st.expander("Sources"):
            for c in result.citations:
                st.write(f"- {c}")
    if result.top_score is not None:
        st.caption(f"Top retrieval score: {result.top_score:.3f}")