import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .config import get_settings
from .extraction import SCHEMAS
from .index import Index, build_index
from .llm import GeminiProvider
from .parsing import ParseError
from .qa import answer_question

log = logging.getLogger("docmind.api")

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "index" not in _state:
        settings = get_settings()
        _state["index"] = build_index()
        _state["llm"] = GeminiProvider(settings.gemini_api_key, settings.gemini_model)
    yield
    _state.clear()


app = FastAPI(title="DocMind AI API", lifespan=lifespan)


def get_index() -> Index:
    return _state["index"]


def get_llm() -> GeminiProvider:
    return _state["llm"]


class DocumentOut(BaseModel):
    doc_id: str
    doc_name: str
    n_chunks: int


class UploadOut(BaseModel):
    doc_id: str
    chunks_indexed: int
    status: str


class AskIn(BaseModel):
    query: str
    k: int = 5


class AskOut(BaseModel):
    answer: str
    abstained: bool
    citations: list[str]
    top_score: float | None


@app.get("/documents", response_model=list[DocumentOut])
def list_documents():
    return [
        DocumentOut(doc_id=d["doc_id"], doc_name=d["doc_name"], n_chunks=d["n_chunks"])
        for d in get_index().documents()
    ]


@app.post("/documents", response_model=UploadOut)
async def upload_document(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / file.filename
        path.write_bytes(await file.read())
        try:
            doc_id, n, status = get_index().add_file(path)
        except ParseError as e:
            raise HTTPException(status_code=422, detail=str(e))
        return UploadOut(doc_id=doc_id, chunks_indexed=n, status=status)


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    deleted = get_index().delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No such document")
    return {"deleted": True}


@app.post("/ask", response_model=AskOut)
def ask(payload: AskIn):
    result = answer_question(get_index(), get_llm(), payload.query, k=payload.k)
    return AskOut(
        answer=result.answer, abstained=result.abstained,
        citations=result.citations, top_score=result.top_score,
    )


@app.post("/extract/{doc_id}/{doc_type}")
def extract(doc_id: str, doc_type: str):
    if doc_type not in SCHEMAS:
        raise HTTPException(status_code=400, detail=f"Unknown doc_type: {doc_type}")
    text = get_index().document_text(doc_id)
    if not text:
        raise HTTPException(status_code=404, detail="No such document, or it has no text")
    try:
        result = get_llm().extract_structured(text, SCHEMAS[doc_type])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Extraction failed: {e}")
    return result.model_dump()