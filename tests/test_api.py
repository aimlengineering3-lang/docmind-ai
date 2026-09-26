import pytest
from fastapi.testclient import TestClient

from docmind import api
from docmind.index import Index
from tests.test_index import FakeEmbedder, FakeVectorStore, make_chunks
from tests.test_qa import FakeLLM


@pytest.fixture
def client():
    api._state["index"] = Index(FakeEmbedder(), FakeVectorStore(), ":memory:")
    api._state["index"].add_chunks(make_chunks())
    api._state["llm"] = FakeLLM("Payment is due within 30 days.")
    with TestClient(api.app) as c:
        yield c
    api._state.clear()


def test_list_documents(client):
    resp = client.get("/documents")
    assert resp.status_code == 200
    assert resp.json()[0]["doc_id"] == "d1"


def test_ask_returns_grounded_answer(client):
    resp = client.post("/ask", json={"query": "payment due days"})
    assert resp.status_code == 200
    body = resp.json()
    assert "30 days" in body["answer"]
    assert body["citations"]


def test_delete_missing_document_404(client):
    resp = client.delete("/documents/does-not-exist")
    assert resp.status_code == 404