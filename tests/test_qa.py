from docmind.index import Index
from docmind.qa import answer_question
from tests.test_index import FakeEmbedder, FakeVectorStore, make_chunks


class FakeLLM:
    def __init__(self, reply="Payment is due within 30 days."):
        self.reply = reply
        self.last_context = None

    def generate_answer(self, query, context):
        self.last_context = context
        return self.reply


def build_index():
    index = Index(FakeEmbedder(), FakeVectorStore(), ":memory:")
    index.add_chunks(make_chunks())
    return index


def test_answers_when_evidence_found():
    index = build_index()
    llm = FakeLLM()
    result = answer_question(index, llm, "payment due days", min_score=0.05)
    assert not result.abstained
    assert result.answer == "Payment is due within 30 days."
    assert result.citations
    assert llm.last_context  # context was actually passed to the LLM


def test_abstains_when_no_relevant_evidence():
    index = build_index()
    llm = FakeLLM()
    result = answer_question(index, llm, "gibberish unrelated query xyz", min_score=0.99)
    assert result.abstained
    assert "don't have enough information" in result.answer
    assert llm.last_context is None  # LLM was never called


def test_citations_are_sorted_and_unique():
    index = build_index()
    result = answer_question(index, FakeLLM(), "payment due days", min_score=0.05, k=5)
    assert result.citations == sorted(set(result.citations))