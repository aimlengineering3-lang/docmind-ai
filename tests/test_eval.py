from docmind.eval import EvalItem, evaluate_item
from docmind.index import Index
from tests.test_index import FakeEmbedder, FakeVectorStore, make_chunks
from tests.test_qa import FakeLLM


def build_index():
    index = Index(FakeEmbedder(), FakeVectorStore(), ":memory:")
    index.add_chunks(make_chunks())
    return index


def test_retrieval_and_keywords_pass():
    index = build_index()
    llm = FakeLLM("Payment is due within 30 days.")
    item = EvalItem(id="q1", query="payment due days", expected_doc="a.pdf",
                     expected_keywords=["30 days"])
    result = evaluate_item(index, llm, item)
    assert result.retrieval_hit is True
    assert result.keywords_found is True
    assert result.abstain_correct is True


def test_keywords_fail_when_missing():
    index = build_index()
    llm = FakeLLM("The document mentions payment terms.")
    item = EvalItem(id="q2", query="payment due days", expected_keywords=["30 days"])
    result = evaluate_item(index, llm, item)
    assert result.keywords_found is False


def test_abstain_expected_and_declined():
    index = build_index()
    llm = FakeLLM("I don't have enough information to answer that.")
    item = EvalItem(id="q3", query="unrelated gibberish query", should_abstain=True)
    result = evaluate_item(index, llm, item)
    assert result.abstain_correct is True


def test_abstain_expected_but_answered_is_incorrect():
    index = build_index()
    llm = FakeLLM("Payment is due within 30 days.")
    item = EvalItem(id="q4", query="payment due days", should_abstain=True)
    result = evaluate_item(index, llm, item)
    assert result.abstain_correct is False