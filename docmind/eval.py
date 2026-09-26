import argparse
import json
import logging
from pathlib import Path

from pydantic import BaseModel

from .index import DEFAULT_DB, Index, build_index
from .qa import answer_question, is_no_answer

log = logging.getLogger("docmind.eval")

DEFAULT_DATASET = "eval/dataset.json"


class EvalItem(BaseModel):
    id: str
    query: str
    expected_doc: str | None = None  # substring match against doc_name
    expected_keywords: list[str] = []
    should_abstain: bool = False


class EvalResult(BaseModel):
    id: str
    query: str
    retrieval_hit: bool | None
    keywords_found: bool | None
    abstain_correct: bool
    answer: str


def load_dataset(path: str | Path) -> list[EvalItem]:
    data = json.loads(Path(path).read_text())
    return [EvalItem(**d) for d in data]


def evaluate_item(index: Index, llm, item: EvalItem, k: int = 5) -> EvalResult:
    hits = index.search(item.query, k=k, mode="hybrid")
    retrieval_hit = (
        any(item.expected_doc.lower() in h.chunk.doc_name.lower() for h in hits)
        if item.expected_doc else None
    )

    result = answer_question(index, llm, item.query, k=k)

    keywords_found = (
        all(kw.lower() in result.answer.lower() for kw in item.expected_keywords)
        if item.expected_keywords else None
    )

    abstain_correct = item.should_abstain == is_no_answer(result)

    return EvalResult(
        id=item.id, query=item.query, retrieval_hit=retrieval_hit,
        keywords_found=keywords_found, abstain_correct=abstain_correct, answer=result.answer,
    )


def run(dataset_path: str, index: Index, llm) -> list[EvalResult]:
    items = load_dataset(dataset_path)
    return [evaluate_item(index, llm, item) for item in items]


def _rate(values: list[bool]) -> str:
    return f"{sum(values)}/{len(values)}" if values else "n/a"


def print_report(results: list[EvalResult]) -> None:
    def mark(b):
        return "-" if b is None else ("PASS" if b else "FAIL")

    for r in results:
        print(
            f"[{r.id}] retrieval={mark(r.retrieval_hit)} keywords={mark(r.keywords_found)} "
            f"abstain={'PASS' if r.abstain_correct else 'FAIL'}  {r.query}"
        )

    retrieval = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    keywords = [r.keywords_found for r in results if r.keywords_found is not None]
    abstain = [r.abstain_correct for r in results]
    print("\n--- Summary ---")
    print(f"Retrieval hit rate:   {_rate(retrieval)}")
    print(f"Answer keyword rate:  {_rate(keywords)}")
    print(f"Abstention accuracy:  {_rate(abstain)}")


def main(argv: list[str] | None = None) -> None:
    from .config import get_settings
    from .llm import GeminiProvider

    ap = argparse.ArgumentParser(prog="python -m docmind.eval")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    settings = get_settings()
    index = build_index(args.db)
    llm = GeminiProvider(settings.gemini_api_key, settings.gemini_model)

    results = run(args.dataset, index, llm)
    print_report(results)


if __name__ == "__main__":
    main()