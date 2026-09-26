import logging
import time

from .index import Index
from .llm import LLMProvider
from .schemas import AnswerResult

log = logging.getLogger("docmind.qa")

# Below this cosine similarity, retrieved chunks are treated as unrelated to
# the question rather than weak evidence - tuned empirically, not a magic constant.
MIN_RELEVANT_SCORE = 0.35
NO_EVIDENCE_MSG = "I don't have enough information in the uploaded documents to answer that."
DECLINE_PHRASES = (
    "does not contain", "don't have enough information", "cannot find",
    "no information", "insufficient information",
)


def is_no_answer(result: AnswerResult) -> bool:
    """True if the system produced no real answer - either it abstained before
    calling the LLM (retrieval score too low), or the LLM itself declined."""
    return result.abstained or any(p in result.answer.lower() for p in DECLINE_PHRASES)


def answer_question(
    index: Index,
    llm: LLMProvider,
    query: str,
    k: int = 5,
    min_score: float = MIN_RELEVANT_SCORE,
) -> AnswerResult:
    t0 = time.perf_counter()
    hits = index.search(query, k=k, mode="hybrid")
    t1 = time.perf_counter()
    log.info("Retrieval took %.2fs", t1 - t0)
    top_score = max((h.semantic_score or 0.0) for h in hits) if hits else None

    relevant = [h for h in hits if (h.semantic_score or 0.0) >= min_score]
    if not relevant:
        log.info("Abstaining: top_score=%s below threshold=%s", top_score, min_score)
        return AnswerResult(query=query, answer=NO_EVIDENCE_MSG, abstained=True, top_score=top_score)

    context = [(h.chunk.citation, h.chunk.text) for h in relevant]
    answer = llm.generate_answer(query, context)
    t2 = time.perf_counter()
    log.info("Gemini generation took %.2fs", t2 - t1)
    citations = sorted({h.chunk.citation for h in relevant})
    return AnswerResult(query=query, answer=answer, abstained=False, citations=citations, top_score=top_score)