import logging
from typing import Protocol

log = logging.getLogger("docmind.llm")

SYSTEM_PROMPT = (
    "You are a document QA assistant. Answer ONLY using the provided context excerpts. "
    "Each excerpt is labeled with a citation - cite the ones you used. "
    "If the context is insufficient, say so plainly instead of guessing."
)


class LLMProvider(Protocol):
    def generate_answer(self, query: str, context: list[tuple[str, str]]) -> str: ...


def _build_prompt(query: str, context: list[tuple[str, str]]) -> str:
    blocks = "\n\n".join(f"[{label}]\n{text}" for label, text in context)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{blocks}\n\nQuestion: {query}\nAnswer:"


class GeminiProvider:
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def generate_answer(self, query: str, context: list[tuple[str, str]]) -> str:
        resp = self._model.generate_content(_build_prompt(query, context))
        return resp.text