import logging
from typing import Protocol

log = logging.getLogger("docmind.llm")

SYSTEM_PROMPT = (
    "You are a document QA assistant. Answer ONLY using the provided context excerpts. "
    "Each excerpt is labeled with its source citation in [brackets] - reference the "
    "relevant ones in your answer. If the context does not contain the answer, say so "
    "plainly instead of guessing or using outside knowledge."
)


class LLMProvider(Protocol):
    def generate_answer(self, query: str, context: list[tuple[str, str]]) -> str: ...


def _build_prompt(query: str, context: list[tuple[str, str]]) -> str:
    blocks = "\n\n".join(f"[{label}]\n{text}" for label, text in context)
    return f"{SYSTEM_PROMPT}\n\nContext:\n{blocks}\n\nQuestion: {query}\nAnswer:"


class GeminiProvider:
    def __init__(self, api_key: str, model: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate_answer(self, query: str, context: list[tuple[str, str]]) -> str:
        try:
            resp = self._client.models.generate_content(
                model=self._model, contents=_build_prompt(query, context)
            )
            return resp.text
        except Exception as e:
            log.error("Gemini generation failed: %s", e)
            return "The answer could not be generated right now (LLM provider error)."