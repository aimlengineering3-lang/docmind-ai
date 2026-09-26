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
        self._model_name = model

    def generate_answer(self, query: str, context: list[tuple[str, str]]) -> str:
        try:
            resp = self._client.models.generate_content(
                model=self._model_name, contents=_build_prompt(query, context)
            )
            return resp.text
        except Exception as e:
            log.error("Gemini generation failed: %s", e)
            return "The answer could not be generated right now (LLM provider error)."

    def extract_structured(self, text: str, schema):
        """schema is a Pydantic BaseModel class. Returns a validated instance;
        raises on API failure or if the model output fails validation."""
        from google.genai import types

        prompt = (
            "Extract the following information from this document. "
            "If a field is not present in the text, leave it null or empty - "
            "never guess or invent a value.\n\n"
            f"{text}"
        )
        resp = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", response_schema=schema
            ),
        )
        return schema.model_validate_json(resp.text)