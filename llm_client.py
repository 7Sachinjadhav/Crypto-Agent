"""
Thin wrapper around OpenRouter (OpenAI-compatible) API.
Used by all Hermes agents as their LLM backend.
"""

from openai import OpenAI
from config.settings import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, LLM_MODEL
from utils.logger import logger


def get_llm_client() -> OpenAI:
    """Return a configured OpenAI client pointing at OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set. Add it to your .env file."
        )
    return OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )


def chat_completion(messages: list[dict], model: str = LLM_MODEL, **kwargs) -> str:
    """
    Simple synchronous chat completion helper.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.
        model:    OpenRouter model string.
        **kwargs: Extra params forwarded to the API (temperature, max_tokens, …).

    Returns:
        The assistant's reply as a plain string.
    """
    client = get_llm_client()
    logger.debug(f"LLM request | model={model} | messages={len(messages)}")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs,
    )
    content = response.choices[0].message.content
    logger.debug(f"LLM response | tokens={response.usage.total_tokens}")
    return content
