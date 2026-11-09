"""
rag/llm.py — Unified LLM chat interface for Ollama, OpenAI, and Anthropic.

All three providers share the same call signature:
    reply = chat(messages, model, provider)

`messages` uses the standard OpenAI format (list of {role, content} dicts).
A leading {"role": "system", ...} entry is handled correctly for every provider —
Anthropic requires it as a separate kwarg; the others accept it inline.
"""

import os
from typing import Literal

Provider = Literal["ollama", "openai", "anthropic"]

# Default model per provider
DEFAULT_MODELS: dict[str, str] = {
    "ollama":    "llama3.2",
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
}


def chat(messages: list[dict], model: str, provider: Provider) -> str:
    """Route a chat request to the correct provider and return the reply text."""
    if provider == "ollama":
        return _ollama(messages, model)
    elif provider == "openai":
        return _openai(messages, model)
    elif provider == "anthropic":
        return _anthropic(messages, model)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose: ollama | openai | anthropic")


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _ollama(messages: list[dict], model: str) -> str:
    import ollama as _lib
    host = os.getenv("OLLAMA_HOST")
    client = _lib.Client(host=host) if host else None
    fn = client.chat if client else _lib.chat
    return fn(model=model, messages=messages).message.content


def _openai(messages: list[dict], model: str) -> str:
    from openai import OpenAI
    # Reads OPENAI_API_KEY from environment automatically
    response = OpenAI().chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def _anthropic(messages: list[dict], model: str) -> str:
    import anthropic
    # Anthropic requires system prompt as a separate parameter, not in messages[]
    system = ""
    convo: list[dict] = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        else:
            convo.append(m)

    # max_tokens covers thinking as well as the reply, so leave plenty of room
    kwargs: dict = dict(model=model, max_tokens=16000, messages=convo)
    if system:
        kwargs["system"] = system

    # Reads ANTHROPIC_API_KEY from environment automatically
    response = anthropic.Anthropic().messages.create(**kwargs)

    # Current models think by default, so content can start with a thinking
    # block; the reply is the text blocks only.
    reply = "".join(b.text for b in response.content if b.type == "text")
    if not reply:
        raise RuntimeError(f"Anthropic returned no text (stop_reason={response.stop_reason})")
    return reply
