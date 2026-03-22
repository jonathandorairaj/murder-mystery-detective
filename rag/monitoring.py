"""
Observability for the Murder Mystery RAG system.

Uses Arize Phoenix (local, open-source) as the tracing backend.
Phoenix launches a local web UI at http://localhost:6006 where you can see:
  - Every detective question as a root CHAIN span
  - RAG retrieval as a RETRIEVER child span (with the retrieved chunks)
  - The LLM call as an LLM child span (model, messages, response, latency)
  - Evaluation scores (context_relevance, groundedness) as span attributes

Start Phoenix before playing:
    python -m phoenix.server.main serve
    # OR: call init_tracing() and it auto-launches an in-process server.

OpenInference semantic convention attributes are used so Phoenix renders the
spans correctly in its built-in LLM trace viewer.
"""

import json
import time
from contextlib import contextmanager, nullcontext

from .llm import chat as llm_chat, Provider

# ---------------------------------------------------------------------------
# Optional Phoenix / OpenTelemetry import — degrades gracefully if not installed
# ---------------------------------------------------------------------------

try:
    import phoenix as px
    from opentelemetry import trace
    from phoenix.otel import register
    _PHOENIX_AVAILABLE = True
except ImportError:
    _PHOENIX_AVAILABLE = False
    # Stub so the rest of the module compiles even without phoenix installed
    class _TraceFallback:  # noqa: E302
        class Span:
            def is_recording(self): return False
            def set_attribute(self, *a, **kw): pass
        def start_as_current_span(self, *a, **kw): return nullcontext()
    class _TracerFallback:  # noqa: E302
        def get_tracer(self, *a): return _TraceFallback()
    trace = _TracerFallback()  # type: ignore

# ---------------------------------------------------------------------------
# Global tracer (None until init_tracing() is called)
# ---------------------------------------------------------------------------

_tracer = None


def init_tracing(project: str = "murder-mystery"):
    """
    Launch Phoenix (if not already running) and register a global OTel
    tracer provider that ships spans to it.
    Call once at the start of main.py or server.py.
    No-ops silently if arize-phoenix is not installed.
    """
    global _tracer
    if _tracer is not None:
        return _tracer
    if not _PHOENIX_AVAILABLE:
        print("⚠  arize-phoenix not installed — tracing disabled. "
              "Install it with: pip install arize-phoenix opentelemetry-sdk")
        return None

    # Launch the Phoenix UI in-process (opens http://localhost:6006)
    px.launch_app()

    # Register the OTel tracer provider → Phoenix
    register(project_name=project)

    _tracer = trace.get_tracer("murder-mystery")
    return _tracer


def get_tracer():
    return _tracer


# ---------------------------------------------------------------------------
# Convenience context manager
# ---------------------------------------------------------------------------

@contextmanager
def span(name: str, kind: str = "CHAIN", **attributes):
    """
    Yields an OTel span if tracing is initialised, otherwise a no-op context.
    `kind` maps to the OpenInference span kind attribute (CHAIN, LLM, RETRIEVER).
    Extra keyword arguments are set as span attributes.
    """
    t = get_tracer()
    if t is None:
        yield None
        return

    with t.start_as_current_span(name) as s:
        s.set_attribute("openinference.span.kind", kind)
        for k, v in attributes.items():
            s.set_attribute(k, str(v))
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            if _PHOENIX_AVAILABLE:
                s.set_status(trace.StatusCode.ERROR, str(exc))
            raise


# ---------------------------------------------------------------------------
# LLM-as-judge evaluator
# ---------------------------------------------------------------------------

_EVAL_PROMPT = """\
You are an evaluation assistant. Answer ONLY with a JSON object — no prose.

Question asked to a character: {question}

Retrieved context the character was given:
{context}

Character's response:
{response}

Score the response on two axes, each from 1 to 5:
- context_relevance: Does the retrieved context actually contain information relevant to the question? (1 = completely irrelevant, 5 = highly relevant)
- groundedness: Is the character's response grounded in the retrieved context, with no invented facts? (1 = completely hallucinated, 5 = fully grounded)

Respond with exactly: {{"context_relevance": <int>, "groundedness": <int>}}"""


def evaluate_response(
    question: str,
    context: str,
    response: str,
    model: str,
    provider: Provider = "ollama",
    current_span=None,
) -> dict[str, int]:
    """
    Run a quick LLM-as-judge evaluation using the active provider and return scores.
    If a span is provided, scores are recorded as attributes on it.

    Note: OpenAI and Anthropic don't support a native JSON-mode equivalent here,
    so we ask the model to reply with only JSON and parse the first {...} block.
    """
    prompt = _EVAL_PROMPT.format(
        question=question,
        context=context[:1500],   # keep the judge prompt short
        response=response,
    )

    try:
        t0 = time.perf_counter()
        raw = llm_chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            provider=provider,
        )
        elapsed = time.perf_counter() - t0

        # Extract the first JSON object from the response
        start = raw.find("{")
        end = raw.rfind("}") + 1
        scores = json.loads(raw[start:end])
        scores = {
            "context_relevance": int(scores.get("context_relevance", 0)),
            "groundedness": int(scores.get("groundedness", 0)),
        }
    except Exception:
        scores = {"context_relevance": -1, "groundedness": -1}
        elapsed = 0.0

    if current_span and current_span.is_recording():
        current_span.set_attribute("eval.context_relevance", scores["context_relevance"])
        current_span.set_attribute("eval.groundedness", scores["groundedness"])
        current_span.set_attribute("eval.latency_s", round(elapsed, 3))

    return scores
