"""
Observability for the Murder Mystery RAG system.

Ships telemetry to the otel-lgtm stack (Grafana + Tempo + Prometheus + Loki)
via the standard OpenTelemetry OTLP HTTP exporter.

Configure via environment variables (set in docker-compose):
    OTEL_EXPORTER_OTLP_ENDPOINT   e.g. http://otel-lgtm:4318
    OTEL_SERVICE_NAME             e.g. murder-mystery (default)

Every detective question produces a root CHAIN span with child spans for:
  - rag.retrieval  — the ChromaDB vector query and retrieved chunks
  - llm.call       — provider, model, latency, input/output messages
  - eval.judge     — LLM-as-judge scores (if --evaluate is passed)
"""

import json
import os
import time
from contextlib import contextmanager, nullcontext

from .llm import chat as llm_chat, Provider

# ---------------------------------------------------------------------------
# OpenTelemetry SDK — degrades gracefully if packages are not installed
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False

    # Minimal stubs so the rest of the module compiles without the SDK installed
    class _Span:
        def is_recording(self): return False
        def set_attribute(self, *a, **kw): pass
        def record_exception(self, *a, **kw): pass

    class _Tracer:
        def start_as_current_span(self, *a, **kw): return nullcontext(_Span())

    class _TraceFallback:
        def get_tracer(self, *a): return _Tracer()

    trace = _TraceFallback()  # type: ignore

# ---------------------------------------------------------------------------
# Global tracer — None until init_tracing() is called
# ---------------------------------------------------------------------------

_tracer = None


def init_tracing(service_name: str = None) -> object:
    """
    Build a TracerProvider that exports spans to the otel-lgtm collector over
    OTLP/HTTP, then register it as the global OTel provider.

    Called once at startup in server.py when OTEL_EXPORTER_OTLP_ENDPOINT is set.
    Safe to call multiple times — returns the existing tracer on subsequent calls.
    """
    global _tracer
    if _tracer is not None:
        return _tracer

    if not _OTEL_AVAILABLE:
        print("⚠  opentelemetry-sdk not installed — tracing disabled.")
        return None

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-lgtm:4318")
    service  = service_name or os.getenv("OTEL_SERVICE_NAME", "murder-mystery")

    # The SDK appends /v1/traces automatically when you pass a base URL
    exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
    resource  = Resource.create({"service.name": service})
    provider  = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # Register as the global provider so opentelemetry.trace.get_tracer() works
    # everywhere, including inside the FastAPI auto-instrumentation
    trace.set_tracer_provider(provider)

    _tracer = trace.get_tracer(service)
    print(f"✓  Tracing → {endpoint}  (service: {service})")
    return _tracer


def get_tracer():
    return _tracer


# ---------------------------------------------------------------------------
# Convenience span context manager
# ---------------------------------------------------------------------------

@contextmanager
def span(name: str, kind: str = "CHAIN", **attributes):
    """
    Yields an OTel span if tracing is initialised, otherwise a silent no-op.

    `kind` is stored as the openinference.span.kind attribute so Grafana Tempo
    can categorise spans as CHAIN / LLM / RETRIEVER in the trace view.
    Extra keyword arguments become span attributes.
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
            raise


# ---------------------------------------------------------------------------
# LLM-as-judge evaluator (optional, activated with --evaluate)
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
    Run an LLM-as-judge evaluation and return context_relevance + groundedness scores.
    Scores are also written to the provided span as attributes if one is active.
    """
    prompt = _EVAL_PROMPT.format(
        question=question,
        context=context[:1500],
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

        start  = raw.find("{")
        end    = raw.rfind("}") + 1
        scores = json.loads(raw[start:end])
        scores = {
            "context_relevance": int(scores.get("context_relevance", 0)),
            "groundedness":      int(scores.get("groundedness", 0)),
        }
    except Exception:
        scores  = {"context_relevance": -1, "groundedness": -1}
        elapsed = 0.0

    if current_span and current_span.is_recording():
        current_span.set_attribute("eval.context_relevance", scores["context_relevance"])
        current_span.set_attribute("eval.groundedness",      scores["groundedness"])
        current_span.set_attribute("eval.latency_s",         round(elapsed, 3))

    return scores
