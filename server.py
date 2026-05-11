"""
server.py — FastAPI backend for the Murder Mystery Detective game.

Run with:
    python server.py
    python server.py --evaluate     # LLM-as-judge eval on each response

Tracing is enabled automatically when OTEL_EXPORTER_OTLP_ENDPOINT is set
in the environment (configured via docker-compose). Traces, metrics, and
HTTP instrumentation are shipped to the otel-lgtm collector over OTLP/HTTP.

Then open http://localhost:7777
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from rag import CharacterVectorStore, RBAC, CharacterAgent
from rag.monitoring import init_tracing
from rag.llm import DEFAULT_MODELS, Provider
from main import MYSTERIES, BACKSTORIES

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend"

# ---------------------------------------------------------------------------
# Parse CLI args early so we can init tracing before creating the app
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Murder Mystery API server")
parser.add_argument("--provider", choices=["ollama", "openai", "anthropic"],
                    default=None, help="LLM provider (default: $LLM_PROVIDER or ollama)")
parser.add_argument("--model", default=None,
                    help="Model name override (default: $LLM_MODEL or per-provider default)")
parser.add_argument("--evaluate", action="store_true", help="Run LLM-as-judge eval after each response")
parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7777")),
                    help="Port to listen on (default: $PORT or 7777)")
args = parser.parse_args()

# CLI flag → env var → hardcoded default (makes Docker configuration clean)
PROVIDER: Provider = args.provider or os.getenv("LLM_PROVIDER", "ollama")
MODEL: str = args.model or os.getenv("LLM_MODEL") or DEFAULT_MODELS[PROVIDER]

# Auto-enable tracing when running in Docker (env var set in docker-compose).
# The --evaluate flag also activates tracing so judge scores appear in spans.
if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or args.evaluate:
    init_tracing()

# ---------------------------------------------------------------------------
# Bootstrap the RAG system (module-level so it's shared across requests)
# ---------------------------------------------------------------------------

store = CharacterVectorStore(db_path=str(BASE_DIR / "chroma_db"))
rbac = RBAC()
_agents: dict[str, CharacterAgent] = {}

for mystery_id, mdata in MYSTERIES.items():
    for char_id, (display_name, _) in mdata["characters"].items():
        rbac.register(mystery_id, char_id, display_name, display_name)


def _get_agent(mystery_id: str, char_id: str) -> CharacterAgent:
    key = f"{mystery_id}:{char_id}"
    if key not in _agents:
        role = rbac.get_role(mystery_id, char_id)
        if not role:
            raise HTTPException(404, detail=f"Character not found: {char_id}")
        backstory = BACKSTORIES[mystery_id][char_id]
        _agents[key] = CharacterAgent(
            role=role,
            backstory=backstory,
            vector_store=store,
            rbac=rbac,
            provider=PROVIDER,
            model=MODEL,
            evaluate=args.evaluate,
        )
    return _agents[key]


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Murder Mystery API", docs_url="/api/docs")

# Auto-instrument FastAPI — emits HTTP request duration + count metrics and
# creates a server span for every request, parented under any incoming trace context.
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)
except ImportError:
    pass  # silently skip if package not installed

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/config")
def get_config():
    """Runtime configuration consumed by the frontend."""
    return {
        "provider": PROVIDER,
        "model": MODEL,
    }


@app.get("/api/mysteries")
def list_mysteries():
    return [{"id": mid, "title": mdata["title"]} for mid, mdata in MYSTERIES.items()]


@app.get("/api/mysteries/{mystery_id}/overview")
def get_overview(mystery_id: str):
    path = BASE_DIR / mystery_id / "case_overview.md"
    if not path.exists():
        raise HTTPException(404, detail="Overview not found")
    return {"markdown": path.read_text()}


@app.get("/api/mysteries/{mystery_id}/characters")
def list_characters(mystery_id: str):
    if mystery_id not in MYSTERIES:
        raise HTTPException(404, detail="Mystery not found")
    personas = MYSTERIES[mystery_id].get("personas", {})
    return [
        {
            "id": char_id,
            "display_name": display_name,
            "persona": personas.get(char_id, ""),
        }
        for char_id, (display_name, _) in MYSTERIES[mystery_id]["characters"].items()
    ]


@app.get("/api/mysteries/{mystery_id}/solution")
def get_solution(mystery_id: str):
    path = BASE_DIR / mystery_id / "solution.md"
    if not path.exists():
        raise HTTPException(404, detail="Solution not found")
    return {"markdown": path.read_text()}


class AskRequest(BaseModel):
    mystery_id: str
    character_id: str
    question: str


@app.post("/api/ask")
def ask(req: AskRequest):
    if req.mystery_id not in MYSTERIES:
        raise HTTPException(404, detail="Mystery not found")
    agent = _get_agent(req.mystery_id, req.character_id)
    reply = agent.ask(req.question)
    return {
        "reply": reply,
        "character": agent.role.display_name,
        "chunks": agent.last_retrieved,   # RAG evidence chunks for the UI
    }


class ResetRequest(BaseModel):
    mystery_id: str
    character_id: str


@app.post("/api/reset")
def reset(req: ResetRequest):
    key = f"{req.mystery_id}:{req.character_id}"
    if key in _agents:
        _agents[key].reset_conversation()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Serve the frontend (must come after API routes)
# ---------------------------------------------------------------------------

@app.get("/")
def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not store.list_collections():
        print("⚠  Database is empty. Run:  python setup_db.py")
        sys.exit(1)

    print(f"\n🔍  Murder Mystery Detective — http://localhost:{args.port}")
    print(f"    Provider : {PROVIDER}")
    print(f"    Model    : {MODEL}\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
