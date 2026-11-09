"""
CharacterAgent — a RAG-powered in-character witness/suspect.

Supports three LLM providers via rag.llm:
  • ollama    (local, default)
  • openai    (requires OPENAI_API_KEY)
  • anthropic (requires ANTHROPIC_API_KEY)

Flow for each detective question:
  1. RBAC check: confirm the vector store collection is the character's own.
  2. RAG retrieval (RETRIEVER span) — top-k chunks from the character's collection.
  3. LLM call (LLM span) — provider generates an in-character reply.
  4. Evaluation (optional) — LLM-as-judge scores context_relevance & groundedness.
  All wrapped in a root CHAIN span visible in the Phoenix UI.
"""

import json
import time

from .vector_store import CharacterVectorStore
from .rbac import RBAC, CharacterRole
from .monitoring import span, evaluate_response, get_tracer
from .llm import chat as llm_chat, Provider, DEFAULT_MODELS


class CharacterAgent:
    def __init__(
        self,
        role: CharacterRole,
        backstory: str,
        vector_store: CharacterVectorStore,
        rbac: RBAC,
        provider: Provider = "ollama",
        model: str | None = None,   # None → use DEFAULT_MODELS[provider]
        n_retrieve: int = 6,
        evaluate: bool = False,   # set True to run LLM-as-judge after each response
    ):
        self.role = role
        self.backstory = backstory
        self.vector_store = vector_store
        self.rbac = rbac
        self.provider: Provider = provider
        self.model: str = model or DEFAULT_MODELS[provider]
        self.n_retrieve = n_retrieve
        self.evaluate = evaluate
        self.conversation_history: list[dict] = []   # used by ask() (CLI) only
        self.last_retrieved: list[str] = []   # set on every ask()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, detective_question: str) -> str:
        """Ask the character a question, remembering the conversation on this agent (CLI)."""
        reply, retrieved = self.respond(detective_question, self.conversation_history)
        self.last_retrieved = retrieved
        self.conversation_history.append({"role": "user", "content": detective_question})
        self.conversation_history.append({"role": "assistant", "content": reply})
        return reply

    def respond(self, detective_question: str, history: list[dict]) -> tuple[str, list[str]]:
        """Answer a question given the prior conversation, without storing any state.

        Returns (reply, retrieved_chunks). Safe to call concurrently from the web
        server, where each player's browser sends its own history.
        """

        # Root span for the full ask() call
        with span(
            "character.ask",
            kind="CHAIN",
            character=self.role.display_name,
            mystery=self.role.mystery_id,
            **{"input.value": detective_question},
        ) as root:

            # 1. RBAC check
            allowed = self.rbac.get_allowed_collection(
                self.role.mystery_id, self.role.character_id
            )
            assert self.rbac.can_access_collection(
                self.role.mystery_id, self.role.character_id, allowed
            ), "RBAC violation"

            # 2. RAG retrieval
            with span(
                "rag.retrieval",
                kind="RETRIEVER",
                character=self.role.character_id,
                query=detective_question,
            ) as rag_span:
                retrieved = self.vector_store.query(
                    self.role.mystery_id,
                    self.role.character_id,
                    detective_question,
                    n_results=self.n_retrieve,
                )
                if rag_span:
                    rag_span.set_attribute("retrieval.num_chunks", len(retrieved))
                    # Log each chunk using OpenInference retrieval convention
                    for i, doc in enumerate(retrieved):
                        rag_span.set_attribute(f"retrieval.documents.{i}.document.content", doc[:500])

            context_block = "\n\n---\n\n".join(retrieved) if retrieved else "(no specific memories retrieved)"
            system_prompt = self._build_system_prompt(context_block)

            # 3. LLM call
            messages = (
                [{"role": "system", "content": system_prompt}]
                + history
                + [{"role": "user", "content": detective_question}]
            )

            with span(
                "llm.call",
                kind="LLM",
                **{
                    "llm.provider": self.provider,
                    "llm.model_name": self.model,
                    "llm.input_messages": json.dumps(messages[-4:]),  # last 4 turns
                },
            ) as llm_span:
                t0 = time.perf_counter()
                reply = llm_chat(messages, self.model, self.provider)
                elapsed = time.perf_counter() - t0

                if llm_span:
                    llm_span.set_attribute("llm.output_messages", json.dumps([{"role": "assistant", "content": reply}]))
                    llm_span.set_attribute("llm.latency_s", round(elapsed, 3))
                    llm_span.set_attribute("output.value", reply)
                    llm_span.set_attribute("llm.provider", self.provider)


            # 4. Optional LLM-as-judge evaluation
            if self.evaluate:
                with span("eval.judge", kind="LLM", model=self.model) as eval_span:
                    scores = evaluate_response(
                        question=detective_question,
                        context=context_block,
                        response=reply,
                        model=self.model,
                        provider=self.provider,
                        current_span=eval_span,
                    )
                if root:
                    root.set_attribute("eval.context_relevance", scores["context_relevance"])
                    root.set_attribute("eval.groundedness", scores["groundedness"])

            if root:
                root.set_attribute("output.value", reply)

        return reply, retrieved

    def reset_conversation(self) -> None:
        self.conversation_history = []

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_system_prompt(self, retrieved_context: str) -> str:
        return f"""You are {self.role.display_name}. You must stay completely in character at all times.

=== YOUR BACKGROUND & PERSONAL KNOWLEDGE ===
{self.backstory}

=== MEMORIES RETRIEVED RELEVANT TO THIS QUESTION ===
{retrieved_context}

=== STRICT RULES ===
1. You are ONLY {self.role.display_name}. Never break character.
2. You may ONLY speak about things mentioned in your background or retrieved memories above.
   If something is not in your knowledge, say so in character ("I wouldn't know about that",
   "I wasn't there", etc.).
3. You are being questioned by a detective about a suspicious death. Be authentic —
   show emotions, hesitations, and personality appropriate to your character.
4. If you have something to hide, be evasive but do NOT outright lie about your core knowledge.
   Omission, deflection, and partial truths are fair; fabricating facts you don't have is not.
5. Keep answers focused and under ~200 words unless a long answer is warranted.
6. Do NOT reveal the murderer outright even if you have suspicions — let the detective deduce.
"""
