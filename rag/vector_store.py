"""
RAG Vector Store — per-character ChromaDB collections.
Each character has their own isolated collection; queries are scoped strictly
to that collection (RBAC is enforced at the call sites in rbac.py / agent.py).
"""

import chromadb
from chromadb.utils import embedding_functions


class CharacterVectorStore:
    def __init__(self, db_path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=db_path)
        # Default embedding function (all-MiniLM-L6-v2, downloaded on first run ~90 MB)
        self.ef = embedding_functions.DefaultEmbeddingFunction()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _collection_name(self, mystery_id: str, character_id: str) -> str:
        return f"{mystery_id}__{character_id}"

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_documents(
        self,
        mystery_id: str,
        character_id: str,
        documents: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Chunk and store documents in a character-specific collection."""
        name = self._collection_name(mystery_id, character_id)
        collection = self.client.get_or_create_collection(
            name=name, embedding_function=self.ef
        )

        ids = [f"{name}_{i}" for i in range(len(documents))]
        metas = metadatas if metadatas else [{"source": "knowledge"} for _ in documents]
        collection.add(documents=documents, ids=ids, metadatas=metas)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        mystery_id: str,
        character_id: str,
        query_text: str,
        n_results: int = 5,
    ) -> list[str]:
        """Retrieve the top-n relevant chunks from a character's collection only."""
        name = self._collection_name(mystery_id, character_id)
        try:
            collection = self.client.get_collection(name=name, embedding_function=self.ef)
            results = collection.query(query_texts=[query_text], n_results=n_results)
            return results["documents"][0] if results["documents"] else []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def collection_exists(self, mystery_id: str, character_id: str) -> bool:
        name = self._collection_name(mystery_id, character_id)
        try:
            self.client.get_collection(name=name, embedding_function=self.ef)
            return True
        except Exception:
            return False

    def delete_collection(self, mystery_id: str, character_id: str) -> None:
        name = self._collection_name(mystery_id, character_id)
        try:
            self.client.delete_collection(name=name)
        except Exception:
            pass

    def list_collections(self) -> list[str]:
        return [c.name for c in self.client.list_collections()]
