"""
Role-Based Access Control for the Murder Mystery RAG system.

Each "role" is a character within a specific mystery.
A character can ONLY query their own ChromaDB collection — this is the core
RBAC guarantee: no character can read another character's private knowledge.

The RBAC class is the single source of truth for:
  - Which characters exist in which mysteries
  - What each character is allowed to access
  - Metadata needed to boot the agent (display name, brief description)
"""

from dataclasses import dataclass, field


@dataclass
class CharacterRole:
    mystery_id: str
    character_id: str          # machine-readable, e.g. "mr_hawkins"
    display_name: str          # human-readable, e.g. "Mr. Hawkins (Butler)"
    short_description: str     # shown in the character selection menu
    allowed_collection: str = field(init=False)

    def __post_init__(self):
        # The collection this role may access — deterministically derived
        self.allowed_collection = f"{self.mystery_id}__{self.character_id}"


class RBAC:
    def __init__(self):
        self._roles: dict[str, CharacterRole] = {}  # key: "mystery_id:character_id"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        mystery_id: str,
        character_id: str,
        display_name: str,
        short_description: str,
    ) -> CharacterRole:
        key = f"{mystery_id}:{character_id}"
        role = CharacterRole(mystery_id, character_id, display_name, short_description)
        self._roles[key] = role
        return role

    # ------------------------------------------------------------------
    # Access control
    # ------------------------------------------------------------------

    def get_role(self, mystery_id: str, character_id: str) -> CharacterRole | None:
        return self._roles.get(f"{mystery_id}:{character_id}")

    def can_access_collection(
        self, mystery_id: str, character_id: str, collection_name: str
    ) -> bool:
        """Strict: a character may only touch their own collection."""
        role = self.get_role(mystery_id, character_id)
        if role is None:
            return False
        return role.allowed_collection == collection_name

    def get_allowed_collection(self, mystery_id: str, character_id: str) -> str | None:
        role = self.get_role(mystery_id, character_id)
        return role.allowed_collection if role else None

    # ------------------------------------------------------------------
    # Listing helpers (for the game UI)
    # ------------------------------------------------------------------

    def characters_for_mystery(self, mystery_id: str) -> list[CharacterRole]:
        return [r for r in self._roles.values() if r.mystery_id == mystery_id]

    def all_mysteries(self) -> list[str]:
        seen = []
        for r in self._roles.values():
            if r.mystery_id not in seen:
                seen.append(r.mystery_id)
        return seen
