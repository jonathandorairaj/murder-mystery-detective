"""
setup_db.py — One-time database initialisation.

Reads every character document from the mystery folders, chunks the text,
and loads each character's chunks into their own ChromaDB collection.

Run once before starting the game:
    python setup_db.py
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.progress import track

from rag import CharacterVectorStore, RBAC
from rag.chunkers import STRATEGIES, DEFAULT_STRATEGY

console = Console()

# ---------------------------------------------------------------------------
# Registry: mystery_id → title, then character_id → (display_name, file)
# ---------------------------------------------------------------------------

MYSTERIES: dict[str, dict] = {
    "mystery_1_thornwood_manor": {
        "title": "The Thornwood Manor Murder",
        "characters": {
            "mr_hawkins":    ("Mr. Hawkins (Butler)",             "mr_hawkins_butler.md"),
            "lady_victoria": ("Lady Victoria Thornwood (Wife)",   "lady_victoria_thornwood.md"),
            "dr_pemberton":  ("Dr. Pemberton (Family Doctor)",    "dr_pemberton.md"),
            "miss_clara":    ("Miss Clara (Governess)",           "miss_clara_governess.md"),
            "thomas":        ("Thomas (Groundskeeper)",           "thomas_groundskeeper.md"),
        },
        "personas": {
            "mr_hawkins":    "Loyal and discreet. Thirty-five years of service. Sees all, admits little.",
            "lady_victoria": "Composed and imperious. Grief sits strangely on her — too controlled, too perfect.",
            "dr_pemberton":  "Charming and evasive. He certified the death but his eyes say he has doubts.",
            "miss_clara":    "Quiet and watchful. Anxious about her position, honest when asked directly.",
            "thomas":        "Blunt and weathered. Loyal to the land and Lord Edmund. Misses nothing outdoors.",
        },
    },
    "mystery_3_silicon_secrets": {
        "title": "Silicon Secrets",
        "characters": {
            "dr_sarah_chen":  ("Dr. Sarah Chen (CTO)",            "dr_sarah_chen_cto.md"),
            "robert_frost":   ("Robert Frost (CFO)",              "robert_frost_cfo.md"),
            "amanda_pierce":  ("Amanda Pierce (HR Director)",     "amanda_pierce_hr.md"),
            "jake_torres":    ("Jake Torres (Security Guard)",    "jake_torres_security.md"),
            "lily_park":      ("Lily Park (Personal Assistant)",  "lily_park_pa.md"),
        },
        "personas": {
            "dr_sarah_chen": "Analytical and precise. If it isn't in the data, she won't say it.",
            "robert_frost":  "Charming and overcomposed. His grief is corporate; his answers too polished.",
            "amanda_pierce": "Professional and concerned. Her HR records are meticulous; so is her memory.",
            "jake_torres":   "Methodical and reliable. He has timestamps, footage, and no reason to lie.",
            "lily_park":     "Loyal and devastated. She found Marcus — and saw who left his office.",
        },
    },
    "mystery_5_crimson_gallery": {
        "title": "The Crimson Gallery",
        "characters": {
            "sophia_laurent":  ("Sophia Laurent (Gallery Owner)", "sophia_laurent_gallery_owner.md"),
            "charles_pembroke":("Charles Pembroke (Art Critic)",  "charles_pembroke_art_critic.md"),
            "danny_obrien":    ("Danny O'Brien (Security Chief)", "danny_obrien_security.md"),
            "nina_reyes":      ("Nina Reyes (Caterer)",           "nina_reyes_caterer.md"),
            "marco_vitelli":   ("Marco Vitelli (Artist)",         "marco_vitelli_artist.md"),
        },
        "personas": {
            "sophia_laurent":   "Elegant and controlled. The gallery is her world; someone violated it.",
            "charles_pembroke": "Incisive and superior. He notices forgeries, inconsistencies, and lies.",
            "danny_obrien":     "Gruff and thorough. Security is his job; he takes failure personally.",
            "nina_reyes":       "Warm and observant. The caterer sees what the guests ignore.",
            "marco_vitelli":    "Volatile and brilliant. His passion is real; so is his resentment.",
        },
    },
    "mystery_6_styles_court": {
        "title": "The Mysterious Affair at Styles Court",
        "characters": {
            "alfred_inglethorp":  ("Alfred Inglethorp (Husband)",       "alfred_inglethorp.md"),
            "john_cavendish":     ("John Cavendish (Stepson)",          "john_cavendish.md"),
            "mary_cavendish":     ("Mary Cavendish (John's Wife)",      "mary_cavendish.md"),
            "evelyn_howard":      ("Evelyn Howard (Companion)",         "evelyn_howard.md"),
            "lawrence_cavendish": ("Lawrence Cavendish (Stepson)",      "lawrence_cavendish.md"),
            "dr_bauerstein":      ("Dr. Bauerstein (Chemist)",          "dr_bauerstein.md"),
        },
        "personas": {
            "alfred_inglethorp":  "Theatrical and too smooth. His helpfulness is calculated; his alibi is airtight.",
            "john_cavendish":     "Troubled and nervous. He looks guilty but is hiding someone else's secret.",
            "mary_cavendish":     "Aloof and guarded. She was in the garden at midnight and will not say why.",
            "evelyn_howard":      "Forceful and righteous. She despised Alfred from the first and says so plainly.",
            "lawrence_cavendish": "Timid and evasive. Not a killer — just a shy man with an embarrassing secret.",
            "dr_bauerstein":      "Precise and withholding. He knows exactly what killed Emily and almost says so.",
        },
    },
    "mystery_7_hound_baskervilles": {
        "title": "The Hound of the Baskervilles",
        "characters": {
            "dr_mortimer":    ("Dr. James Mortimer (Family Doctor)", "dr_mortimer.md"),
            "barrymore":      ("Barrymore (Butler)",                 "barrymore.md"),
            "mrs_barrymore":  ("Mrs. Barrymore (Housekeeper)",       "mrs_barrymore.md"),
            "jack_stapleton": ("Jack Stapleton (Naturalist)",        "jack_stapleton.md"),
            "laura_lyons":    ("Laura Lyons (Secretary)",            "laura_lyons.md"),
        },
        "personas": {
            "dr_mortimer":    "Earnest and scientific. He only trusts what he can measure.",
            "barrymore":      "Formal and haunted. Loyal to the family but hiding something on the moors.",
            "mrs_barrymore":  "Frightened and devoted. Her secret is not what you expect.",
            "jack_stapleton": "Charming and knowledgeable. His smile never quite reaches his eyes.",
            "laura_lyons":    "Defiant and ashamed. Stapleton used her and she is beginning to realise it.",
        },
    },
    "mystery_8_disappearing_cat": {
        "title": "The Mystery of the Disappearing Cat",
        "characters": {
            "leo":          ("Leo (Garden Boy)",              "leo.md"),
            "mr_napper":    ("Mr. Napper (Head Gardener)",    "mr_napper.md"),
            "miss_palmer":  ("Miss Palmer (Lady's Companion)","miss_palmer.md"),
            "lady_hartley": ("Lady Hartley (Cat Owner)",      "lady_hartley.md"),
            "mrs_napper":   ("Mrs. Napper (Gardener's Wife)", "mrs_napper.md"),
        },
        "personas": {
            "leo":          "Simple-hearted and frightened. He didn't do it, but no one believes him.",
            "mr_napper":    "Surly and resentful. He has a grudge and a secret he is desperate to keep.",
            "miss_palmer":  "Nervous and observant. She noticed more than she thought to mention.",
            "lady_hartley": "Imperious and distressed. Midnight Pearl was her pride and joy.",
            "mrs_napper":   "Quiet and apologetic. She suspects her husband but fears his temper.",
        },
    },
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Index character documents into ChromaDB.")
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default=DEFAULT_STRATEGY,
        help=f"Chunking strategy (default: {DEFAULT_STRATEGY})",
    )
    args = parser.parse_args()
    chunk_fn = STRATEGIES[args.strategy]
    console.print(f"[bold]Chunking strategy:[/bold] [cyan]{args.strategy}[/cyan]")

    base_dir = Path(__file__).parent
    store = CharacterVectorStore(db_path=str(base_dir / "chroma_db"))
    rbac = RBAC()

    total_chunks = 0
    errors: list[str] = []

    for mystery_id, mystery_data in MYSTERIES.items():
        console.rule(f"[bold cyan]{mystery_data['title']}")
        char_dir = base_dir / mystery_id / "characters"

        for char_id, (display_name, filename) in track(
            mystery_data["characters"].items(),
            description=f"  Loading {mystery_id}…",
        ):
            doc_path = char_dir / filename
            if not doc_path.exists():
                errors.append(f"  [red]MISSING:[/red] {doc_path}")
                continue

            text = doc_path.read_text(encoding="utf-8")
            chunks = chunk_fn(text)

            if not chunks:
                errors.append(f"  [yellow]EMPTY:[/yellow] {doc_path}")
                continue

            # Wipe and re-create so setup is idempotent
            store.delete_collection(mystery_id, char_id)
            store.add_documents(
                mystery_id,
                char_id,
                chunks,
                metadatas=[{"character": char_id, "mystery": mystery_id} for _ in chunks],
            )
            rbac.register(mystery_id, char_id, display_name, display_name)
            total_chunks += len(chunks)
            console.print(f"    [green]✓[/green] {display_name} — {len(chunks)} chunks")

    console.rule()
    console.print(f"[bold green]Setup complete.[/bold green] {total_chunks} total chunks indexed.")

    if errors:
        console.print("\n[bold red]Warnings / Errors:[/bold red]")
        for e in errors:
            console.print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
