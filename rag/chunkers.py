"""
Chunking strategies for character knowledge documents.

paragraph     — splits on blank lines, merges tiny paragraphs (good for
                structured markdown with clear section breaks)
sliding_window — fixed character-size windows with overlap (better for
                dense continuous prose; ensures every fact is covered by
                at least one chunk even across paragraph boundaries)
"""

import re

# ---------------------------------------------------------------------------
# Strategy 1: Paragraph-based (original)
# ---------------------------------------------------------------------------

def paragraph_chunk(text: str, min_chunk: int = 80, max_chunk: int = 600) -> list[str]:
    """
    Split on double-newline paragraph boundaries.
    Tiny paragraphs are merged into their neighbour; oversized ones are left as-is.
    """
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < max_chunk:
            buffer = (buffer + "\n\n" + para).strip()
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < min_chunk:
            merged[-1] += "\n\n" + chunk
        else:
            merged.append(chunk)

    return [c for c in merged if c.strip()]


# ---------------------------------------------------------------------------
# Strategy 2: Sliding window with overlap
# ---------------------------------------------------------------------------

def sliding_window_chunk(
    text: str,
    window: int = 400,
    overlap: int = 100,
    tolerance: int = 60,
) -> list[str]:
    """
    Carve text into fixed-size character windows that overlap by `overlap` chars.

    Before cutting, we look back up to `tolerance` characters from the nominal
    cut point for a sentence boundary ('. ', '! ', '? ') so chunks don't split
    mid-sentence where avoidable.

    Overlap means consecutive chunks share ~25% of their content — this helps
    the retriever find facts that straddle a boundary in either chunk.
    """
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) <= window:
        return [text] if text else []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + window, len(text))

        # Snap to a sentence boundary if we're not at the very end
        if end < len(text):
            best = -1
            for pat in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
                idx = text.rfind(pat, end - tolerance, end)
                if idx > best:
                    best = idx + len(pat) - 1  # keep the punctuation in the chunk
            if best > start:
                end = best

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Advance by (window - overlap), but stop if we've reached the end
        step = max(window - overlap, 1)
        start = start + step if end < len(text) else len(text)

    return chunks


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, callable] = {
    "paragraph": paragraph_chunk,
    "sliding_window": sliding_window_chunk,
}

DEFAULT_STRATEGY = "paragraph"
