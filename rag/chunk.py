"""Stage 2 — chunking.

Splits pages into retrieval units. Two constraints drive the design:

* BGE-small truncates at 512 tokens, so a chunk that overflows silently loses
  its tail from the embedding while still returning it to the model. Chunks are
  therefore measured with the embedder's own tokenizer, not a character guess.
* Annual reports are full of short, fact-dense lines (figures, table rows).
  Splitting on paragraphs alone produces chunks of wildly uneven size, so
  paragraphs are packed together up to a target size instead.

Each chunk keeps its page number and the nearest preceding heading, which gives
the model something to cite and gives retrieval extra lexical signal.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List

from .ingest import Page

TARGET_TOKENS = 380     # comfortably inside BGE's 512 limit after the prefix
OVERLAP_TOKENS = 64     # carries context across a boundary mid-explanation
MIN_TOKENS = 24         # below this a chunk is noise (stray page furniture)


@dataclass
class Chunk:
    id: int
    text: str
    page: int
    source: str
    heading: str = ""

    # --- hierarchical metadata, populated for structured documents ---------
    # These default to empty so that a flat document indexed by `chunk_pages`
    # still produces valid chunks.
    label: str = ""        # printed page number, which is what a reader cites
    part: str = ""         # "Chapter 3 - A Rising Challenge: ..."
    subheading: str = ""
    container: str = ""    # "Box 1.1", "Table 2.3", "Annex 3.1"
    kind: str = "body"     # body | table | figure | box | annex | reference
    path: str = ""         # full breadcrumb, root to leaf

    def to_dict(self) -> Dict:
        return asdict(self)

    def page_ref(self) -> str:
        """How the page should be cited.

        Reports number their own pages, and the front matter is usually
        numbered separately in roman numerals, so the PDF's index and the
        number printed on the page diverge — by 22 in the GEP report. Citing
        the PDF index would send a reader to the wrong page, so the printed
        label wins whenever ingestion recovered one.
        """
        if self.label and self.label != str(self.page):
            return f"p.{self.label} (PDF p.{self.page})"
        return f"p.{self.page}"

    def cite(self) -> str:
        """Short provenance label shown to the model and to the user."""
        where = self.page_ref()
        trail = self.path or self.heading
        return f"{self.source}, {where}" + (f" — {trail}" if trail else "")

    def context_header(self) -> str:
        """The breadcrumb line prepended to the text before embedding.

        Carrying the hierarchy into the embedded text is what lets a passage
        be found by a question that names its chapter, its box or its region,
        none of which the passage itself repeats.
        """
        return self.path or self.heading


# Headings in reports: ALL CAPS lines, numbered sections, or short Title Case.
_NUMBERED = re.compile(r"^\s*(\d+(\.\d+)*)[.)]?\s+\S")
_ALLCAPS = re.compile(r"^[A-Z0-9][A-Z0-9 &/,'\-()]{3,70}$")


def _looks_like_data_row(s: str) -> bool:
    """True for table rows such as "1980 6,450" or "2018 79.5 million".

    Annual reports are full of these. Treating one as a heading is doubly
    damaging: it splits the table into fragments, and the row itself is
    recorded as metadata instead of text, so the figures vanish from the
    index entirely.
    """
    tokens = s.split()
    if not tokens:
        return True
    numeric = sum(1 for t in tokens if re.fullmatch(r"[$(]?[\d,.%–-]+[)]?", t))
    return numeric >= max(1, len(tokens) // 2)


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 90 or s.endswith((".", ";", ",")):
        return False
    if _looks_like_data_row(s):
        return False
    if _ALLCAPS.match(s):
        return True
    # "3.1 Scope" — the separator is required, so "1980 6,450" cannot match.
    if _NUMBERED.match(s) and len(s.split()) <= 12:
        return True
    # Title Case line with no terminal punctuation, e.g. "Financial Highlights"
    words = s.split()
    if 1 < len(words) <= 9 and sum(w[:1].isupper() for w in words) >= len(words) - 1:
        return True
    return False


def _split_blocks(text: str) -> List[str]:
    """Paragraph-ish blocks.

    PDF extractors are inconsistent about whether paragraphs are separated by
    one newline or two, so both are treated as block boundaries. Ingestion has
    already rejoined margin-wrapped lines, meaning a surviving newline is a
    real boundary.
    """
    return [line.strip() for line in text.split("\n") if line.strip()]


def chunk_pages(
    pages: List[Page],
    count_tokens: Callable[[str], int],
    target: int = TARGET_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> List[Chunk]:
    """Pack page text into overlapping, token-bounded chunks.

    `count_tokens` is injected so chunk sizes are measured with the same
    tokenizer that will embed them.
    """
    chunks: List[Chunk] = []
    next_id = 0
    heading = ""

    for page in pages:
        buf: List[str] = []
        buf_tokens = 0
        buf_heading = heading

        def flush() -> None:
            nonlocal buf, buf_tokens, next_id, buf_heading
            if not buf:
                return
            text = "\n\n".join(buf).strip()
            if count_tokens(text) >= MIN_TOKENS:
                chunks.append(Chunk(next_id, text, page.number, page.source, buf_heading))
                next_id += 1
            # Carry the tail of this chunk into the next one so a statement
            # split across the boundary is still retrievable from both sides.
            tail: List[str] = []
            tail_tokens = 0
            for block in reversed(buf):
                t = count_tokens(block)
                if tail_tokens + t > overlap:
                    break
                tail.insert(0, block)
                tail_tokens += t
            buf = tail
            buf_tokens = tail_tokens
            buf_heading = heading

        for block in _split_blocks(page.text):
            first_line = block.split("\n", 1)[0]
            if _is_heading(first_line) and len(block.split("\n")) <= 2:
                flush()
                heading = first_line.strip()
                buf_heading = heading
                # Keep the heading in the chunk text as well. Recording it only
                # as metadata would drop those words from the searchable body.
                buf.append(block)
                buf_tokens += count_tokens(block)
                continue

            tokens = count_tokens(block)

            # A single oversized block (a long table) must still be split.
            if tokens > target:
                flush()
                for piece in _split_oversized(block, count_tokens, target):
                    chunks.append(Chunk(next_id, piece, page.number, page.source, heading))
                    next_id += 1
                buf, buf_tokens, buf_heading = [], 0, heading
                continue

            if buf_tokens + tokens > target:
                flush()
            buf.append(block)
            buf_tokens += tokens

        flush()

    return chunks


def _split_oversized(block: str, count_tokens: Callable[[str], int], target: int) -> List[str]:
    """Break a too-large block on sentence/line boundaries."""
    units = re.split(r"(?<=[.!?])\s+|\n", block)
    out: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for u in units:
        u = u.strip()
        if not u:
            continue
        t = count_tokens(u)
        if cur and cur_tokens + t > target:
            out.append(" ".join(cur))
            cur, cur_tokens = [], 0
        cur.append(u)
        cur_tokens += t
    if cur:
        out.append(" ".join(cur))
    return out
