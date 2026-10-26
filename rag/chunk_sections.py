"""Stage 2b - turn a document's section tree into retrieval units.

The flat chunker in `chunk.py` packs a page's blocks up to a token budget. That
is the right approach for a short, single-column report, but it has no notion
of where a passage sits, and on a 200-page document that costs accuracy in two
ways. Chunks run across section boundaries, mixing the end of one argument with
the start of another; and every chunk looks alike to the retriever, because the
words that would distinguish them - the chapter title, the region, the box
number - are on a different page.

This chunker works from the hierarchy instead:

  * a chunk never spans two sections, so it is about one thing;
  * a chunk carries its breadcrumb into the embedded text, so it can be found
    by a question that names its chapter, its region or its box;
  * tables are split by rows with their header repeated, so a row stays
    readable on its own;
  * reference lists are kept but marked, so retrieval can hold them back.
"""
from __future__ import annotations

import re
from typing import Callable, List

from .chunk import MIN_TOKENS, OVERLAP_TOKENS, TARGET_TOKENS, Chunk
from .structure import ROLE_REFERENCE, ROLE_TABLE, Element, Section

# A table's own header rows, repeated at the top of every piece it is split
# into. Without them a piece reads "Angola 2.4 3.1 2.8 ..." with no way to
# tell which years those are, and no way for a question about 2026 to match.
TABLE_HEADER_LINES = 4

# A line that ends a paragraph: full stop and comfortably short of the measure.
_COLUMN_MEASURE = 46


def _is_sentence_end(line: str) -> bool:
    return line.rstrip().endswith((".", ":", "!", "?", '."', ".'", ".)"))


def _paragraphs(elements: List[Element]) -> List[str]:
    """Reassemble justified column lines into paragraphs.

    Each element is one typeset line, so a paragraph has to be rebuilt from
    the run of lines that fills the measure. A line that both ends a sentence
    and stops short of the measure is the last line of its paragraph.
    """
    out: List[str] = []
    buf: List[str] = []

    for element in elements:
        text = element.text.strip()
        if not text:
            continue
        buf.append(text)
        if _is_sentence_end(text) and len(text) < _COLUMN_MEASURE:
            out.append(" ".join(buf))
            buf = []
        elif text.startswith(("•", "-", "–", "—")) and len(buf) > 1:
            out.append(" ".join(buf[:-1]))
            buf = [text]

    if buf:
        out.append(" ".join(buf))
    return [p for p in (" ".join(p.split()) for p in out) if p]


def _rows(elements: List[Element]) -> List[str]:
    """Table and reference matter, kept one record per line."""
    return [" ".join(e.text.split()) for e in elements if e.text.strip()]


def _pack(units: List[str], count_tokens: Callable[[str], int],
          target: int, overlap: int, repeat: List[str]) -> List[List[str]]:
    """Group units into token-bounded pieces with a trailing overlap.

    `repeat` is prefixed to every piece after the first; it carries a table's
    column headers down into each of its parts.
    """
    pieces: List[List[str]] = []
    buf: List[str] = []
    used = 0
    repeat_cost = sum(count_tokens(r) for r in repeat)

    for unit in units:
        cost = count_tokens(unit)
        if cost > target:                       # one very long unit
            if buf:
                pieces.append(buf)
                buf, used = [], 0
            for part in _split_long(unit, count_tokens, target):
                pieces.append([part])
            continue

        budget = target - (repeat_cost if pieces else 0)
        if buf and used + cost > budget:
            pieces.append(buf)
            tail, tail_used = [], 0
            for prev in reversed(buf):
                t = count_tokens(prev)
                if tail_used + t > overlap:
                    break
                tail.insert(0, prev)
                tail_used += t
            buf, used = tail, tail_used
        buf.append(unit)
        used += cost

    if buf:
        pieces.append(buf)

    if repeat:
        pieces = [pieces[0]] + [repeat + p for p in pieces[1:]] if pieces else []
    return pieces


def _split_long(text: str, count_tokens: Callable[[str], int],
                target: int) -> List[str]:
    """Break one oversized unit on sentence boundaries."""
    units = [u.strip() for u in re.split(r"(?<=[.!?])\s+", text) if u.strip()]
    out: List[str] = []
    buf: List[str] = []
    used = 0
    for unit in units:
        cost = count_tokens(unit)
        if buf and used + cost > target:
            out.append(" ".join(buf))
            buf, used = [], 0
        buf.append(unit)
        used += cost
    if buf:
        out.append(" ".join(buf))
    return out or [text]


def chunk_sections(
    sections: List[Section],
    source: str,
    count_tokens: Callable[[str], int],
    target: int = TARGET_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> List[Chunk]:
    """Build chunks from a section tree, one section at a time."""
    chunks: List[Chunk] = []
    next_id = 0

    for section in sections:
        elements = section.elements
        if not elements:
            continue

        tabular = section.kind in (ROLE_TABLE, "reference") or \
            all(e.role in (ROLE_TABLE, ROLE_REFERENCE) for e in elements)
        units = _rows(elements) if tabular else _paragraphs(elements)
        if not units:
            continue

        repeat: List[str] = []
        if section.kind == ROLE_TABLE:
            repeat = units[:TABLE_HEADER_LINES]

        # The breadcrumb is counted against the budget, since it is prepended
        # before the text is embedded and BGE truncates at 512 tokens.
        path = section.path()
        budget = max(target - count_tokens(path) - 8, 96)

        pieces = _pack(units, count_tokens, budget, overlap, repeat)
        # A section short enough to be a single piece is a complete thought --
        # a numbered endnote, a one-line table note -- and MIN_TOKENS, which
        # exists to drop stray page furniture from the middle of a section,
        # would throw the whole thing away. "Renewable energy emissions are
        # too small to be visible on the chart." is 13 tokens and is the only
        # place the report explains a gap in its own chart.
        whole = len(pieces) == 1 and bool(section.container or section.heading)

        for piece in pieces:
            text = "\n".join(piece).strip() if tabular else "\n\n".join(piece).strip()
            if not text or (count_tokens(text) < MIN_TOKENS and not whole):
                continue
            first = elements[0]
            page = _page_of(elements, piece)
            chunks.append(Chunk(
                id=next_id,
                text=text,
                page=page.page,
                source=source,
                heading=section.heading or section.container or section.part,
                label=page.label or "",
                part=section.part,
                subheading=section.subheading,
                container=section.container,
                kind=section.kind,
                path=path,
            ))
            next_id += 1

    return chunks


def _page_of(elements: List[Element], piece: List[str]) -> Element:
    """The element a piece starts on, so the citation points at the right page.

    A section can run over a page break, and citing the section's first page
    for every one of its chunks would send a reader several pages away from
    the sentence they were shown.
    """
    if not piece:
        return elements[0]
    head = piece[0][:40]
    for element in elements:
        if element.text[:40] == head or element.text.startswith(head[:24]):
            return element
    return elements[0]
