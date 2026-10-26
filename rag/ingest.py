"""Stage 1 — document ingestion.

Reads a PDF into plain text, one record per page, and cleans up the artefacts
that PDF text extraction always produces (hyphen-broken words, hard-wrapped
lines, repeated headers/footers). Page numbers are preserved all the way
through the pipeline so answers can cite where a fact came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class Page:
    number: int          # 1-based, as printed in a PDF viewer
    text: str
    source: str          # file name, so a multi-document index stays traceable


# A word split across a line break: "govern-\nment" -> "government".
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")

# A line this long almost certainly hit the right margin, so the break after it
# is PDF wrapping rather than a real paragraph end. Short lines are headings,
# table rows or genuine paragraph ends and must keep their break — joining them
# is what destroys heading detection downstream.
_WRAP_WIDTH = 55


def _unwrap(text: str) -> str:
    """Rejoin margin-wrapped lines while preserving real line breaks."""
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if not out or not out[-1]:
            out.append(stripped)
            continue

        prev = out[-1]
        continues = (
            len(prev) >= _WRAP_WIDTH                 # previous line reached the margin
            and not prev.endswith((".", ":", ";", "!", "?", "•"))
            and not stripped[0] in "•-–—"            # not the start of a list item
            and not re.match(r"^\d+[.)]\s", stripped)  # not a numbered item
        )
        if continues:
            out[-1] = prev + " " + stripped
        else:
            out.append(stripped)
    return "\n".join(out)


def clean(text: str) -> str:
    """Undo PDF line-breaking so chunks contain readable prose."""
    text = text.replace("\r\n", "\n").replace("\xa0", " ")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = _unwrap(text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def _strip_repeating_lines(pages: List[Page], threshold: float = 0.12) -> None:
    """Remove running headers/footers in place.

    A short line repeated across many pages is furniture ("GPO Annual Report
    2018 | U.S. Government Publishing Office", a page-number band). Left in, it
    pollutes chunk text and drags unrelated chunks together at retrieval time,
    because they all share the same boilerplate.

    The threshold is deliberately low. Reports commonly alternate between two
    header variants on odd and even pages, and change them again per section,
    so any single variant may cover well under half the document. Deleting real
    prose is guarded against by `_looks_like_content` rather than by requiring
    a high repeat count.
    """
    if len(pages) < 4:
        return

    counts: dict[str, int] = {}
    for p in pages:
        lines = [l.strip() for l in p.text.split("\n")]
        # Headers/footers sit at the page edges; look a little way in, because
        # extraction order is not always exact.
        for line in lines[:4] + lines[-4:]:
            if 0 < len(line) <= 90:
                counts[line] = counts.get(line, 0) + 1

    cutoff = max(3, int(len(pages) * threshold))
    furniture = {
        line for line, n in counts.items()
        if n >= cutoff and not _looks_like_content(line)
    }
    if not furniture:
        return

    # Strip anywhere on the page, not only at the edges: once a line is known
    # to be furniture, a copy that extraction placed mid-page is still noise.
    for p in pages:
        kept = [l for l in p.text.split("\n") if l.strip() not in furniture]
        p.text = "\n".join(kept).strip()


def _looks_like_content(line: str) -> bool:
    """Guard against deleting a real sentence that happens to repeat."""
    return len(line.split()) > 12 or line.rstrip().endswith((".", ":", ";"))


def load_pdf(path: str | Path) -> List[Page]:
    """Extract every page of a PDF as cleaned text."""
    path = Path(path)
    reader = PdfReader(str(path))

    pages: List[Page] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""  # a single unreadable page must not abort the whole ingest
        pages.append(Page(number=i, text=clean(raw), source=path.name))

    _strip_repeating_lines(pages)
    return [p for p in pages if p.text.strip()]


def load_documents(folder: str | Path) -> List[Page]:
    """Ingest every PDF (and .txt) in a folder."""
    folder = Path(folder)
    pages: List[Page] = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() == ".pdf":
            pages.extend(load_pdf(f))
        elif f.suffix.lower() in {".txt", ".md"}:
            pages.append(Page(1, clean(f.read_text(encoding="utf-8", errors="replace")), f.name))
    return pages
