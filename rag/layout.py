"""Coordinate-aware page extraction for multi-column reports.

pypdf's plain `extract_text()` returns fragments in content-stream order. For a
single-column document that is reading order, but the World Bank GEP report is
typeset in two columns with full-width tables and figure banners dropped in
between. Stream order there interleaves the columns, so sentences from the left
column are spliced into the middle of the right column's paragraph. Chunks built
from that text are unreadable, and the embedding of a spliced paragraph points
nowhere useful.

This module rebuilds reading order from glyph coordinates instead:

    fragments -> lines (cluster by baseline)
                   -> bands (split at full-width lines)
                      -> left column, then right column, within each band

Running headers and footers are identified by their position on the page rather
than by matching their text, which matters because this report's header is
letter-spaced ("C H A PT ER  1") and carries the printed page number, so no two
pages share an identical header string.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Baselines within this many points belong to the same visual line.
_LINE_TOL = 3.0

# Fraction of page width where the gutter between the two columns sits.
_GUTTER = 0.48

# Bands at the very top/bottom of the page hold running headers and footers.
_HEADER_BAND = 0.93
_FOOTER_BAND = 0.07

# A line must span this fraction of the text block to count as full-width
# rather than as a long line that merely starts left and runs past the gutter.
_FULL_WIDTH_SPAN = 0.55


@dataclass
class Fragment:
    x: float
    y: float
    text: str
    size: float = 0.0   # effective point size, after the text matrix scale
    font: str = ""      # base font name, subset prefix stripped


@dataclass
class Line:
    y: float
    x0: float
    x1: float
    text: str
    frags: List["Fragment"] = field(default_factory=list)

    @property
    def size(self) -> float:
        """Point size of the line, taken from its widest-set fragment."""
        return max((f.size for f in self.frags), default=0.0)

    @property
    def font(self) -> str:
        """Font of the fragment that contributes the most characters."""
        if not self.frags:
            return ""
        return max(self.frags, key=lambda f: len(f.text)).font


@dataclass
class PageLayout:
    number: int                    # 1-based PDF page index
    body: str                      # text in reading order, furniture removed
    lines: List[Line] = field(default_factory=list)  # body lines, with style
    header: str = ""               # raw running header, if any
    footer: str = ""
    columns: int = 1
    label: Optional[str] = None    # printed page number ("5", "xvii")
    section: Optional[str] = None  # running-header section ("Chapter 1")


def _collect(page) -> List[Fragment]:
    frags: List[Fragment] = []

    def visit(text, cm, tm, font_dict, font_size):
        if not text or not text.strip():
            return
        name = ""
        if isinstance(font_dict, dict):
            # "/ABCDEF+AGaramondPro-Bold" -> "AGaramondPro-Bold"; the subset
            # prefix is regenerated per chapter and carries no information.
            name = str(font_dict.get("/BaseFont", "")).lstrip("/").split("+")[-1]
        try:
            size = float(font_size) * abs(float(tm[0]))
        except (TypeError, ValueError):
            size = 0.0
        frags.append(Fragment(x=tm[4], y=tm[5], text=text, size=size, font=name))

    try:
        page.extract_text(visitor_text=visit)
    except Exception:
        return []
    return frags


# Mean glyph advance as a fraction of point size, across the serif and sans
# faces this report uses. Only an estimate: the exact widths live in the font
# programs, and parsing those to place a space would be a poor trade.
_ADVANCE = 0.46


def _width(text: str, size: float = 10.0) -> float:
    """Rough advance width; only the relative span matters here."""
    return len(text.rstrip()) * _ADVANCE * (size or 10.0)


def _join(frags: List[Fragment]) -> str:
    """Join fragments left to right, inserting a space where the PDF left a gap.

    Fragments must be assembled in x order, never in the order the content
    stream happened to emit them: on these pages the right-hand page number is
    drawn a fraction above the running head, so stream order puts it in front
    of the header and the printed page number is read as the "6" of "JUNE 2026".
    """
    out = ""
    prev_end: Optional[float] = None
    for frag in sorted(frags, key=lambda f: f.x):
        text = frag.text.replace("\n", " ")
        # A gap only means a space if it is wide relative to the type size.
        # Judged against a fixed number of points, ordinary kerning in small
        # type reads as a word break and "potential" comes out "poten tial" —
        # which costs the keyword index the word entirely.
        if (prev_end is not None and not out.endswith(" ")
                and frag.x - prev_end > 0.3 * (frag.size or 10.0)):
            out += " "
        out += text
        prev_end = frag.x + _width(text, frag.size)
    return out.rstrip()


def _to_lines(frags: List[Fragment]) -> List[Line]:
    """Cluster fragments sharing a baseline into single visual lines."""
    if not frags:
        return []

    rows: List[Line] = []
    for frag in sorted(frags, key=lambda f: (-f.y, f.x)):
        placed = False
        # Fragments arrive top-to-bottom, so only the most recent rows can
        # still be open. Scanning all of them would merge unrelated lines.
        for line in reversed(rows[-4:]):
            if abs(line.y - frag.y) <= _LINE_TOL:
                line.frags.append(frag)
                line.x0 = min(line.x0, frag.x)
                line.x1 = max(line.x1, frag.x + _width(frag.text, frag.size))
                placed = True
                break
        if not placed:
            rows.append(Line(y=frag.y, x0=frag.x,
                             x1=frag.x + _width(frag.text, frag.size), text="",
                             frags=[frag]))

    out: List[Line] = []
    for line in rows:
        out.extend(_split_by_size(line))
    for line in out:
        line.text = _join(line.frags)
    return [l for l in out if l.text.strip()]


# A display heading set in the margin sits on the same baseline as the first
# line of body copy beside it. Merged into one line, the whole thing inherits
# the heading's size, and a paragraph of body text is read as a heading - on
# Apple's endnotes page that swallowed the entire first endnote. Real typeset
# lines do not mix sizes this far apart, so the baseline is cut where the
# size changes.
_SIZE_SPLIT = 1.5


def _split_by_size(line: Line) -> List[Line]:
    """Cut one baseline into runs of comparable type size."""
    frags = sorted(line.frags, key=lambda f: f.x)
    if len(frags) < 2:
        return [line]

    runs: List[List[Fragment]] = [[frags[0]]]
    for frag in frags[1:]:
        last = runs[-1][-1]
        big, small = max(frag.size, last.size), min(frag.size, last.size)
        if small > 0 and big / small >= _SIZE_SPLIT:
            runs.append([frag])
        else:
            runs[-1].append(frag)
    if len(runs) == 1:
        return [line]

    return [Line(y=line.y, x0=run[0].x,
                 x1=run[-1].x + _width(run[-1].text, run[-1].size),
                 text="", frags=run)
            for run in runs]


def _is_two_column(lines: List[Line], gutter: float) -> bool:
    """Decide whether a page is set in two columns or runs full width.

    The test is how many lines stay wholly on one side of the gutter. Prose
    pages have almost every line confined to one column; a page given over to
    a wide forecast table has rows whose cells straddle the gutter, and those
    rows must not be torn in half.
    """
    if len(lines) < 6:
        return False

    frags = [f for l in lines for f in l.frags]
    if len(frags) < 8:
        return False

    # Test where fragments *start*, never where an estimated advance width
    # says they end: glyph widths are not known without parsing the font, and
    # a long prose line overshooting the gutter would be read as a table row.
    #
    # The two columns are typeset on a shared baseline grid, so counting
    # lines that stay on one side proves nothing — on a well-aligned page
    # every baseline carries a line from each column. What does separate the
    # cases is alignment: a real column has a left edge, so nearly every one
    # of its lines begins at the very same x. A wide table starts cells at as
    # many different x positions as it has columns, and must be left whole —
    # splitting its rows at the gutter would strand the forecast years in one
    # half of the index and the country names in the other.
    left = [f.x for f in frags if f.x < gutter]
    right = [f.x for f in frags if f.x >= gutter]
    if len(left) < 4 or len(right) < 4:
        return False

    # One well-aligned side is enough. A page often pairs a column of prose
    # with a figure whose panel labels sit at a dozen different offsets, and
    # demanding alignment on both sides would send that page down the
    # full-width path and interleave the prose with the figure.
    return max(_alignment(left), _alignment(right)) >= 0.75


def _alignment(xs: List[float], tolerance: float = 3.0) -> float:
    """Share of these x positions that sit on the single commonest left edge."""
    rounded = [round(x / tolerance) for x in xs]
    edge = max(set(rounded), key=rounded.count)
    return rounded.count(edge) / len(rounded)


def _sub_line(line: Line, frags: List[Fragment]) -> Line:
    """One column's half of a line that straddles the gutter."""
    return Line(y=line.y, x0=min(f.x for f in frags),
                x1=max(f.x + _width(f.text, f.size) for f in frags),
                text=_join(frags), frags=frags)


# A heading set in the left margin, beside the content column rather than
# above it. Apple's report puts every section heading there, so "Carbon" and
# "Footprint" sit two baselines apart with body copy between them, and are
# read as two separate headings around a stray fragment of the table.
_MARGIN_GAP = 24.0          # points clear of the content column
_MARGIN_SHARE = 0.4         # at most this fraction of a page may be margin


def _hoist_margin(lines: List[Line]) -> List[Line]:
    """Read a left-margin heading block before the column it labels.

    The content column has a left edge that most lines start at. Anything
    lying wholly to the left of it is marginal: a heading, a pull quote, a
    standfirst. Read in plain top-to-bottom order it is interleaved with the
    copy beside it, which breaks a two-line heading into two headings.
    """
    if len(lines) < 4:
        return lines

    edges = [round(l.x0) for l in lines]
    content = max(set(edges), key=edges.count)

    margin = [l for l in lines if l.x1 < content - _MARGIN_GAP]
    if not margin or len(margin) > len(lines) * _MARGIN_SHARE:
        return lines
    body = [l for l in lines if l.x1 >= content - _MARGIN_GAP]
    return margin + body


def _reading_order(lines: List[Line], page_width: float) -> Tuple[List[Line], int]:
    """Order lines as a human reads them, honouring the column gutter.

    On a two-column page the left column is read in full, then the right. The
    lines are returned rather than joined text so that later stages can still
    see each line's point size and font, which is how headings are found.
    """
    if not lines:
        return [], 1

    gutter = page_width * _GUTTER
    if not _is_two_column(lines, gutter):
        return _hoist_margin(lines), 1

    left: List[Line] = []
    right: List[Line] = []

    for line in lines:                                   # already top-to-bottom
        near = [f for f in line.frags if f.x < gutter]
        far = [f for f in line.frags if f.x >= gutter]
        # The two columns are typeset on a shared grid, so most baselines carry
        # a line from each. Such a row is split at the gutter and its halves
        # sent to their own columns; emitting it whole is what splices the
        # start of a left-column sentence into the middle of a right-column
        # paragraph. Pages that are genuinely full width never reach here,
        # having failed the two-column test above.
        if near and far:
            left.append(_sub_line(line, near))
            right.append(_sub_line(line, far))
        elif near:
            left.append(line)
        else:
            right.append(line)

    return left + right, 2


_ROMAN = re.compile(r"^[ivxlcdm]{1,7}$", re.I)
_TRAILING_NUM = re.compile(r"(\d{1,3})\s*$")
_LEADING_NUM = re.compile(r"^\s*(\d{1,3})\b")

# Matched against the de-spaced header, so no spacing variant can evade it.
_BOILERPLATE = re.compile(
    r"GLOBALECONOMICPROSPECTS\|?(JANUARY|JUNE)?\d{4}", re.I)
_CHAPTER_TOKEN = re.compile(r"CHAPTER(\d+)")

_SECTION_NAMES = {
    "STATISTICALAPPENDIX": "Statistical Appendix",
    "SELECTEDTOPICS": "Selected Topics",
    "EXECUTIVESUMMARY": "Executive Summary",
    "REFERENCES": "References",
}


def _despace(text: str) -> str:
    """Collapse the letter-spacing used in the running head.

    "C H A PT ER  1" and "STA TISTICA L A PPEN DIX" only become recognisable
    once every space is removed; the spacing is inconsistent between pages.
    """
    return re.sub(r"\s+", "", text)


def _parse_running_head(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (printed page label, section name) from a running header/footer."""
    stripped = text.strip()
    if not stripped:
        return None, None

    # Front matter: the page label is the whole line, in roman numerals.
    if _ROMAN.match(stripped) and stripped.lower() != "i":
        return stripped.lower(), "Front matter"
    if stripped.isdigit() and len(stripped) <= 3:
        return stripped, None

    flat = _despace(stripped).upper()

    # The running head reads "<section> GLOBAL ECONOMIC PROSPECTS | JUNE 2026"
    # with the printed page number on the outer edge: to the right on odd
    # pages, to the left on even ones.
    section = None
    bare = _BOILERPLATE.sub(" ", flat)
    chapter = _CHAPTER_TOKEN.search(bare)
    if chapter:
        section = "Chapter " + chapter.group(1)
        bare = _CHAPTER_TOKEN.sub(" ", bare)
    else:
        for key, name in _SECTION_NAMES.items():
            if key in bare:
                section = name
                bare = bare.replace(key, " ")
                break

    # Whatever digits survive the title, the date and the section number are
    # the printed page number.
    digits = re.findall(r"\d{1,3}", bare)
    label = digits[0] if digits else None
    return label, section


# A printed page number has to be at least this common among the labels found
# before the document is believed to be numbered at all, and a label that does
# not sit at the agreed offset is a false positive.
_LABEL_AGREEMENT = 0.5
_LABEL_MINIMUM = 3


def calibrate_labels(pages: List["PageLayout"]) -> None:
    """Discard printed page labels that do not form a consistent sequence.

    `_parse_running_head` takes the digits left over after the boilerplate is
    stripped, which is right for a report whose running head is its title and
    its page number -- and wrong for one whose header band holds ordinary
    text. The GPO annual report produced "p.732" for PDF page 3 and "p.201"
    for page 5, and those numbers were shown to the reader as the page to go
    and check.

    A real printed number tracks the PDF page at a constant offset: the GEP
    report's front matter is 22 pages long, so printed 89 is PDF 111, on every
    page of the body. That offset is the test. Labels agreeing with the
    document's modal offset are kept, the rest are dropped, and a document
    where no offset commands a majority is treated as unnumbered -- its
    citations fall back to the PDF page, which is at least true.
    """
    numbered = [p for p in pages if p.label and p.label.isdigit()]
    if not numbered:
        return

    offsets = Counter(int(p.label) - p.number for p in numbered)
    offset, agreed = offsets.most_common(1)[0]
    if agreed < _LABEL_MINIMUM or agreed < _LABEL_AGREEMENT * len(numbered):
        for page in numbered:
            page.label = None
        return

    for page in numbered:
        if int(page.label) - page.number != offset:
            page.label = None


def extract_page(page, number: int) -> PageLayout:
    """Extract one page as ordered body text plus structural metadata."""
    frags = _collect(page)
    if not frags:
        return PageLayout(number=number, body="")

    try:
        page_w = float(page.mediabox.width)
        page_h = float(page.mediabox.height)
    except Exception:
        page_w, page_h = 612.0, 792.0

    lines = _to_lines(frags)
    if not lines:
        return PageLayout(number=number, body="")

    top = page_h * _HEADER_BAND
    bottom = page_h * _FOOTER_BAND

    header = " ".join(l.text for l in lines if l.y >= top).strip()
    footer = " ".join(l.text for l in lines if l.y <= bottom).strip()
    body_lines = [l for l in lines if bottom < l.y < top]

    label, section = _parse_running_head(header)
    if label is None:
        label, footer_section = _parse_running_head(footer)
        section = section or footer_section

    ordered, columns = _reading_order(body_lines, page_w)
    body = "\n".join(l.text for l in ordered)
    return PageLayout(number=number, body=body, lines=ordered, header=header,
                      footer=footer, columns=columns, label=label,
                      section=section)
