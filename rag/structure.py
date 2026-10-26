"""Stage 1b - recover a document's structure before it is chunked.

A 200-page flagship report is not a bag of paragraphs. It is four chapters,
each with headings and sub-headings, carrying numbered boxes that run across
several pages, annexes, dozens of figures and tables, a statistical appendix
and per-chapter reference lists. Chunking that as flat text throws away the
only information that makes retrieval tractable at this length: a passage
about "the share fell sharply after the pandemic" means nothing without
knowing that it sits under "Chapter 4 > Fiscal positions > commodity
exporters".

This module turns a PDF into an ordered list of `Element`s, each tagged with

  * where it sits in the hierarchy (chapter, heading, sub-heading, container),
  * what kind of thing it is (prose, table row, figure note, reference, ...),
  * which printed page it came from, as the report numbers its own pages.

Roles are decided from typography rather than from text patterns. The report
sets body copy in 11pt Garamond, section headings in 16pt Helvetica Bold,
sub-headings in 14pt, table and figure banners in 10pt bold, and chart axis
labels in 5.5pt plain Helvetica. Reading those sizes is both more accurate and
far more robust than guessing from capitalization, which is what defeats
heading detection on documents like this one.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .layout import Line, PageLayout, calibrate_labels, extract_page
from .textfix import build_vocabulary, rejoin_split_words, repair

# --- role thresholds, calibrated per document ------------------------------
# Absolute point sizes cannot be used here. The World Bank report sets body
# copy at 11pt and section headings at 16pt; Apple's product report sets body
# at 8.5pt and section headings at 24pt. A threshold that finds one document's
# headings classifies the other's as something else entirely, which is how a
# perfectly good hierarchy silently becomes "no hierarchy found".
#
# What is stable across both is the *ratio* to the document's own body copy,
# because that ratio is what makes a heading look like a heading to a reader.
# So the body size is measured first - it is simply the commonest size on the
# page, since most of a document is body copy - and every other role is a
# multiple of it.
TITLE_RATIO = 2.00      # a part/chapter title, above the heading tier
HEADING_RATIO = 1.35    # minimum for a section heading
SUBHEADING_RATIO = 1.15 # minimum for a sub-heading
BANNER_RATIO = 0.85     # "TABLE 1.1 Real GDP" is set near body size
SMALL_RATIO = 0.95      # below this is small print: tables, figures, notes
NOTE_RATIO = 0.60       # "Source:", "Note:" under a table
TINY_RATIO = 0.45       # too small to carry meaning

ROLE_TITLE = "title"
ROLE_HEADING = "heading"
ROLE_SUBHEADING = "subheading"
ROLE_BANNER = "banner"
ROLE_BODY = "body"
ROLE_TABLE = "table"
ROLE_FIGURE = "figure"
ROLE_NOTE = "note"
ROLE_REFERENCE = "reference"
ROLE_NOISE = "noise"

# Plain "Helvetica" (as opposed to the HelveticaLTStd family used for real
# copy) is only ever used for chart axis labels, which extract as runs of bare
# numbers. They carry no retrievable meaning and would otherwise pad out
# chunks and skew their embeddings.
_CHART_FONT = re.compile(r"^Helvetica$|^Arial(MT)?$", re.I)

# A line of contents, recognisable by its dot leader.
_DOT_LEADER = re.compile(r"\.{4,}\s*[\divxlc]+\s*$")

_BANNER = re.compile(
    r"^(TABLE|FIGURE|BOX|ANNEX|MAP|SPOTLIGHT)\s+"
    r"([A-Z]?\d+(?:\.\d+)*(?:\.[A-Z])?)\b\s*(.*)$", re.I)

# "BOX 1.1 How much will AI affect global growth? (continued)". The closing
# bracket is sometimes a separate text fragment and can be lost, so it is
# optional here.
_CONTINUED = re.compile(r"\(\s*continued\s*\)?\s*$", re.I)

# Containers that hold running prose, as opposed to a caption and its panels.
_PROSE_CONTAINERS = {"box", "annex"}

_REFERENCES_HEAD = re.compile(r"^references\s*$", re.I)

# A reference entry: "World Bank. 2025a. Global Economic Prospects. ..."
_REFERENCE_ENTRY = re.compile(r"^[A-Z][^.]{1,60}[.,]\s*(19|20)\d{2}[a-z]?[.,]")


# Weights used for display type. "Bold" alone is not enough: Apple's headings
# are set in SFHello-Semibold, which contains no "Bold".
_DISPLAY_WEIGHT = re.compile(r"Bold|Semibold|Black|Heavy|Demi|Medium", re.I)
_OBLIQUE = re.compile(r"Obl|Italic|It$", re.I)

# A value in a table cell: "0.27W", "2.8", "-0.1", "<1%", "346", "91.1%".
_NUMERIC_TOKEN = re.compile(r"^[<>~≤≥(-]?\d[\d,.–—/]*[A-Za-z%°]{0,3}\)?$")


def _family(font: str) -> str:
    """The typeface a font belongs to, ignoring its weight and slant."""
    return re.split(r"[-,+]", font or "", 1)[0]


def _looks_tabular(text: str) -> bool:
    """Whether a line of small print reads as a row of a table.

    Tables are the one kind of small print worth tagging as such, because the
    chunker repeats a table's header row with every piece it cuts. Font family
    identifies them in a document that sets tables in a second typeface, but
    not in one that sets the whole document in a single family. What does
    carry across both is shape: a table row is mostly numbers.
    """
    tokens = text.split()
    if len(tokens) < 3:
        return False
    numeric = sum(1 for t in tokens if _NUMERIC_TOKEN.match(t))
    return numeric >= 2 and numeric / len(tokens) >= 0.3


@dataclass
class Style:
    """One document's measured style sheet."""
    body: float = 11.0
    family: str = ""
    title: float = 1e9      # unreachable when the document has no part tier
    heading: float = 1e9
    subheading: float = 1e9
    banner: float = 0.0
    small: float = 0.0
    note: float = 0.0
    tiny: float = 0.0


def profile_style(pages: List[PageLayout]) -> Style:
    """Measure a document's own style sheet, so roles can be read off it.

    The display tiers are taken from the sizes the document actually uses,
    not from a fixed table, and the heading tier is identified by how widely
    it is spread rather than by how large it is. Headings are the one kind of
    display type that recurs throughout a document, because every section has
    one; a cover title is enormous but appears on a single page. Picking the
    largest size instead reads Apple's section headings as chapter titles and
    leaves the document with no headings at all, and reads the World Bank
    cover as its heading tier.

    Two filters keep the measurement honest: a tier has to appear on several
    pages, which excludes one-off cover and divider type, and it has to be a
    small share of the document's lines, which excludes a second body size
    masquerading as display type.
    """
    sizes: Dict[float, int] = {}
    spread: Dict[float, set] = {}
    fonts: Dict[Tuple[float, str], int] = {}
    for page in pages:
        for line in page.lines:
            if not line.text.strip():
                continue
            size = round(line.size, 1)
            sizes[size] = sizes.get(size, 0) + 1
            spread.setdefault(size, set()).add(page.number)
            key = (size, _family(line.font))
            fonts[key] = fonts.get(key, 0) + 1
    if not sizes:
        return Style()

    body = max(sizes, key=lambda s: sizes[s])
    family = max((k for k in fonts if k[0] == body),
                 key=lambda k: fonts[k], default=(body, ""))[1]

    style = Style(
        body=body, family=family,
        banner=body * BANNER_RATIO, small=body * SMALL_RATIO,
        note=body * NOTE_RATIO, tiny=body * TINY_RATIO,
    )

    min_pages = 3 if len(pages) >= 12 else 2
    tiers = [
        s for s, n in sizes.items()
        if s >= body * SUBHEADING_RATIO
        and n >= 3 and len(spread[s]) >= min_pages
        and n < 0.15 * sizes[body]
    ]
    if not tiers:
        return style

    heading = max(tiers, key=lambda s: (len(spread[s]), s))
    above = [s for s in tiers if s > heading]
    below = [s for s in tiers if s < heading]

    style.heading = heading - 0.4
    style.title = (min(above) - 0.4) if above else 1e9
    style.subheading = (max(below) - 0.4) if below else style.heading
    return style


@dataclass
class Element:
    """One typeset line, once its role in the document is known."""
    text: str
    page: int                       # 1-based PDF page index
    role: str
    size: float = 0.0
    font: str = ""
    label: Optional[str] = None     # printed page number, as the report prints it
    part: Optional[str] = None      # "Chapter 3", "Executive Summary", ...


@dataclass
class Section:
    """A run of elements sharing one place in the hierarchy."""
    part: str = ""                  # "Chapter 3 - A Rising Challenge: ..."
    heading: str = ""               # 16pt section heading
    subheading: str = ""            # 14pt sub-heading
    container: str = ""             # "Box 1.1", "Table 2.3", "Annex 3.1"
    container_title: str = ""
    kind: str = ROLE_BODY           # body | table | figure | box | annex | reference
    elements: List[Element] = field(default_factory=list)

    def path(self) -> str:
        """Breadcrumb from document root down to this section.

        This string is prepended to every chunk before it is embedded, so a
        passage carries its own context into the vector space. It is the
        cheapest large win available on a long structured document: two
        paragraphs that read alike but sit in different chapters stop
        colliding, and a question naming a chapter or a box matches directly.
        """
        parts = [p for p in (self.part, self.heading, self.subheading) if p]
        if self.container:
            label = self.container
            if self.container_title:
                label = f"{label}: {self.container_title}"
            parts.append(label)
        return " > ".join(parts)


# "445 kg", "-99 kg", "76%", ">20%", "346 kg CO2e": a figure set in display
# type. Infographic numbers are typeset as large as a heading, but a heading
# they are not -- made one, the number becomes a breadcrumb and leaves the body
# text that explains it ("of the packaging is fiber-based") under a nameless
# parent, and the number itself is searchable only as a breadcrumb.
_MEASURE_TOKEN = re.compile(
    r"^[<>~≤≥–—+-]?\s*\d[\d,.–—]*\s*"
    r"(kg|g|w|wh|mm|cm|%|percent)?$", re.I)
_MEASURE_UNIT = re.compile(r"^(kg|g|w|wh|mm|cm|%|percent|co2e)$", re.I)


def _is_measure(text: str) -> bool:
    """Whether a display-type line is nothing but figures and their units."""
    tokens = text.split()
    if not tokens or len(tokens) > 6:
        return False
    return all(_MEASURE_TOKEN.match(t) or _MEASURE_UNIT.match(t) for t in tokens)


def _classify(line: Line, style: Style) -> str:
    """Assign a role to one line from its typography and its text."""
    text = line.text.strip()
    size = line.size
    font = line.font or ""

    if not text:
        return ROLE_NOISE
    if _CHART_FONT.match(font) or size <= style.tiny:
        return ROLE_NOISE
    if _DOT_LEADER.search(text):
        return ROLE_NOISE                      # table-of-contents entry

    if _BANNER.match(text) and size >= style.banner:
        return ROLE_BANNER
    if size >= style.subheading and _is_measure(text):
        return ROLE_BODY                       # a chart figure, not a heading
    if size >= style.title:
        return ROLE_TITLE
    if size >= style.heading:
        # Size alone decides here: the tier was measured from this document,
        # so anything reaching it is display type whatever its weight. Apple
        # sets "Endnotes" in a light 24pt face, and it is still a heading.
        return ROLE_HEADING
    if size >= style.subheading and _DISPLAY_WEIGHT.search(font):
        return ROLE_SUBHEADING

    if size <= style.note:
        return ROLE_NOTE
    if size < style.small:
        # Small print. Tabular matter is worth separating from it, because
        # the chunker carries a table's header row into every piece it cuts.
        if _looks_tabular(text):
            return ROLE_TABLE
        if _family(font) != style.family:
            return ROLE_FIGURE if _OBLIQUE.search(font) else ROLE_TABLE
        return ROLE_BODY
    return ROLE_BODY


def _chapter_map(reader, pages: List[PageLayout]) -> Dict[int, str]:
    """Map every PDF page to a human-readable part name.

    The running header already names the part ("Chapter 3"), and the PDF
    outline supplies that part's actual title. Combining them gives
    "Chapter 3 - A Rising Challenge: Sovereign Debt Levels and Interest Rates
    in EMDEs", which is far more useful in a breadcrumb, and more likely to be
    matched by a question, than either source alone.
    """
    titles: Dict[str, str] = {}
    starts: List[Tuple[int, str]] = []

    def walk(items) -> None:
        for item in items:
            if isinstance(item, list):
                walk(item)
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
                raw = " ".join(str(item.title).split())
            except Exception:
                continue
            if raw:
                starts.append((page, raw))

    try:
        walk(reader.outline or [])
    except Exception:
        starts = []

    for page, raw in starts:
        chapter = re.match(r"^Ch\.?\s*(\d+)\s*(.*)$", raw, re.I)
        if chapter:
            titles[f"Chapter {chapter.group(1)}"] = chapter.group(2).strip()

    # The header gives the part for every page; outline entries only give the
    # page a part begins on, and the two disagree on front matter.
    mapping: Dict[int, str] = {}
    current = "Front matter"
    for page in pages:
        if page.section:
            current = page.section
        name = current
        title = titles.get(current)
        if title:
            name = f"{current} - {title}"
        mapping[page.number] = name
    return mapping


def _merge_heading(lines: List[Element], start: int) -> Tuple[str, int]:
    """Join a heading that the column width broke over several lines.

    "Major economies: Recent" / "developments and outlook" are one heading set
    on two lines. Recording them separately would produce two sections, each
    labelled with half a title.
    """
    role = lines[start].role
    words = [lines[start].text.strip()]
    i = start + 1
    while (i < len(lines) and lines[i].role == role
           and lines[i].page == lines[start].page
           and abs(lines[i].size - lines[start].size) < 0.6):
        words.append(lines[i].text.strip())
        i += 1
    return " ".join(w for w in words if w).strip(), i


def read_elements(path: str | Path) -> List[Element]:
    """Extract one PDF as a flat, ordered, role-tagged element list."""
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))

    pages = [extract_page(page, i) for i, page in enumerate(reader.pages, 1)]
    # Printed page numbers are only trusted when they agree with each other;
    # see calibrate_labels. Without this a header that merely contains a
    # number is cited as the page to go and look at.
    calibrate_labels(pages)
    parts = _chapter_map(reader, pages)
    style = profile_style(pages)

    # The extractor's spurious mid-word spaces can only be found by comparing
    # a word against the rest of the document, so the vocabulary is built from
    # the whole PDF before any line is kept.
    vocab = build_vocabulary([repair(l.text) for p in pages for l in p.lines])

    elements: List[Element] = []
    for page in pages:
        for line in page.lines:
            text = rejoin_split_words(repair(line.text), vocab).strip()
            if not text:
                continue
            role = _classify(line, style)
            if role == ROLE_NOISE:
                continue
            elements.append(Element(
                text=text, page=page.number, role=role,
                size=round(line.size, 1), font=line.font,
                label=page.label, part=parts.get(page.number),
            ))
    return elements


def _trim_title(raw: str) -> str:
    """Keep a banner's own title, dropping caption text that ran into it.

    Splitting a two-column line at the gutter sometimes leaves the first line
    of a figure's description attached to its banner. A title is short; the
    description that follows starts a new sentence, so the cut is made there.
    """
    # A dangling "(" is the remains of a "(continued)" marker whose closing
    # bracket was a separate text fragment; left in, it makes a box's
    # continuation look like a different box.
    title = " ".join(raw.split()).rstrip("( ").strip(" .:-")
    if len(title) > 70:
        head = re.split(r"(?<=[a-z])\s{2,}|\.\s", title)[0]
        title = head if 3 < len(head) <= 70 else title[:70].rsplit(" ", 1)[0]
    return title.strip(" .:-")


# "Endnotes", "Footnotes", "Notes" - the section a numbered reference marker
# in the body points to.
_NOTES_HEADING = re.compile(r"^(end ?notes?|foot ?notes?|notes)\b", re.I)

# The start of one numbered note: "18 Energy consumption and energy ...".
# The start of one numbered note: "18 Energy consumption and energy ...".
# The text after the number may begin in lower case - Apple's sixth note
# opens with "iMac" - so the marker cannot be required to precede a
# capital. What keeps a measurement like "15 percent" from being read as
# the fifteenth note is that the numbers have to run forwards in step.
_NOTE_START = re.compile(r"^(\d{1,2})\s*([A-Za-z“‘(].*)$", re.S)

# Words that follow a number when it is a quantity rather than a marker.
_UNIT_AFTER = re.compile(
    r"^(percent|per\s?cent|kg|g|w|wh|mm|cm|in|percentage|million|billion|trillion|times|years?|months?|days?|hours?|pages?|parts?)", re.I)


def split_notes(sections: List["Section"]) -> List["Section"]:
    """Give every numbered endnote a section of its own.

    A product report carries much of its substance in its endnotes: which
    power adapter a figure was measured with, what "recycled content" counts,
    which configuration a carbon footprint describes. Left as one block they
    are a single 5,000-character section, so a question about any one of them
    retrieves a chunk that is mostly about the other eighteen - and the note
    that answers it may not even be in the part of the block that was kept.

    Notes are found by their own numbering rather than by position, and the
    numbers must not go backwards, so a figure like "15 percent" at the start
    of a continuation line cannot be mistaken for the sixteenth note.
    """
    out: List[Section] = []
    for section in sections:
        if not _NOTES_HEADING.match(section.heading or ""):
            out.append(section)
            continue

        current = Section(part=section.part, heading=section.heading,
                          subheading=section.subheading, kind=section.kind)
        last = 0
        for element in section.elements:
            match = _NOTE_START.match(element.text.strip())
            number = int(match.group(1)) if match else 0
            ok = (match and number > last
                  and (last == 0 or number <= last + 3)
                  and not _UNIT_AFTER.match(match.group(2)))
            if ok:
                if current.elements:
                    out.append(current)
                last = number
                current = Section(
                    part=section.part, heading=section.heading,
                    subheading=section.subheading,
                    container=f"Note {number}", kind=ROLE_NOTE)
                element = Element(text=match.group(2), page=element.page,
                                  role=element.role, size=element.size,
                                  font=element.font, label=element.label,
                                  part=element.part)
            current.elements.append(element)
        if current.elements:
            out.append(current)
    return out


def build_sections(elements: List[Element]) -> List[Section]:
    """Group elements into hierarchical sections.

    Walks the element list once, keeping a stack of the current chapter,
    heading, sub-heading and container. A new heading closes everything below
    it; a banner opens a container that lasts until the next banner or
    heading. Boxes are the exception: they run across page breaks and
    re-announce themselves as "(continued)", so a continuation re-opens the
    box it names instead of starting a new one.
    """
    sections: List[Section] = []
    part = heading = subheading = ""
    container = container_title = ""
    kind = ROLE_BODY
    in_references = False

    # A box's title is announced once; its "(continued)" banners on later
    # pages often carry nothing but the number, so titles are remembered.
    titles: Dict[str, str] = {}
    current: Optional[Section] = None

    def open_section() -> Section:
        section = Section(part=part, heading=heading, subheading=subheading,
                          container=container, container_title=container_title,
                          kind=kind)
        sections.append(section)
        return section

    i = 0
    while i < len(elements):
        element = elements[i]

        if element.part and element.part != part:
            part, heading, subheading = element.part, "", ""
            container = container_title = ""
            kind, in_references, current = ROLE_BODY, False, None

        role = element.role

        if role == ROLE_TITLE:
            # Front matter has no chapter number for the running header to
            # report, so its divider titles ("Executive Summary", "Foreword")
            # are the only thing naming those pages. Chapter openers merely
            # repeat the title that `part` already holds.
            if part.startswith("Front matter") and len(element.text) > 3:
                part = element.text.title() if element.text.isupper() \
                    else element.text
                heading = subheading = ""
                container = container_title = ""
                kind, current = ROLE_BODY, None
            i += 1
            continue

        if role in (ROLE_HEADING, ROLE_SUBHEADING):
            text, i = _merge_heading(elements, i)
            if role == ROLE_HEADING:
                heading, subheading = text, ""
            else:
                subheading = text
            container = container_title = ""
            in_references = bool(_REFERENCES_HEAD.match(text))
            kind = ROLE_REFERENCE if in_references else ROLE_BODY
            current = None
            continue

        if role == ROLE_BANNER:
            match = _BANNER.match(element.text)
            label = f"{match.group(1).title()} {match.group(2)}"
            title = _trim_title(_CONTINUED.sub("", match.group(3)))
            if title:
                titles.setdefault(label, title)

            if _CONTINUED.search(element.text) and label == container:
                i += 1                      # same container, carry on in place
                continue

            container = label
            container_title = title or titles.get(label, "")
            kind = {"box": "box", "annex": "annex", "table": ROLE_TABLE,
                    "figure": ROLE_FIGURE}.get(match.group(1).lower(), ROLE_BODY)
            current = open_section()
            if title:
                current.elements.append(element)
            i += 1
            continue

        # A figure or table container is a caption, its panel labels and its
        # source note - nothing more. When body copy resumes it belongs to the
        # enclosing section, not to the figure, and filing it under the figure
        # would both mislabel the citation and bury the prose behind a caption
        # in the index. Boxes and annexes genuinely do contain running prose,
        # so they keep absorbing until the next banner or heading.
        if (role == ROLE_BODY and container
                and kind not in _PROSE_CONTAINERS):
            container = container_title = ""
            kind, current = ROLE_BODY, None

        if in_references or kind == ROLE_REFERENCE:
            element.role = ROLE_REFERENCE

        if current is None:
            current = open_section()
        current.elements.append(element)
        i += 1

    return split_notes([s for s in sections if s.elements])
