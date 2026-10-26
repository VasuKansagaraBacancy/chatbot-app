"""Unit tests for the ingestion and retrieval pipeline.

These cover the parts of the pipeline that are pure functions over text and
geometry -- layout reconstruction, role classification, section splitting,
chunk packing and query parsing. None of them loads the language model or the
embedder, so the whole file runs in about a second and is usable in the
edit-and-check loop.

What is deliberately not here: answer quality. That is measured against the
real documents by `rag.recall` (retrieval only, seconds) and `rag.evaluate`
(full generation, minutes), because it is a property of the corpus and the
model rather than of a function.

    .venv\\Scripts\\python.exe -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import textfix
from rag.chunk import Chunk
from rag.chunk_sections import chunk_sections
from rag.layout import (Fragment, Line, PageLayout, _hoist_margin,
                        _parse_running_head, _split_by_size, calibrate_labels)
from rag.retrieve import _diversify, _named_targets, needs_rewrite
from rag.store import Hit
from rag.structure import (ROLE_BODY, ROLE_HEADING, ROLE_NOTE, ROLE_TABLE,
                           Element, Section, Style, _classify, _is_measure,
                           _looks_tabular, profile_style, split_notes)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def line(text: str, size: float, font: str = "Body-Regular",
         y: float = 500.0, x0: float = 72.0, x1: float = 400.0) -> Line:
    frag = Fragment(x=x0, y=y, text=text, size=size, font=font)
    return Line(y=y, x0=x0, x1=x1, text=text, frags=[frag])


def page(number: int, lines, label=None) -> PageLayout:
    return PageLayout(number=number, body="\n".join(l.text for l in lines),
                      lines=lines, label=label)


def body_page(number: int, size: float = 10.0, count: int = 20) -> PageLayout:
    return page(number, [line(f"body text line {i}", size) for i in range(count)])


# --------------------------------------------------------------------------
# printed page numbers
# --------------------------------------------------------------------------

def test_labels_at_a_constant_offset_are_kept():
    """Front matter shifts the printed number; the offset stays constant."""
    pages = [page(n, [], label=str(n - 22)) for n in range(23, 60)]
    calibrate_labels(pages)
    assert [p.label for p in pages[:3]] == ["1", "2", "3"]


def test_stray_numbers_in_a_header_are_discarded():
    """The defect this guards: a header containing any number was cited.

    The GPO annual report has no running page numbers, but its header band
    holds ordinary text. Every digit run in it was being read as the printed
    page, so PDF page 3 was cited to the reader as "p.732".
    """
    pages = [page(3, [], label="732"), page(4, [], label="220"),
             page(5, [], label="201"), page(6, [], label="420")]
    calibrate_labels(pages)
    assert all(p.label is None for p in pages)


def test_a_single_outlier_is_dropped_but_the_run_survives():
    pages = [page(n, [], label=str(n - 10)) for n in range(11, 25)]
    pages[4].label = "999"
    calibrate_labels(pages)
    assert pages[4].label is None
    assert sum(p.label is not None for p in pages) == len(pages) - 1


def test_roman_front_matter_is_read_from_the_running_head():
    assert _parse_running_head("vii") == ("vii", "Front matter")


def test_a_bare_page_number_is_read_as_a_label():
    assert _parse_running_head("42") == ("42", None)


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------

def test_one_baseline_carrying_two_sizes_is_split():
    """A 24pt margin heading set beside 7pt body is not one line."""
    frags = [Fragment(x=40, y=700, text="Endnotes", size=24.0, font="Display"),
             Fragment(x=180, y=700, text="1 Recycled content is", size=7.0,
                      font="Body")]
    merged = Line(y=700, x0=40, x1=400, text="Endnotes 1 Recycled content is",
                  frags=frags)
    out = _split_by_size(merged)
    assert len(out) == 2
    assert out[0].size == 24.0
    assert out[1].size == 7.0


def test_a_line_of_one_size_is_left_alone():
    assert len(_split_by_size(line("ordinary body copy", 10.0))) == 1


def test_margin_headings_are_read_before_the_column_they_label():
    """In y-order the heading interleaves with the column and splits it."""
    content = [line(f"column line {i}", 9.0, y=700 - i * 12, x0=200.0, x1=520.0)
               for i in range(6)]
    heading = line("Carbon", 22.0, y=688, x0=40.0, x1=120.0)
    ordered = _hoist_margin(content[:2] + [heading] + content[2:])
    assert ordered[0].text == "Carbon"


def test_a_narrow_column_is_not_mistaken_for_a_margin():
    lines = [line(f"left {i}", 9.0, y=700 - i * 12, x0=40.0, x1=120.0)
             for i in range(6)]
    assert _hoist_margin(lines) == lines


# --------------------------------------------------------------------------
# typographic calibration
# --------------------------------------------------------------------------

def test_thresholds_are_measured_from_the_document():
    """Roles are ratios of the body size, not absolute points.

    Apple sets body copy at 8.5pt and headings at 24pt; the World Bank sets
    11pt and 16pt. A fixed threshold works for one and finds nothing in the
    other.
    """
    pages = []
    for n in range(1, 13):
        lines = [line(f"body {i}", 8.5) for i in range(40)]
        lines.append(line("Carbon Footprint", 24.0, font="Display-Light"))
        pages.append(page(n, lines))

    style = profile_style(pages)
    assert style.body == 8.5
    assert style.heading < 24.0
    assert _classify(line("Carbon Footprint", 24.0, font="Display-Light"),
                     style) == ROLE_HEADING


def test_a_second_body_size_is_not_a_heading_tier():
    """969 lines of 12pt against a 9.8pt body is body copy, not headings."""
    pages = []
    for n in range(1, 13):
        lines = [line(f"body {i}", 9.8) for i in range(40)]
        lines += [line(f"also body {i}", 12.0) for i in range(30)]
        pages.append(page(n, lines))

    style = profile_style(pages)
    assert _classify(line("also body 1", 12.0), style) == ROLE_BODY


def test_a_cover_title_does_not_become_the_heading_tier():
    """70pt type used once on the cover is not the document's heading size."""
    pages = [page(1, [line("GLOBAL ECONOMIC PROSPECTS", 70.0, font="Display")])]
    for n in range(2, 14):
        lines = [line(f"body {i}", 11.0) for i in range(40)]
        lines.append(line("Recent developments", 16.0, font="Helvetica-Bold"))
        pages.append(page(n, lines))

    style = profile_style(pages)
    assert style.heading < 16.0
    assert style.heading > 11.0


@pytest.mark.parametrize("text", ["445 kg", "-99 kg", "76%", "346 kg CO2e", "100%"])
def test_display_numbers_are_data_not_headings(text):
    assert _is_measure(text)


@pytest.mark.parametrize("text", ["Carbon Footprint", "Note 18", "2026 outlook ahead"])
def test_real_headings_are_not_mistaken_for_measures(text):
    assert not _is_measure(text)


def test_a_row_of_figures_is_recognised_as_tabular():
    assert _looks_tabular("World 2.8 2.9 2.9 2.5")
    assert not _looks_tabular("Growth is expected to slow in 2026.")


# --------------------------------------------------------------------------
# endnotes
# --------------------------------------------------------------------------

def notes_section(*texts: str) -> Section:
    section = Section(heading="Endnotes")
    section.elements = [Element(text=t, page=11, role=ROLE_NOTE) for t in texts]
    return section


def test_each_numbered_note_becomes_its_own_section():
    out = split_notes([notes_section(
        "1 Recycled content is the mass of certified material.",
        "2 We estimate the percentage of electricity-related emissions.",
        "3 Breakdown of U.S. retail packaging by weight.")])
    assert [s.container for s in out] == ["Note 1", "Note 2", "Note 3"]


def test_a_measurement_is_not_read_as_the_next_note():
    """"15 percent" opening a line is a figure, not note 15."""
    out = split_notes([notes_section(
        "1 Recycled content is the mass of certified material.",
        "15 percent of the enclosure is recycled aluminium.")])
    assert [s.container for s in out] == ["Note 1"]
    assert "15 percent" in out[0].elements[-1].text


def test_notes_resume_after_a_page_break():
    """A second block restarts the counter, so note 17 must be accepted."""
    out = split_notes([notes_section("17 Responsible sourcing of wood fiber.",
                                     "18 Energy consumption values.")])
    assert [s.container for s in out] == ["Note 17", "Note 18"]


def test_a_section_that_is_not_notes_is_untouched():
    section = Section(heading="Carbon Footprint")
    section.elements = [Element(text="1 kg of CO2e per unit", page=2, role=ROLE_BODY)]
    assert split_notes([section]) == [section]


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def tokens(text: str) -> int:
    """Stand-in for the embedder's tokenizer: close enough for packing."""
    return max(1, len(text.split()))


def test_a_short_but_complete_section_is_kept():
    """Three endnotes were being deleted for being under MIN_TOKENS.

    The floor exists to drop stray page furniture from the middle of a
    section. Applied to a section that is one piece long, it threw away the
    only place the report explained a gap in its own chart.
    """
    section = Section(heading="Endnotes", container="Note 11")
    section.elements = [Element(
        text="Renewable energy emissions are too small to be visible on the chart.",
        page=11, role=ROLE_BODY)]
    chunks = chunk_sections([section], "imac.pdf", tokens)
    assert len(chunks) == 1
    assert "too small to be visible" in chunks[0].text


def test_a_chunk_never_spans_two_sections():
    first = Section(heading="One")
    first.elements = [Element(text="alpha " * 40, page=1, role=ROLE_BODY)]
    second = Section(heading="Two")
    second.elements = [Element(text="beta " * 40, page=2, role=ROLE_BODY)]
    chunks = chunk_sections([first, second], "doc.pdf", tokens)
    assert all(("alpha" in c.text) != ("beta" in c.text) for c in chunks)


def test_every_chunk_carries_its_breadcrumb():
    section = Section(part="Chapter 3", heading="Key trends")
    section.elements = [Element(text="prose " * 60, page=5, role=ROLE_BODY)]
    chunks = chunk_sections([section], "doc.pdf", tokens)
    assert chunks and all(c.path == "Chapter 3 > Key trends" for c in chunks)


def test_a_tables_header_row_travels_with_every_piece():
    """Without it a piece reads "Angola 2.4 3.1" with no year to match on."""
    section = Section(heading="Forecasts", kind=ROLE_TABLE)
    rows = ["Country 2025 2026 2027"] + [f"Country{i} 2.{i} 3.{i} 4.{i}"
                                         for i in range(60)]
    section.elements = [Element(text=r, page=9, role=ROLE_TABLE) for r in rows]
    chunks = chunk_sections([section], "doc.pdf", tokens, target=60)
    assert len(chunks) > 1
    assert all("Country 2025 2026 2027" in c.text for c in chunks)


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("What does Table 1.1 show for China?", {"table 1.1"}),
    ("What is Box 3.1 about?", {"box 3.1"}),
    ("What does endnote 18 say?", {"note 18"}),
    ("What does footnote 6 say?", {"note 6"}),
    ("What is Figure B1.2.3?", {"figure b1.2.3"}),
])
def test_a_named_block_is_resolved_to_its_container(question, expected):
    blocks, _ = _named_targets(question)
    assert blocks == expected


def test_a_named_chapter_is_resolved():
    _, parts = _named_targets("What does chapter 3 conclude?")
    assert parts == {"chapter 3"}


def test_an_ordinary_question_names_nothing():
    assert _named_targets("What is global growth in 2026?") == (set(), set())


def hit(path: str, score: float) -> Hit:
    return Hit(chunk=Chunk(id=0, text="t", page=1, source="d.pdf", path=path),
               score=score, dense=score, lexical=0.0)


def test_one_section_cannot_crowd_out_the_report():
    hits = [hit("A", 0.9), hit("A", 0.8), hit("A", 0.7), hit("B", 0.6)]
    kept = _diversify(hits, k=4, per_section=2)
    assert [h.chunk.path for h in kept] == ["A", "A", "B"]


@pytest.mark.parametrize("question", ["why?", "and the second one?",
                                      "what about it?", "tell me more about that"])
def test_an_elliptical_follow_up_is_rewritten(question):
    assert needs_rewrite(question, [{"role": "user", "content": "earlier"}])


def test_a_standalone_question_is_left_alone():
    history = [{"role": "user", "content": "earlier"}]
    assert not needs_rewrite(
        "What was GPO's total revenue in fiscal year 2018?", history)


def test_the_first_question_of_a_session_is_never_rewritten():
    assert not needs_rewrite("why?", [])


# --------------------------------------------------------------------------
# text repair
# --------------------------------------------------------------------------

def test_ligatures_are_normalized():
    """A BM25 index over "ﬁscal" holds no token "fiscal"."""
    assert textfix.normalize("ﬁscal policy") == "fiscal policy"


def test_the_document_supplies_its_own_spelling_dictionary():
    """pypdf reads wide kerning as a space: "poten tial" for "potential"."""
    vocab = textfix.build_vocabulary(["potential " * 50])
    assert "potential" in textfix.rejoin_split_words("poten tial growth", vocab)


def test_a_genuine_word_pair_is_not_welded_together():
    vocab = textfix.build_vocabulary(["in " * 80, "to " * 80, "into " * 5])
    assert textfix.rejoin_split_words("in to the report", vocab) == "in to the report"
