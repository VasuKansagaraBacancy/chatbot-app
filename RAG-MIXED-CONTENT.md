# RAG for documents whose facts live in tables, charts and footnotes

The baseline pipeline ([RAG.md](RAG.md)) was built for a report that is mostly
prose; the long-document work ([RAG-LONG-DOCUMENTS.md](RAG-LONG-DOCUMENTS.md))
added hierarchy for a 200-page two-column report. This document covers the
third corpus: Apple's *iMac Product Environmental Report, October 2024*.

It is twelve pages. That makes it the easiest of the three to read and by some
distance the hardest to ingest, because almost nothing in it is a paragraph:

* the carbon footprint is a **table** (`Total product footprint 346 kg CO2e`,
  and a four-line percentage split);
* the emissions story is a **waterfall chart** whose only text is the labels
  `445 kg`, `-99 kg`, `346 kg`;
* the power figures are a **three-column measurement table** by voltage;
* and the qualifications that make any of the headline claims meaningful —
  what "recycled content" counts, which countries are exempt from the
  PVC-free claim, which configuration the copper claim applies to — are in
  **nineteen numbered endnotes**, several of them one sentence long.

Running the structured pipeline over it unmodified produced
`no usable hierarchy found` and fell back to flat page chunking. Four separate
defects were behind that, and each one is a general defect that the first two
corpora simply never exposed.

## 1. Thresholds in points do not transfer between documents

`rag/structure.py` read roles off absolute font sizes tuned to the World Bank
report: body 11pt, headings 16pt and up. Apple sets body copy at **8.5pt** and
section headings at **24pt**. Every Apple heading was over the "title"
threshold, and nothing landed in the heading tier at all.

The fix is to measure the document before classifying it. `profile_style()`
takes the modal body size as the unit and expresses every role as a ratio of
it, then picks the tiers actually present:

| | body | title | heading | sub | small | note |
|---|---|---|---|---|---|---|
| GEP (World Bank) | 11.0 AGaramondPro | ≥25.6 | ≥15.6 | ≥13.6 | <10.4 | ≤6.6 |
| iMac (Apple) | 8.5 SFHello | — | ≥23.6 | ≥11.6 | <8.1 | ≤5.1 |
| GPO annual report | 9.8 ITCGaramondStd | ≥23.6 | ≥15.6 | ≥13.6 | <9.3 | ≤5.9 |

Two guards were needed to make tier selection behave:

* **A cover page is not a heading tier.** The GEP cover sets its title at
  70pt, the largest size in the document, and uses it once. A tier must occur
  on at least three pages before it counts, and the heading tier is chosen by
  *page spread* rather than by size — headings recur throughout a document,
  cover type does not.
* **A second body size is not a heading tier.** The GPO report sets 969 lines
  in 12pt against a 9.8pt body. A tier holding more than 15% as many lines as
  the body is body copy, whatever its size.

Weight was also dropped as a requirement for the heading tier: Apple sets
"Endnotes" in a 24pt *light* face, and it is still a heading.

This change alone flipped the GPO report from flat chunking (312 page chunks)
to a recovered hierarchy (29 sections, 129 chunks) with **no change in
retrieval accuracy** — 18/20 either way, the same two questions missed — which
is the regression check that made it safe to ship.

## 2. One typeset baseline can hold two different things

Apple prints section headings in the left margin, on the same baseline as the
first line of the content column. `Endnotes` (24pt) and the first endnote's
text (7pt) were clustered into one line, which inherited the 24pt size, so the
breadcrumb for all nineteen notes read:

```
Front matter > Endnotes device, not including packaging or in-box accessories.
```

`_split_by_size()` in `rag/layout.py` cuts a line wherever the size ratio
along it reaches 1.5x, so a margin heading and the body beside it become two
lines again.

## 3. Marginal headings must be read before the column they label

Even split, the margin heading and the content column interleave in y-order,
so `Carbon` and `Footprint` arrived as two separate sections with the table
orphaned between them. `_hoist_margin()` finds the modal left edge of a page
(the content column), treats lines that end well clear of it as margin
furniture, and reads them first — but only when they are a minority of the
page, so a genuinely narrow column is not reordered.

## 4. A short section is a complete thought, not noise

`MIN_TOKENS = 24` exists to drop stray page furniture out of the middle of a
section. Applied to a section that is *one piece long*, it deletes the whole
thing. Three endnotes were being discarded outright:

> Renewable energy emissions are too small to be visible on the chart.
> Recycled copper in the thermal module applies to iMac (four ports) only.
> All cobalt in the battery claims or references use mass balance allocation.

Each is the only place the report qualifies a claim it makes elsewhere in
large type. `chunk_sections` now keeps a single-piece section that has a name
of its own, and all nineteen notes are retrievable.

### Endnotes are split per note

A page of endnotes is one typographic block, so all nineteen landed in one
section and a question about note 6 retrieved a chunk mostly about notes 4, 5,
7 and 8. `split_notes()` cuts the block on a leading number into one Section
per note, filed as `container = "Note 6"`, which the retriever already knows
how to match by name ("What does endnote 6 say?" resolves to `Note 6`). The
cut is conservative, because a note's own text is full of numbers: the number
must be the next one in sequence, must not jump by more than three (notes
resume across a page break), and must not be followed by a unit —
`15 percent` at the start of a line is a measurement, not note 15.

## 5. Display-type numbers are data, not headings

An infographic sets `76%` as large as a heading. Classified as one, the number
became a breadcrumb and left the sentence that explains it ("recycled content
in fiber packaging") filed under a nameless parent — and the number itself was
searchable only as part of a path. `_is_measure()` routes a display-size line
that is nothing but figures and units back to body text, which is what
recovered `445 kg` and `-99 kg` from the waterfall chart into the text of the
section that explains them.

## 6. The relevance floor belongs to the corpus

`MIN_DENSE = 0.66` separates covered from uncovered questions cleanly on a
200-page report spanning four subjects. On a twelve-page single-product
brochure it does not, because every chunk shares the product's vocabulary:

| question | top dense |
|---|---|
| "When was this iMac introduced?" (covered) | 0.649 |
| "What processor does the iMac use?" (not covered) | 0.697 |
| "What is the capital of France?" (not covered) | 0.407 |

No floor separates the first two. The floor is therefore no longer a constant
in the code: `build_index --min-dense` writes it into the index's `meta.json`
and `retrieve()` reads it from the store, so documents of different breadth
can be served side by side. The iMac index is built at 0.60, which keeps
out-of-domain questions out and lets in-document questions through. Questions
about the product that the report does not answer are left to the generator,
which does decline them:

```
Q: What processor does the iMac use?
The report does not appear to cover this.
```

## What is retrievable now

```
.venv\Scripts\python.exe -m rag.build_index --data data_imac --index index_imac --min-dense 0.60
.venv\Scripts\python.exe -m rag.recall index_imac data_imac\questions.json
```

37 chunks from 12 pages, median 100 tokens, every one carrying a breadcrumb;
19 of them are individual endnotes.

| where the fact lives | example question | answer |
|---|---|---|
| Footprint table | "What is the carbon footprint of the iMac (four ports) with 512GB?" | 391 kg CO2e |
| Percentage split | "How is the footprint split across production, transportation and use?" | 50% / 5% / 45% |
| Waterfall chart labels | "What were the baseline emissions before reductions?" | 445 kg |
| Power table, by column | "What is the adapter efficiency at 115V?" | 91.5% |
| Endnote | "Which countries are exempt from the PVC-free claim?" | India, Thailand, South Korea |
| One-line endnote | "Why are renewable energy emissions not visible on the chart?" | too small to be visible |
| Body copy | "How much less energy does it use than ENERGY STAR?" | 58 percent |

`data_imac/questions.json` holds 34 questions — 12 table and chart lookups,
10 endnote questions, 9 from body copy, 3 abstention probes.

## Results across all three corpora

Retrieval, measured with `rag.recall`: the fact must appear in a returned
passage or its breadcrumb, and an abstention probe must return nothing.

| corpus | pages | chunks | retrieved |
|---|---|---|---|
| GPO annual report | 78 | 129 | 18/20 |
| GEP (World Bank) | ~200 | 596 | 25/26 |
| iMac environmental report | 12 | 37 | 32/34 |

The remaining misses are all the same shape: a question the document does not
cover, where retrieval returns passages and the generator, rather than the
floor, is what declines to answer.

## Known limits

* **The power table is recovered as rows, not as a grid.** "0.57W at 115V"
  works because the header row travels with the chunk and the values sit in
  order, but nothing in the pipeline knows that 0.57 is in the 115V *column*.
  A question needing a two-dimensional lookup is a coincidence away from
  being wrong.
* **Only the chart's labels are text.** The waterfall's bars are vector art.
  `445 kg` and `-99 kg` are recoverable because Apple sets them as type; a
  chart that rendered its numbers as paths would still be opaque.
* **The floor is set by hand per corpus.** `--min-dense` is a measurement
  someone has to make. Deriving it from the index — the spread of
  chunk-to-chunk similarity is the obvious candidate — is unfinished.
