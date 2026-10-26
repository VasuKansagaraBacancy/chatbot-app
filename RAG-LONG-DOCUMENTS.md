# RAG for long, structured documents

The baseline pipeline (see [RAG.md](RAG.md)) was built against a 78-page annual
report: one column, few tables, a handful of headings. This document covers the
work done to handle the World Bank *Global Economic Prospects, June 2026*
report — 200 pages, two columns, four chapters, numbered boxes that run across
page breaks, 50-odd figures, 40-odd tables, per-chapter reference lists and a
statistical appendix.

Running the baseline pipeline over that document does not fail. It produces an
index, and the index returns passages. That is what makes the failure modes
worth writing down: every one of them degrades answers silently.

```
                   pypdf
                     |
     rag/layout.py   |   glyph coordinates -> lines -> columns -> reading order
                     |   running header -> printed page number + section
                     v
     rag/textfix.py      broken ligatures, split words
                     |
                     v
   rag/structure.py      typography -> roles -> section hierarchy
                     |
                     v
rag/chunk_sections.py    one section per chunk, breadcrumb carried into the text
                     |
                     v
      rag/embed.py       BGE-small, breadcrumb + body
                     |
                     v
      rag/store.py       dense cosine + BM25, both over breadcrumb + body
                     |
                     v
   rag/retrieve.py       structure boost, reference penalty, diversity,
                     |   absolute relevance floor, token-budgeted context
                     v
     rag/answer.py       grounded generation with [n] citations
```

## What went wrong with flat ingestion, and why it was invisible

**Columns interleaved.** `pypdf.extract_text()` returns text in content-stream
order. On a two-column page that order alternates between the columns, so the
first half of a left-column sentence lands in the middle of a right-column
paragraph. The text still looks like prose at a glance. Chunks built from it
are nonsense, and their embeddings point nowhere in particular.

*Fix:* `rag/layout.py` rebuilds reading order from glyph coordinates. Fragments
are clustered into lines by baseline, the page is classified as one-column or
two, and two-column pages are read left column first.

Deciding which is which turned out to be the hard part. Counting lines that
stay on one side of the gutter proves nothing, because both columns share a
baseline grid and almost every baseline carries a line from each. The signal
that does work is alignment: a real column has a left edge, so nearly all of
its lines start at exactly the same x. A wide forecast table starts cells at as
many x positions as it has columns. Splitting such a table at the gutter would
strand the country names in one half of the index and the forecast years in the
other.

**Printed page numbers differ from PDF page numbers.** The front matter is
numbered in roman numerals, so PDF page 111 is printed page 89 — an offset of
22. A citation of "p.111" sends a reader 22 pages past the sentence they were
shown. The PDF's own page labels are no help; it declares them all as plain
arabic. The running header carries the real number, along with the chapter
name, which is how both are recovered.

**Broken font encodings.** The report embeds subsetted Type1 fonts with
incorrect ToUnicode maps, and each chapter was typeset with a different subset.
"The" extracts as `7'_he`, `—ó<he` or `Fﬂähe`, about 315 times. Worse and much
more common, `fi`/`fl`/`ff` stay as single ligature code points: 936 instances
of `ﬁ`. A BM25 index built over `ﬁscal` contains no token `fiscal`, so a user
typing "fiscal" matches nothing by keyword, and the embedder tokenizes a glyph
it has never seen. Normalizing ligatures was the single highest-value text
repair in the pipeline.

**Words split by the extractor.** pypdf infers spaces from kerning adjustments.
In this report's justified serif columns those adjustments grow large enough
mid-word to read as word breaks, so "potential" arrives as "poten tial" and
"these" as "th ese". No spelling dictionary is available offline, so the
document supplies its own: the same words occur correctly elsewhere, hundreds
of times. A pair is rejoined only when the whole is well attested and both
halves are much rarer than the whole, which leaves ordinary pairs like "in to"
alone, since "in" and "to" are each far commoner than "into".

## Recovering the hierarchy

Headings are found from typography, not from text patterns. The report's style
sheet is consistent and easy to read off:

| Role | Style | Count |
|---|---|---|
| Section heading | HelveticaLTStd-Bold 16pt | 56 |
| Sub-heading | HelveticaLTStd-Bold 14pt | 4 |
| Table / box / figure banner | HelveticaLTStd-Bold 10pt | 104 |
| Body copy | AGaramondPro-Regular 11pt | 9,402 |
| Table matter | HelveticaLTStd-Roman 7.5pt | 1,502 |
| Figure description | HelveticaLTStd-Obl 8pt | 423 |
| Source and note lines | HelveticaLTStd-Roman 6pt | 739 |
| Chart axis labels | plain Helvetica ≤6pt | ~850, discarded |

Discarding the chart axis labels matters: they extract as runs of bare numbers
(`-0.4 -0.2 0.0 0.2 0.4 2026 2027`) that pad chunks and skew their embeddings
while carrying nothing retrievable.

Guessing headings from capitalization — which is what the flat chunker does —
fails badly here. Headings wrap across two lines in a narrow column
("Major economies: Recent" / "developments and outlook") and have to be
rejoined, while table rows like `World 2.8 2.9 2.9 2.5` look like numbered
headings to a text heuristic.

The result is a tree. Chapter names come from the running header combined with
the PDF outline, so a breadcrumb reads in full:

```
Chapter 3 - A Rising Challenge: Sovereign Debt Levels and Interest Rates in
EMDEs > Key trends in EMDE government debt and interest rates
```

Two container rules were needed:

* **Boxes and annexes contain running prose** and re-announce themselves as
  "(continued)" on later pages. A continuation re-opens the box it names rather
  than starting a new one, so all four parts of Box 1.1 share one breadcrumb.
* **Figures and tables do not.** A figure container is its caption, its panel
  labels and its source note. When body copy resumes it belongs to the
  enclosing section. Letting figures absorb the prose that follows them was the
  single worst structural bug found: Figure 1.2's "section" had swallowed 9,166
  characters of chapter text, which was then cited as a figure caption.

## Chunking by section

`rag/chunk_sections.py` replaces page-at-a-time packing:

* **A chunk never spans two sections**, so it is about one thing.
* **The breadcrumb is prepended to the text before embedding**, and indexed by
  BM25 as well. This is the cheapest large win available on a long document: a
  passage reading "the share fell sharply after the pandemic" carries its
  chapter and section into the vector space, so a question naming either one
  matches it. The chunker reserves the breadcrumb's tokens inside BGE's 512
  limit rather than overflowing it.
* **Tables are split by rows with their header repeated.** Without that, a
  piece reads `Angola 2.4 3.1 2.8 ...` with no way to tell which years those
  are, and no way for a question about 2026 to match it.
* **Paragraphs are rebuilt from typeset lines** before packing, since each
  element is one line of a justified column.

596 chunks, median 282 tokens, none over the embedder's limit.

## Retrieval

| Change | Why |
|---|---|
| `FETCH_K` 12 → 30 | At 596 chunks a 12-candidate pool fills with near-duplicates from one section before a better passage elsewhere is reached. |
| Structure boost (+0.30) | Readers navigate a long report by name: "what does table 1.1 show for China?". Matching the recovered `container` metadata is far more reliable than hoping the embedding picked the right one of forty forecast tables. |
| Chapter boost (+0.15) | Same, for "what does chapter 3 conclude about...". |
| Reference penalty (−0.35) | A bibliography answers almost nothing but matches strongly on any question sharing an author's subject matter. Kept in the index so "which studies does chapter 3 cite?" still works, and pushed down otherwise. |
| Max 2 chunks per section | Overlapping chunks from one section crowd out the rest of the report, which is how a question several chapters speak to gets answered from only one. |
| Absolute relevance floor | See below. |

### The relevance floor was broken

The fused score is min-max normalized across the candidate set, so the best
candidate scores near 1.0 **whatever the question**. "What is the capital of
France?" scored 0.752 against the old floor of 0.32 and returned six confident
passages about commodity exporters.

The raw cosine against the embedding is absolute and does separate the cases.
Measured over this index:

| | top dense score |
|---|---|
| Questions the report answers | 0.785, 0.818, 0.917, 0.804 |
| Questions it does not | 0.487, 0.445, 0.598, 0.597 |

`MIN_DENSE = 0.66` sits in the gap with margin on both sides. Retrieval now
returns an empty list for an uncovered question, and the generator is asked to
say so.

## Context management

The context window is 4,096 tokens and has to hold the system prompt, a few
turns of history, the retrieved passages and the answer. The old budget was
6,000 *characters*, which is the wrong unit: a chunk of table rows is a third
the characters of a chunk of prose for the same number of tokens, so a
character budget either overflows the window or wastes most of it.

`build_context` is now budgeted in tokens, derived in `config.py` from the
window rather than guessed:

```
DOC_CONTEXT_TOKENS = N_CTX - MAX_TOKENS - 700
```

Passages are added strongest first and the budget is a hard stop, so a long
table cannot push the rest of the evidence out of the prompt.

## Running it

```bat
.venv\Scripts\python.exe -m rag.build_index --data data_gep --index index_gep
.venv\Scripts\python.exe -m rag.ask --index index_gep "What is global growth projected to be in 2026?"
.venv\Scripts\python.exe -m rag.evaluate --questions data_gep\questions.json --index index_gep
```

`build_index` chooses its strategy automatically: it uses the section
hierarchy when one can be recovered and falls back to flat chunking when it
cannot, so the GPO report still indexes exactly as before. Force either with
`--mode structured` or `--mode flat`.

The app reads `DATA_DIR` and `INDEX_DIR` from `config.py`, which now default to
the GEP pair. To switch back to the GPO report:

```bat
set DATA_DIR=data
set INDEX_DIR=index
```

`docmode.get_store()` caches the index per process, so **the server must be
restarted after re-indexing** for the change to appear in the UI.

## Known limits

* **Chart data is not extractable.** The numbers behind a figure live in the
  vector art, not in the text layer. Only the caption, the panel titles and the
  descriptive summary are indexed, so "what value does figure 3.2.A show for
  2015?" cannot be answered. The figure's description usually can be.
* **Table cells lose their grid.** A row is recovered as a line of values in
  the right order, which supports lookups along a row, but the pipeline has no
  model of which column a value sits in. Questions naming a specific year
  usually work because the header row travels with every piece; questions
  requiring a two-dimensional lookup are unreliable.
* **Sub-headings are sparse.** The report only uses the 14pt tier four times,
  in Chapter 2. Deeper structure within a long chapter section is not
  available, so the longest sections produce several chunks sharing one
  breadcrumb.
* **Thresholds in points did not transfer.** The style-sheet table above was
  measured on this report and hard-coded. A third corpus (Apple's iMac
  environmental report, 8.5pt body) found nothing at all, which is what led
  to per-document typographic calibration -- see
  [RAG-MIXED-CONTENT.md](RAG-MIXED-CONTENT.md).
* **The two reports must not share an index.** Retrieval across both pulls the
  wrong document into roughly half the answers, which is why the corpus folders
  are kept separate rather than merged.
