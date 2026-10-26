# How it works — the three tasks, the logic, and worked examples

This is the plain-language walkthrough. For the engineering detail behind each
decision see [RAG.md](RAG.md), [RAG-LONG-DOCUMENTS.md](RAG-LONG-DOCUMENTS.md)
and [RAG-MIXED-CONTENT.md](RAG-MIXED-CONTENT.md).

## The idea in one paragraph

The language model has never seen your PDF. So the PDF is cut into pieces, the
pieces that match the question are found, and only those are shown to the model,
which answers from them and cites them. Everything hard about this is in the
cutting and the finding: **if the pieces are assembled wrongly, no amount of
model quality rescues the answer.**

```
PDF ──> read layout ──> find structure ──> cut into chunks ──> embed
                                                                 │
question ──> embed ──> search (meaning + words) ──> rank ──> filter
                                                                 │
                                        passages ──> model ──> answer + [1][2]
```

## What each file does

| File | Stage | Responsibility |
|---|---|---|
| [rag/layout.py](rag/layout.py) | Read | Glyph coordinates to lines, columns, reading order, printed page numbers |
| [rag/textfix.py](rag/textfix.py) | Read | Broken ligatures, words split by the extractor |
| [rag/structure.py](rag/structure.py) | Structure | Typography to roles (heading, body, table, note) to a section tree |
| [rag/chunk_sections.py](rag/chunk_sections.py) | Cut | One section at a time, breadcrumb carried into the text |
| [rag/chunk.py](rag/chunk.py) | Cut | Flat fallback for documents with no recoverable hierarchy |
| [rag/embed.py](rag/embed.py) | Embed | BGE-small-en-v1.5, 384 dimensions, local |
| [rag/store.py](rag/store.py) | Search | Dense cosine fused with BM25 keyword search |
| [rag/retrieve.py](rag/retrieve.py) | Rank | Structure boosts, diversity, the relevance floor, context budget |
| [rag/answer.py](rag/answer.py) | Generate | The grounding prompt and streaming |
| [rag/recall.py](rag/recall.py) | Check | Retrieval-only validation, seconds per question set |
| [rag/evaluate.py](rag/evaluate.py) | Check | Full generation, minutes per question set |

---

# Task 1 — GPO Annual Report

**The document:** 78 pages of ordinary narrative prose, with the facts written
into the sentences. The closest of the three to a normal business document.

**The logic:** when the text is ordinary, there is only one serious risk left —
the system answering questions the document never addresses. A search always
returns *something*, because it returns the best matches it has, not the good
ones. Confident invention is worse than silence.

### Worked example 1 — searching two ways

> **"What was GPO's total revenue in fiscal year 2018?"**

Searching only by *meaning* returns passages that read as financial — budgets,
appropriations, expenses — most of them without the number. Searching only by
*exact words* matches "fiscal year 2018" on dozens of pages.

The two are fused (65% meaning, 35% keyword), so the chunk that wins has to
score on both:

> "FY 2018 Financial Results **Revenue totaled $874.5 million** while expenses
> charged against GPO's budget were $821.9 million, for an overall net income
> of $52.6 million from operations."

**Answer: $874.5 million [2]**

### Worked example 2 — declining to answer

> **"What is GPO's policy on employee remote working from overseas?"**

The report says nothing about this. The search still returns six passages about
staff, pensions and benefits, and an unguarded system writes a plausible
paragraph from them.

The fix is to look at the *raw* similarity rather than the ranked score. The
ranked score is normalised, so the best match is near 1.0 for any question at
all; the raw cosine is absolute:

| Question | Raw score |
|---|---|
| GPO's revenue (in the report) | 0.80 |
| Who won the 2022 World Cup (not in it) | 0.51 |
| Capital of France (not in it) | 0.47 |

Below the floor, retrieval returns an empty list and the model is told to say so.

**Answer: "The report does not appear to cover this."**

### Worked example 3 — a defect this found

> **"What new digital system replaced FDsys, and when was it fully deployed?"**

This is one of the brief's own sample questions, and it returned **nothing** —
even though thirteen chunks mention FDsys. It scored 0.656 against a floor of
0.66. Meanwhile questions GPO genuinely cannot answer scored 0.665 and 0.725.

The floor had been measured on a different document. It is now stored per index
instead of being a constant in the code, and GPO uses 0.60.

**Answer: govinfo, fully deployed December 2018 [2][3]**

---

# Task 2 — World Bank, Global Economic Prospects

**The document:** ~200 pages, two columns, four chapters, numbered boxes that
run across page breaks, 40-odd tables, per-chapter reference lists.

**The logic:** at this length the problem is no longer finding *a* relevant
passage — it is choosing the right one among many that look alike. Six regions
each write "growth is expected to soften to X percent in 2026". What
distinguishes them is not in the paragraph; **it is where the paragraph sits in
the document.** So the structure has to be recovered, and then put somewhere the
search can actually see it.

### Worked example 1 — the columns

Raw extraction returns text in file order, which on a two-column page alternates
between the columns:

```
"Growth in South Asia is Inflation remained elevated expected to soften to
 across most economies in"
```

It still looks like prose at a glance, which is what makes it dangerous. Chunks
built from this are meaningless and their embeddings point nowhere.

`layout.py` rebuilds the page from each glyph's x/y position, clusters
fragments into lines, decides whether the page is one column or two, and reads
the left column first:

```
"Growth in SAR is expected to soften to 6.3 percent in 2026, mainly reflecting
 the adverse impact of the conflict in the Middle East..."
```

### Worked example 2 — the breadcrumb

> **"What is the growth outlook for South Asia in this report?"**

Every regional section uses nearly the same sentence, so the embedding cannot
separate them. The fix is to attach the location to the chunk **before** it is
embedded and indexed:

```
Chapter 2 - Regional Outlooks > South Asia
Growth in SAR is expected to soften to 6.3 percent in 2026...
```

"South Asia" is now literally part of the text being searched, so a question
naming the region matches it even though the paragraph says "SAR".

**Answer: 6.3 percent in 2026, picking up to around 7 percent in 2027–28 [2]**

### Worked example 3 — asking by name

> **"What does Box 3.1 examine?"**

The box never contains the phrase "Box 3.1 examine" — it just discusses hidden
debt. By meaning alone it scores *below* the relevance floor, so the system
would announce that the report does not cover a box the report prints a title
for.

Naming a block is exact evidence, not a similarity judgement, so a named match
bypasses the floor entirely.

**Answer: the costs of hidden debt, cited as**
`Chapter 3 - A Rising Challenge... > Box 3.1: The costs of hidden debt`

### Worked example 4 — a defect this found

Figures and boxes had been treated the same way. But a box *contains* running
prose and re-announces itself as "(continued)" on later pages, whereas a figure
is only its caption, panel labels and source note — when body copy resumes, it
belongs to the enclosing section.

Letting figures absorb what followed them meant Figure 1.2's "section" had
swallowed **9,166 characters** of chapter text, all of it then cited to readers
as a figure caption.

---

# Task 3 — Apple iMac Product Environmental Report

**The document:** 12 pages. The shortest of the three and by far the hardest to
ingest, because almost nothing in it is a paragraph.

**The logic:** the facts live in a table, in the labels of a chart, and in
nineteen numbered endnotes that qualify every headline claim. A pipeline that
reads paragraphs collects perhaps a third of this document. And because the
task-2 code had hard-coded the World Bank's type sizes, this document produced
*"no usable hierarchy found"* and silently fell back to flat page chunking.

### Worked example 1 — font sizes do not transfer

| | Body text | Section headings |
|---|---|---|
| World Bank | 11 pt | 16 pt |
| **Apple** | **8.5 pt** | **24 pt** |

Every Apple heading sat *above* the threshold that meant "this is cover
typography, ignore it", so **no headings were found at all**.

The fix is to stop hard-coding. Each document's most common body size is
measured first, and every role is expressed as a ratio of it. Two guards were
needed:

* **A cover title is not a heading tier.** The GEP cover sets its title at 70 pt
  and uses it once. A tier must appear on at least three pages, and the heading
  tier is chosen by how widely it is spread across pages, not by size.
* **A second body size is not a heading tier.** The GPO report sets 969 lines in
  12 pt against a 9.8 pt body. A tier holding more than 15% as many lines as the
  body is body copy, whatever its size.

### Worked example 2 — the chart

The emissions chart is vector art, but its labels are real text: `445 kg`,
`-99 kg`, `346 kg`. Because they are set large, they were classified as
**headings** — so "445 kg" became a section title, and the sentence explaining
it was filed under a nameless parent.

A display-size line that is nothing but figures and units is data, not a
heading.

> **"What were the baseline emissions before reductions, and the footprint now?"**
>
> **Answer: 445 kg CO₂e, reduced to 346 kg CO₂e [1][2]**

### Worked example 3 — the table

> **"What is the total carbon footprint, and how much comes from product use
> versus manufacturing?"**

This answer appears in no sentence anywhere in the document. It is a table:

```
iMac (two ports) 256GB
Total product footprint        346 kg CO2e
  • Production         50%
  • Transportation      5%
  • Product use        45%
  • End-of-life        <1%
```

**Answer: 346 kg CO₂e; product use 45%, production 50% [1]**

### Worked example 4 — the footnotes

All nineteen endnotes were one block, so a question about note 6 returned a
chunk mostly about notes 4, 5, 7 and 8. Worse, three notes were **deleted
outright** for being shorter than the minimum chunk size:

> "Renewable energy emissions are too small to be visible on the chart."

Eleven words, and the only place the report explains a gap in its own chart.

Two fixes: each numbered note becomes its own retrievable section, and a section
that is short *but complete* is kept rather than discarded.

Splitting has to be careful, because a note's own text is full of numbers:

```
"1 Recycled content is the mass of certified material..."   -> Note 1
"15 percent of the enclosure is recycled aluminium..."      -> not note 15
```

A number only starts a new note if it is the next one in sequence, does not jump
by more than three, and is not followed by a unit such as "percent".

### Worked example 5 — the floor belongs to the corpus

On a single-product brochure every chunk shares the product's vocabulary, so no
threshold separates covered from uncovered questions:

| Question | Raw score |
|---|---|
| "When was this iMac introduced?" (covered) | 0.649 |
| "What processor does the iMac use?" (not covered) | 0.697 |
| "What is the capital of France?" (not covered) | 0.407 |

No floor can admit the first and reject the second. So the floor stopped being a
constant in the code and became a property of the index, written to `meta.json`
at build time. Out-of-domain questions are still refused; questions about the
product that the report does not answer are left to the model, which does
decline them.

---

# Results

The exercise brief supplies three sample questions per document as the stated
pass/fail bar. All nine are answered correctly, each with a citation.

| Task | Sample questions | Full question set |
|---|---|---|
| 1 — GPO | 3/3 | 23/23 |
| 2 — GEP | 3/3 | 28/29 |
| 3 — iMac | 3/3 | 35/37 |

The three not retrieved are abstention probes, where the model declines to
answer rather than the retriever filtering first — the intended behaviour,
scored strictly.

```
.venv\Scripts\python.exe -m pytest tests -q                              # 46 tests, ~1s
.venv\Scripts\python.exe -m rag.recall index_imac data_imac\questions.json
```

# The thread running through all three

Every fix made for one document had to leave the other two working. The
per-document type calibration built for Apple is what flipped the GPO report
from flat page chunks to a proper section hierarchy. The per-index relevance
floor built for Apple is what fixed GPO's FDsys question.

That is why the three corpora stay indexed separately and are re-checked after
every change. **The regression check is the design.**
