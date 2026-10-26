# RAG pipeline

Retrieval-augmented generation over a PDF, running entirely offline on the same
llama.cpp runtime as the chat model. No PyTorch, no vector database, no API keys.

## Quick start

```
1. Put the PDF in              data\
2. .venv\Scripts\python.exe -m rag.build_index
3. .venv\Scripts\python.exe -m rag.ask "What was total revenue?"
```

Interactive mode (keeps conversation context, so follow-ups work):

```
.venv\Scripts\python.exe -m rag.ask
```

Run a question set and get a scored report:

```
.venv\Scripts\python.exe -m rag.evaluate --questions data\questions.json
```

## The pipeline

```
data\*.pdf
   │
   ├─ 1. ingest.py    pypdf extraction → per-page text
   │                  de-hyphenation, margin-unwrapping, header/footer removal
   │
   ├─ 2. chunk.py     pack paragraphs to ~380 tokens, 64-token overlap
   │                  heading detection; page number carried through
   │
   ├─ 3. embed.py     BGE-small-en-v1.5 (384-dim) via llama.cpp
   │                  asymmetric: query prefix applied, passages not
   │
   ├─ 4. store.py     vectors.npy + chunks.json
   │                  dense cosine  ⊕  BM25 lexical, min-max fused
   │
   ├─ 5. retrieve.py  follow-up rewriting → hybrid search → relevance floor
   │
   └─ 6. answer.py    grounded prompt → Qwen3-4B → cited answer
```

## Design decisions

**Embeddings run on llama.cpp, not sentence-transformers.** BGE-small is a
33M-parameter, 384-dimension model with strong retrieval scores for its size.
Serving it through the runtime already in the project avoids a ~2.5 GB PyTorch
install and keeps the system offline. Cost: embedding is CPU-bound, but at a
few hundred chunks indexing takes seconds.

**No vector database.** An annual report yields hundreds to a few thousand
chunks. Brute-force cosine over a NumPy matrix of that size is sub-millisecond —
measured at 22 ms median including BM25 and query embedding. FAISS or Chroma
would add a dependency and a failure mode for no measurable gain. `VectorStore.search`
exposes the interface an ANN index would, so swapping one in later is local.

**Hybrid retrieval, not dense alone.** Annual reports are dense with exact
tokens — figures, programme names, acronyms, section titles — which dense
embeddings handle poorly and BM25 handles well. Scores are min-max normalised
before fusing, because a raw BM25 score and a cosine similarity are not on
comparable scales. Weighting is `alpha=0.65` toward dense.

**Chunks are measured with the embedder's tokenizer.** BGE truncates at 512
tokens *silently* — an oversized chunk still returns a vector, computed from
only its first half, while the full text is handed to the generator. Sizing
chunks with a character heuristic would let this happen invisibly, so
`count_tokens` comes from the embedding model itself, and `build_index` warns
if any chunk exceeds the limit.

**A relevance floor, so the system can say no.** Returning the top 6 chunks
regardless of score is what makes RAG systems answer confidently about things
the document never mentions. Hits below `MIN_SCORE = 0.32` are dropped, and the
prompt instructs the model to state plainly that the report does not cover the
question.

**Follow-ups are rewritten before retrieval.** "Why did it fall?" embeds to
nothing useful. When a question looks elliptical, it is first resolved against
recent turns into a standalone query, then retrieved on. This is why the chat
behaviour survives the move to RAG.

## Validation

The pipeline is validated against a synthetic annual report
(`tests/fixtures/sample_annual_report.pdf`) whose figures are known, so
retrieval regressions surface as assertion failures rather than as vague
quality complaints.

```
.venv\Scripts\python.exe tests\make_fixture.py
.venv\Scripts\python.exe -m rag.build_index --data tests\fixtures --index tests\fixture_index
.venv\Scripts\python.exe -m rag.evaluate --questions tests\fixture_questions.json --index tests\fixture_index
```

Results as committed:

| Suite | Checks | Result |
|---|---|---|
| Factual retrieval (`fixture_questions.json`) | 6 expected-value assertions | 6/6 passed |
| Abstention (question not covered by the document) | 1 | correctly declined |
| Conversational follow-ups (`fixture_followups.json`) | 4 expected-value assertions | 4/4 passed |

Median retrieval latency 22 ms. Generation is the slow part, governed by the
chat model's ~5 tok/s on this CPU.

The abstention case matters most: asked about a topic absent from the document,
the system answered *"The report does not appear to cover this"* and named what
the retrieved passages did discuss, instead of inventing a policy.

## Evaluation report format

`rag.evaluate` writes a JSON row per question:

| Field | Meaning |
|---|---|
| `cited` | the answer referenced a passage by number |
| `abstained` | the answer declined, saying the document does not cover it |
| `top_score` | fused relevance of the best retrieved chunk |
| `pages` | which document pages the context came from |
| `expect_pass` | every required substring appeared in the answer |
| `retrieval_ms`, `generation_s` | timings |

`cited` and `abstained` are tracked separately because a correct refusal is a
success, not a miss.

## Tuning

| Setting | Where | Default | Effect |
|---|---|---|---|
| `TARGET_TOKENS` | `rag/chunk.py` | 380 | smaller = more precise retrieval, less context per hit |
| `OVERLAP_TOKENS` | `rag/chunk.py` | 64 | guards against facts split across a boundary |
| `MIN_SCORE` | `rag/retrieve.py` | 0.32 | raise if it answers things the report doesn't cover |
| `alpha` | `store.search` | 0.65 | lower to favour exact-term (BM25) matching |
| `KEEP_K` | `rag/retrieve.py` | 6 | passages passed to the model |

## Known limitations

- **Scanned PDFs are not handled.** `pypdf` extracts embedded text only. A
  scanned report needs OCR first; `build_index` reports this rather than
  producing an empty index.
- **Tables lose their structure.** Extraction flattens a table to text, so a
  figure may lose its column header. Numbers stated in prose retrieve reliably;
  numbers that only make sense within a grid are weaker.
- **The generator can do arithmetic the document does not state.** Asked why
  headcount fell, it correctly reported attrition but also computed the
  difference (61) itself. The figure was right; it is still derived rather than
  quoted.
- **One generation at a time.** Inherited from the chat model: llama.cpp keeps a
  single KV cache per instance, so requests serialise.
