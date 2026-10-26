# Local Chatbot

A ChatGPT-style chat application running entirely on your machine, with no API keys
and no data leaving the computer. Built on an open-source LLM with session-based
conversation history and context-aware follow-up handling.

## Quick start

```
run.bat
```

Then open <http://127.0.0.1:8000>.

Or manually:

```
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

The model loads at startup (~5–10 s), so the first message isn't slowed by it.

## The model

| | |
|---|---|
| Model | Qwen3-4B-Instruct-2507, Apache-2.0 |
| Quantisation | Q4_K_M (GGUF) |
| Runtime | `llama-cpp-python` 0.3.35, CPU build |
| Disk | ~2.5 GB, stored at `D:\llm-models\` |
| RAM at runtime | ~3 GB |
| Context window | 4096 tokens |
| Observed speed | 6.0 tok/s generating, 21 tok/s reading the prompt (i5-10210U, 4 cores) |

**Why this model.** The machine has 24 GB of RAM but a 2 GB MX230, too small to be
worth offloading layers to, so inference is CPU-only and the model has to be small
enough to stay responsive. Weights live on `D:` because `C:` only had ~5 GB free.

This started on Qwen2.5-3B, which had a real recall problem: asked "What is RAG?"
it invented a different expansion every time ("Reusable", "Relevance-Augmented",
"Reformer-Augmenter-Guided") instead of *Retrieval-Augmented Generation*, and
lowering the temperature did not help. Qwen3-4B-Instruct-2507 answers it correctly
every time and, measured on this CPU, runs at the same ~6.0 tok/s despite being
larger — so the upgrade cost nothing in speed.

**Switching models.** Any GGUF chat model works — point `MODEL_PATH` at it:

```
set MODEL_PATH=D:\llm-models\qwen2.5-1.5b-instruct-q4_k_m.gguf
run.bat
```

The Qwen2.5 3B and 1.5B files are still on disk. The 1.5B is roughly twice as fast
at noticeably lower answer quality. A 7B model would be a little more accurate but
drops to roughly 3 tok/s on this CPU, which is slow to read along with.

## How context works

Every session keeps its own message list. On each turn the server sends the model
a system prompt plus the **entire conversation so far**, which is what lets it
resolve follow-up questions:

```
User: What is RAG?
User: What are its main components?          <- "its" resolves to RAG
User: Can you explain the second one?        <- "the second one" resolves to the list above
```

When history outgrows the context window, `llm.build_messages()` trims from the
**oldest** end, so the recent turns that follow-ups depend on are always preserved.
The trim keeps a token budget of `N_CTX - MAX_TOKENS - 256` and never opens the
prompt on an assistant turn.

The system prompt (in `config.py`) additionally tells the model to drop the old
thread when the user changes subject, and not to refuse harmless everyday requests.

## Sessions

- Each session has a UUID and its own history; a browser cookie remembers the last one.
- **New chat** creates an empty session — it inherits nothing from previous ones.
- The sidebar lists sessions, titled after their opening question, and lets you
  switch between or delete them.
- Storage is in-memory, so history clears when the server restarts. Sessions idle
  for more than `SESSION_TTL` (default 6 h) are evicted.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/session` | Current session (creates one on first visit) |
| `POST` | `/api/session` | Start a fresh session |
| `GET` | `/api/sessions` | List all sessions |
| `GET` | `/api/sessions/{id}` | One session's history |
| `POST` | `/api/sessions/{id}/clear` | Wipe history, keep the id |
| `DELETE` | `/api/sessions/{id}` | Delete a session |
| `POST` | `/api/chat` | Send a message, get an SSE token stream |
| `GET` | `/api/health` | Model status |

`/api/chat` streams server-sent events: `start`, then `token` per piece, then
`done` (or `error`).

## Document mode (RAG)

Besides open chat, the app answers questions about indexed PDFs and shows the
page and section every fact came from. Three documents are indexed, each one
chosen to break the pipeline in a different way:

| Document | Pages | Chunks | What it exercises | Retrieval |
|---|---|---|---|---|
| GPO Annual Report 2018 | 78 | 129 | The baseline: ingest, chunk, embed, retrieve, cite | 18/20 |
| World Bank *Global Economic Prospects* | ~200 | 596 | Two columns, chapters, boxes, 40+ tables, section hierarchy | 25/26 |
| Apple *iMac Product Environmental Report* | 12 | 37 | Facts that live in tables, chart labels and footnotes | 32/34 |

Pick the document from the sidebar in Document mode. Answers cite passages by
number, and the citation panel shows the printed page and the full breadcrumb
(`Chapter 3 - A Rising Challenge... > Box 3.1: The costs of hidden debt`).

Write-ups of the work each one needed:

- [RAG.md](RAG.md) - the baseline pipeline
- [RAG-LONG-DOCUMENTS.md](RAG-LONG-DOCUMENTS.md) - columns, hierarchy, the relevance floor
- [RAG-MIXED-CONTENT.md](RAG-MIXED-CONTENT.md) - typographic calibration, endnotes, chart labels

Recorded demos of each are in [demo/](demo/), two questions per document.

### Building and checking an index

```
.venv\Scripts\python.exe -m rag.build_index --data data_imac --index index_imac --min-dense 0.60
.venv\Scripts\python.exe -m rag.ask --index index_imac "What is the iMac's carbon footprint?"
.venv\Scripts\python.exe -m rag.recall index_imac data_imac\questions.json
.venv\Scripts\python.exe -m rag.evaluate --questions data_imac\questions.json --index index_imac
```

`rag.recall` checks retrieval only and runs in seconds; `rag.evaluate` generates
an answer for every question and takes a while on CPU. The server caches each
index per process, so **restart it after re-indexing**.

## Layout

```
config.py       model paths, generation settings, system prompt
llm.py          model loading, prompt building, history trimming, streaming
sessions.py     in-memory per-session conversation store
docmode.py      multi-corpus document mode, wired for the web app
app.py          FastAPI routes and SSE streaming
rag/            the retrieval pipeline (layout, structure, chunking, retrieval)
data*/          source PDFs and their question sets
index*/         built indexes (vectors + chunk metadata)
static/         chat UI (no build step, no framework)
demo/           recorded walkthroughs, one per document
test_context.py end-to-end check of follow-ups and session isolation
```

## Tests

With the server running:

```
.venv\Scripts\python.exe test_context.py
```

This exercises the follow-up chain, a mid-session topic change, and confirms a
fresh session cannot see an earlier one's history.

## Known limitations

- **Factual accuracy.** A 4-bit 4B model still has limited world knowledge and can
  state things confidently and wrongly. It handles common technical topics (RAG,
  HTTP, Python) correctly, but treat niche facts with suspicion. Context handling
  is sound; recall is the weaker part.
- **One generation at a time.** llama.cpp keeps a single KV cache per model
  instance, so concurrent requests are serialised behind a lock.
- **History is not persisted** across restarts.
