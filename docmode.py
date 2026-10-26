"""Document mode: the RAG pipeline, wired for the web app.

Kept separate from app.py so the server still starts, and plain chat still
works, when no index has been built yet. Everything here degrades to
`available() is False` rather than raising at import time.

Several indexes can be served at once, one per document. They are kept apart
rather than merged into a single index because retrieval across two unrelated
reports pulls the wrong document into a good share of the answers: an annual
report and a macroeconomic outlook share enough vocabulary — revenue, growth,
forecast, risk — to look alike to an embedding, but a question is almost
always about one of them. The caller picks which to search; the indexes are
loaded lazily and cached per process.
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import config

_stores: Dict[str, object] = {}
_lock = threading.Lock()


def _index_dirs() -> List[Path]:
    """Every folder in the project that holds a built index.

    Discovered rather than configured, so building a new index with
    `--index index_something` makes that document selectable without anyone
    having to edit a list.
    """
    from rag.store import VectorStore

    root = Path(__file__).resolve().parent
    found = []
    for path in sorted(root.glob("index*")):
        if path.is_dir() and VectorStore.exists(path):
            found.append(path)

    # Honour an explicit INDEX_DIR even if it sits outside the project.
    configured = Path(config.INDEX_DIR)
    if VectorStore.exists(configured):
        resolved = configured.resolve()
        if resolved not in {p.resolve() for p in found}:
            found.insert(0, configured)
    return found


def corpus_ids() -> List[str]:
    return [p.name for p in _index_dirs()]


def default_corpus() -> Optional[str]:
    """The corpus used when the caller does not name one."""
    ids = corpus_ids()
    if not ids:
        return None
    preferred = Path(config.INDEX_DIR).name
    return preferred if preferred in ids else ids[0]


def available() -> bool:
    """True when at least one index exists on disk and could be loaded."""
    try:
        return bool(_index_dirs())
    except Exception:
        return False


def get_store(corpus: Optional[str] = None):
    """Load an index once per process, keyed by its folder name."""
    corpus = corpus or default_corpus()
    if not corpus:
        raise FileNotFoundError("no index has been built")

    if corpus not in _stores:
        with _lock:
            if corpus not in _stores:
                from rag.store import VectorStore
                folder = next((p for p in _index_dirs() if p.name == corpus), None)
                if folder is None:
                    raise FileNotFoundError(f"unknown corpus {corpus!r}")
                _stores[corpus] = VectorStore.load(folder)
    return _stores[corpus]


def _describe(corpus: str) -> Dict:
    """Summarise one index for the document picker."""
    store = get_store(corpus)
    chunks = store.chunks
    sources = sorted({c.source for c in chunks})

    # Preserve document order rather than sorting: a reader recognises
    # "Chapter 1, Chapter 2, ..." and not an alphabetical shuffle of them.
    parts: List[str] = []
    for chunk in chunks:
        if chunk.part and chunk.part not in parts:
            parts.append(chunk.part)

    kinds: Dict[str, int] = {}
    for chunk in chunks:
        kinds[chunk.kind] = kinds.get(chunk.kind, 0) + 1

    return {
        "id": corpus,
        "label": ", ".join(Path(s).stem for s in sources) or corpus,
        "documents": sources,
        "chunks": len(chunks),
        "pages": max((c.page for c in chunks), default=0),
        "parts": parts,
        "kinds": kinds,
        "structured": any(c.path for c in chunks),
    }


def status() -> Dict:
    """Describe every indexed corpus for the UI."""
    if not available():
        return {
            "available": False,
            "detail": "No index found. Put a PDF in data\\ and run: "
                      "python -m rag.build_index",
        }

    corpora = []
    for corpus in corpus_ids():
        try:
            corpora.append(_describe(corpus))
        except Exception as exc:               # one bad index must not hide the rest
            corpora.append({"id": corpus, "label": corpus, "error": str(exc)})

    usable = [c for c in corpora if "error" not in c]
    if not usable:
        detail = corpora[0].get("error", "unknown error") if corpora else "none"
        return {"available": False, "detail": f"Index failed to load: {detail}"}

    current = default_corpus()
    if current not in {c["id"] for c in usable}:
        current = usable[0]["id"]

    # The top-level fields describe the default corpus, so a caller that
    # predates the picker keeps working unchanged.
    head = next(c for c in usable if c["id"] == current)
    return {
        "available": True,
        "corpora": corpora,
        "default": current,
        "chunks": head["chunks"],
        "documents": head["documents"],
        "pages": head["pages"],
        "parts": head["parts"],
        "kinds": head["kinds"],
        "structured": head["structured"],
    }


def answer_stream(
    question: str,
    history: List[Dict[str, str]],
    corpus: Optional[str] = None,
) -> Tuple[Iterator[str], List[Dict], str]:
    """Retrieve from one corpus, then stream a grounded answer.

    Returns the token iterator, the citation list and the standalone form of
    the question. Retrieval happens eagerly so the UI can show sources before
    the answer finishes streaming.
    """
    import llm
    from rag import answer as answer_mod
    from rag import retrieve as retrieve_mod

    store = get_store(corpus)

    # Resolve follow-ups against the conversation before retrieving.
    prior = [m for m in history[:-1] if m["role"] in ("user", "assistant")]
    standalone = retrieve_mod.rewrite_question(question, prior, llm.complete)

    hits = retrieve_mod.retrieve(store, standalone)
    # Report only the passages that fit the prompt. Listing a source the
    # model was never shown invites the reader to check a citation against
    # an answer that could not have been based on it.
    hits = retrieve_mod.fit_context(hits, config.DOC_CONTEXT_TOKENS)
    citations = [
        {
            "n": i,
            "source": h.chunk.source,
            "page": h.chunk.page,
            "label": h.chunk.label or str(h.chunk.page),
            "heading": h.chunk.heading,
            "path": h.chunk.path,
            "kind": h.chunk.kind,
            "score": round(h.score, 3),
            "snippet": h.chunk.text[:260],
        }
        for i, h in enumerate(hits, 1)
    ]
    stream = answer_mod.stream_answer(standalone, hits, prior)
    return stream, citations, standalone
