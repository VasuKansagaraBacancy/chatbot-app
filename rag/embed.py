"""Stage 3 — embeddings.

Wraps BGE-small-en-v1.5 running under llama.cpp, the same runtime that already
serves the chat model, so the whole system stays offline with no extra
dependencies (no PyTorch, no sentence-transformers).

Two details matter for retrieval quality:

* BGE is an asymmetric model. Queries must carry the instruction prefix it was
  trained with; passages must not. Embedding both the same way measurably
  degrades recall, and it is a silent failure, so it is handled here rather
  than left to callers.
* Vectors are L2-normalised on the way out, which makes cosine similarity a
  plain dot product at search time.
"""
from __future__ import annotations

import threading
from typing import Iterable, List

import numpy as np
from llama_cpp import Llama

import config

# The prefix BGE-v1.5 was trained with for the retrieval task.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model: Llama | None = None
_lock = threading.Lock()


def get_embedder() -> Llama:
    """Load the embedding model once per process."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = Llama(
                    model_path=config.EMBED_MODEL_PATH,
                    embedding=True,
                    n_ctx=config.EMBED_N_CTX,
                    n_threads=config.N_THREADS,
                    n_gpu_layers=0,
                    verbose=False,
                )
    return _model


def count_tokens(text: str) -> int:
    """Token length as the embedder sees it — used to size chunks."""
    return len(get_embedder().tokenize(text.encode("utf-8"), add_bos=False))


def _as_vector(raw) -> np.ndarray:
    """Normalise llama.cpp's embedding shape.

    Depending on the pooling configuration it returns either a flat vector or a
    per-token matrix; mean-pool the latter so both cases yield one vector.
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


def embed_passages(texts: Iterable[str], progress: bool = False) -> np.ndarray:
    """Embed document chunks. Returns an (n, dim) float32 matrix."""
    model = get_embedder()
    texts = list(texts)
    out: List[np.ndarray] = []
    for i, t in enumerate(texts, 1):
        res = model.create_embedding(t)
        out.append(_as_vector(res["data"][0]["embedding"]))
        if progress and (i % 25 == 0 or i == len(texts)):
            print(f"  embedded {i}/{len(texts)} chunks", flush=True)
    return np.vstack(out) if out else np.zeros((0, 0), dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a user question, with the instruction prefix BGE expects."""
    res = get_embedder().create_embedding(QUERY_PREFIX + text)
    return _as_vector(res["data"][0]["embedding"])


def embed_text(chunk) -> str:
    """The text actually embedded for a chunk: its breadcrumb, then its body.

    Prepending the hierarchy is what lets a passage be retrieved by a question
    that names its chapter, its region or its box number — words the passage
    itself never repeats, because the page it sits on already said them. The
    chunker reserves room for this header inside the 512-token limit.
    """
    header = chunk.context_header() if hasattr(chunk, "context_header") else ""
    return (header + "\n" + chunk.text) if header else chunk.text
