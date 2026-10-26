"""Stage 4 — the index.

A deliberately small vector store: a NumPy matrix plus the chunk metadata,
saved as .npy + .json.

An annual report produces a few hundred to a few thousand chunks. Brute-force
cosine over that is sub-millisecond, so FAISS or Chroma would add a dependency,
a build step and a failure mode in exchange for nothing measurable. The search
interface below is the one an ANN index would expose, so swapping one in later
is a local change if the corpus ever grows by orders of magnitude.

Alongside the dense vectors it keeps a BM25 lexical index. Annual reports are
full of exact tokens — figures, programme names, acronyms — that dense
retrieval alone handles poorly, and fusing the two is the single biggest
quality win available at this scale.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chunk import Chunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9.\-%/]*")
# Words too common to discriminate between chunks of the same report.
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at", "by",
    "is", "are", "was", "were", "be", "been", "it", "its", "as", "that", "this",
    "with", "from", "we", "our", "their", "they", "has", "have", "had", "will",
    "what", "which", "who", "how", "when", "where", "does", "do", "did", "can",
}


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense: float = 0.0
    lexical: float = 0.0


class BM25:
    """Okapi BM25 over the chunk texts."""

    def __init__(self, docs: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        self.lens = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg_len = float(self.lens.mean()) if self.n else 0.0
        self.tf: List[Counter] = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        # Standard BM25 idf, floored so a term in every document cannot go negative.
        self.idf = {
            t: max(0.0, math.log((self.n - c + 0.5) / (c + 0.5) + 1.0))
            for t, c in df.items()
        }

    def scores(self, query: List[str]) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return out
        for term in query:
            idf = self.idf.get(term)
            if not idf:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(term)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: List[Chunk],
                 min_dense: Optional[float] = None):
        self.vectors = vectors
        self.chunks = chunks
        # The relevance floor is a property of the corpus, not of the code.
        # It is carried with the index so documents of different breadth can
        # be served side by side; see rag/retrieve.py for what it decides.
        self.min_dense = min_dense
        # The breadcrumb is indexed alongside the body for keyword search as
        # well as for the embedding, so "Box 1.1" or "Sub-Saharan Africa"
        # matches the passages filed under it even when the passage itself
        # never spells the name out.
        self.bm25 = BM25([
            tokenize(" ".join((c.text, c.path or c.heading, c.container)))
            for c in chunks
        ])

    # ---------- persistence ----------
    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        (folder / "chunks.json").write_text(
            json.dumps([c.to_dict() for c in self.chunks], ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        if self.min_dense is not None:
            (folder / "meta.json").write_text(
                json.dumps({"min_dense": self.min_dense}), encoding="utf-8")

    @classmethod
    def load(cls, folder: str | Path) -> "VectorStore":
        folder = Path(folder)
        vectors = np.load(folder / "vectors.npy")
        raw = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
        meta_file = folder / "meta.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        return cls(vectors, [Chunk(**c) for c in raw], meta.get("min_dense"))

    @staticmethod
    def exists(folder: str | Path) -> bool:
        folder = Path(folder)
        return (folder / "vectors.npy").exists() and (folder / "chunks.json").exists()

    # ---------- search ----------
    def search(
        self,
        query_vector: np.ndarray,
        query_text: str,
        k: int = 6,
        alpha: float = 0.65,
    ) -> List[Hit]:
        """Hybrid search: dense cosine fused with BM25.

        `alpha` weights dense against lexical. Both score distributions are
        min-max normalised first, because a raw BM25 score and a cosine
        similarity are not on comparable scales.
        """
        if not len(self.chunks):
            return []

        dense = self.vectors @ query_vector              # cosine: vectors are normalised
        lex = self.bm25.scores(tokenize(query_text))

        fused = alpha * _minmax(dense) + (1 - alpha) * _minmax(lex)

        top = np.argsort(-fused)[:k]
        return [
            Hit(self.chunks[i], float(fused[i]), float(dense[i]), float(lex[i]))
            for i in top
        ]


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)
