"""Build the retrieval index.

    .venv\\Scripts\\python.exe -m rag.build_index
    .venv\\Scripts\\python.exe -m rag.build_index --data data --index index

Runs the full ingest -> chunk -> embed -> store pipeline and prints enough
statistics to tell whether the document was parsed sensibly before anyone
starts debugging answer quality.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from rag import embed
from rag.embed import embed_text
from rag.chunk import Chunk, chunk_pages
from rag.chunk_sections import chunk_sections
from rag.ingest import load_documents
from rag.store import VectorStore
from rag.structure import build_sections, read_elements


def _structured_chunks(data_dir: Path, target: int, overlap: int):
    """Chunk every PDF in the folder through the section hierarchy.

    Returns None when the documents have no recoverable structure, so the
    caller can fall back to flat chunking rather than produce a worse index.
    """
    chunks = []
    sections_total = 0
    for pdf in sorted(data_dir.glob("*.pdf")):
        elements = read_elements(pdf)
        if not elements:
            continue
        sections = build_sections(elements)
        headed = sum(1 for s in sections if s.heading or s.container)
        if headed < 4:
            return None            # no usable hierarchy; flat chunking is honest
        sections_total += len(sections)
        part = chunk_sections(sections, pdf.name, embed.count_tokens,
                              target, overlap)
        for chunk in part:         # ids must be unique across documents
            chunk.id = len(chunks)
            chunks.append(chunk)
    if not chunks:
        return None
    print(f"      {sections_total} sections recovered from the document hierarchy")
    return chunks


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the RAG index.")
    ap.add_argument("--data", default=config.DATA_DIR)
    ap.add_argument("--index", default=config.INDEX_DIR)
    ap.add_argument("--target-tokens", type=int, default=380)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--min-dense", type=float, default=None,
                    help="relevance floor for this corpus (default 0.66); "
                         "lower it for a narrow single-topic document, where "
                         "every chunk shares the same vocabulary")
    ap.add_argument("--mode", choices=("auto", "structured", "flat"),
                    default="auto",
                    help="auto uses the section hierarchy when one is found")
    args = ap.parse_args()

    data_dir = Path(args.data)
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print(f"No documents found in {data_dir.resolve()}")
        print("Put the PDF there and run this again.")
        return 1

    t0 = time.time()
    chunks = None

    if args.mode in ("auto", "structured"):
        print(f"[1/4] Reading structure from {data_dir.resolve()} ...")
        chunks = _structured_chunks(data_dir, args.target_tokens, args.overlap)
        if chunks is None and args.mode == "structured":
            print("No section hierarchy could be recovered. Try --mode flat.")
            return 1
        if chunks is None:
            print("      no usable hierarchy found; falling back to flat chunking")

    if chunks is None:
        print(f"[1/4] Ingesting from {data_dir.resolve()} ...")
        pages = load_documents(data_dir)
        if not pages:
            print("No readable text extracted. If the PDF is scanned images, it needs OCR first.")
            return 1
        chars = sum(len(p.text) for p in pages)
        sources = sorted({p.source for p in pages})
        print(f"      {len(pages)} pages, {chars:,} characters, from: {', '.join(sources)}")
        print("[2/4] Chunking ...")
        chunks = chunk_pages(pages, embed.count_tokens, args.target_tokens, args.overlap)
    else:
        print("[2/4] Chunking by section ...")

    if not chunks:
        print("Chunking produced nothing — the extracted text may be unusable.")
        return 1

    lens = [embed.count_tokens(c.text) for c in chunks]
    with_path = sum(1 for c in chunks if c.path)
    print(f"      {len(chunks)} chunks | tokens min {min(lens)} / median "
          f"{sorted(lens)[len(lens)//2]} / max {max(lens)} | {with_path} carry a breadcrumb")
    over = sum(1 for l in lens if l > 512)
    if over:
        print(f"      WARNING: {over} chunks exceed the embedder's 512-token limit and will be truncated")

    print(f"[3/4] Embedding {len(chunks)} chunks (CPU, this is the slow part) ...")
    t1 = time.time()
    vectors = embed.embed_passages([embed_text(c) for c in chunks], progress=True)
    print(f"      {vectors.shape[0]} x {vectors.shape[1]} matrix in {time.time()-t1:.1f}s")

    print(f"[4/4] Saving to {Path(args.index).resolve()} ...")
    VectorStore(vectors, chunks, args.min_dense).save(args.index)

    print(f"\nDone in {time.time()-t0:.1f}s. Ask questions with:")
    print("      .venv\\Scripts\\python.exe -m rag.ask \"your question\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
