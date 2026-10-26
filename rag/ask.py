"""Ask the indexed document a question from the command line.

    .venv\\Scripts\\python.exe -m rag.ask "What were total revenues?"
    .venv\\Scripts\\python.exe -m rag.ask            # interactive, keeps context

Interactive mode holds a conversation history, so follow-up questions are
resolved against earlier turns before retrieval runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
import llm
from rag import answer as answer_mod
from rag import retrieve as retrieve_mod
from rag.store import VectorStore


def show_sources(hits) -> None:
    print("\n  sources:")
    for i, h in enumerate(hits, 1):
        print(f"    [{i}] {h.chunk.cite():<52} "
              f"score {h.score:.3f} (dense {h.dense:+.3f} / lex {h.lexical:.2f})")


def ask_once(store: VectorStore, question: str, history, show: bool = True) -> str:
    standalone = retrieve_mod.rewrite_question(question, history, llm.complete)
    if standalone != question and show:
        print(f"  (resolved to: {standalone})")

    t0 = time.time()
    hits = retrieve_mod.retrieve(store, standalone)
    retrieval_ms = (time.time() - t0) * 1000

    if show:
        print()
    parts = []
    for piece in answer_mod.stream_answer(standalone, hits, history):
        parts.append(piece)
        if show:
            print(piece, end="", flush=True)
    reply = "".join(parts).strip()

    if show:
        print()
        show_sources(hits)
        print(f"  retrieval {retrieval_ms:.0f} ms | generation {time.time()-t0:.1f} s")
    return reply


def main() -> int:
    ap = argparse.ArgumentParser(description="Query the indexed document.")
    ap.add_argument("question", nargs="*", help="omit for interactive mode")
    ap.add_argument("--index", default=config.INDEX_DIR)
    args = ap.parse_args()

    if not VectorStore.exists(args.index):
        print(f"No index at {Path(args.index).resolve()}. Build it first:")
        print("    .venv\\Scripts\\python.exe -m rag.build_index")
        return 1

    store = VectorStore.load(args.index)
    print(f"Index: {len(store.chunks)} chunks from "
          f"{len({c.source for c in store.chunks})} document(s)\n")

    history: list[dict] = []

    if args.question:
        q = " ".join(args.question)
        print(f"Q: {q}")
        ask_once(store, q, history)
        return 0

    print("Interactive mode. Blank line or Ctrl+C to exit.\n")
    while True:
        try:
            q = input("Q: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not q:
            return 0
        reply = ask_once(store, q, history)
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    raise SystemExit(main())
