r"""Retrieval-only validation: does the right passage come back at all?

Separate from `rag.evaluate`, which generates an answer for every question
and takes a minute each on CPU. Most retrieval regressions are visible
without the model: if the chunk holding the fact is not in the hit list, no
amount of generation will recover it. This runs the whole question set in a
second or two, which makes it usable in the edit-rebuild-check loop.

    .venv\Scripts\python.exe -m rag.recall index_imac data_imac\questions.json
"""
from __future__ import annotations

import json
import sys

from .retrieve import retrieve
from .store import VectorStore


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    index, qfile = argv
    store = VectorStore.load(index)
    questions = json.loads(open(qfile, encoding="utf-8").read())

    ok = miss = 0
    for q in questions:
        hits = retrieve(store, q["question"])
        expect = q.get("expect") or []
        # The breadcrumb is shown to the model on the citation line, so a
        # fact that lives only in the path -- a chapter title -- counts as
        # retrieved.
        blob = " ".join(h.chunk.cite() + " " + h.chunk.text for h in hits).lower()
        if q.get("probe") == "abstain":
            good = not hits
        elif expect:
            good = all(e.lower() in blob for e in expect)
        else:
            good = bool(hits)
        ok += good
        miss += not good
        print(f"{'ok  ' if good else 'MISS'} [{len(hits)}] {q['question'][:62]:64} exp={expect}")

    print(f"\n{ok}/{len(questions)} retrieved, {miss} missed")
    return 0 if miss == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
