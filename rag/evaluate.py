"""Run a set of questions against the index and report on the results.

    .venv\\Scripts\\python.exe -m rag.evaluate                      # uses data/questions.txt
    .venv\\Scripts\\python.exe -m rag.evaluate --questions qs.json  # with expected answers

Question file formats
---------------------
Plain text: one question per line (blank lines and #comments ignored).
JSON: [{"question": "...", "expect": ["substring", "that must appear"]}, ...]

With `expect` present each answer is checked for those substrings, which gives
a blunt but useful regression signal: it catches a retrieval change that stops
surfacing the passage holding a known figure.

Reported per question: whether the answer was grounded (cited a passage),
whether it abstained, retrieval scores, and timings. Abstention is tracked
separately from failure, because correctly saying "not in the report" is the
behaviour we want, not a miss.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config
import llm
from rag import answer as answer_mod
from rag import retrieve as retrieve_mod
from rag.store import VectorStore

_CITED = re.compile(r"\[\d+\]")
_ABSTAINED = re.compile(
    r"does not (appear to )?(cover|contain|mention|include|address)|"
    r"not (found|covered|mentioned|stated|specified) in the (report|document|passages)|"
    r"no (information|mention|details) (about|on|regarding)",
    re.I,
)


def load_questions(path: Path) -> List[Dict]:
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [q if isinstance(q, dict) else {"question": q} for q in raw]
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append({"question": line})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--index", default=config.INDEX_DIR)
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--conversational", action="store_true",
                    help="carry history between questions, so follow-ups resolve")
    args = ap.parse_args()

    qpath = Path(args.questions) if args.questions else None
    if qpath is None:
        for cand in (Path(config.DATA_DIR) / "questions.txt",
                     Path(config.DATA_DIR) / "questions.json",
                     Path("questions.txt")):
            if cand.exists():
                qpath = cand
                break
    if qpath is None or not qpath.exists():
        print("No questions file found. Create data/questions.txt (one per line)")
        print("or pass --questions <file>.")
        return 1

    if not VectorStore.exists(args.index):
        print(f"No index at {Path(args.index).resolve()}. Run: python -m rag.build_index")
        return 1

    questions = load_questions(qpath)
    store = VectorStore.load(args.index)
    print(f"Index   : {len(store.chunks)} chunks")
    print(f"Questions: {len(questions)} from {qpath}")
    print("=" * 78)

    history: List[Dict[str, str]] = []
    results = []
    grounded = abstained = expect_pass = expect_total = 0

    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"\n[{i}/{len(questions)}] {q}")

        standalone = retrieve_mod.rewrite_question(q, history, llm.complete) \
            if args.conversational else q
        if standalone != q:
            print(f"    resolved -> {standalone}")

        t0 = time.time()
        hits = retrieve_mod.retrieve(store, standalone)
        t_ret = time.time() - t0

        t1 = time.time()
        reply = answer_mod.answer(standalone, hits, history if args.conversational else None)
        t_gen = time.time() - t1

        cited = bool(_CITED.search(reply))
        abst = bool(_ABSTAINED.search(reply))
        grounded += cited
        abstained += abst

        print(f"    {reply[:400]}{'...' if len(reply) > 400 else ''}")
        if hits:
            print(f"    top score {hits[0].score:.3f} | {len(hits)} passages | "
                  # A printed page label is a string ("iv", "12"); a chunk
                  # with no label falls back to the integer PDF page, and the
                  # two cannot be sorted against each other.
                  f"pages {sorted({str(h.chunk.label or h.chunk.page) for h in hits})}")
        else:
            print("    no passage cleared the relevance floor")
        print(f"    cited={cited} abstained={abst} | "
              f"retrieval {t_ret*1000:.0f}ms gen {t_gen:.1f}s")

        row = {
            "question": q, "standalone": standalone, "answer": reply,
            "probe": item.get("probe", ""),
            "cited": cited, "abstained": abst,
            "top_score": round(hits[0].score, 4) if hits else 0.0,
            "top_dense": round(hits[0].dense, 4) if hits else 0.0,
            "pages": sorted({h.chunk.page for h in hits}),
            "sources": [h.chunk.cite() for h in hits],
            "retrieval_ms": round(t_ret * 1000), "generation_s": round(t_gen, 1),
        }

        # A question the document cannot answer is scored on whether the
        # system said so, not on whether it produced a citation.
        if item.get("probe") == "abstain":
            row["expect_pass"] = abst
            expect_total += 1
            expect_pass += abst
            print(f"    ABSTAIN {'PASS' if abst else 'FAIL -> answered anyway'}")

        if item.get("expect"):
            expect_total += 1
            missing = [e for e in item["expect"] if e.lower() not in reply.lower()]
            row["expect"] = item["expect"]
            row["expect_missing"] = missing
            row["expect_pass"] = not missing
            expect_pass += not missing
            print(f"    EXPECT {'PASS' if not missing else 'FAIL -> missing ' + str(missing)}")

        results.append(row)
        if args.conversational:
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": reply})

    print("\n" + "=" * 78)
    n = len(results)
    print(f"Answered with a citation : {grounded}/{n}")
    print(f"Abstained ('not covered'): {abstained}/{n}")
    if expect_total:
        print(f"Checks passed            : {expect_pass}/{expect_total}")
    print(f"Median retrieval          : "
          f"{sorted(r['retrieval_ms'] for r in results)[n//2]} ms")

    # Break the score down by what each question was probing, so a regression
    # can be traced to one part of the document rather than to the whole.
    probes: Dict[str, List[int]] = {}
    for row in results:
        if "expect_pass" in row:
            probes.setdefault(row["probe"] or "other", []).append(
                int(row["expect_pass"]))
    if probes:
        print("\nBy probe:")
        for name in sorted(probes):
            hits_ = probes[name]
            print(f"  {name:12} {sum(hits_)}/{len(hits_)}")

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nFull results written to {Path(args.out).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
