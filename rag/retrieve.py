"""Stage 5 — retrieval.

Turns a user question into a ranked set of chunks, with two refinements over a
plain top-k:

* **Conversational rewriting.** In a chat, follow-ups are elliptical: "what
  about the second one?" embeds to nothing useful on its own. The question is
  first resolved against recent turns into a standalone query.
* **A relevance floor.** Returning the best six chunks regardless of score is
  what makes a RAG system confidently answer questions the document never
  covers. Hits below the floor are dropped, which lets the generator say the
  report does not contain the answer.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from . import embed
from .store import Hit, VectorStore

# Below this fused score a chunk is treated as a weak tail behind the leaders.
MIN_SCORE = 0.32

# The fused score is min-max normalized across the candidates, so the best
# candidate scores near 1.0 whatever the question - which makes a floor on it
# useless for deciding that the report simply does not cover something. The
# raw cosine against the embedding is absolute and does separate the cases:
# measured on this report, questions it answers score 0.75 and above, while
# questions it has nothing to say about peak below 0.60.
MIN_DENSE = 0.66
# Pull a wider net than we keep, then drop the weak tail. The net widens with
# the corpus: at 588 chunks a 12-candidate pool is routinely filled by near
# duplicates from one section before a better passage elsewhere is reached.
FETCH_K = 30
# Six are ranked and shown as citations; how many the model actually reads is
# decided by DOC_CONTEXT_TOKENS, because the cost is tokens rather than hits.
# Capping the hits instead was tried and lost a question outright (the answer
# for South Asia sits in hit six), whereas a token budget keeps the tail when
# the chunks are short enough to afford it.
KEEP_K = 6

# At most this many chunks from any one section, so the context shows the
# report's several views on a question rather than one page of it four times.
MAX_PER_SECTION = 2

# A reference list answers almost nothing, but matches strongly on any
# question that shares an author's subject matter. It is kept in the index
# (so "which studies does chapter 3 cite?" still works) and pushed down.
REFERENCE_PENALTY = 0.35

# How far a chunk is lifted when the question names its table, box or chapter.
STRUCTURE_BOOST = 0.30

# "table 1.1", "box 2.1", "figure B1.2.3", "annex 3.1", "endnote 18"
_NAMED_BLOCK = re.compile(
    r"\b(table|box|figure|annex|(?:end|foot)?note)\s*"
    r"([A-Z]?\d+(?:\.\d+)*(?:\.[A-Z])?)\b", re.I)

# A numbered note is recorded as "Note 18" whether the document calls them
# endnotes or footnotes, so both spellings must resolve to that label.
_NOTE_WORD = re.compile(r"^(?:end|foot)?note$", re.I)

_NAMED_PART = re.compile(r"\bchapter\s*(\d+)\b", re.I)

_ASKS_SOURCES = re.compile(
    r"\b(reference|references|bibliograph\w*|cite[sd]?|citation\w*|"
    r"source[sd]?|studies|literature)\b", re.I)

_FOLLOW_UP = re.compile(
    r"^\s*(and\s+)?(what|how|why|who|when|where|which|can|could|is|are|does|do|tell)?\s*"
    r"\b(it|its|that|this|they|them|those|these|the (first|second|third|last|next) one|"
    r"the same|there|he|she|his|her)\b",
    re.I,
)


def needs_rewrite(question: str, history: List[Dict[str, str]]) -> bool:
    """Cheap test for whether a question stands alone."""
    if not history:
        return False
    q = question.strip()
    if len(q.split()) <= 4:          # "and the second?" / "why?"
        return True
    return bool(_FOLLOW_UP.search(q))


def rewrite_question(question: str, history: List[Dict[str, str]], generate) -> str:
    """Rewrite a follow-up into a self-contained query.

    `generate(messages) -> str` is injected so this module does not depend on
    the chat model directly (and so tests can stub it).
    """
    if not needs_rewrite(question, history):
        return question

    recent = history[-6:]
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content'][:400]}" for m in recent
    )
    prompt = (
        "Rewrite the user's latest question so it can be understood on its own, "
        "resolving every pronoun and reference using the conversation below. "
        "Keep it short. Reply with the rewritten question only, nothing else.\n\n"
        f"{transcript}\n\nLATEST QUESTION: {question}\n\nREWRITTEN:"
    )
    try:
        out = generate([{"role": "user", "content": prompt}]).strip()
    except Exception:
        return question

    out = out.strip().strip('"').split("\n")[0].strip()
    # Guard against the model refusing, explaining itself, or rambling.
    if not out or len(out.split()) > 60:
        return question
    return out


def _named_targets(question: str):
    """The tables, boxes, notes and chapters a question names by name."""
    blocks = {
        (f"Note {num}" if _NOTE_WORD.match(kind) else f"{kind.title()} {num}").lower()
        for kind, num in _NAMED_BLOCK.findall(question)
    }
    parts = {f"chapter {n}".lower() for n in _NAMED_PART.findall(question)}
    return blocks, parts


def _named_hits(hits: List[Hit], question: str) -> List[Hit]:
    """Candidates the question named outright, by table, box, note or chapter.

    Naming a block is exact evidence, and it does not depend on the question
    reading like the passage it asks about. "What is Box 1.1 about?" shares
    almost no vocabulary with the box itself, so it scores below the floor
    and the reader is told the report does not cover it - about a box the
    report prints a title for. A named match is therefore allowed through
    the floor, which only ever existed to catch questions the document says
    nothing about.
    """
    blocks, parts = _named_targets(question)
    if not blocks and not parts:
        return []
    named = []
    for hit in hits:
        container = (hit.chunk.container or "").lower()
        part = (hit.chunk.part or "").lower()
        if (blocks and container in blocks) or (parts and part.startswith(tuple(parts))):
            named.append(hit)
    return named


def _rescore(hits: List[Hit], question: str) -> List[Hit]:
    """Re-rank candidates using what the question says about structure.

    A long report is navigated as much by name as by meaning. Readers ask
    "what does table 1.1 show for China?" or "what does chapter 3 conclude
    about hidden debt", and those names are precisely the metadata the
    hierarchy recovered. Matching them is far more reliable than hoping the
    embedding picked the right one of six near-identical forecast tables.
    """
    blocks, parts = _named_targets(question)
    wants_sources = bool(_ASKS_SOURCES.search(question))

    for hit in hits:
        chunk = hit.chunk
        if blocks and (chunk.container or "").lower() in blocks:
            hit.score += STRUCTURE_BOOST
        if parts and (chunk.part or "").lower().startswith(tuple(parts)):
            hit.score += STRUCTURE_BOOST / 2
        if chunk.kind == "reference" and not wants_sources:
            hit.score -= REFERENCE_PENALTY

    hits.sort(key=lambda h: -h.score)
    return hits


def _diversify(hits: List[Hit], k: int, per_section: int) -> List[Hit]:
    """Take the best k hits, allowing only so many from one section.

    Overlapping chunks from a single section crowd out the rest of the
    report, which is how a question that several chapters speak to gets
    answered from only one of them.
    """
    kept: List[Hit] = []
    seen: Dict[str, int] = {}
    for hit in hits:
        key = hit.chunk.path or hit.chunk.heading or str(hit.chunk.page)
        if seen.get(key, 0) >= per_section:
            continue
        seen[key] = seen.get(key, 0) + 1
        kept.append(hit)
        if len(kept) >= k:
            break
    return kept


def retrieve(
    store: VectorStore,
    question: str,
    k: int = KEEP_K,
    min_score: float = MIN_SCORE,
    per_section: int = MAX_PER_SECTION,
    min_dense: Optional[float] = None,
) -> List[Hit]:
    """Rank chunks for a standalone question.

    Returns an empty list when nothing in the report is close enough to the
    question. That is the answer to a question the document does not cover,
    and handing the generator six weak passages instead is what produces a
    confident citation of something irrelevant.
    """
    # The floor belongs to the corpus. A 200-page report covering four
    # chapters of macroeconomics separates covered from uncovered questions
    # cleanly at 0.66; a 12-page single-product brochure does not, because
    # every chunk in it shares the product's vocabulary, so an index may
    # carry its own calibrated floor.
    if min_dense is None:
        min_dense = getattr(store, "min_dense", None) or MIN_DENSE

    qv = embed.embed_query(question)
    candidates = store.search(qv, question, k=FETCH_K)
    if not candidates:
        return []

    named = _named_hits(candidates, question)
    if not named and max(h.dense for h in candidates) < min_dense:
        return []

    hits = _rescore(candidates, question)
    strong = [h for h in hits if h.score >= min_score and h.dense >= min_dense - 0.1]
    if named:
        seen = {id(h) for h in named}
        strong = named + [h for h in strong if id(h) not in seen]
    kept = _diversify(strong, k, per_section)
    if not kept and hits:
        kept = hits[:1]
    return kept


def fit_context(
    hits: List[Hit],
    max_tokens: int = 1800,
    count_tokens=None,
) -> List[Hit]:
    """The leading hits that fit the prompt's budget for passages.

    Budgeted in tokens rather than characters, because that is the unit the
    context window is actually measured in: a chunk of table rows is a third
    the characters of a chunk of prose for the same number of tokens, so a
    character budget either overflows the window or wastes most of it.

    A prefix is returned rather than the best-fitting subset, so that the
    numbering the model is shown runs [1][2][3] without gaps, and so the
    citation panel can list exactly the passages the model actually read.
    """
    if count_tokens is None:
        count_tokens = embed.count_tokens

    kept: List[Hit] = []
    used = 0
    for hit in hits:
        cost = count_tokens(hit.chunk.text) + 24    # allowance for the label
        if kept and used + cost > max_tokens:
            break
        kept.append(hit)
        used += cost
    return kept


def build_context(
    hits: List[Hit],
    max_tokens: int = 1800,
    count_tokens=None,
) -> str:
    """Format retrieved chunks for the prompt, numbered so they can be cited."""
    chosen = fit_context(hits, max_tokens, count_tokens)
    return "\n\n".join(
        f"[{i}] ({hit.chunk.cite()})\n{hit.chunk.text}"
        for i, hit in enumerate(chosen, 1)
    )
