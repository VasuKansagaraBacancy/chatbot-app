"""Stage 6 — grounded generation.

Builds the prompt that forces the chat model to answer from the retrieved
passages rather than from memory, and streams the reply.

The system prompt here is deliberately stricter than the general chat one in
config.py. A 4B model will happily answer a question about a report it was
never shown, so the instructions have to make abstention the easy path:
cite the numbered passages, and say so plainly when they do not contain the
answer.
"""
from __future__ import annotations

from typing import Dict, Iterator, List

import config
import llm
from .retrieve import build_context
from .store import Hit

RAG_SYSTEM_PROMPT = """You answer questions about a specific document using only the numbered passages supplied with each question.

Rules:
- Base every factual statement on the passages. Do not use outside knowledge, and do not guess.
- Cite the passages you used with their bracketed numbers, like [1] or [2][3], immediately after the statement they support.
- Quote figures, dates and names exactly as they appear. Never round, convert or estimate a number that is not written in the passages.
- If the passages do not contain the answer, say so directly: "The report does not appear to cover this." Then, if the passages are related but insufficient, say briefly what they do cover. Do not pad this out.
- If the passages conflict, point that out rather than silently picking one.
- Be brief. Give the answer in two or three sentences, or a few short bullets. Do not restate the question, do not explain your reasoning, and do not add a closing summary.
- Earlier conversation turns are context for understanding the question. The passages are the only source for facts."""


def build_messages(
    question: str,
    hits: List[Hit],
    history: List[Dict[str, str]] | None = None,
    max_history: int = 4,
) -> List[Dict[str, str]]:
    """Assemble system prompt + recent turns + the retrieved context."""
    messages: List[Dict[str, str]] = [{"role": "system", "content": RAG_SYSTEM_PROMPT}]

    # A couple of prior turns let the model follow the thread of the
    # conversation without letting an old answer act as a source of fact.
    if history:
        for m in history[-max_history:]:
            if m["role"] in ("user", "assistant"):
                messages.append({"role": m["role"], "content": m["content"]})

    context = build_context(hits, max_tokens=config.DOC_CONTEXT_TOKENS)
    if context:
        user = (
            f"PASSAGES FROM THE DOCUMENT:\n\n{context}\n\n"
            f"---\nQUESTION: {question}\n\n"
            "Answer using only the passages above, citing them by number."
        )
    else:
        user = (
            f"QUESTION: {question}\n\n"
            "No passages from the document matched this question. "
            "Say that the report does not appear to cover it."
        )
    messages.append({"role": "user", "content": user})
    return messages


def stream_answer(
    question: str,
    hits: List[Hit],
    history: List[Dict[str, str]] | None = None,
) -> Iterator[str]:
    """Stream a grounded answer token by token."""
    return llm.stream_messages(build_messages(question, hits, history),
                               max_tokens=config.DOC_MAX_TOKENS)


def answer(
    question: str,
    hits: List[Hit],
    history: List[Dict[str, str]] | None = None,
) -> str:
    """Non-streaming convenience wrapper, used by the evaluation script."""
    return "".join(stream_answer(question, hits, history)).strip()
