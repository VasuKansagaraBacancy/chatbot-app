"""Wrapper around the local GGUF model.

Loads the model once per process and exposes a streaming chat completion that
takes the full message history so the model can resolve follow-up questions.
"""
from __future__ import annotations

import threading
from typing import Iterator, List, Dict

from llama_cpp import Llama

import config

_llm: Llama | None = None
_load_lock = threading.Lock()
# llama.cpp keeps one KV cache per model instance, so generations must not
# interleave. Requests are serialised; each one still streams token by token.
_infer_lock = threading.Lock()


def get_llm() -> Llama:
    """Load the model on first use (subsequent calls reuse the instance)."""
    global _llm
    if _llm is None:
        with _load_lock:
            if _llm is None:
                _llm = Llama(
                    model_path=config.MODEL_PATH,
                    n_ctx=config.N_CTX,
                    n_threads=config.N_THREADS,
                    n_gpu_layers=config.N_GPU_LAYERS,
                    verbose=False,
                )
    return _llm


def count_tokens(text: str) -> int:
    return len(get_llm().tokenize(text.encode("utf-8"), add_bos=False))


def build_messages(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Prompt template: a system instruction followed by the session history.

    The history is trimmed from the oldest end so the prompt fits the context
    window. Trimming keeps whole user/assistant pairs and always preserves the
    most recent turns, which are what follow-up questions depend on.
    """
    budget = config.HISTORY_TOKEN_BUDGET - count_tokens(config.SYSTEM_PROMPT)

    kept: List[Dict[str, str]] = []
    used = 0
    for msg in reversed(history):
        # +8 approximates the per-message chat-template markup.
        cost = count_tokens(msg["content"]) + 8
        if used + cost > budget and kept:
            break
        kept.append(msg)
        used += cost
    kept.reverse()

    # Never open the prompt on an assistant turn.
    while kept and kept[0]["role"] == "assistant":
        kept.pop(0)

    return [{"role": "system", "content": config.SYSTEM_PROMPT}, *kept]


def stream_reply(history: List[Dict[str, str]]) -> Iterator[str]:
    """Yield the assistant's reply token by token, from raw chat history."""
    return stream_messages(build_messages(history))


def complete(messages: List[Dict[str, str]], max_tokens: int = 96) -> str:
    """Short, non-streaming completion. Used for query rewriting."""
    with _infer_lock:
        out = get_llm().create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=0.0, stream=False,
        )
    return out["choices"][0]["message"]["content"]


def stream_messages(messages: List[Dict[str, str]],
                    max_tokens: int | None = None) -> Iterator[str]:
    """Yield a reply for an already-assembled message list.

    `max_tokens` is a per-call cap, because the two modes want different
    lengths: an open chat answer may run on, a grounded answer is a figure
    and a citation and should stop.
    """
    with _infer_lock:
        stream = get_llm().create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            repeat_penalty=config.REPEAT_PENALTY,
            stream=True,
        )
        for chunk in stream:
            piece = chunk["choices"][0]["delta"].get("content")
            if piece:
                yield piece
