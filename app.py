"""FastAPI server for the local chatbot."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import docmode
import llm
from sessions import store

SESSION_COOKIE = "chat_session"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup so the first message isn't slowed by it.
    llm.get_llm()
    yield


app = FastAPI(title="Local Chatbot", lifespan=lifespan)


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    """Stop the browser caching the UI.

    This is a local dev app whose HTML, CSS and JS are edited in place. A
    cached stylesheet or script against freshly-served HTML produces a broken
    page -- unstyled controls, dead click handlers -- that looks like a code
    bug, so caching is simply disabled.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".css", ".js")):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    mode: str = "chat"          # "chat" = general assistant, "document" = RAG
    corpus: str | None = None   # which indexed document to search, if several


@app.get("/api/session")
def current_session(request: Request, response: Response):
    """Return the caller's session, creating one on first visit."""
    store.purge_expired()
    session = store.get_or_create(request.cookies.get(SESSION_COOKIE))
    response.set_cookie(SESSION_COOKIE, session.id, httponly=True, samesite="lax")
    return {"session_id": session.id, "title": session.title, "messages": session.messages}


@app.post("/api/session")
def new_session(response: Response):
    """Start a fresh conversation with no history from previous sessions."""
    session = store.create()
    response.set_cookie(SESSION_COOKIE, session.id, httponly=True, samesite="lax")
    return {"session_id": session.id, "title": session.title, "messages": []}


@app.get("/api/sessions")
def list_sessions():
    return {
        "sessions": [
            {"session_id": s.id, "title": s.title, "updated_at": s.updated_at,
             "message_count": len(s.messages)}
            for s in store.list_sessions()
        ]
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session.id, "title": session.title, "messages": session.messages}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    store.delete(session_id)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/clear")
def clear_session(session_id: str):
    store.clear(session_id)
    return {"ok": True}


@app.post("/api/chat")
def chat(req: ChatRequest, request: Request):
    """Stream a reply, using this session's history as context."""
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Message is empty")

    session_id = req.session_id or request.cookies.get(SESSION_COOKIE)
    session = store.get_or_create(session_id)
    store.add_message(session.id, "user", text)

    # Snapshot taken after appending the user turn: the model sees the whole
    # conversation so far, which is what makes follow-ups work.
    history = store.history(session.id)

    def event_stream():
        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"

        yield sse("start", {"session_id": session.id, "title": store.get(session.id).title,
                            "mode": req.mode})
        parts: list[str] = []
        try:
            if req.mode == "document":
                if not docmode.available():
                    yield sse("error", {"detail": docmode.status()["detail"]})
                    return
                # Retrieval runs first so the citations can be shown while the
                # answer is still streaming.
                stream, citations, standalone = docmode.answer_stream(
                    text, history, req.corpus)
                yield sse("sources", {"citations": citations,
                                      "standalone": standalone,
                                      "corpus": req.corpus or docmode.default_corpus()})
            else:
                stream = llm.stream_reply(history)

            for piece in stream:
                parts.append(piece)
                yield sse("token", {"text": piece})
        except Exception as exc:  # surface generation failures in the UI
            yield sse("error", {"detail": str(exc)})
            return
        reply = "".join(parts).strip()
        store.add_message(session.id, "assistant", reply)
        yield sse("done", {"session_id": session.id})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/document")
def document_status():
    """Whether document mode is usable, and what is indexed."""
    return docmode.status()


@app.get("/api/health")
def health():
    return {"status": "ok", "model": config.MODEL_PATH, "n_ctx": config.N_CTX}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
