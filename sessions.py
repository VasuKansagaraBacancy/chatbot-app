"""In-memory, per-session conversation store.

Each session has its own message list. A new session starts empty, so it never
inherits context from an earlier conversation.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config


@dataclass
class Session:
    id: str
    title: str = "New chat"
    messages: List[Dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self) -> Session:
        session = Session(id=uuid.uuid4().hex)
        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def get_or_create(self, session_id: Optional[str]) -> Session:
        if session_id:
            existing = self.get(session_id)
            if existing is not None:
                return existing
        return self.create()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.messages.append({"role": role, "content": content})
            session.updated_at = time.time()
            # Name the conversation after its opening question.
            if role == "user" and session.title == "New chat":
                session.title = content.strip().splitlines()[0][:48] or "New chat"

    def history(self, session_id: str) -> List[Dict[str, str]]:
        session = self.get(session_id)
        return list(session.messages) if session else []

    def clear(self, session_id: str) -> None:
        """Wipe a session's history while keeping its id."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.messages.clear()
                session.title = "New chat"
                session.updated_at = time.time()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> List[Session]:
        with self._lock:
            return sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)

    def purge_expired(self) -> None:
        cutoff = time.time() - config.SESSION_TTL
        with self._lock:
            for sid in [s.id for s in self._sessions.values() if s.updated_at < cutoff]:
                del self._sessions[sid]


store = SessionStore()
