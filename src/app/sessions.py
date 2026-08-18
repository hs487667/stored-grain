"""One set of lots per visitor, and a bound on how many are kept.

A session used to be the process. That is right for one photographer on a LAN
and wrong for a published URL, where it would put every visitor into the same
ranking. Each visitor now gets a store of their own, keyed by an id carried in
a cookie.

Nothing is persisted. A session dies when it goes quiet or when the process
does, which is the honest lifetime: a reading that outlived the models that
produced it would be a reading nobody could interpret later.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field

from src.app.store import SessionStore

#: Sessions are dropped after this long without a request. Long enough to
#: photograph several trays, short enough that a public URL does not
#: accumulate abandoned overlays indefinitely.
IDLE_TIMEOUT_S = 30 * 60

#: Hard ceiling on concurrent sessions. Overlay PNGs are held in memory, so
#: this is the difference between a busy day and an exhausted process.
MAX_SESSIONS = 200


@dataclass
class Session:
    """One visitor's lots, and the images behind them."""

    store: SessionStore = field(default_factory=SessionStore)
    overlays: dict[str, bytes] = field(default_factory=dict)
    photographs: dict[str, bytes] = field(default_factory=dict)
    touched: float = 0.0


class SessionRegistry:
    def __init__(
        self,
        idle_timeout_s: float = IDLE_TIMEOUT_S,
        max_sessions: int = MAX_SESSIONS,
        clock=time.monotonic,
    ) -> None:
        self._sessions: dict[str, Session] = {}
        #: Read by the HTTP layer to keep the cookie's max-age in step with
        #: the server's own idea of when a session has gone quiet.
        self.idle_timeout_s = idle_timeout_s
        self._max_sessions = max_sessions
        self._clock = clock

    def get_or_create(self, session_id: str | None) -> tuple[str, Session]:
        """Return the caller's session, minting one if they have no valid id.

        An id that is not already registered never becomes a session id. A
        visitor who sends an arbitrary cookie gets a fresh session under a
        server-chosen id instead, so nobody can land in another visitor's
        session by guessing or replaying one.
        """
        now = self._clock()
        self._evict_idle(now)

        session = self._sessions.get(session_id) if session_id else None
        if session is None:
            session_id = secrets.token_urlsafe(32)
            # Stamped before the overflow sweep runs: the sweep drops the least
            # recently touched session, and a session still sitting at its
            # default timestamp would be the one it chose.
            session = Session(touched=now)
            self._sessions[session_id] = session
            self._evict_overflow()

        session.touched = now
        return session_id, session

    def _evict_idle(self, now: float) -> None:
        expired = [
            key
            for key, session in self._sessions.items()
            if now - session.touched > self.idle_timeout_s
        ]
        for key in expired:
            del self._sessions[key]

    def _evict_overflow(self) -> None:
        while len(self._sessions) > self._max_sessions:
            oldest = min(self._sessions, key=lambda key: self._sessions[key].touched)
            del self._sessions[oldest]

    def __len__(self) -> int:
        return len(self._sessions)
