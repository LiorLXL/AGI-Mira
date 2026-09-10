"""Per-session short-term windows; serialize complete turns within one session."""
from dataclasses import dataclass, field
import threading

from internal.request_context import DEFAULT_SESSION_ID, normalize_session_id
from .memory import ShortTerm


@dataclass
class Session:
    stm: ShortTerm
    loaded: bool = False
    lock: object = field(default_factory=threading.RLock)


class SessionStore:
    def __init__(self, default_stm, max_turns):
        self.max_turns = max_turns
        self._lock = threading.Lock()
        # Startup restore already loaded the legacy/default conversation.
        self._sessions = {DEFAULT_SESSION_ID: Session(default_stm, loaded=True)}

    def get(self, session_id):
        session_id = normalize_session_id(session_id)
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session(ShortTerm(self.max_turns))
            return self._sessions[session_id]
