"""Request-local state shared explicitly with runtime worker contexts."""
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from typing import Any
import threading
from uuid import uuid4

DEFAULT_SESSION_ID = "default"


def normalize_session_id(value):
    value = str(value or DEFAULT_SESSION_ID).strip()
    if not value or len(value) > 128:
        raise ValueError("session_id must contain 1-128 characters")
    return value


@dataclass
class RequestState:
    owner: Any
    session_id: str
    stm: Any
    task_mem: Any
    tool_tracker: Any
    request_id: str = field(default_factory=lambda: uuid4().hex)
    task: Any = None
    snapshots: list = field(default_factory=list)
    last_subagent_task: Any = None


current_request = ContextVar("mira_request", default=None)


def context_thread(target, args=(), **kwargs):
    """Each thread gets a distinct Context sharing the originating request state."""
    return threading.Thread(target=copy_context().run, args=(target, *args), **kwargs)
