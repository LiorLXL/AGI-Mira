"""Feature 001 T401-T406: API, persistence and concurrent request isolation."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from internal.agent.agent import ChatOptions
from internal.handler.handler import ChatRequest, setup_routes
from internal.memory.memory import LongTerm
from internal.repo.chathistory import PGRepo, Entry
from internal.request_context import current_request
from internal.tools.tools import Tool
from test_frontend_main_alignment import _request, _Infra
from test_prompt_cache_friendly import _agent_shell, _Preference, _RecordingLLM
from test_memory_correctness import DurableRepo


class History:
    def __init__(self):
        self.rows = {}
        self.lock = threading.Lock()
        self.loads = []

    def save(self, role, content, session_id="default"):
        with self.lock:
            self.rows.setdefault(session_id, []).append(Entry(role, content))

    def load(self, limit, session_id="default"):
        with self.lock:
            self.loads.append(session_id)
            return list(self.rows.get(session_id, [])[-limit:])


class LLM(_RecordingLLM):
    def __init__(self, barrier=None):
        super().__init__()
        self.by_session = {}
        self.barrier = barrier

    def chat(self, messages, system_prompt=""):
        state = current_request.get()
        self.by_session[state.session_id] = (system_prompt, list(messages))
        if self.barrier:
            self.barrier.wait(timeout=5)
        return "answer-" + state.session_id

    def chat_stream_context(self, token, system_prompt, messages, on_token=None):
        answer = self.chat(messages, system_prompt)
        if on_token:
            on_token(answer)
        return answer


def agent_with_history(llm=None, tools=None):
    agent = _agent_shell(llm=llm or LLM(), tools=tools)
    agent.chat_repo = History()
    return agent


def call(agent, session_id, query="hello", **options):
    return agent.process_with_options(query, ChatOptions(
        session_id=session_id, explicit=True, **options
    ))


@pytest.mark.parametrize("invalid", ["", " " * 5, "x" * 129, None])
def test_chat_session_validation(invalid):
    with pytest.raises(ValidationError):
        ChatRequest(message="hello", session_id=invalid)


def test_missing_session_id_uses_default_history():
    assert ChatRequest(message="hello").session_id == "default"
    agent = agent_with_history()
    agent.stm.add("user", "legacy history")
    response = agent.process_with_options("hello", ChatOptions(explicit=True))
    assert response.session_id == "default"
    assert agent.llm.by_session["default"][1][0].content == "legacy history"
    assert current_request.get() is None


@pytest.mark.parametrize("path", ["/api/chat", "/api/chat/stream"])
def test_two_http_sessions_can_generate_concurrently_without_shared_history(path):
    agent = agent_with_history(LLM(threading.Barrier(2)))
    app = setup_routes(agent, _Infra(), agent.cfg)
    def send(sid):
        status, data = _request(app, "POST", path, json.dumps({
            "message": "private-" + sid, "explicit": True, "session_id": sid
        }).encode())
        assert status == 200
        if path.endswith("/stream"):
            blocks = [b for b in data.decode().split("\n\n") if b.startswith("event: done")]
            payload = json.loads(blocks[-1].split("data: ", 1)[1])
        else:
            payload = json.loads(data)
        assert payload["session_id"] == sid
        assert payload["request_id"]
        return payload
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(send, ["a", "b"]))
    assert results[0]["request_id"] != results[1]["request_id"]
    for sid in ("a", "b"):
        messages = agent.llm.by_session[sid][1]
        assert [m.content for m in messages] == ["private-" + sid]
        assert [m.content for m in agent.chat_repo.rows[sid]] == [
            "private-" + sid, "answer-" + sid
        ]
    assert agent.stm.count() == 0
    assert agent._cancel_registry._tokens == {}


def test_switch_back_and_restart_only_restore_selected_session():
    agent = agent_with_history()
    call(agent, "a", "A1")
    call(agent, "b", "B1")
    call(agent, "a", "A2")
    assert [m.content for m in agent.llm.by_session["a"][1]] == ["A1", "answer-a", "A2"]
    restored = agent_with_history()
    restored.chat_repo = agent.chat_repo
    call(restored, "b", "B2")
    assert [m.content for m in restored.llm.by_session["b"][1]] == ["B1", "answer-b", "B2"]
    assert restored.chat_repo.loads == ["a", "b", "b"]


def test_same_session_turns_are_serialized():
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    class BlockingLLM(LLM):
        def chat(self, messages, system_prompt=""):
            if messages[-1].content == "first":
                entered.set()
                assert release.wait(5)
            return super().chat(messages, system_prompt)
    agent = agent_with_history(BlockingLLM())
    def second():
        second_started.set()
        return call(agent, "a", "second")
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(call, agent, "a", "first")
        assert entered.wait(5)
        next_turn = pool.submit(second)
        assert second_started.wait(5)
        assert not next_turn.done()
        release.set()
        first.result(5)
        next_turn.result(5)
    assert [m.content for m in agent.chat_repo.rows["a"]] == [
        "first", "answer-a", "second", "answer-a"
    ]


def test_concurrent_react_tasks_keep_observations_planner_and_traces_local():
    barrier = threading.Barrier(2)
    task_ids = {}
    def execute(_params):
        state = current_request.get()
        snapshot = agent._planner_snapshot()
        task_ids[state.session_id] = snapshot.task_id
        barrier.wait(5)
        assert agent._cancel_registry.current_task()["session_id"] == state.session_id
        return "observation-" + state.session_id
    tool = Tool("custom", "custom", [], execute)
    agent = agent_with_history(tools=[tool])
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(
            lambda sid: call(agent, sid, "work-" + sid, selected_tools=["custom"]), ["a", "b"]
        ))
    assert task_ids["a"] != task_ids["b"]
    for response in responses:
        sid = response.session_id
        system = agent.llm.by_session[sid][0]
        assert "observation-" + sid in system  # Source worker inherited request state
        assert "observation-" + ("b" if sid == "a" else "a") not in system
        assert response.task["task_id"] == task_ids[sid]
    assert agent.task_mem.snapshot() == []
    assert agent.tool_tracker.snapshot() == []
    assert agent._cancel_registry.current_task() is None
    assert current_request.get() is None


def test_task_context_is_released_after_exception():
    agent = agent_with_history()
    def fail(*args):
        assert current_request.get().session_id == "a"
        raise RuntimeError("failed")
    agent._dispatch_mode = fail
    with pytest.raises(RuntimeError):
        call(agent, "a")
    assert current_request.get() is None
    assert agent._cancel_registry._tokens == {}


def test_concurrent_prompt_telemetry_only_contains_own_request():
    from config.config import APIConfig
    from internal.llm.llm import Client
    llm = Client(APIConfig())
    barrier = threading.Barrier(2)
    def generate(_messages):
        barrier.wait(5)
        return "answer"
    llm._mock = generate
    agent = agent_with_history(llm)
    with ThreadPoolExecutor(2) as pool:
        responses = list(pool.map(lambda sid: call(agent, sid), ["a", "b"]))
    for response in responses:
        assert len(response.prompt_trace) == 1
        trace = response.prompt_trace[0]
        assert trace["request_id"] == response.request_id
        assert trace["session_id"] == response.session_id


def test_cancelled_queued_turn_does_not_enter_history():
    entered, release = threading.Event(), threading.Event()
    class BlockingLLM(LLM):
        def chat(self, messages, system_prompt=""):
            entered.set()
            assert release.wait(5)
            return super().chat(messages, system_prompt)
    agent = agent_with_history(BlockingLLM())
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(call, agent, "a", "first")
        assert entered.wait(5)
        queued_token, unregister = agent._cancel_registry.register("a")
        queued_token.cancel()
        second = pool.submit(agent.process_stream, "second",
                             ChatOptions(explicit=True, session_id="a"), lambda _: None,
                             queued_token)
        release.set()
        first.result(5)
        assert second.result(5).interrupted
        unregister()
    assert [r.content for r in agent.chat_repo.rows["a"]] == ["first", "answer-a"]


def test_rag_rewriter_only_receives_selected_session_history():
    agent = agent_with_history()
    call(agent, "a", "private-A")
    call(agent, "b", "private-B")
    history = []
    def query(question, recent, context_prefix=""):
        history.extend(recent)
        return "rag", []
    agent.rag = SimpleNamespace(loaded=True, query_with_history=query)
    call(agent, "b", "follow-up", use_rag=True)
    assert [m.content for m in history] == ["private-B", "answer-b", "follow-up"]


def test_preference_and_long_term_memory_remain_shared():
    agent = agent_with_history()
    agent.preference = _Preference({"城市": "上海"})
    # The real Preference pipeline can update a confirmed shared profile.
    agent.preference.set = lambda k, v: agent.preference.values.__setitem__(k, v)
    repo = DurableRepo()
    agent.ltm = LongTerm(agent.cfg, SimpleNamespace(repo=SimpleNamespace(ltm=repo)))
    agent.ltm.store_classified("coffee beans", .7, None, "fact", ["source:user"], "recall_memory")
    agent._build_prompt_context()
    call(agent, "a", "我住在北京")
    call(agent, "b", "coffee beans")
    system, history = agent.llm.by_session["b"]
    assert "城市: 北京" in system
    assert "coffee beans" in system
    assert "我住在北京" not in [m.content for m in history]
    assert len(agent.ltm.items) == 1


def test_cancel_endpoint_targets_requested_session_only():
    agent = agent_with_history()
    a, end_a = agent._cancel_registry.register("a")
    b, end_b = agent._cancel_registry.register("b")
    app = setup_routes(agent, _Infra(), agent.cfg)
    try:
        status, _ = _request(app, "POST", "/api/chat/cancel", b'{"session_id":"a"}')
        assert status == 200
        assert a.is_cancelled()
        assert not b.is_cancelled()
    finally:
        end_a()
        end_b()


def test_history_repo_queries_include_session_and_preserve_chronology():
    calls = []
    client = SimpleNamespace(
        is_real=lambda: True,
        exec=lambda sql, params: calls.append((sql, params)),
        query=lambda sql, params: calls.append((sql, params)) or [
            ("assistant", "second", ""), ("user", "first", "")
        ],
    )
    repo = PGRepo(client)
    repo.save("user", "first", "a")
    rows = repo.load(10, "a")
    assert calls[0][1] == ("user", "first", "a")
    assert "WHERE session_id = %s" in calls[1][0]
    assert calls[1][1] == ("a", 10)
    assert [r.content for r in rows] == ["first", "second"]


def test_frontend_pins_stream_reply_and_cancellation_to_originating_session():
    source = (Path(__file__).parents[1] / "frontend/index.html").read_text(encoding="utf-8")
    assert "const requestSessionId = currentId;" in source
    assert "session_id: requestSessionId" in source
    assert "pushMessage('ai', replyHtml, requestSessionId)" in source
    assert "activeChatSessionId || currentId" in source
