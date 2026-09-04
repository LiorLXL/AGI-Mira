"""Feature 001 Phase 1 characterization tests.

This module intentionally captures the current runtime behaviour before the
prompt/context and memory changes are implemented.  Desired behaviours that
the current implementation does not satisfy are strict xfails: they document
the gap without making the baseline suite fail, and turn into an XPASS failure
as soon as production code changes without updating the expectation.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from internal.agent.agent import ReActStep, Response, StepType, UnifiedAgent
from internal.agent.memory_writer import async_update_memory, extract_memory_from_reply
from internal.agent.planner import llm_plan_graph
from internal.llm.llm import Message
from internal.memory.memory import Item, LongTerm, ShortTerm
from internal.promptctx import (
    ContextAssembler,
    ProfileSource,
    Query,
    RecallSource,
    SourceRegistry,
)
from internal.rag.rag import Engine
from internal.rag.rewriter import HistoryMessage, LLMRewriter
from internal.tools.tools import Tool, ToolExecutor


_GOLDEN_PATH = (
    Path(__file__).parent / "fixtures" / "feature_001_prompt_baseline.json"
)
_CACHE_GOLDEN_PATH = (
    Path(__file__).parent / "fixtures" / "feature_001_cache_friendly_prompt.json"
)
_PROJECT_ROOT = Path(__file__).parents[1]


class _RecordingLLM:
    def __init__(self, reply: str = "baseline-answer") -> None:
        self.reply = reply
        self.calls = []

    def chat(self, messages, system_prompt=""):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in messages
                ],
            }
        )
        return self.reply


class _PreferenceSnapshot:
    def snapshot(self):
        # Intentionally unsorted to prove the current ProfileSource render order.
        return {"语言": "中文", "城市": "上海"}


class _RecallRecorder:
    def recall_by_filter(self, query, query_embedding, filter):
        return [
            Item(
                content="用户偏好简洁回答",
                importance=0.8,
                category="preference",
                score=0.9,
            )
        ]


def _capture_generate(calls, purpose: str, reply: str):
    def _generate(system_prompt: str, user_msg: str) -> str:
        calls[purpose] = {
            "system_prompt": system_prompt,
            "messages": [{"role": "user", "content": user_msg}],
        }
        return reply

    return _generate


def _capture_cache_friendly_prompt_snapshot():
    calls = {}

    # Chat generation.
    chat_llm = _RecordingLLM()
    chat_agent = object.__new__(UnifiedAgent)
    chat_agent.llm = chat_llm
    chat_agent._chat_response(
        '用户偏好: {"语言":"中文"}',
        [Message(role="user", content="请介绍这个项目")],
    )
    calls["chat.generate"] = chat_llm.calls[-1]

    # Tool-result generation.
    tool_llm = _RecordingLLM()
    tool_agent = object.__new__(UnifiedAgent)
    tool_agent.llm = tool_llm
    tool_agent.preference = SimpleNamespace(get_all=lambda: {})
    weather = Tool(
        name="get_weather",
        description="查询天气",
        params=[{"name": "city", "type": "string", "required": True}],
        func=lambda _params: "上海：晴，25℃",
    )
    tool_agent.tool_executor = ToolExecutor([weather])
    tool_agent._run_tool_from_set(
        "天气 上海",
        {"get_weather": weather},
        "相关记忆:\n- 用户偏好中文回答",
        [Message(role="user", content="天气 上海")],
    )
    calls["tool.generate"] = tool_llm.calls[-1]

    # React planner generation.  The insertion order is deliberate and part of
    # the current golden: the implementation does not sort this catalog yet.
    planner_llm = _RecordingLLM(reply="[]")
    planner_agent = SimpleNamespace(
        cfg=SimpleNamespace(is_real_llm=lambda: True),
        llm=planner_llm,
    )
    tools = {
        "search_web": Tool(
            name="search_web",
            description="搜索网页",
            params=[{"name": "query", "type": "string", "required": True}],
            func=lambda _params: "web",
        ),
        "get_weather": weather,
    }
    llm_plan_graph(
        planner_agent,
        "先查天气再搜索",
        tools,
        '用户偏好: {"城市":"上海"}',
    )
    calls["react.plan"] = planner_llm.calls[-1]

    # React final generation.
    final_llm = _RecordingLLM()
    final_agent = object.__new__(UnifiedAgent)
    final_agent.llm = final_llm
    final_agent._generate_final_answer(
        "汇总结果",
        [
            ReActStep(
                type=StepType.OBSERVATION,
                content="天气晴朗",
                tool="get_weather",
                params={},
            )
        ],
        "相关记忆:\n- 用户偏好中文回答",
    )
    calls["react.generate"] = final_llm.calls[-1]

    # RAG query rewrite.
    rewriter = LLMRewriter(
        _capture_generate(
            calls,
            "rag.rewrite",
            json.dumps({"queries": ["独立查询", "查询变体"]}, ensure_ascii=False),
        ),
        num_queries=2,
    )
    rewriter.rewrite(
        "它如何工作？",
        [HistoryMessage(role="user", content="上一轮问了 Agent")],
    )

    # RAG answer generation without constructing external infrastructure.
    engine = object.__new__(Engine)
    engine._generate_fn = _capture_generate(calls, "rag.generate", "rag-answer")
    engine._compose_answer(
        "TaskGraph 如何执行？",
        [
            {
                "pg_id": 1,
                "content": "TaskGraph 使用拓扑排序。",
                "score": 0.9,
                "source": "baseline",
            }
        ],
    )

    # Current promptctx rendering and its currently-empty trace.
    registry = SourceRegistry()
    registry.register(ProfileSource(_PreferenceSnapshot(), None))
    registry.register(RecallSource(_RecallRecorder()))
    runtime_context = ContextAssembler(registry=registry).assemble(
        Query(text="请简洁回答", mode="chat")
    )

    return {
        "schema_version": 1,
        "trace_contract": [
            "mode",
            "phase",
            "prompt_version",
            "stable_prefix_hash",
            "prompt_chars",
            "slots",
        ],
        "calls": calls,
        "context_render": runtime_context.render(),
        "current_runtime_context_trace": runtime_context.trace,
    }


def _load_golden():
    return json.loads(_GOLDEN_PATH.read_text(encoding="utf-8"))


def test_phase_1_llm_prompt_baseline_documents_original_order():
    """The before-snapshot stays immutable after production code improves."""
    baseline = _load_golden()
    calls = baseline["calls"]

    assert set(calls) == {
        "chat.generate",
        "tool.generate",
        "react.plan",
        "react.generate",
        "rag.rewrite",
        "rag.generate",
    }
    assert calls["chat.generate"]["system_prompt"].startswith("用户偏好:")
    planner_user = calls["react.plan"]["messages"][0]["content"]
    assert planner_user.index("用户问题：") < planner_user.index("可用工具：")


def test_phase_1_context_baseline_documents_missing_trace():
    baseline = _load_golden()

    assert "【用户画像】" in baseline["context_render"]
    assert "【相关回忆】" in baseline["context_render"]
    assert baseline["current_runtime_context_trace"] == []


def test_feature_001_trace_contract_is_declared_in_baseline():
    expected = _load_golden()

    assert expected["schema_version"] == 1
    assert expected["trace_contract"] == [
        "mode",
        "phase",
        "prompt_version",
        "stable_prefix_hash",
        "prompt_chars",
        "slots",
    ]


def test_cache_friendly_prompt_calls_match_feature_001_golden():
    actual = _capture_cache_friendly_prompt_snapshot()
    # Context source timings are telemetry, not stable Prompt snapshot data.
    actual.pop("current_runtime_context_trace")
    expected = json.loads(_CACHE_GOLDEN_PATH.read_text(encoding="utf-8"))

    assert actual == expected


def test_current_frontend_sessions_are_ui_only_baseline():
    """Capture why two visible frontend sessions still share backend STM."""
    frontend = (_PROJECT_ROOT / "frontend" / "index.html").read_text(
        encoding="utf-8"
    )
    handler = (_PROJECT_ROOT / "internal" / "handler" / "handler.py").read_text(
        encoding="utf-8"
    )

    request_body_line = next(
        line for line in frontend.splitlines() if "const body = { message: msg" in line
    )
    chat_request_block = handler.split("class ChatRequest", 1)[1].split(
        "class MCPParam", 1
    )[0]

    assert "session_id" not in request_body_line
    assert "session_id" not in chat_request_block


class _ImmediateWriter:
    def submit(self, fn):
        fn()


class _RawLTMRecorder:
    def __init__(self):
        self.added = []

    def add(self, content, importance=0.5):
        self.added.append((content, importance))


@pytest.mark.xfail(
    strict=True,
    reason="Feature 001 T201: ordinary user queries must not be copied to LTM",
)
def test_future_ordinary_query_is_not_written_to_long_term_memory():
    ltm = _RawLTMRecorder()
    agent = SimpleNamespace(
        llm=SimpleNamespace(extract_preferences=lambda _text: {}),
        preference=SimpleNamespace(save_batch=lambda _items: None),
        ltm=ltm,
        memory_writer=_ImmediateWriter(),
    )

    async_update_memory(agent, "今天天气怎么样", Response(query="今天天气怎么样"))

    assert ltm.added == []


class _PreferenceRecorder:
    def __init__(self):
        self.saved = []

    def set(self, key, value):
        self.saved.append((key, value))


class _ClassifiedLTMRecorder:
    _embed_fn = None

    def __init__(self):
        self.stored = []

    def store_classified(self, content, importance, emb, category, tags, slot_hint):
        self.stored.append(
            (content, importance, emb, category, tuple(tags), slot_hint)
        )
        return True

    def last_id(self):
        return 1


@pytest.mark.xfail(
    strict=True,
    reason="Feature 001 T202: assistant output must not activate user memory",
)
def test_future_assistant_reply_is_not_a_memory_source():
    preference = _PreferenceRecorder()
    ltm = _ClassifiedLTMRecorder()
    agent = SimpleNamespace(
        cfg=SimpleNamespace(is_real_llm=lambda: True),
        llm=_RecordingLLM(reply='{"姓名":"小林"}'),
        preference=preference,
        ltm=ltm,
        graph_memory=None,
    )

    extract_memory_from_reply(agent, "根据上下文，用户的姓名是小林。")

    assert preference.saved == []
    assert ltm.stored == []


class _StoredRows:
    def load(self):
        return [
            SimpleNamespace(
                id=7,
                content="memory-seven",
                importance=0.7,
                embedding=[],
                created_at=100.0,
                last_accessed=100.0,
                category="fact",
                tags=[],
                slot_hint="recall_memory",
                score=0.0,
            ),
            SimpleNamespace(
                id=42,
                content="memory-forty-two",
                importance=0.8,
                embedding=[],
                created_at=200.0,
                last_accessed=200.0,
                category="fact",
                tags=[],
                slot_hint="recall_memory",
                score=0.0,
            ),
        ]


@pytest.mark.xfail(
    strict=True,
    reason="Feature 001 T301: restore must preserve PostgreSQL memory IDs",
)
def test_future_restore_preserves_non_contiguous_postgres_ids():
    inf = SimpleNamespace(repo=SimpleNamespace(ltm=_StoredRows()))
    ltm = LongTerm(SimpleNamespace(), inf)

    ltm.load_from_storage()

    assert [item.id for item in ltm.items] == [7, 42]
    assert ltm.last_id() == 42


@pytest.mark.xfail(
    strict=True,
    reason="Feature 001 T403: short-term history must be isolated by session",
)
def test_future_session_b_history_excludes_session_a_messages():
    agent = object.__new__(UnifiedAgent)
    agent.stm = ShortTerm(max_turns=5)
    agent.stm.add("user", "session-a-private-message")
    agent.stm.add("assistant", "session-a-answer")

    # The current method has no session argument, so a new frontend session
    # still receives the singleton Agent's existing short-term history.
    session_b_history = agent._build_history_messages("session-b-question")

    assert all("session-a" not in message.content for message in session_b_history)
