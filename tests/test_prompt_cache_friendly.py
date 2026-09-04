"""Feature 001 P0 cache-friendly Prompt/Context tests."""

from __future__ import annotations

from types import SimpleNamespace

from internal.agent.agent import ChatOptions, UnifiedAgent
from internal.agent.cancel import CancelRegistry
from internal.agent.planner import llm_plan_graph
from internal.llm.llm import Client, Message
from internal.memory.memory import ShortTerm
from internal.promptctx import (
    ContextAssembler,
    ContextItem,
    ContextSource,
    Query,
    RuntimeContextSchema,
    Slot,
    SlotConstraints,
    SlotFilter,
    SlotProfile,
    SourceRegistry,
    StepObservation,
)
from internal.promptctx.prompts import (
    CHAT_SYSTEM_PROMPT,
    PROMPT_VERSION,
    RUNTIME_CONTEXT_MARKER,
    compose_system_prompt,
    stable_prompt_prefix,
)
from internal.tools.tools import Tool, ToolExecutor


class _RecordingLLM:
    def __init__(self, reply="answer", embedding=None):
        self.reply = reply
        self.embedding = list(embedding or [])
        self.calls = []
        self.embed_calls = []

    def chat(self, messages, system_prompt=""):
        self.calls.append((system_prompt, list(messages)))
        return self.reply

    def embed(self, text):
        self.embed_calls.append(text)
        return list(self.embedding)


class _Preference:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def snapshot(self):
        return dict(self.values)

    def get_all(self):
        return dict(self.values)

    def save_batch(self, values):
        self.values.update(values or {})


class _LTM:
    def __init__(self):
        self.items = []
        self.recall_embeddings = []

    def filter_by_category(self, _categories, _limit):
        return []

    def recall_by_filter(self, _query, query_embedding, _filter):
        self.recall_embeddings.append(list(query_embedding or []))
        return []


class _MemoryWriter:
    def submit(self, _fn):
        pass


class _Events:
    def publish(self, _event, _payload):
        pass


class _Cfg:
    short_term_max_turns = 5
    long_term_top_k = 3
    snapshot_every_turns = 99
    max_retries = 1
    retry_delay_ms = 0
    graph_max_parallel = 2
    graph_race_timeout_ms = 30000
    graph_enable_racing = True

    def __init__(self, real_embedding=False):
        self._real_embedding = real_embedding

    def is_real_llm(self):
        return False

    def is_real_embedding(self):
        return self._real_embedding


def _agent_shell(*, preference=None, llm=None, tools=None, rag=None, cfg=None):
    agent = object.__new__(UnifiedAgent)
    agent.cfg = cfg or _Cfg()
    agent.llm = llm or _RecordingLLM()
    agent.stm = ShortTerm(5)
    agent.ltm = _LTM()
    agent.preference = preference or _Preference()
    agent.memory_writer = _MemoryWriter()
    agent.chat_repo = None
    agent.rag = rag
    agent.inf = SimpleNamespace(repo=SimpleNamespace(events=_Events()))
    agent.tool_executor = ToolExecutor(list(tools or []))
    agent._cancel_registry = CancelRegistry()
    agent._turn_count = 0
    agent._snapshot_every = 99
    agent._build_prompt_context()
    return agent


def test_runtime_context_changes_do_not_change_stable_system_prefix():
    first = compose_system_prompt(CHAT_SYSTEM_PROMPT, "用户画像：上海")
    second = compose_system_prompt(CHAT_SYSTEM_PROMPT, "用户画像：北京")

    assert first.startswith(CHAT_SYSTEM_PROMPT)
    assert second.startswith(CHAT_SYSTEM_PROMPT)
    assert stable_prompt_prefix(first) == CHAT_SYSTEM_PROMPT
    assert stable_prompt_prefix(second) == CHAT_SYSTEM_PROMPT
    assert first.index(RUNTIME_CONTEXT_MARKER) > len(CHAT_SYSTEM_PROMPT)


def test_planner_catalog_is_sorted_stable_and_precedes_query():
    llm = _RecordingLLM(reply="[]")
    agent = SimpleNamespace(
        cfg=SimpleNamespace(is_real_llm=lambda: True),
        llm=llm,
    )
    weather = Tool(
        name="get_weather",
        description="weather",
        params=[
            {"name": "unit", "type": "string"},
            {"name": "city", "type": "string", "required": True},
        ],
        func=lambda _params: "weather",
    )
    search = Tool(
        name="search_web",
        description="search",
        params=[{"name": "query", "type": "string", "required": True}],
        func=lambda _params: "search",
    )

    llm_plan_graph(agent, "first query", {"search_web": search, "get_weather": weather}, "profile")
    llm_plan_graph(agent, "second query", {"get_weather": weather, "search_web": search}, "profile")

    first_system, first_messages = llm.calls[0]
    second_system, second_messages = llm.calls[1]
    assert first_system == second_system
    assert first_system.index("- get_weather:") < first_system.index("- search_web:")
    assert first_system.index("city(string)") < first_system.index("unit(string)")
    assert "first query" not in first_system
    assert first_messages[0].content == "用户问题：first query"
    assert second_messages[0].content == "用户问题：second query"


class _FailingProfileSource(ContextSource):
    def id(self):
        return "failing"

    def supports(self, kind):
        return kind == SlotProfile

    def fetch(self, slot, query):
        raise RuntimeError("source unavailable")


class _WorkingProfileSource(ContextSource):
    def id(self):
        return "working"

    def supports(self, kind):
        return kind == SlotProfile

    def fetch(self, slot, query):
        return [
            ContextItem(text="alpha", score=0.5, source=self.id()),
            ContextItem(text="bravo-long", score=0.9, source=self.id()),
            ContextItem(text="charlie", score=0.7, source=self.id()),
        ]


def test_assembler_continues_after_source_error_and_traces_budget():
    schema = RuntimeContextSchema(
        mode="test",
        slots=[
            Slot(
                kind=SlotProfile,
                filter=SlotFilter(top_k=2, char_budget=10),
            )
        ],
    )
    registry = SourceRegistry()
    registry.register(_FailingProfileSource())
    registry.register(_WorkingProfileSource())

    context = ContextAssembler(
        schemas={"test": schema}, registry=registry, global_limit=100
    ).assemble(Query(text="q", mode="test", phase="generate"))

    assert [item.text for item in context.filled[0].items] == ["bravo-long"]
    trace = context.trace[0]
    assert trace["mode"] == "test"
    assert trace["phase"] == "generate"
    assert trace["candidate_count"] == 3
    assert trace["kept_count"] == 1
    assert trace["dropped_count"] == 2
    assert trace["reason"] == "top_k,char_budget"
    assert [source["status"] for source in trace["sources"]] == ["error", "ok"]


def test_required_slot_missing_has_explicit_trace_status():
    schema = RuntimeContextSchema(
        mode="required-test",
        slots=[Slot(kind=SlotConstraints, required=True)],
    )

    context = ContextAssembler(schemas={"required-test": schema}).assemble(
        Query(text="q", mode="required-test")
    )

    assert context.filled[0].skipped is True
    assert context.trace[0]["status"] == "required_missing"
    assert context.trace[0]["reason"] == "required source missing"


def test_prepare_computes_query_embedding_once_and_does_not_build_legacy_prefix():
    llm = _RecordingLLM(embedding=[0.1, 0.2])
    agent = _agent_shell(llm=llm, cfg=_Cfg(real_embedding=True))

    prepared = agent._prepare("你好", ChatOptions(explicit=True))
    agent._request_context_prefix(
        "你好",
        mode="chat",
        phase="generate",
        query_embedding=prepared["query_embedding"],
    )
    agent._request_context_prefix(
        "你好",
        mode="react",
        phase="plan",
        query_embedding=prepared["query_embedding"],
    )

    assert llm.embed_calls == ["你好"]
    assert prepared["query_embedding"] == [0.1, 0.2]
    assert "mem_prefix" not in prepared
    assert agent.ltm.recall_embeddings == [[0.1, 0.2]]


def test_chat_and_tool_main_paths_use_cache_friendly_context():
    chat_llm = _RecordingLLM()
    chat_agent = _agent_shell(
        llm=chat_llm,
        preference=_Preference({"语言": "中文"}),
    )

    chat_response = chat_agent.process_with_options(
        "你好", ChatOptions(explicit=True)
    )

    assert chat_llm.calls[-1][0].startswith("[AGI-Mira Prompt chat.generate")
    assert "【用户画像】\n- 语言: 中文" in chat_llm.calls[-1][0]
    assert chat_response.context_trace[-1]["phase"] == "generate"

    tool_llm = _RecordingLLM()
    weather = Tool(
        name="get_weather",
        description="weather",
        params=[{"name": "city", "type": "string"}],
        func=lambda _params: "sunny",
    )
    tool_agent = _agent_shell(
        llm=tool_llm,
        preference=_Preference({"语言": "中文"}),
        tools=[weather],
    )

    tool_response = tool_agent.process_with_options(
        "天气 上海", ChatOptions(explicit=False)
    )

    assert tool_response.mode == "tool"
    assert tool_llm.calls[-1][0].startswith("[AGI-Mira Prompt tool.generate")
    assert "【用户画像】\n- 语言: 中文" in tool_llm.calls[-1][0]


def test_react_resets_task_memory_and_reassembles_final_context():
    llm = _RecordingLLM()
    custom = Tool(
        name="custom_tool",
        description="custom",
        params=[],
        func=lambda _params: "custom-result",
    )
    agent = _agent_shell(llm=llm, tools=[custom])
    agent.task_mem.push(
        StepObservation(
            step_id=99,
            tool_name="old_tool",
            result="stale-result",
            success=True,
        )
    )
    context_trace = []

    answer, steps, task = agent._run_react_with_tools(
        "使用自定义工具",
        {"custom_tool": custom},
        [Message(role="user", content="使用自定义工具")],
        [],
        None,
        context_trace=context_trace,
    )

    final_system = llm.calls[-1][0]
    assert answer == "answer"
    assert steps
    assert task["status"] == "completed"
    assert "custom-result" in final_system
    assert "stale-result" not in final_system
    assert [trace["phase"] for trace in context_trace] == ["plan", "generate"]


class _RAGRecorder:
    loaded = True

    def __init__(self):
        self.context_prefix = None

    def query_with_history(self, query, history, context_prefix=""):
        self.context_prefix = context_prefix
        return "rag-answer", [{"content": query, "history": len(history)}]


def test_rag_main_path_receives_profile_without_recall_duplication():
    rag = _RAGRecorder()
    agent = _agent_shell(
        preference=_Preference({"语言": "中文"}),
        rag=rag,
    )

    response = agent.process_with_options(
        "知识库问题",
        ChatOptions(explicit=True, use_rag=True),
    )

    assert response.mode == "rag"
    assert "【用户画像】\n- 语言: 中文" in rag.context_prefix
    assert "【相关回忆】" not in rag.context_prefix
    assert response.context_trace[-1]["mode"] == "rag"


class _UsageResponse:
    status_code = 200
    text = "ok"

    def json(self):
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 80},
            },
        }


def test_llm_records_prompt_prefix_and_provider_usage(monkeypatch):
    cfg = SimpleNamespace(
        llm_api_url="https://example.com/chat",
        llm_api_key="key",
        llm_model="model",
        temperature=0.2,
        is_real_llm=lambda: True,
    )
    client = Client(cfg)
    monkeypatch.setattr("internal.llm.llm.requests.post", lambda *args, **kwargs: _UsageResponse())

    first_system = compose_system_prompt(CHAT_SYSTEM_PROMPT, "城市: 上海")
    second_system = compose_system_prompt(CHAT_SYSTEM_PROMPT, "城市: 北京")
    client.chat([Message(role="user", content="first")], first_system)
    client.chat([Message(role="user", content="second")], second_system)

    first, second = client.prompt_traces()
    assert first["purpose"] == "chat.generate"
    assert first["prompt_version"] == PROMPT_VERSION
    assert first["stable_prefix_hash"] == second["stable_prefix_hash"]
    assert second["common_prefix_chars"] >= second["stable_prefix_chars"]
    assert second["input_tokens"] == 100
    assert second["output_tokens"] == 10
    assert second["cached_input_tokens"] == 80
