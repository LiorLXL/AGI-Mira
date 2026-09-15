"""Feature 002 T001/T004: executable HTTP/SSE and design-baseline contracts."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import APIConfig
from internal.agent.agent import Response, ReActStep, StepType, _emit
from internal.document.library import Document, DocumentVersion
from internal.handler.handler import setup_routes
from internal.promptctx.prompts import PROMPT_VERSION
from test_frontend_main_alignment import _request


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "frontend-contracts.json"


class _DocumentRepo:
    def __init__(self, agent):
        self.agent = agent

    def get_version(self, version_id):
        assert version_id == "ver_fixture"
        return self.agent.version


class _SnapshotRepo:
    def list(self, limit=50):
        return []


class _Infra:
    ready = SimpleNamespace(
        milvus="connected", postgresql="connected",
        elasticsearch="connected", kafka="disconnected",
    )

    def __init__(self, agent):
        self.repo = SimpleNamespace(
            snapshot=_SnapshotRepo(), documents=_DocumentRepo(agent),
        )


class ContractAgent:
    """Deterministic double; handler serialization/framing remains production code."""

    def __init__(self, mode="chat", fail_stream=False):
        fixed = datetime(2026, 9, 10, 1, 2, 3, tzinfo=timezone.utc)
        self.mode = mode
        self.fail_stream = fail_stream
        self.cancelled_session = None
        self.registered = None
        self.doc = Document(
            id="doc_fixture", title="Fixture document", doc_type="report",
            source="agent_generated", status="active", created_by="agent",
            created_at=fixed, updated_at=fixed,
            latest_version=1, latest_version_id="ver_fixture",
        )
        self.version = DocumentVersion(
            id="ver_fixture", document_id="doc_fixture", version=1,
            content_md="# Fixture\n\nContract content.", summary="Fixture summary",
            metadata={"filename": "fixture.md", "parser": "plain_text", "text_chars": 29},
            created_at=fixed,
        )
        self.inf = None

    def _response(self, message, opts):
        steps = []
        tool_call = None
        search_results = []
        if self.mode == "react":
            steps = [
                ReActStep(StepType.THOUGHT, "检查输入"),
                ReActStep(StepType.ACTION, "读取资料", "read_document", {"id": "doc_fixture"}),
                ReActStep(StepType.OBSERVATION, "已读取 1 份资料"),
            ]
        if self.mode == "tool":
            tool_call = {
                "tool_name": "get_time", "params": {},
                "tool_result": "10:00", "success": True, "error": "",
            }
        if self.mode == "rag":
            search_results = [
                {"content": "Fixture source", "score": 0.72, "source": "fixture.md"}
            ]
        return Response(
            query=message, answer="示例回答", mode=self.mode,
            steps=steps, tool_call=tool_call, search_results=search_results,
            session_id=opts.session_id, request_id="request_fixture",
        )

    def process_with_options(self, message, opts):
        return self._response(message, opts)

    def process_stream(self, message, opts, on_event):
        if self.fail_stream:
            raise RuntimeError("fixture failure")
        response = self._response(message, opts)
        _emit(on_event, "route", {"mode": response.mode})
        _emit(on_event, "context", {
            "mode": response.mode, "phase": "generate",
            "prompt_version": PROMPT_VERSION, "prompt_chars": 0, "slots": [],
        })
        for step in response.steps:
            _emit(on_event, "step", asdict(step))
        if response.tool_call:
            _emit(on_event, "tool_call", response.tool_call)
        if response.search_results:
            _emit(on_event, "rag_result", {"search_results": response.search_results})
        _emit(on_event, "token", {"content": "示例"})
        _emit(on_event, "token", {"content": "回答"})
        _emit(on_event, "done", asdict(response))
        return response

    def cancel(self, session_id=None):
        self.cancelled_session = session_id

    def get_tools(self):
        return [{
            "name": "rag_search", "description": "检索知识库",
            "params": [{"name": "query", "type": "string", "description": "问题"}],
            "is_mcp": False,
        }]

    def register_mcp_tool(self, name, description, params, func):
        self.registered = (name, description, params, func({"input": "fixture"}))

    def status(self):
        return {
            "rag_loaded": True, "rag_mode": "hybrid", "rag_chunks": [],
            "short_term_count": 2, "long_term_count": 1,
            "preferences": {"回答风格": "简洁"}, "tools_count": 1,
            "llm_model": "fixture-llm", "embedding_model": "fixture-embedding",
            "is_mock": True,
            "infrastructure": {
                "milvus": "connected", "postgresql": "connected",
                "elasticsearch": "connected", "kafka": "disconnected",
            },
        }

    def list_documents(self):
        return [self.doc]

    def get_document(self, document_id):
        if document_id != self.doc.id:
            raise LookupError(f"document not found: {document_id}")
        return {"document": self.doc, "version": self.version}

    def write_document(self, req, ingest_to_rag=False):
        document = Document(
            **{**asdict(self.doc), "title": req.title, "doc_type": req.doc_type,
               "source": req.source, "created_by": req.created_by}
        )
        version = DocumentVersion(
            **{**asdict(self.version), "content_md": req.content_md,
               "metadata": req.metadata}
        )
        result = {"document": document, "version": version, "created": True}
        if ingest_to_rag:
            result["ingest"] = self.ingest_document(document.id, version.id)
        return result

    def ingest_document(self, document_id, version_id=""):
        return {
            "chunk_count": 2, "parent_count": 0, "indexed_count": 2,
            "doc_hash": "fixture_hash", "document_id": document_id,
            "version_id": version_id or self.version.id, "section": "upload",
        }


def _app(agent):
    cfg = APIConfig()
    agent.inf = _Infra(agent)
    return setup_routes(agent, agent.inf, cfg)


def _json_request(agent, method, path, body=None):
    payload = b"" if body is None else json.dumps(body, ensure_ascii=False).encode()
    status, raw = _request(_app(agent), method, path, payload)
    return {"status": status, "body": json.loads(raw) if raw else None}


def _parse_sse(raw):
    events = []
    terminal = None
    normalized = raw.decode().replace("\r\n", "\n")
    for block in normalized.split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        data = "\n".join(data_lines)
        if data == "[DONE]":
            terminal = data
        elif data:
            events.append({"event": event_name, "data": json.loads(data)})
    return {"events": events, "terminal": terminal}


def _stream_case(mode="chat", fail=False):
    agent = ContractAgent(mode, fail_stream=fail)
    request = {
        "session_id": "session_fixture", "message": "示例问题",
        "use_rag": mode == "rag",
        "selected_tools": ["get_time"] if mode == "tool" else [],
        "explicit": True,
    }
    status, raw = _request(
        _app(agent), "POST", "/api/chat/stream",
        json.dumps(request, ensure_ascii=False).encode(),
    )
    return {"request": request, "status": status, **_parse_sse(raw)}


def capture_contracts():
    sync_request = {
        "session_id": "session_fixture", "message": "示例问题",
        "use_rag": False, "selected_tools": [], "explicit": True,
    }
    sync = _json_request(ContractAgent(), "POST", "/api/chat", sync_request)
    sync["request"] = sync_request

    cancel_agent = ContractAgent()
    cancel_request = {"session_id": "session_fixture"}
    cancel = _json_request(cancel_agent, "POST", "/api/chat/cancel", cancel_request)
    assert cancel_agent.cancelled_session == "session_fixture"
    cancel["request"] = cancel_request

    upload = _json_request(
        ContractAgent(), "POST", "/api/upload", {"content": "fixture upload"}
    )
    documents = _json_request(ContractAgent(), "GET", "/api/documents")
    detail = _json_request(
        ContractAgent(), "GET", "/api/documents/doc_fixture"
    )
    ingest = _json_request(
        ContractAgent(), "POST", "/api/documents/doc_fixture/ingest",
        {"version_id": "ver_fixture"},
    )
    tools = _json_request(ContractAgent(), "GET", "/api/tools")
    mcp = _json_request(
        ContractAgent(), "POST", "/api/tools/mcp",
        {"name": "fixture_tool", "description": "Fixture", "endpoint": "http://fixture.invalid/tool", "params": []},
    )
    status = _json_request(ContractAgent(), "GET", "/api/status")
    invalid = _json_request(
        ContractAgent(), "POST", "/api/chat",
        {"session_id": "", "message": "bad"},
    )
    default_session = _json_request(
        ContractAgent(), "POST", "/api/chat", {"message": "legacy"}
    )
    return {
        "meta": {
            "version": 1,
            "provenance": "Captured from current setup_routes with a deterministic Agent double; no provider or database calls.",
            "notes": [
                "HTTP status, serialization, SSE framing and validation are production handler behavior.",
                "Agent decisions and fixture content are deterministic examples, not real model output.",
                "A normal streaming done event has no success field; an exception done may have no session_id/request_id.",
            ],
        },
        "cases": {
            "chat_json": sync,
            "chat_stream": _stream_case("chat"),
            "tool_stream": _stream_case("tool"),
            "rag_stream": _stream_case("rag"),
            "react_stream": _stream_case("react"),
            "exception_stream": _stream_case("chat", fail=True),
            "cancel_session": cancel,
            "upload_json": upload,
            "documents_list": documents,
            "document_detail": detail,
            "document_ingest": ingest,
            "tools_list": tools,
            "mcp_register": mcp,
            "status": status,
            "invalid_session": invalid,
            "default_session": default_session,
        },
    }


def test_contract_fixture_matches_current_handler():
    expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert capture_contracts() == expected


def test_contract_fixture_covers_required_frontend_edges():
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]
    normal_done = fixture["chat_stream"]["events"][-1]
    error_done = fixture["exception_stream"]["events"][-1]
    assert normal_done["event"] == "done" and "success" not in normal_done["data"]
    assert fixture["chat_stream"]["events"][0]["data"].keys() == {"message", "session_id"}
    assert error_done == {
        "event": "done",
        "data": {"answer": "请求失败: fixture failure", "interrupted": False, "success": False},
    }
    assert fixture["cancel_session"]["request"] == {"session_id": "session_fixture"}
    assert fixture["upload_json"]["body"]["document"]["id"] == "doc_fixture"
    assert fixture["documents_list"]["body"]["documents"][0]["latest_version"] == 1
    assert fixture["mcp_register"]["body"] == {"success": True, "ok": True}


def _relative_luminance(hex_color):
    values = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in values]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(left, right):
    first, second = _relative_luminance(left), _relative_luminance(right)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def _theme_tokens(css, selector):
    block = re.search(selector + r"\s*\{(?P<body>[^}]+)\}", css).group("body")
    return dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})\s*;", block))


def test_design_tokens_keep_normal_text_contrast_at_least_4_5():
    css = (ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    light = _theme_tokens(css, r":root,\s*:root\[data-theme=\"light\"\]")
    dark = _theme_tokens(css, r":root\[data-theme=\"dark\"\]")
    pairs = [
        (light["text"], light["canvas"]),
        (light["muted"], light["canvas"]),
        (light["muted"], light["surface"]),
        (light["muted"], light["sidebar"]),
        (light["accent"], light["canvas"]),
        (light["error"], light["error-surface"]),
        (dark["text"], dark["canvas"]),
        (dark["muted"], dark["canvas"]),
        (dark["muted"], dark["surface"]),
        (dark["muted"], dark["sidebar"]),
        (dark["accent"], dark["canvas"]),
        (dark["error"], dark["error-surface"]),
    ]
    assert min(_contrast(*pair) for pair in pairs) >= 4.5


def test_production_frontend_preserves_the_design_baseline():
    html = (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    app = (ROOT / "frontend" / "app.js").read_text(encoding="utf-8")
    ui = (ROOT / "frontend" / "ui.js").read_text(encoding="utf-8")
    css = (ROOT / "frontend" / "styles.css").read_text(encoding="utf-8")
    assert all(f'data-page="{page}"' in html for page in ("chat", "documents", "settings"))
    assert '<dialog class="mobile-drawer"' in html
    assert 'data-set-theme="light"' in html and 'data-set-theme="dark"' in html
    assert "isComposing" in app
    assert "aria-current" in app and "aria-current" in ui
    assert "aria-pressed" in app and "aria-pressed" in ui
    assert "@media (max-width: 767px)" in css and ".sidebar { display: none; }" in css
    assert "prefers-reduced-motion: reduce" in css and ":focus-visible" in css


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    if not args.update:
        raise SystemExit("Use --update only after reviewing intentional backend contract changes")
    FIXTURE_PATH.write_text(
        json.dumps(capture_contracts(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"updated {FIXTURE_PATH}")
