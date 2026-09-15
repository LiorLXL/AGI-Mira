"""Deterministic local browser-QA server. Never calls a model or external service."""
from pathlib import Path
import sys
import time
from dataclasses import asdict

ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from test_frontend_p0_contracts import ContractAgent, _app
from internal.agent.agent import Response, ReActStep, StepType, _emit


class BrowserAgent(ContractAgent):
    def __init__(self):
        super().__init__()
        self.extra_tools = []
        self.documents = [self.doc]
        self.versions = {self.version.id: self.version}
        self.cancelled_sessions = set()

    def _mode_for(self, message):
        for mode in ("tool", "rag", "react"):
            if f"[{mode}]" in message:
                return mode
        return "chat"

    def _browser_response(self, message, opts):
        mode = self._mode_for(message)
        steps = [ReActStep(StepType.THOUGHT, "检查输入"), ReActStep(StepType.OBSERVATION, "完成分析")] if mode == "react" else []
        tool_call = {"tool_name": "rag_search", "params": {"query": message}, "tool_result": "工具结果", "success": True, "error": ""} if mode == "tool" else None
        search_results = [{"content": "检索片段", "score": .8, "source": "fixture.md"}] if mode == "rag" else []
        return Response(query=message, answer=f"{mode} 示例回答", mode=mode, steps=steps, tool_call=tool_call, search_results=search_results, session_id=opts.session_id, request_id=f"request_{opts.session_id}")

    def process_with_options(self, message, opts):
        return self._browser_response(message, opts)

    def process_stream(self, message, opts, on_event):
        response = self._browser_response(message, opts)
        _emit(on_event, "route", {"mode": response.mode})
        for step in response.steps:
            _emit(on_event, "step", asdict(step))
        if response.tool_call:
            _emit(on_event, "tool_call", response.tool_call)
        if response.search_results:
            _emit(on_event, "rag_result", {"search_results": response.search_results})
        for content in (response.mode, " 示例回答"):
            if "[slow]" in message:
                time.sleep(.3)
            if opts.session_id in self.cancelled_sessions:
                response.interrupted = True
                response.answer = content if content == response.mode else response.mode
                _emit(on_event, "done", asdict(response))
                return response
            _emit(on_event, "token", {"content": content})
        _emit(on_event, "done", asdict(response))
        return response

    def cancel(self, session_id=None):
        if session_id:
            self.cancelled_sessions.add(session_id)
        else:
            super().cancel(session_id)

    def get_tools(self):
        return super().get_tools() + self.extra_tools

    def register_mcp_tool(self, name, description, params, func):
        super().register_mcp_tool(name, description, params, func)
        self.extra_tools = [tool for tool in self.extra_tools if tool["name"] != name]
        self.extra_tools.append({"name": name, "description": description, "params": params, "is_mcp": True})

    def write_document(self, req, ingest_to_rag=False):
        result = super().write_document(req, ingest_to_rag)
        document, version = result["document"], result["version"]
        self.documents = [item for item in self.documents if item.id != document.id] + [document]
        self.versions[version.id] = version
        return result

    def list_documents(self):
        return list(self.documents)

    def get_document(self, document_id):
        document = next((item for item in self.documents if item.id == document_id), None)
        if document is None:
            raise LookupError(f"document not found: {document_id}")
        return {"document": document, "version": self.versions[document.latest_version_id]}


app = _app(BrowserAgent())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8091, log_level="warning")
