# llm — LLM 客户端（OpenAI 兼容 Chat Completions + Embedding，与 main 分支 Go 版协议对齐）
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional

import requests

from config.config import APIConfig
from internal.request_context import current_request
from internal.promptctx.prompts import (
    PREFERENCE_EXTRACT_SYSTEM_PROMPT,
    prompt_identity,
    stable_prompt_prefix,
)

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str


@dataclass
class PromptCallTrace:
    sequence: int
    purpose: str
    prompt_version: str
    stable_prefix_hash: str
    stable_prefix_chars: int
    prompt_chars: int
    common_prefix_chars: int
    stream: bool
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cached_input_tokens: Optional[int] = None
    fallback: bool = False
    error_type: str = ""
    request_id: str = ""
    session_id: str = ""


class Client:
    """LLM 客户端封装：OpenAI 兼容 Chat Completions + 火山方舟多模态 Embedding。"""

    def __init__(self, cfg: APIConfig):
        self.cfg = cfg
        self._timeout = 60
        self._mock_responses = {
            "你是谁": "我是一个全能 AI 助手，具备知识库、工具调用、推理、记忆和稳定执行能力。",
            "后端工程师": "后端工程师负责服务器端逻辑开发：API 设计、数据库、业务逻辑、系统架构、性能优化。",
        }
        self._trace_lock = threading.RLock()
        self._trace_sequence = 0
        self._prompt_traces: List[PromptCallTrace] = []
        self._previous_prompt: Dict[str, str] = {}

    # ── Chat ────────────────────────────────────────────────────────────────

    def chat(self, messages: List[Message], system_prompt: str = "") -> str:
        """OpenAI 兼容 /chat/completions 调用。"""
        trace = self._begin_prompt_trace(system_prompt, messages, stream=False)
        if not self.cfg.is_real_llm():
            return self._mock(messages)
        try:
            return self._call_chat(system_prompt, messages, trace=trace)
        except Exception as e:
            self._mark_trace_error(trace, e, fallback=True)
            logger.error("LLM API 调用失败: %s，回退到 Mock", e)
            return self._mock(messages)

    def chat_context(self, ctx, system_prompt: str, messages: List[Message]) -> str:
        """兼容主分支 Go 版 ChatContext。"""
        if getattr(ctx, "cancelled", False):
            return "[已中断]"
        return self.chat(messages, system_prompt=system_prompt)

    def chat_stream_context(
        self,
        ctx,
        system_prompt: str,
        messages: List[Message],
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """流式对话调用，对齐 main 分支 Go ChatStreamContext。

        - mock 模式按字符 sleep 0.02s 推送，模拟流式体感。
        - 真实 API 走 OpenAI 兼容 SSE：``data: {json}\\n``，``data: [DONE]`` 终止。
        - ``ctx`` 提供 ``is_cancelled()`` 时通过关闭 ``requests.Session`` 触发 ``iter_lines`` 异常。
        - 异常时已发出的 token 不会回滚；返回累积的 full_text（失败回退到同步 chat）。
        """
        is_cancelled = getattr(ctx, "is_cancelled", None)
        trace = self._begin_prompt_trace(system_prompt, messages, stream=True)

        def _cancelled() -> bool:
            try:
                return bool(is_cancelled and is_cancelled())
            except Exception:
                return False

        if not self.cfg.is_real_llm():
            reply = self._mock(messages)
            sent = []
            for ch in reply:
                if _cancelled():
                    return "".join(sent)
                if on_token:
                    try:
                        on_token(ch)
                    except Exception as e:
                        logger.warning("on_token 回调异常: %s", e)
                sent.append(ch)
                time.sleep(0.02)
            return reply

        try:
            return self._call_chat_stream(
                ctx, system_prompt, messages, on_token, trace=trace
            )
        except Exception as e:
            if _cancelled():
                self._mark_trace_error(trace, e)
                return "[已中断]"
            logger.warning("LLM 流式调用失败: %s，回退到同步", e)
            self._mark_trace_error(trace, e, fallback=True)
            try:
                return self._call_chat(system_prompt, messages)
            except Exception as e2:
                logger.error("同步回退仍失败: %s", e2)
                return self._mock(messages)

    def _call_chat_stream(
        self,
        ctx,
        system_prompt: str,
        messages: List[Message],
        on_token: Optional[Callable[[str], None]],
        trace: Optional[PromptCallTrace] = None,
    ) -> str:
        if trace is None:
            trace = self._begin_prompt_trace(system_prompt, messages, stream=True)
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend({"role": m.role, "content": m.content} for m in messages)

        payload = {
            "model": self.cfg.llm_model,
            "messages": msgs,
            "temperature": self.cfg.temperature,
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.llm_api_key}",
            "Accept": "text/event-stream",
        }

        session = requests.Session()
        # ctx 透传：cancel 时 close session 触发 iter_lines 异常
        unbind = self._bind_session_to_ctx(ctx, session)

        full_parts: List[str] = []
        try:
            resp = session.post(
                self.cfg.llm_api_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=(10, 120),
            )
            if resp.status_code != 200:
                body = ""
                try:
                    body = resp.text
                except Exception:
                    pass
                raise RuntimeError(f"API 返回错误状态 {resp.status_code}, body: {body}")

            for raw in resp.iter_lines(decode_unicode=True):
                if raw is None:
                    continue
                line = raw.strip() if isinstance(raw, str) else raw
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                if isinstance(chunk, dict) and chunk.get("error"):
                    err = chunk["error"]
                    msg = err.get("message") if isinstance(err, dict) else str(err)
                    raise RuntimeError(f"API 流式错误: {msg}")
                self._apply_usage(
                    trace, chunk.get("usage") if isinstance(chunk, dict) else None
                )
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content") or ""
                if not content:
                    continue
                full_parts.append(content)
                if on_token:
                    try:
                        on_token(content)
                    except Exception as e:
                        logger.warning("on_token 回调异常: %s", e)
            return "".join(full_parts)
        finally:
            try:
                unbind()
            except Exception:
                pass
            try:
                session.close()
            except Exception:
                pass

    @staticmethod
    def _bind_session_to_ctx(ctx, session: "requests.Session"):
        """把 ctx 的 cancel 信号绑到 session.close() 上，返回解绑函数。

        ctx 可暴露 ``register_cancel(callback)``（推荐）或后台轮询
        ``is_cancelled()``；都不存在时返回 no-op 解绑。
        """
        if ctx is None:
            return lambda: None

        # 优先使用 register_cancel hook
        register = getattr(ctx, "register_cancel", None)
        if callable(register):
            try:
                handle = register(lambda: _safe_close(session))
                if callable(handle):
                    return handle
                return lambda: None
            except Exception:
                pass

        is_cancelled = getattr(ctx, "is_cancelled", None)
        if not callable(is_cancelled):
            return lambda: None

        stop_evt = threading.Event()

        def _watch():
            while not stop_evt.is_set():
                try:
                    if is_cancelled():
                        _safe_close(session)
                        return
                except Exception:
                    return
                if stop_evt.wait(0.1):
                    return

        t = threading.Thread(target=_watch, name="llm-stream-cancel", daemon=True)
        t.start()
        return lambda: stop_evt.set()

    def _call_chat(
        self,
        system_prompt: str,
        messages: List[Message],
        trace: Optional[PromptCallTrace] = None,
    ) -> str:
        if trace is None:
            trace = self._begin_prompt_trace(system_prompt, messages, stream=False)
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend({"role": m.role, "content": m.content} for m in messages)

        payload = {
            "model": self.cfg.llm_model,
            "messages": msgs,
            "temperature": self.cfg.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.llm_api_key}",
        }
        resp = requests.post(self.cfg.llm_api_url, headers=headers, json=payload, timeout=self._timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"API 返回错误状态 {resp.status_code}, body: {resp.text}")
        data = resp.json()
        self._apply_usage(trace, data.get("usage"))
        if data.get("error"):
            raise RuntimeError(f"API 错误: {data['error'].get('message')}")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"API 返回空结果, body: {resp.text}")
        return choices[0].get("message", {}).get("content", "")

    # ── Embedding ───────────────────────────────────────────────────────────

    def embed(self, text: str) -> List[float]:
        """文本向量化；与 Go 主分支一致：失败时抛错，由调用方决定是否降级。"""
        if not self.cfg.is_real_embedding():
            raise RuntimeError("embedding API 未配置")
        return self._call_embed(text)

    def _call_embed(self, text: str) -> List[float]:
        api_url = self.cfg.embedding_api_url
        is_multimodal = "/embeddings/multimodal" in api_url

        # 火山方舟多模态 embedding 端点：input 为结构化数组，data 为单对象
        if is_multimodal:
            input_payload = [{"type": "text", "text": text}]
        else:
            input_payload = text

        payload = {"model": self.cfg.embedding_model, "input": input_payload}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.cfg.embedding_api_key}",
        }
        resp = requests.post(api_url, headers=headers, json=payload, timeout=self._timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"embedding API 返回错误状态 {resp.status_code}, body: {resp.text}")
        result = resp.json()
        if result.get("error"):
            raise RuntimeError(f"embedding API 错误: {result['error'].get('message')}")

        if is_multimodal:
            embedding = (result.get("data") or {}).get("embedding") or []
        else:
            data_list = result.get("data") or []
            if not data_list:
                raise RuntimeError("embedding 返回空结果")
            embedding = data_list[0].get("embedding") or []

        if not embedding:
            raise RuntimeError("embedding 返回空向量")
        return embedding

    # ── Preference Extraction ──────────────────────────────────────────────

    def extract_preferences(self, msg: str) -> Dict[str, str]:
        """对齐 main 分支：优先用 LLM 抽取偏好 JSON，失败时规则兜底。"""
        if not msg:
            return {}
        if not self.cfg.is_real_llm():
            return _extract_rule_based(msg)

        prompt = f"消息：{msg}"
        try:
            raw = self._call_chat(
                PREFERENCE_EXTRACT_SYSTEM_PROMPT,
                [Message(role="user", content=prompt)],
            )
        except Exception:
            return _extract_rule_based(msg)

        raw = raw.strip()
        for prefix in ("```json", "```"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):].strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except Exception:
            pass
        return _extract_rule_based(msg)

    # ── Prompt/cache telemetry ────────────────────────────────────────────

    def prompt_trace_count(self) -> int:
        with self._trace_lock:
            return self._trace_sequence

    def prompt_traces_since(self, sequence: int) -> List[Dict[str, Any]]:
        state = current_request.get()
        with self._trace_lock:
            return [
                asdict(trace)
                for trace in self._prompt_traces
                if trace.sequence > int(sequence or 0)
                and (state is None or trace.request_id == state.request_id)
            ]

    def prompt_traces(self) -> List[Dict[str, Any]]:
        return self.prompt_traces_since(0)

    def _begin_prompt_trace(
        self,
        system_prompt: str,
        messages: List[Message],
        *,
        stream: bool,
    ) -> PromptCallTrace:
        purpose, version = prompt_identity(system_prompt)
        stable_prefix = stable_prompt_prefix(system_prompt)
        wire_messages: List[Dict[str, str]] = []
        if system_prompt:
            wire_messages.append({"role": "system", "content": system_prompt})
        wire_messages.extend(
            {"role": message.role, "content": message.content}
            for message in messages
        )
        serialized = json.dumps(
            wire_messages,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        scope = f"{purpose}@{version}"
        state = current_request.get()
        if state is not None:
            scope = f"{state.session_id}:{scope}"
        with self._trace_lock:
            previous = self._previous_prompt.get(scope, "")
            common = _common_prefix_length(previous, serialized) if previous else 0
            self._previous_prompt[scope] = serialized
            self._trace_sequence += 1
            trace = PromptCallTrace(
                sequence=self._trace_sequence,
                purpose=purpose,
                prompt_version=version,
                stable_prefix_hash=hashlib.sha256(
                    stable_prefix.encode("utf-8")
                ).hexdigest()[:16],
                stable_prefix_chars=len(stable_prefix),
                prompt_chars=len(serialized),
                common_prefix_chars=common,
                stream=stream,
                request_id=state.request_id if state is not None else "",
                session_id=state.session_id if state is not None else "",
            )
            self._prompt_traces.append(trace)
            if len(self._prompt_traces) > 200:
                self._prompt_traces = self._prompt_traces[-200:]
            return trace

    def _apply_usage(
        self, trace: Optional[PromptCallTrace], usage: Optional[Dict[str, Any]]
    ) -> None:
        if trace is None or not isinstance(usage, dict):
            return
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        if cached is None:
            cached = usage.get("cached_input_tokens", usage.get("cached_tokens"))
        with self._trace_lock:
            trace.input_tokens = _optional_int(
                usage.get("prompt_tokens", usage.get("input_tokens"))
            )
            trace.output_tokens = _optional_int(
                usage.get("completion_tokens", usage.get("output_tokens"))
            )
            trace.cached_input_tokens = _optional_int(cached)

    def _mark_trace_error(
        self,
        trace: Optional[PromptCallTrace],
        error: Exception,
        *,
        fallback: bool = False,
    ) -> None:
        if trace is None:
            return
        with self._trace_lock:
            trace.error_type = type(error).__name__
            trace.fallback = fallback

    # ── Mock ────────────────────────────────────────────────────────────────

    def _mock(self, messages: List[Message]) -> str:
        user_query = ""
        for m in messages:
            if m.role == "user":
                user_query = m.content
        q = user_query.lower()
        for key, response in self._mock_responses.items():
            if key in q:
                return response
        return f"收到：「{user_query}」——这是模拟 LLM 回复，接入真实 API 后会更智能。"


def _extract_rule_based(msg: str) -> Dict[str, str]:
    """规则兜底，与 Go 版 extractRuleBased 一致。"""
    result: Dict[str, str] = {}
    if "我喜欢" in msg:
        parts = msg.split("喜欢", 1)
        if len(parts) == 2 and parts[1].strip():
            result["喜好"] = parts[1].strip()
    elif "我爱" in msg:
        parts = msg.split("爱", 1)
        if len(parts) == 2 and parts[1].strip():
            result["喜好"] = parts[1].strip()
    if "我叫" in msg:
        parts = msg.split("叫", 1)
        if len(parts) == 2 and parts[1].strip():
            result["姓名"] = parts[1].strip()
    return result


def _safe_close(session: "requests.Session") -> None:
    try:
        session.close()
    except Exception:
        pass


def _common_prefix_length(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    index = 0
    while index < limit and a[index] == b[index]:
        index += 1
    return index


def _optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
