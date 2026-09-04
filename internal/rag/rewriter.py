import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional

from internal.promptctx.prompts import rag_rewrite_system_prompt

logger = logging.getLogger(__name__)

GenerateFn = Callable[[str, str], str]


@dataclass
class HistoryMessage:
    role: str
    content: str


class LLMRewriter:
    """用 LLM 做 history-aware multi-query 改写，失败时回退原 query。"""

    def __init__(
        self,
        generate_fn: Optional[GenerateFn],
        num_queries: int = 3,
        strict: bool = False,
    ):
        self.generate_fn = generate_fn
        self.num_queries = num_queries if num_queries > 0 else 3
        self.strict = strict

    def rewrite(self, query: str, history: List[HistoryMessage]) -> List[str]:
        query = (query or "").strip()
        if not query:
            return []
        if self.generate_fn is None or self.num_queries <= 1:
            return [query]

        user_msg = self._build_user_msg(query, history)
        system_prompt = rag_rewrite_system_prompt(self.num_queries)
        raw = ""
        try:
            raw = self.generate_fn(system_prompt, user_msg)
            queries = _parse_queries(raw)
        except Exception as e:
            if self.strict:
                preview = (raw or "")[:500].replace("\n", "\\n")
                raise RuntimeError(
                    f"Query rewrite failed: {e}; raw_preview={preview!r}"
                ) from e
            logger.warning("⚠️  Query rewrite 失败，回退原查询: %s", e)
            return [query]
        if not queries:
            if self.strict:
                raise RuntimeError("Query rewrite returned an empty queries array")
            return [query]
        return _dedup_keep_order(queries + [query])[:self.num_queries]

    def _build_user_msg(self, query: str, history: List[HistoryMessage]) -> str:
        lines: List[str] = ["最近对话历史："]
        if history:
            recent = history[-6:]
            for msg in recent:
                role = msg.role or "user"
                content = (msg.content or "").strip()
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"[{role}] {content}")
        else:
            lines.append("（无历史，直接改写当前问题）")
        lines.append("")
        lines.append(f"当前问题：{query}")
        return "\n".join(lines)


def _parse_queries(raw: str) -> List[str]:
    raw = _strip_json_fence(raw)
    data = json.loads(raw)
    queries = data.get("queries", []) if isinstance(data, dict) else []
    return [str(q).strip() for q in queries if str(q).strip()]


def _strip_json_fence(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"^```json\s*", "", raw)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def _dedup_keep_order(values: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        key = value.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
