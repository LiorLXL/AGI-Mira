# memory_writer — 从用户明确自述中异步提取可信记忆
#
# 对应 main 分支 internal/application/chat/mem_writer.go：
#   - assistant 回复不是记忆来源；extract_memory_from_reply 仅保留兼容 no-op。
#   - 用户明确自述先经安全检查和 key 规范化，再分流 Preference / LTM。
#   - classify_memory_content：4 条规则 (identity/preference/tool_failure/policy)。
#   - llm_classify_memory：7 类 6 槽 LLM 兜底。
#   - sync_consolidation_to_db：把 ConsolidationResult 落到 PG（批删 + 逐条 update）。
#
# Python 在 Go 的 goroutine + channel 基础上额外提供 AsyncMemoryWriter：
# 后台线程 + queue.Queue 串行化所有记忆写入，避免 PG/Milvus 并发竞争。
import json
import logging
import queue
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from internal.llm.llm import Message
from internal.memory.preference import normalize_preference_key
from internal.promptctx.prompts import (
    MEMORY_CLASSIFY_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)


def _publish_event(agent, event_type: str, payload: Dict[str, Any]) -> None:
    try:
        repo = getattr(getattr(agent, "inf", None), "repo", None) or getattr(agent, "repo", None)
        events = getattr(repo, "events", None) if repo is not None else None
        if events is not None and hasattr(events, "publish"):
            events.publish(event_type, json.dumps(payload, ensure_ascii=False))
    except Exception as e:
        logger.warning("⚠️  publish_event(%s) 失败: %s", event_type, e)


# ── 异步记忆写入器 ──────────────────────────────────────────────────────────

class AsyncMemoryWriter:
    """后台线程串行化记忆写入（对应 Go 的 goroutine + channel 模型）。

    使用 queue.Queue 排队写任务，单 worker 线程消费，避免 ltm/preference 同时
    被多线程改写。stop() 触发优雅退出。
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._stopped = threading.Event()
        self._worker = threading.Thread(target=self._run, name="memory-writer", daemon=True)
        self._worker.start()

    def submit(self, fn):
        """提交一个无参可调用，最终在 worker 线程执行。"""
        if self._stopped.is_set():
            return
        try:
            self._queue.put_nowait(fn)
        except Exception as e:
            logger.warning("⚠️  memory-writer 提交失败: %s", e)

    def stop(self):
        self._stopped.set()
        try:
            self._queue.put_nowait(None)  # 唤醒 worker
        except Exception:
            pass

    def _run(self):
        while not self._stopped.is_set():
            try:
                fn = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if fn is None:
                break
            try:
                fn()
            except Exception as e:
                logger.warning("⚠️  memory-writer 任务异常: %s", e)


# ── 公共工具 ───────────────────────────────────────────────────────────────


@dataclass
class MemoryInspection:
    risk: str = "safe"
    reason: str = ""
    matched: str = ""

    @property
    def safe(self) -> bool:
        return self.risk == "safe"


_PII_PATTERNS = [
    ("password_keyword", re.compile(r"(密\s*码|password|passwd|passphrase)\s*(是|为|=|:)\s*\S{3,}", re.I)),
    ("api_key", re.compile(r"(api[\s_\-]?key|access[\s_\-]?key|secret[\s_\-]?key)\s*(是|为|=|:)\s*\S{6,}", re.I)),
    ("token", re.compile(r"(bearer|jwt|access[\s_\-]?token|refresh[\s_\-]?token)\s*(是|为|=|:)?\s*[\w\-\.]{20,}", re.I)),
    ("private_key_block", re.compile(r"-----BEGIN\s+(RSA|OPENSSH|DSA|EC|PRIVATE)\s+PRIVATE\s+KEY-----", re.I)),
    ("id_card_cn", re.compile(r"\b\d{17}[\dXx]\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("aws_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{36,255}")),
]


_INJECTION_PATTERNS = [
    ("ignore_previous", re.compile(r"(忽略|无视|disregard|ignore)\s*(之前|前面|所有|previous|all\s+prior|above)\s*(指令|内容|规则|instructions?|rules?)?", re.I)),
    ("role_override_zh", re.compile(r"你\s*(现在|从现在起|从此|以后)\s*(是|扮演|作为|当)")),
    ("role_override_en", re.compile(r"you\s+are\s+now\s+(a|an|the)\s+", re.I)),
    ("system_role_inject", re.compile(r"^\s*(system|assistant|user)\s*[:：]\s*", re.I)),
    ("jailbreak_prompt", re.compile(r"(DAN|do\s+anything\s+now|developer\s+mode|越狱)", re.I)),
    ("persistent_command", re.compile(r"(永远|从今以后|每次|总是|always|forever|from\s+now\s+on)\s*(回复|回答|说|拒绝|reply|answer|say|refuse)", re.I)),
    ("memory_injection", re.compile(r"(请\s*)?(记住|牢记|永远记住|remember\s+(this|that|always))[：:、，,]", re.I)),
]


_EPHEMERAL_PATTERNS = [
    ("now_words", re.compile(r"(今天|今晚|刚才|这次|此刻|现在|马上|稍后|just\s+now|right\s+now|today|tonight)", re.I)),
    ("weather_smalltalk", re.compile(r"(天气|温度|气温).{0,10}(怎么样|如何|不错|很好|很差)")),
]


def _match_any(content: str, patterns) -> MemoryInspection:
    for name, pattern in patterns:
        match = pattern.search(content)
        if not match:
            continue
        snippet = match.group(0)
        if len(snippet) > 40:
            snippet = snippet[:40] + "..."
        return MemoryInspection(reason=name, matched=snippet)
    return MemoryInspection()


def inspect_memory_content(content: str) -> MemoryInspection:
    text = str(content or "").strip()
    if not text:
        return MemoryInspection()
    hit = _match_any(text, _PII_PATTERNS)
    if hit.reason:
        hit.risk = "pii"
        return hit
    hit = _match_any(text, _INJECTION_PATTERNS)
    if hit.reason:
        hit.risk = "injection"
        return hit
    hit = _match_any(text, _EPHEMERAL_PATTERNS)
    if hit.reason:
        hit.risk = "ephemeral"
        return hit
    return MemoryInspection()


def inspect_kv_pair(key: str, value: str) -> MemoryInspection:
    for text in (f"{key}={value}", str(key or ""), str(value or "")):
        hit = inspect_memory_content(text)
        if not hit.safe:
            return hit
    return MemoryInspection()


def _strip_code_fence(raw: str) -> str:
    raw = (raw or "").strip()
    raw = re.sub(r"^```json", "", raw)
    raw = re.sub(r"^```", "", raw)
    raw = re.sub(r"```$", "", raw)
    return raw.strip()


def _embed(agent, content: str) -> Optional[List[float]]:
    """优先复用 LongTerm._embed_fn（避免重复构造 embedder）。"""
    fn = getattr(getattr(agent, "ltm", None), "_embed_fn", None)
    if fn is None:
        fn = getattr(agent, "_embed_fn", None)
    if fn is None:
        return None
    try:
        return fn(content)
    except Exception as e:
        logger.warning("⚠️  embed 失败: %s", e)
        return None


# ── 用户消息 → 可信记忆 ────────────────────────────────────────────────────


@dataclass
class UserMemoryUpdate:
    preferences: Dict[str, str]
    long_term: List[str]
    rejected: Dict[str, str]


_EXPLICIT_SELF_PATTERNS = [
    re.compile(r"(?:^|[，。；,;\s])我(?:叫|是|住在|来自|喜欢|爱|偏好|讨厌|不喜欢|从事|有|通常|经常)"),
    re.compile(r"我的(?:姓名|名字|城市|居住地|所在地|时区|语言|国家|职业|工作|爱好|兴趣|偏好|回答风格|回复风格)\s*(?:是|为|叫|[:：])"),
    re.compile(r"请记住我(?:叫|的|喜欢|爱|住在|来自|是)"),
    re.compile(r"(?:以后)?(?:请)?(?:用|使用).{0,16}(?:回答|回复)"),
    re.compile(r"我在.{1,30}(?:工作|上班|居住|生活)"),
]

_QUESTION_HINT = re.compile(r"(?:什么|怎么|如何|哪个|哪里|是否|能否|可以吗|吗|呢)")
_NON_AUTHORITATIVE_SELF_REFERENCE = re.compile(
    r"(?:假设|如果|例如|比如|引用|他说|她说|别人说|对方说).{0,24}我"
)
_VALUE = r"([^\n,，。！？!?;；]{1,80})"


def is_explicit_user_memory_statement(text: str) -> bool:
    """Return True only for likely first-person stable statements/preferences."""
    value = str(text or "").strip()
    if not value or re.search(r"[?？]\s*$", value):
        return False
    if _QUESTION_HINT.search(value) and "请记住" not in value:
        return False
    if _NON_AUTHORITATIVE_SELF_REFERENCE.search(value):
        return False
    return any(pattern.search(value) for pattern in _EXPLICIT_SELF_PATTERNS)


def extract_explicit_user_facts(text: str) -> Dict[str, str]:
    """Fast deterministic extraction for common supported profile keys."""
    value = str(text or "").strip()
    if not is_explicit_user_memory_statement(value):
        return {}

    facts: Dict[str, str] = {}
    patterns = [
        (
            "姓名",
            re.compile(
                rf"(?:我叫|我的(?:姓名|名字)(?:是|为|叫)|请记住我的(?:姓名|名字)(?:是|为|叫)){_VALUE}"
            ),
        ),
        (
            "城市",
            re.compile(
                rf"(?:我住在|我的(?:城市|居住地|所在地)(?:是|为)|请记住我的城市(?:是|为)){_VALUE}"
            ),
        ),
        (
            "时区",
            re.compile(rf"(?:我的时区(?:是|为)|请记住我的时区(?:是|为)){_VALUE}"),
        ),
        (
            "职业",
            re.compile(rf"(?:我从事|我的(?:职业|工作|职位)(?:是|为)){_VALUE}"),
        ),
        (
            "国家",
            re.compile(rf"(?:我的(?:国家|国籍)(?:是|为)){_VALUE}"),
        ),
    ]
    for key, pattern in patterns:
        match = pattern.search(value)
        if match:
            facts[key] = match.group(1).strip()

    language = re.search(
        r"(?:我的(?:语言|偏好语言|回复语言)(?:是|为)|(?:以后)?(?:请)?(?:用|使用))\s*(中文|英文|英语|日文|日语|法文|法语)",
        value,
    )
    if language:
        facts["语言"] = language.group(1)

    style = re.search(
        r"(?:我(?:喜欢|偏好)|以后请|请)(简洁|详细|专业|口语化|分点|直接).{0,8}(?:回答|回复)",
        value,
    )
    if style:
        facts["回答风格"] = style.group(1)

    like = re.search(rf"我(?:喜欢|爱){_VALUE}", value)
    if like:
        liked = like.group(1).strip()
        if not re.search(r"(?:回答|回复)$", liked):
            facts["喜好"] = liked

    dislike = re.search(rf"我(?:不喜欢|讨厌){_VALUE}", value)
    if dislike:
        facts["禁忌"] = dislike.group(1).strip()

    return facts


def extract_memory_from_user(agent, user_input: str) -> UserMemoryUpdate:
    """Extract, validate and persist memory from one authoritative user turn."""
    update = UserMemoryUpdate(preferences={}, long_term=[], rejected={})
    text = str(user_input or "").strip()
    if not is_explicit_user_memory_statement(text):
        update.rejected["message"] = "not_explicit_self_statement"
        return update

    inspection = inspect_memory_content(text)
    if not inspection.safe:
        update.rejected["message"] = inspection.risk
        _publish_event(
            agent,
            "memory.user.rejected",
            {"reason": inspection.reason, "risk": inspection.risk},
        )
        return update

    candidates = extract_explicit_user_facts(text)
    extractor = getattr(getattr(agent, "llm", None), "extract_preferences", None)
    if callable(extractor):
        try:
            extracted = extractor(text) or {}
        except Exception as e:
            logger.warning("用户记忆抽取失败: %s", e)
            extracted = {}
        if isinstance(extracted, dict):
            for key, value in extracted.items():
                candidates.setdefault(str(key), value)

    _persist_user_candidates(agent, candidates, update, evidence_text=text)
    return update


def _persist_user_candidates(
    agent,
    candidates: Dict[str, Any],
    update: UserMemoryUpdate,
    evidence_text: str = "",
) -> None:
    for raw_key in sorted(candidates, key=lambda item: str(item)):
        key = str(raw_key or "").strip()
        value = _candidate_value(candidates[raw_key])
        if not key or value is None:
            continue

        if evidence_text and value.lower() not in evidence_text.lower():
            update.rejected[key] = "ungrounded_candidate"
            continue

        inspection = inspect_kv_pair(key, value)
        if not inspection.safe:
            update.rejected[key] = inspection.risk
            _publish_event(
                agent,
                "memory.user.rejected",
                {"key": key, "reason": inspection.reason, "risk": inspection.risk},
            )
            continue

        canonical_key = normalize_preference_key(key)
        if canonical_key:
            if _set_preference_if_changed(agent, canonical_key, value):
                update.preferences[canonical_key] = value
                _publish_event(
                    agent,
                    "memory.preference.saved",
                    {"key": canonical_key, "source": "user"},
                )
            continue

        content = f"用户{key}: {value}"
        content_inspection = inspect_memory_content(content)
        if not content_inspection.safe:
            update.rejected[key] = content_inspection.risk
            continue

        if any(
            marker in key.lower()
            for marker in ("规则", "指令", "prompt", "system", "工具结果", "tool_result")
        ):
            update.rejected[key] = "unsupported_or_untrusted_category"
            continue
        # Supported profile categories were handled above. Any remaining
        # first-person, safety-checked attribute is a classified user fact.
        category = "fact"
        stable_tags = ["source:user", f"memory-key:{key}"]
        try:
            inserted = _store_classified_with_graph(
                agent,
                content,
                0.7,
                _embed(agent, content),
                category,
                stable_tags,
                "recall_memory",
            )
        except Exception as e:
            logger.warning("用户长期记忆写入失败: %s", e)
            update.rejected[key] = "store_failed"
            continue
        if inserted:
            update.long_term.append(content)
            _publish_event(
                agent,
                "memory.longterm.user_saved",
                {"category": category, "source": "user"},
            )


def _set_preference_if_changed(agent, key: str, value: str) -> bool:
    preference = getattr(agent, "preference", None)
    if preference is None:
        return False
    getter = getattr(preference, "get", None)
    if callable(getter):
        try:
            if str(getter(key, "")) == value:
                return False
        except Exception:
            pass
    setter = getattr(preference, "set", None)
    if not callable(setter):
        return False
    setter(key, value)
    return True


def _candidate_value(value: Any) -> Optional[str]:
    if isinstance(value, (str, int, float, bool)):
        normalized = str(value).strip()
    elif isinstance(value, list) and all(
        isinstance(item, (str, int, float, bool)) for item in value
    ):
        normalized = "、".join(str(item).strip() for item in value if str(item).strip())
    else:
        return None
    if not normalized or len(normalized) > 200:
        return None
    return normalized


# ── Assistant compatibility boundary ───────────────────────────────────────

def extract_memory_from_reply(agent, answer: str):
    """Deprecated no-op: assistant output is never an authoritative memory source."""
    return None


def _store_classified_with_graph(
    agent,
    content: str,
    importance: float,
    emb: Optional[List[float]],
    category: str,
    tags: List[str],
    slot_hint: str,
) -> bool:
    """LongTerm assigns the PG ID before graph publication; no last-item rewrite."""
    return agent.ltm.store_classified(
        content, importance, emb, category or "general",
        list(tags or []), slot_hint or "",
    )


def classify_memory_content(key: str, value: str) -> Tuple[str, List[str], str]:
    """用规则快速分类；返回空字符串表示规则未命中，由 LLM 兜底。"""
    combined = f"{key}{value}"
    if _contains_any(combined, "叫", "名字", "姓名", "是我", "我是"):
        return "identity", ["name"], "profile"
    if _contains_any(combined, "喜欢", "偏好", "习惯", "爱好", "讨厌", "不喜欢"):
        return "preference", ["preference"], "profile"
    if _contains_any(combined, "工具", "失败", "错误", "报错", "异常"):
        return "tool_failure", ["tool", "error"], "tool_state"
    if _contains_any(combined, "禁止", "不要", "不能", "必须", "强制"):
        return "policy", ["constraint"], "constraints"
    return "", [], ""


def _contains_any(s: str, *subs: str) -> bool:
    return any(sub in s for sub in subs)


def llm_classify_memory(agent, content: str) -> Tuple[str, List[str], str]:
    """LLM 兜底分类（7 类 6 槽）；失败时回退到 'general'。"""
    if not agent.cfg.is_real_llm():
        return "general", [], ""

    prompt = f"记忆内容：{content}"
    try:
        raw = agent.llm.chat(
            [Message(role="user", content=prompt)],
            system_prompt=MEMORY_CLASSIFY_SYSTEM_PROMPT,
        )
    except Exception:
        return "general", [], ""
    raw = _strip_code_fence(raw)
    try:
        result = json.loads(raw)
    except Exception:
        return "general", [], ""
    if not isinstance(result, dict):
        return "general", [], ""
    cat = result.get("category") or "general"
    return cat, list(result.get("tags") or []), result.get("slot_hint") or ""


# ── consolidate 后落库 ─────────────────────────────────────────────────────


def sync_consolidation_to_db(agent, result) -> bool:
    """Confirm PG commit, then synchronize graph projections. Return success."""
    if result is None:
        return True
    repo = getattr(getattr(agent, "inf", None), "repo", None) or getattr(agent, "repo", None)
    ltm_repo = getattr(repo, "ltm", None) if repo is not None else None
    if ltm_repo is None:
        return False

    apply_result = getattr(ltm_repo, "apply_consolidation", None)
    if callable(apply_result):
        if not getattr(result, "persisted", False) and not apply_result(result):
            return False
        graph = getattr(agent, "graph_memory", None)
        if graph is not None:
            for memory_id in result.delete_from_db:
                graph.delete_from_graph(memory_id)
            for item in result.update_in_db:
                graph.update_node(item)
        return True

    delete_ids = list(getattr(result, "delete_from_db", []) or [])
    if delete_ids:
        try:
            ltm_repo.delete(delete_ids)
            _publish_event(agent, "memory.consolidate.delete", {"ids": delete_ids, "count": len(delete_ids)})
            logger.info("🧹 记忆合并：删除 %d 条 (ids=%s)", len(delete_ids), delete_ids)
        except Exception as e:
            logger.warning("⚠️  sync_consolidation_to_db delete 失败: %s", e)

    for item in getattr(result, "update_in_db", []) or []:
        item_id = getattr(item, "id", None)
        if item_id is None or item_id <= 0:
            continue
        try:
            emb_json = json.dumps(item.embedding) if item.embedding else "null"
            ltm_repo.update(int(item_id), item.content, float(item.importance), emb_json)
            _publish_event(agent, "memory.consolidate.update", {
                "id": int(item_id),
                "importance": float(item.importance),
                "content": item.content,
            })
            logger.info("🔗 记忆合并：更新 id=%d", int(item_id))
        except Exception as e:
            logger.warning("⚠️  sync_consolidation_to_db update id=%s 失败: %s", item_id, e)

    return True


# ── 请求入口：同步常用偏好 + 异步完整抽取 ──────────────────────────────────

def async_update_memory(agent, user_input: str, resp: Any) -> None:
    """Persist safe rule hits now, then run the full user extractor async."""
    text = str(user_input or "").strip()
    if not is_explicit_user_memory_statement(text):
        return
    inspection = inspect_memory_content(text)
    if not inspection.safe:
        _publish_event(
            agent,
            "memory.user.rejected",
            {"reason": inspection.reason, "risk": inspection.risk},
        )
        return

    quick = extract_explicit_user_facts(text)
    quick_update = UserMemoryUpdate(preferences={}, long_term=[], rejected={})
    _persist_user_candidates(agent, quick, quick_update, evidence_text=text)
    if quick_update.preferences and hasattr(resp, "extracted_info"):
        resp.extracted_info = "已记住：" + ", ".join(
            f"{key}={value}" for key, value in sorted(quick_update.preferences.items())
        )

    def _bg():
        try:
            extract_memory_from_user(agent, text)
        except Exception as e:
            logger.warning("异步更新用户记忆失败: %s", e)

    writer = getattr(agent, "memory_writer", None)
    if writer is not None:
        writer.submit(_bg)
    else:
        threading.Thread(target=_bg, name="memory-fallback", daemon=True).start()


def maybe_consolidate_memory(agent):
    """达到触发阈值时合并/去重/衰减/淘汰长期记忆，并把结果同步到 PG。

    与 main 分支 finalize 的 consolidate 分支对齐：
    有 graph_memory 时走 ``graph_aware_consolidate``（保护高中心度节点 + 同步删 Neo4j），
    否则走纯内存 ``ltm.consolidate``。
    """
    pending = getattr(agent, "_pending_memory_consolidation", None)
    if pending is not None:
        if sync_consolidation_to_db(agent, pending):
            agent._pending_memory_consolidation = None
        return
    try:
        if not agent.ltm.need_consolidation():
            return
        gm = getattr(agent, "graph_memory", None)
        if gm is not None and hasattr(gm, "graph_aware_consolidate"):
            result = gm.graph_aware_consolidate()
        else:
            result = agent.ltm.consolidate()
    except Exception as e:
        logger.warning("记忆合并失败: %s", e)
        return
    try:
        agent._pending_memory_consolidation = result
        if sync_consolidation_to_db(agent, result):
            agent._pending_memory_consolidation = None
    except Exception as e:
        logger.warning("记忆合并落库失败: %s", e)
