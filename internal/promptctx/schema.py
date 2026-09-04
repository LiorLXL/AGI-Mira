"""promptctx.schema — RuntimeContextSchema 与四个内置 Schema 定义。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .slot import (
    Slot,
    SlotConstraints,
    SlotFilter,
    SlotKind,
    SlotPlanner,
    SlotProfile,
    SlotRecall,
    SlotTaskMem,
    SlotToolState,
)


@dataclass
class RuntimeContextSchema:
    """定义某个 Mode 下需要装配的认知槽位与顺序。
    槽位顺序即 Render 时的输出顺序。
    """

    mode: str
    slots: List[Slot] = field(default_factory=list)


# 全局总预算是字符数，不等同于模型 token。
DEFAULT_GLOBAL_CHAR_BUDGET = 2400
# 向后兼容旧导入名。
DEFAULT_GLOBAL_TOKEN_BUDGET = DEFAULT_GLOBAL_CHAR_BUDGET


# ChatSchema 普通对话：偏好 + 兜底召回；不需要 Planner / TaskMem / ToolState
CHAT_SCHEMA = RuntimeContextSchema(
    mode="chat",
    slots=[
        Slot(
            kind=SlotProfile,
            required=False,
            filter=SlotFilter(
                categories=["identity", "preference"],
                char_budget=300,
                top_k=10,
            ),
        ),
        Slot(
            kind=SlotRecall,
            required=False,
            filter=SlotFilter(
                categories=["episodic", "fact", "general"],
                top_k=3,
                min_score=0.4,
                char_budget=400,
            ),
        ),
    ],
)


# Tool result generation: profile + current tool result (the result is supplied
# by the call site, not duplicated in ToolState).
TOOL_SCHEMA = RuntimeContextSchema(
    mode="tool",
    slots=[
        Slot(
            kind=SlotProfile,
            required=False,
            filter=SlotFilter(
                categories=["identity", "preference"],
                char_budget=250,
                top_k=8,
            ),
        ),
    ],
)


# ReactSchema 多步推理：装配全部 5 类槽位
REACT_SCHEMA = RuntimeContextSchema(
    mode="react",
    slots=[
        Slot(
            kind=SlotConstraints,
            required=True,
            filter=SlotFilter(char_budget=280),
        ),
        Slot(
            kind=SlotPlanner,
            required=True,
            filter=SlotFilter(char_budget=300),
        ),
        Slot(
            kind=SlotTaskMem,
            required=False,
            filter=SlotFilter(char_budget=350, top_k=8, max_age_hours=24),
        ),
        Slot(
            kind=SlotToolState,
            required=True,
            filter=SlotFilter(char_budget=350, top_k=8),
        ),
        Slot(
            kind=SlotProfile,
            required=False,
            filter=SlotFilter(
                categories=["identity", "preference"],
                char_budget=250,
                top_k=6,
            ),
        ),
        Slot(
            kind=SlotRecall,
            required=False,
            filter=SlotFilter(
                categories=["episodic", "fact", "general", "tool_failure"],
                top_k=2,
                min_score=0.5,
                char_budget=200,
            ),
        ),
    ],
)


# RagSchema 知识库检索：弱化 Planner/TaskMem，保留 Profile/Constraints/Recall
RAG_SCHEMA = RuntimeContextSchema(
    mode="rag",
    slots=[
        Slot(
            kind=SlotConstraints,
            required=False,
            filter=SlotFilter(char_budget=200),
        ),
        Slot(
            kind=SlotProfile,
            required=False,
            filter=SlotFilter(
                categories=["identity", "preference"],
                char_budget=300,
                top_k=8,
            ),
        ),
        Slot(
            kind=SlotRecall,
            required=False,
            filter=SlotFilter(
                categories=["episodic", "fact", "general"],
                top_k=3,
                min_score=0.4,
                char_budget=400,
            ),
        ),
    ],
)


# Phase-specific schemas used by production call sites.  The legacy mode-only
# schemas remain available for compatibility with existing callers/tests.
REACT_PLAN_SCHEMA = RuntimeContextSchema(
    mode="react.plan",
    slots=[
        Slot(
            kind=SlotConstraints,
            required=True,
            filter=SlotFilter(char_budget=280),
        ),
        Slot(
            kind=SlotProfile,
            required=False,
            filter=SlotFilter(
                categories=["identity", "preference"],
                char_budget=250,
                top_k=6,
            ),
        ),
    ],
)


REACT_GENERATE_SCHEMA = RuntimeContextSchema(
    mode="react.generate",
    slots=[
        Slot(
            kind=SlotProfile,
            required=False,
            filter=SlotFilter(
                categories=["identity", "preference"],
                char_budget=250,
                top_k=6,
            ),
        ),
        Slot(
            kind=SlotTaskMem,
            required=False,
            filter=SlotFilter(char_budget=500, top_k=8, max_age_hours=24),
        ),
    ],
)


RAG_GENERATE_SCHEMA = RuntimeContextSchema(
    mode="rag.generate",
    slots=[
        Slot(
            kind=SlotProfile,
            required=False,
            filter=SlotFilter(
                categories=["identity", "preference"],
                char_budget=300,
                top_k=8,
            ),
        ),
    ],
)


def default_schemas() -> Dict[str, RuntimeContextSchema]:
    """返回 mode 兼容 Schema 与生产调用使用的 phase Schema。"""
    return {
        "chat": CHAT_SCHEMA,
        "tool": TOOL_SCHEMA,
        "react": REACT_SCHEMA,
        "rag": RAG_SCHEMA,
        "react.plan": REACT_PLAN_SCHEMA,
        "react.generate": REACT_GENERATE_SCHEMA,
        "rag.generate": RAG_GENERATE_SCHEMA,
    }


def slot_priority(kind: SlotKind) -> int:
    """slot_priority 决定全局预算超限时的裁剪优先级（数字越小越优先保留）。"""
    if kind == SlotConstraints:
        return 0
    if kind == SlotPlanner:
        return 1
    if kind == SlotTaskMem:
        return 2
    if kind == SlotToolState:
        return 3
    if kind == SlotProfile:
        return 4
    if kind == SlotRecall:
        return 5
    return 99
