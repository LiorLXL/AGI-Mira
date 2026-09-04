"""promptctx.source — Query 与 ContextSource 抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from .slot import ContextItem, Slot, SlotKind


@dataclass
class Query:
    """装配一次上下文时的输入快照。"""

    text: str = ""                                  # 用户当前输入
    embedding: List[float] = field(default_factory=list)  # 已计算的 query embedding（可为空）
    task_id: str = ""                               # 当前任务 ID（用于 Task Memory）
    mode: str = ""                                  # chat / tool / react / rag
    phase: str = ""                                 # generate / plan 等调用阶段
    session_id: str = ""                            # 当前会话 ID（后续会话隔离使用）


class ContextSource(ABC):
    """某类认知槽位的数据提供者。

    一个 source 可声明支持多个 SlotKind（例如 Profile source 同时填 Profile/Recall 都行）。
    """

    @abstractmethod
    def id(self) -> str:
        """返回 source 标识。"""

    @abstractmethod
    def supports(self, kind: SlotKind) -> bool:
        """判断是否支持指定 SlotKind。"""

    @abstractmethod
    def fetch(self, slot: Slot, q: Query) -> List[ContextItem]:
        """返回适合该槽位的候选 ContextItem。

        TopK 与字符预算由 ContextAssembler 统一执行；Source 可做查询下推优化。
        失败可抛异常，由 Assembler 记录 trace 并继续其他 Source。
        """
