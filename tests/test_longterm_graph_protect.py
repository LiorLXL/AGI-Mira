"""LongTerm.consolidate 的图中心度保护单元测试（Task 14）。

图中心度不再否决删除；PG 提交后才同步图，避免重启复活。
"""
import time
from types import SimpleNamespace
from typing import List

from internal.memory.memory import Item, LongTerm


class _LtmRepo:
    def load(self):
        return []

    def save(self, *args, **kwargs):
        return 0

    def update_classified(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass


class _Cfg:
    memory_consolidation_similarity = 0.85
    memory_consolidation_dedup = 0.95
    memory_consolidation_ttl_days = 30
    memory_consolidation_decay_rate = 1.0
    memory_consolidation_min_import = 0.1
    memory_consolidation_trigger = 5
    graph_protect_indegree = 3


class _StubGraph:
    """记录 filter_protected / delete_from_graph / update_node 调用。"""

    def __init__(self, protected_ids: List[int]):
        self._protected = list(protected_ids)
        self.filter_calls: List[List[int]] = []
        self.deleted: List[int] = []
        self.updated: List[int] = []

    def filter_protected(self, candidate_ids, threshold):
        self.filter_calls.append(list(candidate_ids))
        # 只返回与候选交集
        cand_set = set(candidate_ids)
        return [pid for pid in self._protected if pid in cand_set]

    def delete_from_graph(self, mem_id):
        self.deleted.append(mem_id)

    def update_node(self, item):
        self.updated.append(getattr(item, "id", None))


def _make_ltm():
    inf = SimpleNamespace(repo=SimpleNamespace(ltm=_LtmRepo()))
    return LongTerm(_Cfg(), inf)


def test_graph_degree_does_not_veto_pg_deletion():
    """All expired IDs must remain in the deletion set regardless of graph degree."""
    ltm = _make_ltm()
    now = time.time()
    ltm.items = [
        # 应淘汰（受保护）
        Item(
            content="hub",
            importance=0.05,
            embedding=[1.0, 0.0, 0.0],
            id=901,
            created_at=now - 60 * 86400.0,
            last_accessed=now,
        ),
        # 应淘汰（不保护）
        Item(
            content="leaf",
            importance=0.05,
            embedding=[0.0, 1.0, 0.0],
            id=902,
            created_at=now - 60 * 86400.0,
            last_accessed=now,
        ),
    ]
    graph = _StubGraph(protected_ids=[901])
    ltm.set_graph_memory(graph)

    result = ltm.consolidate()

    # 内存层：两条都被淘汰
    assert result.expired == 2
    assert ltm.items == []

    # PG 删除列表必须包含全部已淘汰条目
    assert result.delete_from_db == [901, 902]

    # 中心度不参与删除决策
    assert graph.filter_calls == []

    # PG 提交前不触发图删除
    assert graph.deleted == []  # Graph writes wait for PG commit.


def test_no_graph_memory_no_protection():
    """未挂载 graph_memory 时，原 delete_from_db 不变。"""
    ltm = _make_ltm()
    now = time.time()
    ltm.items = [
        Item(
            content="orphan",
            importance=0.05,
            embedding=[1.0, 0.0],
            id=910,
            created_at=now - 60 * 86400.0,
            last_accessed=now,
        ),
        Item(
            content="anchor",
            importance=0.9,
            embedding=[0.0, 1.0],
            id=911,
            created_at=now - 1 * 86400.0,
            last_accessed=now,
        ),
    ]

    result = ltm.consolidate()

    assert result.expired == 1
    assert result.delete_from_db == [910]
