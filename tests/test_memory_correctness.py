"""Feature 001 T301-T307: durable IDs, read-only recall and decay checkpoints."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from config.config import APIConfig
from internal.agent.agent import UnifiedAgent
from internal.agent.memory_writer import sync_consolidation_to_db
from internal.memory.memory import Item, LongTerm, RecallFilter
from internal.memory.graph_memory import GraphMemory
from internal.promptctx import (
    ContextAssembler, SourceRegistry, RecallSource, RuntimeContextSchema,
    Slot, SlotFilter, SlotRecall,
)
from internal.promptctx.prompts import CHAT_SYSTEM_PROMPT, compose_system_prompt
from internal.repo.longterm import PGRepo
from internal.memory.memory import ConsolidationResult


class DurableRepo:
    def __init__(self):
        self.rows = {}
        self.next_id = 42
        self.fail = False
        self.touches = []

    def load(self):
        return [deepcopy(it) for it in self.rows.values()]

    def save(self, content, importance, embedding, **fields):
        if self.fail:
            return -1
        import json
        memory_id = self.next_id
        self.next_id += 7
        self.rows[memory_id] = Item(
            id=memory_id, content=content, importance=importance,
            embedding=json.loads(embedding), **fields,
            last_decayed_at=fields["created_at"],
        )
        return memory_id

    def touch(self, ids, now):
        if self.fail:
            return False
        self.touches.append(list(ids))
        for memory_id in ids:
            self.rows[memory_id].last_accessed = now
        return True

    def update_classified(self, *args):
        return not self.fail

    def apply_consolidation(self, result):
        if self.fail:
            return False
        rows = deepcopy(self.rows)
        for item in result.update_in_db:
            rows[item.id] = deepcopy(item)
        for memory_id in result.delete_from_db:
            rows.pop(memory_id, None)
        self.rows = rows
        return True


def make_ltm(repo=None):
    repo = repo or DurableRepo()
    cfg = APIConfig()
    cfg.memory_consolidation_decay_rate = 0.9
    cfg.memory_consolidation_min_import = 0.01
    ltm = LongTerm(cfg, SimpleNamespace(repo=SimpleNamespace(ltm=repo)))
    return ltm, repo


@pytest.mark.parametrize("classified", [False, True])
def test_save_uses_pg_id_before_graph_and_preserves_gaps_on_restart(classified):
    ltm, repo = make_ltm()
    seen = []
    ltm.graph_memory = SimpleNamespace(
        add_to_graph=lambda item, **_: seen.append((item.id, item.id in repo.rows))
    )
    if classified:
        ltm.store_classified("cat", .7, None, "fact", ["source:user"], "recall_memory")
    else:
        ltm.add("cat")
    ltm.add("dog")
    assert seen == [(42, True), (49, True)]
    repo.rows.pop(42)
    restored, _ = make_ltm(repo)
    restored.load_from_storage()
    assert [it.id for it in restored.items] == [49]
    assert restored._next_id == 50
    assert restored.add("bird").id == 56


@pytest.mark.parametrize("result", [-1, 0, None, True])
@pytest.mark.parametrize("classified", [False, True])
def test_invalid_save_result_does_not_publish_memory(result, classified):
    ltm, repo = make_ltm()
    repo.save = lambda *args, **kwargs: result
    graph = []
    ltm.graph_memory = SimpleNamespace(add_to_graph=lambda *a, **k: graph.append(a))
    with pytest.raises(RuntimeError, match="not persisted"):
        if classified:
            ltm.store_classified("cat", .7, None, "fact", [], "")
        else:
            ltm.add("cat")
    assert ltm.items == []
    assert ltm._items_since_last == 0
    assert graph == []


def test_save_exception_and_dedup_failure_leave_state_unchanged():
    ltm, repo = make_ltm()
    ltm.store_classified("cat", .3, [1, 0], "fact", [], "")
    before = ltm.snapshot()
    repo.fail = True
    with pytest.raises(RuntimeError):
        ltm.store_classified("cat", .9, [1, 0], "fact", ["new"], "")
    assert ltm.snapshot() == before
    def fail(*args, **kwargs):
        raise OSError("offline")
    repo.save = fail
    with pytest.raises(OSError):
        ltm.add("dog")
    assert ltm.snapshot() == before


@pytest.mark.parametrize("embedding", [None, [], [1, 0, 0]])
def test_fallback_ranks_related_memory_without_touching_candidates(embedding):
    ltm, repo = make_ltm()
    ltm.items = [
        Item(id=1, content="weather sunshine", embedding=[1, 0], last_accessed=10),
        Item(id=2, content="coffee beans", embedding=None, last_accessed=20),
    ]
    hits = ltm.recall_by_filter("coffee beans", embedding, RecallFilter(top_k=1))
    assert [it.id for it in hits] == [2]
    assert [it.last_accessed for it in ltm.items] == [10, 20]
    ltm.set_embed_fn(lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
    assert [it.id for it in ltm.recall("coffee beans", 1)] == [2]
    assert ltm.recall("completely unrelated", 1) == []


def test_only_prompt_retained_items_are_touched_at_llm_boundary(monkeypatch):
    ltm, repo = make_ltm()
    for content in ["coffee", "coffee beans", "coffee beans roasted"]:
        ltm.store_classified(content, .7, None, "fact", [], "")
    for it in ltm.items:
        it.last_accessed = 10
        repo.rows[it.id].last_accessed = 10
    registry = SourceRegistry()
    registry.register(RecallSource(ltm))
    schema = RuntimeContextSchema(mode="chat", slots=[
        Slot(kind=SlotRecall, filter=SlotFilter(top_k=2, char_budget=8))
    ])
    agent = object.__new__(UnifiedAgent)
    agent.ltm = ltm
    agent.prompt_assembler = ContextAssembler({"chat": schema}, registry)
    agent.llm = SimpleNamespace(chat=lambda *a, **k: "answer")
    prefix = agent._request_context_prefix(
        "coffee", mode="chat", phase="generate", query_embedding=[]
    )
    assert repo.touches == []
    monkeypatch.setattr("internal.memory.memory.time.time", lambda: 1234.)
    system = compose_system_prompt(CHAT_SYSTEM_PROMPT, prefix)
    assert repo.touches == []
    agent._chat_llm(system, [])
    agent._chat_llm(system, [])  # same context acknowledgement is one-shot
    assert repo.touches == [[42]]
    assert [it.last_accessed for it in ltm.items] == [1234., 10, 10]
    assert repo.rows[42].last_accessed == 1234.


def test_decay_is_incremental_and_survives_restart(monkeypatch):
    clock = [86400.]
    monkeypatch.setattr("internal.memory.memory.time.time", lambda: clock[0])
    ltm, repo = make_ltm()
    ltm.add("coffee", .8)
    clock[0] += 86400
    ltm.consolidate()  # A singleton must decay too.
    assert ltm.items[0].importance == pytest.approx(.8 * .9)
    ltm.consolidate()
    assert ltm.items[0].importance == pytest.approx(.8 * .9)
    restored, _ = make_ltm(repo)
    restored.load_from_storage()
    restored.consolidate()
    assert restored.items[0].importance == pytest.approx(.8 * .9)
    clock[0] += 86400
    restored.consolidate()
    assert restored.items[0].importance == pytest.approx(.8 * .9 ** 2)


def test_consolidation_commit_failure_restores_memory_and_retry_state(monkeypatch):
    clock = [86400.]
    monkeypatch.setattr("internal.memory.memory.time.time", lambda: clock[0])
    ltm, repo = make_ltm()
    ltm.add("coffee", .8)
    before = ltm.snapshot()
    count = ltm._items_since_last
    clock[0] += 86400
    repo.fail = True
    with pytest.raises(RuntimeError):
        ltm.consolidate()
    assert ltm.snapshot() == before
    assert ltm._items_since_last == count
    repo.fail = False
    ltm.consolidate()
    assert ltm.items[0].importance == pytest.approx(.72)


def test_dedup_metadata_and_expiry_match_storage_after_restart(monkeypatch):
    ltm, repo = make_ltm()
    monkeypatch.setattr("internal.memory.memory.time.time", lambda: 86400.)
    ltm.add("same fact", .5)
    ltm.add("same fact", .7)
    ltm.items[1].tags = ["source:user"]
    result = ltm.consolidate()
    assert result.delete_from_db == [49]
    assert result.update_in_db[0].tags == ["source:user"]
    restored, _ = make_ltm(repo)
    restored.load_from_storage()
    assert restored.snapshot() == ltm.snapshot()
    monkeypatch.setattr("internal.memory.memory.time.time", lambda: 86400. * 400)
    graph = SimpleNamespace(
        delete_from_graph=lambda mid: graph.deleted.append(mid),
        update_node=lambda it: None, deleted=[],
        filter_protected=lambda *args: [42],
    )
    restored.graph_memory = graph
    result = restored.consolidate()
    assert restored.items == []
    assert result.delete_from_db == [42]
    assert graph.deleted == []
    agent = SimpleNamespace(inf=restored.inf, graph_memory=graph)
    assert sync_consolidation_to_db(agent, result) is True
    assert graph.deleted == [42]
    restored.load_from_storage()
    assert restored.items == []


def test_delayed_graph_insert_cannot_resurrect_deleted_node(monkeypatch):
    jobs, writes = [], []
    monkeypatch.setattr("internal.memory.graph_memory._go_safe", lambda name, fn: jobs.append(fn))
    neo = SimpleNamespace(is_real=lambda: True, run_cypher=lambda *a: writes.append(a))
    graph = GraphMemory(APIConfig(), neo)
    graph.add_to_graph(Item(id=42, content="old"))
    graph.delete_from_graph(42)
    jobs[0]()
    assert len(writes) == 1
    assert "DELETE" in writes[0][0]


def test_pg_repo_restores_decay_clock():
    client = SimpleNamespace(
        is_real=lambda: True,
        query=lambda _: [(42, "cat", .7, None, 10., 12., "fact", [], "", 0., 11.)],
    )
    row = PGRepo(client).load()[0]
    assert (row.id, row.last_decayed_at, row.created_at) == (42, 11., 10.)


@pytest.mark.parametrize("fail_delete", [False, True])
def test_pg_consolidation_transaction_commits_or_rolls_back(fail_delete):
    class Connection:
        def __init__(self):
            self.sql = []
            self.rows = {1: "old", 2: "duplicate"}
            self.rowcount = 1
            self.rolled_back = False

        def __enter__(self):
            self.before = deepcopy(self.rows)
            return self

        def __exit__(self, kind, error, traceback):
            if error:
                self.rolled_back = True
                self.rows = self.before

        def cursor(self):
            from contextlib import nullcontext
            return nullcontext(self)

        def execute(self, sql, params):
            self.sql.append((sql, params))
            if sql.startswith("UPDATE"):
                self.rows[params[-1]] = params[0]
            elif sql.startswith("DELETE"):
                if fail_delete:
                    raise OSError("delete failed")
                for mid in params[0]:
                    self.rows.pop(mid, None)

    from contextlib import nullcontext
    conn = Connection()
    client = SimpleNamespace(is_real=lambda: True, transaction=lambda: nullcontext(conn))
    result = ConsolidationResult(
        delete_from_db=[2], update_in_db=[
            Item(id=1, content="merged", last_decayed_at=123., last_accessed=122.)
        ]
    )
    assert PGRepo(client).apply_consolidation(result) is (not fail_delete)
    assert conn.rolled_back is fail_delete
    assert conn.rows == ({1: "old", 2: "duplicate"} if fail_delete else {1: "merged"})
    sql, params = conn.sql[0]
    assert "last_decayed_at=%s" in sql
    assert params[-3:] == (123., 122., 1)


def test_failed_pg_sync_never_touches_graph():
    graph_calls = []
    repo = DurableRepo()
    repo.fail = True
    agent = SimpleNamespace(
        inf=SimpleNamespace(repo=SimpleNamespace(ltm=repo)),
        graph_memory=SimpleNamespace(delete_from_graph=graph_calls.append)
    )
    assert sync_consolidation_to_db(
        agent, ConsolidationResult(delete_from_db=[42])
    ) is False
    assert graph_calls == []


def test_delayed_graph_add_does_not_overwrite_consolidated_content(monkeypatch):
    jobs, writes = [], []
    monkeypatch.setattr("internal.memory.graph_memory._go_safe", lambda name, fn: jobs.append(fn))
    neo = SimpleNamespace(is_real=lambda: True, run_cypher=lambda *a: writes.append(a))
    graph = GraphMemory(APIConfig(), neo)
    graph.add_to_graph(Item(id=42, content="old"))
    graph.update_node(Item(id=42, content="merged", importance=.4))
    jobs[0]()
    assert len(writes) == 1
    assert writes[0][1]["content"] == "merged"
