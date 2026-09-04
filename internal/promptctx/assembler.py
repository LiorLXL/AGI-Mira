"""Schema-driven runtime context assembly with deterministic budgets/traces."""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .context import RuntimeContext
from .schema import (
    DEFAULT_GLOBAL_CHAR_BUDGET,
    RuntimeContextSchema,
    default_schemas,
    slot_priority,
)
from .slot import (
    ContextItem,
    FilledSlot,
    Slot,
    SlotConstraints,
    SlotKind,
    SlotPlanner,
    SlotProfile,
    SlotRecall,
    SlotTaskMem,
    SlotToolState,
)
from .source import ContextSource, Query

logger = logging.getLogger(__name__)


_ALL_SLOT_KINDS: List[SlotKind] = [
    SlotProfile,
    SlotPlanner,
    SlotTaskMem,
    SlotToolState,
    SlotConstraints,
    SlotRecall,
]


class SourceRegistry:
    """Hold ContextSources grouped by SlotKind in registration order."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sources: Dict[SlotKind, List[ContextSource]] = {}

    def register(self, source: ContextSource) -> None:
        with self._lock:
            for kind in _ALL_SLOT_KINDS:
                if source.supports(kind):
                    self._sources.setdefault(kind, []).append(source)

    def for_kind(self, kind: SlotKind) -> List[ContextSource]:
        with self._lock:
            return list(self._sources.get(kind, []))


class ContextAssembler:
    """Assemble ContextItems for a mode/phase and apply central budgets."""

    def __init__(
        self,
        schemas: Optional[Dict[str, RuntimeContextSchema]] = None,
        registry: Optional[SourceRegistry] = None,
        global_limit: int = DEFAULT_GLOBAL_CHAR_BUDGET,
    ) -> None:
        self.schemas = schemas or default_schemas()
        self.registry = registry if registry is not None else SourceRegistry()
        self.global_limit = max(0, int(global_limit or 0))

    def assemble(self, q: Query) -> RuntimeContext:
        """Build one RuntimeContext with a structured, non-sensitive trace."""
        schema_key = f"{q.mode}.{q.phase}" if q.phase else q.mode
        schema = self.schemas.get(schema_key) or self.schemas.get(q.mode)
        if schema is None:
            schema = self.schemas.get("chat")
        if schema is None:
            return RuntimeContext(
                schema=RuntimeContextSchema(mode=schema_key),
                trace=[
                    {
                        "mode": q.mode,
                        "phase": q.phase,
                        "status": "schema_missing",
                        "schema": schema_key,
                    }
                ],
            )

        rc = RuntimeContext(
            schema=schema,
            filled=[FilledSlot(kind=slot.kind) for slot in schema.slots],
        )
        slots = list(schema.slots)

        if slots:
            with ThreadPoolExecutor(max_workers=max(1, len(slots))) as pool:
                futures = {
                    pool.submit(self._fill_slot, slot, q): index
                    for index, slot in enumerate(slots)
                }
                for future, index in futures.items():
                    try:
                        filled, trace = future.result()
                    except Exception as exc:  # defensive boundary around a slot
                        logger.warning("promptctx fill_slot failed: %s", exc)
                        filled = FilledSlot(
                            kind=slots[index].kind,
                            skipped=True,
                            reason=f"slot error: {type(exc).__name__}",
                        )
                        trace = self._slot_trace(
                            slots[index],
                            q,
                            status="error",
                            reason=f"slot error: {type(exc).__name__}",
                        )
                    rc.filled[index] = filled
                    rc.trace.append(trace)

        self._apply_global_budget(rc)
        return rc

    def _fill_slot(self, slot: Slot, q: Query) -> Tuple[FilledSlot, Dict[str, Any]]:
        started = time.perf_counter()
        sources = self.registry.for_kind(slot.kind)
        source_trace: List[Dict[str, Any]] = []
        if not sources:
            reason = "required source missing" if slot.required else "no source registered"
            return (
                FilledSlot(kind=slot.kind, skipped=True, reason=reason),
                self._slot_trace(
                    slot,
                    q,
                    status="required_missing" if slot.required else "empty",
                    reason=reason,
                    elapsed_ms=_elapsed_ms(started),
                ),
            )

        candidates: List[ContextItem] = []
        for source in sources:
            source_started = time.perf_counter()
            source_id = _source_id(source)
            try:
                items = list(source.fetch(slot, q) or [])
            except Exception as exc:
                logger.warning("promptctx source %s fetch failed: %s", source_id, exc)
                source_trace.append(
                    {
                        "source": source_id,
                        "status": "error",
                        "candidate_count": 0,
                        "elapsed_ms": _elapsed_ms(source_started),
                        "error_type": type(exc).__name__,
                    }
                )
                continue
            source_trace.append(
                {
                    "source": source_id,
                    "status": "ok",
                    "candidate_count": len(items),
                    "elapsed_ms": _elapsed_ms(source_started),
                }
            )
            candidates.extend(item for item in items if item and item.text.strip())

        candidate_count = len(candidates)
        candidates = _stable_item_order(slot.kind, candidates)

        top_k_dropped = 0
        if slot.filter.top_k > 0 and len(candidates) > slot.filter.top_k:
            top_k_dropped = len(candidates) - slot.filter.top_k
            candidates = candidates[: slot.filter.top_k]

        budget = slot.filter.effective_char_budget()
        budget_dropped = 0
        truncated = False
        if budget > 0:
            candidates, budget_dropped, truncated = _trim_by_budget(candidates, budget)

        if not candidates:
            reason = "all candidates filtered or trimmed" if candidate_count else "source returned empty"
            status = "required_missing" if slot.required else "empty"
            return (
                FilledSlot(kind=slot.kind, skipped=True, reason=reason),
                self._slot_trace(
                    slot,
                    q,
                    status=status,
                    reason=reason,
                    sources=source_trace,
                    candidate_count=candidate_count,
                    dropped_count=top_k_dropped + budget_dropped,
                    char_count=0,
                    elapsed_ms=_elapsed_ms(started),
                ),
            )

        reasons = []
        if top_k_dropped:
            reasons.append("top_k")
        if budget_dropped or truncated:
            reasons.append("char_budget")
        trace = self._slot_trace(
            slot,
            q,
            status="trimmed" if reasons else "ok",
            reason=",".join(reasons),
            sources=source_trace,
            candidate_count=candidate_count,
            kept_count=len(candidates),
            dropped_count=top_k_dropped + budget_dropped,
            char_count=sum(len(item.text) for item in candidates),
            elapsed_ms=_elapsed_ms(started),
        )
        return FilledSlot(kind=slot.kind, items=candidates), trace

    def _apply_global_budget(self, rc: RuntimeContext) -> None:
        if self.global_limit <= 0 or len(rc.render()) <= self.global_limit:
            return

        order = list(range(len(rc.filled)))
        order.sort(key=lambda index: slot_priority(rc.filled[index].kind), reverse=True)
        for index in order:
            filled = rc.filled[index]
            while filled.items and len(rc.render()) > self.global_limit:
                filled.items.pop()
            if not filled.items:
                filled.skipped = True
                filled.reason = "global char budget exceeded"
            trace = _trace_for_slot(rc.trace, filled.kind)
            if trace is not None:
                trace["kept_count"] = len(filled.items)
                trace["char_count"] = sum(len(item.text) for item in filled.items)
                trace["dropped_count"] = max(
                    int(trace.get("dropped_count", 0)),
                    int(trace.get("candidate_count", 0)) - len(filled.items),
                )
                trace["status"] = "trimmed"
                existing_reason = str(trace.get("reason", "") or "")
                trace["reason"] = ",".join(
                    part
                    for part in [existing_reason, "global_char_budget"]
                    if part
                )
            if len(rc.render()) <= self.global_limit:
                break

    @staticmethod
    def _slot_trace(
        slot: Slot,
        q: Query,
        *,
        status: str,
        reason: str = "",
        sources: Optional[List[Dict[str, Any]]] = None,
        candidate_count: int = 0,
        kept_count: int = 0,
        dropped_count: int = 0,
        char_count: int = 0,
        elapsed_ms: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "mode": q.mode,
            "phase": q.phase,
            "slot": slot.kind,
            "required": slot.required,
            "status": status,
            "sources": list(sources or []),
            "candidate_count": candidate_count,
            "kept_count": kept_count,
            "dropped_count": dropped_count,
            "char_count": char_count,
            "char_budget": slot.filter.effective_char_budget(),
            "reason": reason,
            "elapsed_ms": elapsed_ms,
        }


def _stable_item_order(kind: SlotKind, items: List[ContextItem]) -> List[ContextItem]:
    # Planner/task observations carry semantic sequence; their buffers already
    # provide deterministic order. Other slots use a stable score/tie-break key.
    if kind in {SlotPlanner, SlotTaskMem}:
        return list(items)
    return sorted(
        items,
        key=lambda item: (-float(item.score or 0.0), item.source or "", item.text),
    )


def _trim_by_budget(
    items: List[ContextItem], budget: int
) -> Tuple[List[ContextItem], int, bool]:
    """Apply one character budget, truncating at most the final kept item."""
    kept: List[ContextItem] = []
    remaining = max(0, int(budget or 0))
    truncated = False
    for item in items:
        text = item.text.strip()
        if not text or remaining <= 0:
            continue
        if len(text) <= remaining:
            kept.append(item)
            remaining -= len(text)
            continue
        if remaining > 1:
            kept.append(
                ContextItem(
                    text=text[: remaining - 1] + "…",
                    score=item.score,
                    source=item.source,
                    meta={**item.meta, "truncated": "true"},
                )
            )
            truncated = True
        break
    dropped = max(0, len(items) - len(kept))
    return kept, dropped, truncated


def _source_id(source: ContextSource) -> str:
    try:
        return str(source.id())
    except Exception:
        return type(source).__name__


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _trace_for_slot(
    traces: List[Dict[str, Any]], kind: SlotKind
) -> Optional[Dict[str, Any]]:
    for trace in traces:
        if trace.get("slot") == kind:
            return trace
    return None
