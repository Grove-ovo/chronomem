"""The four competing memory systems on ChronoBench-100.

All four systems observe the *same* conversation stream (turns carry
structured statement metadata so that no system is advantaged or penalized
on NLU — the benchmark isolates memory management, not extraction quality):

``ChronoMemSystem``   — MVCC state chain + 4-state resolver + anchor watcher
                        + temporal hybrid retriever (this project).
``NaiveVectorRAG``    — every statement is a chunk in a TF-IDF vector store;
                        Top-K cosine recall, no version/time awareness.
``RollingSummary``    — per-session digests kept in a bounded window; older
                        digests are discarded (information loss by design).
``FullContext``       — the whole conversation is scanned at answer time;
                        perfect recall, maximal context & poisoning.

Answers carry the *context items* that entered the agent's context, which
the runner audits for staleness (Stale_Poisoning_Rate) and token cost.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chronomem.anchor_watcher import AnchorWatcher, compute_anchor_fingerprint
from chronomem.resolver import ConflictResolver, Resolution
from chronomem.retriever import Retriever, make_summary_text
from chronomem.store import MemoryStore
from chronomem.tokenizer import normalize_tokens

__all__ = ["Answer", "ChronoMemSystem", "NaiveVectorRAG", "RollingSummary",
           "FullContext", "count_tokens"]

# --------------------------------------------------------------------- #
# token metering
# --------------------------------------------------------------------- #
try:  # real BPE tokenization when tiktoken is available
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:  # pragma: no cover — offline fallback meter
    def count_tokens(text: str) -> int:
        import re

        cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]", text))
        words = len(re.findall(r"[A-Za-z0-9]+", text))
        return cjk + words


@dataclass
class Answer:
    abstained: bool
    reason: Optional[str]  # retracted | no_evidence | anchor_drifted
    value: Any = None
    slots: Optional[Dict[str, Any]] = None
    context: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def context_tokens(self) -> int:
        return sum(item.get("tokens", 0) for item in self.context)


def _ctx_item(source: str, turn_id: int, entity: str, slot: str,
              value: Any, text: str) -> Dict[str, Any]:
    return {
        "source": source,
        "turn_id": turn_id,
        "entity": entity,
        "slot": slot,
        "value": value,
        "text": text,
        "tokens": count_tokens(text),
    }


# ===================================================================== #
# Ours
# ===================================================================== #
class ChronoMemSystem:
    """ChronoMem: MVCC version chains + 4-state resolver + anchors."""

    name = "ChronoMem"

    def __init__(self, workdir: str) -> None:
        self.store = MemoryStore(f"{workdir}/chronomem_bench.db")
        self.resolver = ConflictResolver(self.store, namespace="bench")
        self.retriever = Retriever(self.store)
        self.watcher = AnchorWatcher(self.store)
        self.workspace: Optional[str] = None

    def ingest_turn(self, turn: Dict[str, Any], workspace_root: str) -> None:
        self.workspace = workspace_root
        st = turn.get("statement")
        if not st:
            return
        if st["op"] == "retract":
            self.resolver.retract(st["entity"], st["slot"], turn["turn_id"])
            return
        anchor_path = st.get("anchor_path")
        anchor_sha = None
        if anchor_path:
            anchor_sha = compute_anchor_fingerprint(workspace_root, anchor_path)
        result = self.resolver.write(
            st["entity"], st["slot"], st["value"], turn["turn_id"],
            source_turn_id=f"s{turn['session_id']}-t{turn['turn_id']}",
            anchor_path=anchor_path,
            anchor_sha256=anchor_sha,
        )
        if result.new_atom is not None:
            # index the *final* atom value (merged snapshot for MERGE_PARTIAL)
            atom = result.new_atom
            self.retriever.index_atom(atom)

    def end_session(self, session_id: int) -> None:  # uniform interface
        pass

    def answer(self, query: Dict[str, Any], workspace_root: str) -> Answer:
        # PRD §3.3: verify anchors *before* retrieval
        self.watcher.verify_anchors(workspace_root, namespace="bench")

        if query["type"] == "as_of":
            res = self.retriever.retrieve_as_of(
                query["as_of_turn"], query["text"], namespace="bench",
                entity=query["entity"], slot=query["slot"])
            key_hits = [s for s in res.results
                        if s.atom.entity == query["entity"]
                        and s.atom.slot == query["slot"]]
            if not key_hits:
                return Answer(True, "no_evidence")
            top = key_hits[0]
            context = [_ctx_item("atom", s.atom.valid_from_turn, s.atom.entity,
                                 s.atom.slot, s.atom.value,
                                 make_summary_text(s.atom.entity, s.atom.slot,
                                                   s.atom.value))
                       for s in res.results]
            return Answer(False, None, value=top.atom.value,
                          slots=dict(top.atom.value) if isinstance(top.atom.value, dict) else None,
                          context=context)

        active, key_notices = self.retriever.retrieve_by_key(
            query["entity"], query["slot"], namespace="bench")
        if key_notices:
            notice = key_notices[0]
            return Answer(True, notice["type"])
        if active is None:
            return Answer(True, "no_evidence")

        # context = the zero-poisoning current-state retrieval view
        res = self.retriever.retrieve_current(
            query["text"], namespace="bench",
            entity=query["entity"], slot=query["slot"])
        context = [_ctx_item("atom", s.atom.valid_from_turn, s.atom.entity,
                             s.atom.slot, s.atom.value,
                             make_summary_text(s.atom.entity, s.atom.slot,
                                               s.atom.value))
                   for s in res.results]
        if active.key not in {(s.atom.entity, s.atom.slot) for s in res.results}:
            # guarantee the authoritative atom itself is in the audited context
            context.insert(0, _ctx_item(
                "atom", active.valid_from_turn, active.entity, active.slot,
                active.value, make_summary_text(active.entity, active.slot,
                                                active.value)))
        return Answer(False, None, value=active.value,
                      slots=dict(active.value) if isinstance(active.value, dict) else None,
                      context=context)

    def close(self) -> None:
        self.store.close()


# ===================================================================== #
# Baseline 1: Naive Vector RAG
# ===================================================================== #
class NaiveVectorRAG:
    """Flat chunk store + TF-IDF cosine Top-K. No versions, no time axis."""

    name = "Naive Vector RAG"
    TOP_K = 3
    THRESHOLD = 0.05

    def __init__(self) -> None:
        self.chunks: List[Dict[str, Any]] = []
        self._dirty = True

    def ingest_turn(self, turn: Dict[str, Any], workspace_root: str) -> None:
        st = turn.get("statement")
        if not st:
            return
        from chronomem.retriever import flatten_value

        text = f"{turn['text']} {st['entity']} {st['slot']} {flatten_value(st['value'])}"
        self.chunks.append({
            "index": len(self.chunks),
            "turn_id": turn["turn_id"],
            "entity": st["entity"], "slot": st["slot"],
            "value": st["value"], "op": st["op"],
            "text": text,
        })
        self._dirty = True

    def end_session(self, session_id: int) -> None:
        pass

    def _build(self) -> None:
        if not self._dirty:
            return
        token_sets = [Counter(normalize_tokens(c["text"])) for c in self.chunks]
        df: Counter = Counter()
        for ts in token_sets:
            df.update(ts.keys())
        n = len(self.chunks)
        self._idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}
        self._vecs = []
        for ts in token_sets:
            vec = {t: cnt * self._idf[t] for t, cnt in ts.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._vecs.append({t: v / norm for t, v in vec.items()})
        self._dirty = False

    def answer(self, query: Dict[str, Any], workspace_root: str) -> Answer:
        self._build()
        from chronomem.retriever import flatten_value

        hints = f"{query.get('entity', '')} {query.get('slot', '')}"
        qtok = Counter(normalize_tokens(query["text"] + " " + hints))
        qvec = {t: c * self._idf.get(t, 0.0) for t, c in qtok.items()}
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scored = []
        for idx, cvec in enumerate(self._vecs):
            dot = sum(w * cvec.get(t, 0.0) for t, w in qvec.items()) / qnorm
            if dot > self.THRESHOLD:
                scored.append((dot, self.chunks[idx]["index"], idx))
        scored.sort(reverse=True, key=lambda x: (x[0], -x[1]))
        top = scored[: self.TOP_K]

        if not top:
            return Answer(True, "no_evidence")
        context = [_ctx_item("chunk", self.chunks[idx]["turn_id"],
                             self.chunks[idx]["entity"], self.chunks[idx]["slot"],
                             self.chunks[idx]["value"], self.chunks[idx]["text"])
                   for _, _, idx in top]
        best = self.chunks[top[0][2]]
        if best["op"] == "retract":
            return Answer(True, "retracted", context=context)
        return Answer(False, None, value=best["value"],
                      slots=dict(best["value"]) if isinstance(best["value"], dict) else None,
                      context=context)


# ===================================================================== #
# Baseline 2: Rolling Summary
# ===================================================================== #
class RollingSummary:
    """Bounded window of per-session digests; older sessions are discarded."""

    name = "Rolling Summary"
    WINDOW = 3

    def __init__(self) -> None:
        self.digests: List[Dict[str, Any]] = []
        self._buffer: List[Dict[str, Any]] = []

    def ingest_turn(self, turn: Dict[str, Any], workspace_root: str) -> None:
        st = turn.get("statement")
        if st:
            self._buffer.append({
                "turn_id": turn["turn_id"],
                "entity": st["entity"], "slot": st["slot"],
                "value": st["value"], "op": st["op"],
                "text": turn["text"],
            })

    def end_session(self, session_id: int) -> None:
        if self._buffer:
            self.digests.append({"session_id": session_id, "items": self._buffer})
            self._buffer = []
            if len(self.digests) > self.WINDOW:
                self.digests = self.digests[-self.WINDOW:]

    def _items(self) -> List[Dict[str, Any]]:
        return [it for d in self.digests for it in d["items"]]

    def answer(self, query: Dict[str, Any], workspace_root: str) -> Answer:
        items = [it for it in self._items()
                 if it["entity"] == query.get("entity")
                 and it["slot"] == query.get("slot")]
        if not items:
            return Answer(True, "no_evidence")
        context = [_ctx_item("digest", it["turn_id"], it["entity"], it["slot"],
                             it["value"], it["text"]) for it in items]
        latest = max(items, key=lambda it: it["turn_id"])
        if latest["op"] == "retract":
            return Answer(True, "retracted", context=context)
        if query.get("expected_slots") is not None:
            # field-wise merge of everything surviving the window
            slots: Dict[str, Any] = {}
            for it in sorted(items, key=lambda it: it["turn_id"]):
                if it["op"] in ("set", "partial") and isinstance(it["value"], dict):
                    slots.update(it["value"])
            return Answer(False, None, slots=slots, context=context)
        if query["type"] == "as_of":
            # no time axis: the summary only knows the latest state
            return Answer(False, None, value=latest["value"], context=context)
        value = latest["value"] if latest["op"] != "partial" else None
        if value is None:  # latest mention was a partial patch — find last full
            for it in reversed(sorted(items, key=lambda it: it["turn_id"])):
                if it["op"] == "set":
                    value = it["value"]
                    break
        return Answer(False, None, value=value,
                      slots=dict(value) if isinstance(value, dict) else None,
                      context=context)


# ===================================================================== #
# Baseline 3: Full-Context Window
# ===================================================================== #
class FullContext:
    """Scans the entire raw conversation at answer time. Perfect recall,
    maximal context size, maximal stale exposure."""

    name = "Full-Context"

    def __init__(self) -> None:
        self.statements: List[Dict[str, Any]] = []

    def ingest_turn(self, turn: Dict[str, Any], workspace_root: str) -> None:
        st = turn.get("statement")
        if st:
            self.statements.append({
                "turn_id": turn["turn_id"],
                "entity": st["entity"], "slot": st["slot"],
                "value": st["value"], "op": st["op"],
                "text": turn["text"],
            })

    def end_session(self, session_id: int) -> None:
        pass

    def answer(self, query: Dict[str, Any], workspace_root: str) -> Answer:
        horizon = query.get("as_of_turn", query["turn_id"]) \
            if query["type"] == "as_of" else query["turn_id"]
        key_stmts = [s for s in self.statements
                     if s["entity"] == query.get("entity")
                     and s["slot"] == query.get("slot")
                     and s["turn_id"] <= horizon]
        # full-context sees *all* statements about the key (its context window)
        all_key = [s for s in self.statements
                   if s["entity"] == query.get("entity")
                   and s["slot"] == query.get("slot")]
        context = [_ctx_item("statement", s["turn_id"], s["entity"], s["slot"],
                             s["value"], s["text"]) for s in all_key]

        if query["type"] == "as_of":
            effective = [s for s in key_stmts if s["op"] in ("set", "partial")]
            if not effective:
                return Answer(True, "no_evidence")
            best = max(effective, key=lambda s: s["turn_id"])
            value = best["value"]
            if query.get("expected_slots") is not None:
                slots: Dict[str, Any] = {}
                for s in sorted(effective, key=lambda s: s["turn_id"]):
                    if isinstance(s["value"], dict):
                        slots.update(s["value"])
                return Answer(False, None, slots=slots, context=context)
            return Answer(False, None, value=value, context=context)

        if not key_stmts:
            return Answer(True, "no_evidence")
        latest = max(key_stmts, key=lambda s: s["turn_id"])
        if latest["op"] == "retract":
            return Answer(True, "retracted", context=context)
        if query.get("expected_slots") is not None:
            slots = {}
            for s in sorted([x for x in key_stmts if x["op"] in ("set", "partial")],
                            key=lambda s: s["turn_id"]):
                if isinstance(s["value"], dict):
                    slots.update(s["value"])
            return Answer(False, None, slots=slots, context=context)
        value = latest["value"]
        return Answer(False, None, value=value,
                      slots=dict(value) if isinstance(value, dict) else None,
                      context=context)
