"""M4 — temporal-aware hybrid retriever & time-travel query (PRD §3.4).

Two retrieval modes over the MVCC store:

``retrieve_current(query)``
    hybrid scoring = (entity, slot) exact-key match + FTS5 BM25 (Chinese
    bigram / stemmed-latin tokenization) + token-overlap semantic score,
    with a **hard filter**: only ``status == 'active'`` atoms can be
    returned. Matching ``retracted`` / ``anchor_drifted`` atoms are surfaced
    as explicit *notices* so the agent can abstain instead of hallucinating.

``retrieve_as_of(turn_id, query)``
    reconstructs the exact world state at historical ``turn_id`` using the
    MVCC visibility windows ``valid_from_turn <= t AND (valid_to_turn IS NULL
    OR valid_to_turn > t)`` — answering "what config did we use before the
    change in turn N" style questions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from chronomem.store import Atom, MemoryStore
from chronomem.tokenizer import fts_query, normalize_tokens, tokens_to_text

__all__ = ["Retriever", "ScoredAtom", "RetrievalResult", "make_summary_text",
           "flatten_value"]

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    summary_text,
    content='memory_atoms',
    content_rowid='atom_id',
    tokenize='unicode61'
);
"""

# score fusion weights
W_FTS = 1.0        # FTS5 BM25 ranking position
W_KEY = 1.5        # exact (entity, slot) match
W_JACCARD = 0.6    # token overlap between query and summary


def flatten_value(value: Any) -> str:
    """Flatten structured values (dict/list/scalar) into retrieval text."""
    if isinstance(value, dict):
        return " ".join(f"{k} {flatten_value(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return " ".join(flatten_value(v) for v in value)
    return str(value)


def make_summary_text(entity: str, slot: str, value: Any) -> str:
    """Normalized (bigram/stemmed) FTS text for an atom (PRD field
    ``summary_text`` — '供 FTS5 检索的归一化文本')."""
    raw = f"{entity} {slot} {flatten_value(value)}"
    return tokens_to_text(normalize_tokens(raw))


@dataclass(frozen=True)
class ScoredAtom:
    atom: Atom
    score: float
    match_reasons: Tuple[str, ...] = ()

    def as_context_item(self) -> Dict[str, Any]:
        """Dict view used to build the agent context / poisoning audit."""
        return {
            "atom_id": self.atom.atom_id,
            "namespace": self.atom.namespace,
            "entity": self.atom.entity,
            "slot": self.atom.slot,
            "value": self.atom.value,
            "status": self.atom.status,
            "valid_from_turn": self.atom.valid_from_turn,
            "anchor_path": self.atom.anchor_path,
            "score": round(self.score, 6),
            "match_reasons": list(self.match_reasons),
        }


@dataclass
class RetrievalResult:
    query: str
    results: List[ScoredAtom] = field(default_factory=list)
    notices: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def top(self) -> Optional[ScoredAtom]:
        return self.results[0] if self.results else None

    @property
    def poisoning_ratio(self) -> float:
        """Share of returned context items that are NOT active — always 0.0
        by construction for the current-state retriever (hard filter)."""
        if not self.results:
            return 0.0
        stale = sum(1 for s in self.results if s.atom.status != "active")
        return stale / len(self.results)

    def context_items(self) -> List[Dict[str, Any]]:
        return [s.as_context_item() for s in self.results]


class Retriever:
    """Hybrid (FTS5 + key match + token overlap) temporal-aware retriever."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store
        with store._lock:  # reuse the store's connection/lock
            store._conn.execute(_FTS_SCHEMA)
            store._conn.commit()

    # ------------------------------------------------------------------ #
    # index maintenance
    # ------------------------------------------------------------------ #
    def index_atom(self, atom: Atom) -> None:
        """Add one atom's normalized summary to the FTS index."""
        text = atom.summary_text or make_summary_text(
            atom.entity, atom.slot, atom.value
        )
        with self.store._lock:
            self.store._conn.execute(
                "INSERT OR REPLACE INTO memory_fts(rowid, summary_text) VALUES (?, ?)",
                (atom.atom_id, text),
            )
            self.store._conn.commit()

    def index_all(self) -> int:
        """(Re-)index every atom; returns the number indexed."""
        atoms = self.store.all_atoms()
        with self.store._lock:
            self.store._conn.execute("DELETE FROM memory_fts")
            for atom in atoms:
                text = atom.summary_text or make_summary_text(
                    atom.entity, atom.slot, atom.value
                )
                self.store._conn.execute(
                    "INSERT INTO memory_fts(rowid, summary_text) VALUES (?, ?)",
                    (atom.atom_id, text),
                )
            self.store._conn.commit()
        return len(atoms)

    # ------------------------------------------------------------------ #
    # scoring helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _token_set(atom: Atom) -> set:
        return set((atom.summary_text or "").split()) | {atom.entity, atom.slot}

    def _score(self, query_tokens: List[str], atom: Atom,
               fts_pos: Optional[int]) -> Tuple[float, Tuple[str, ...]]:
        reasons: List[str] = []
        score = 0.0
        qset = set(query_tokens)
        overlap = self._token_set(atom) & qset
        if overlap:
            reasons.append("token_overlap")

        if fts_pos is not None:
            score += W_FTS * (1.0 / (1.0 + fts_pos))
            reasons.append("fts")

        if qset and overlap:
            score += W_JACCARD * (len(overlap) / len(qset))

        return score, tuple(reasons)

    def _key_bonus(self, query_tokens: List[str], atom: Atom) -> float:
        """Exact (entity, slot) match bonus: both identifiers appear in query."""
        q = set(query_tokens)
        hit = {atom.entity, atom.slot} <= q
        return W_KEY if hit else 0.0

    # ------------------------------------------------------------------ #
    # current-state retrieval
    # ------------------------------------------------------------------ #
    def retrieve_current(
        self,
        query: str,
        *,
        namespace: Optional[str] = None,
        top_k: int = 5,
        entity: Optional[str] = None,
        slot: Optional[str] = None,
    ) -> RetrievalResult:
        """Zero-poisoning current-state retrieval with abstention notices."""
        ns = namespace if namespace is not None else "default"
        query_tokens = normalize_tokens(query)

        fts_pos: Dict[int, int] = {}
        if query_tokens:
            try:
                rows = self.store._conn.execute(
                    """
                    SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?
                    ORDER BY bm25(memory_fts) LIMIT 200
                    """,
                    (fts_query(query_tokens),),
                ).fetchall()
                fts_pos = {r[0]: i for i, r in enumerate(rows)}
            except Exception:
                fts_pos = {}

        candidates = {
            a.atom_id: a for a in self.store.all_atoms(namespace=ns)
            if a.atom_id in fts_pos or self._token_set(a) & set(query_tokens)
            or (entity and a.entity == entity and slot and a.slot == slot)
        }

        scored: List[ScoredAtom] = []
        notices: List[Dict[str, Any]] = []

        for atom_id, atom in candidates.items():
            score, reasons = self._score(query_tokens, atom, fts_pos.get(atom_id))
            key_bonus = self._key_bonus(query_tokens, atom)
            if key_bonus:
                reasons += ("key_match",)
                score += key_bonus
            if entity is not None and atom.entity == entity:
                score += 0.8
                reasons += ("entity_match",)
            if slot is not None and atom.slot == slot:
                score += 0.8
                reasons += ("slot_match",)
            if score <= 0.0:
                continue
            sa = ScoredAtom(atom, score, reasons)
            if atom.status == "active":
                scored.append(sa)
            elif atom.status in ("retracted", "anchor_drifted"):
                # never returned as a result — surfaced as an abstention notice
                notices.append({
                    "type": atom.status,
                    "entity": atom.entity,
                    "slot": atom.slot,
                    "atom_id": atom.atom_id,
                    "valid_from_turn": atom.valid_from_turn,
                    "valid_to_turn": atom.valid_to_turn,
                    "anchor_path": atom.anchor_path,
                    "score": round(score, 6),
                })

        scored.sort(key=lambda s: s.score, reverse=True)
        notices.sort(key=lambda n: n["score"], reverse=True)
        return RetrievalResult(query, scored[:top_k], notices[:top_k])

    # ------------------------------------------------------------------ #
    # time travel
    # ------------------------------------------------------------------ #
    def retrieve_as_of(
        self,
        turn_id: int,
        query: Optional[str] = None,
        *,
        namespace: Optional[str] = None,
        top_k: Optional[int] = None,
        entity: Optional[str] = None,
        slot: Optional[str] = None,
    ) -> RetrievalResult:
        """Exact world-state snapshot at historical ``turn_id``."""
        ns = namespace if namespace is not None else "default"
        snapshot = [
            a for a in self.store.snapshot_as_of(turn_id, namespace=ns)
            if a.key[0] == ns
        ]
        if query is None:
            ordered = sorted(
                snapshot, key=lambda a: (a.entity, a.slot, -a.valid_from_turn)
            )
            results = [ScoredAtom(a, 1.0, ("snapshot",)) for a in ordered]
            return RetrievalResult(query or f"as_of:{turn_id}",
                                   results[: top_k or len(results)], [])

        query_tokens = normalize_tokens(query)
        fts_pos: Dict[int, int] = {}
        try:
            rows = self.store._conn.execute(
                "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ? "
                "ORDER BY bm25(memory_fts) LIMIT 500",
                (fts_query(query_tokens),),
            ).fetchall()
            fts_pos = {r[0]: i for i, r in enumerate(rows)}
        except Exception:
            fts_pos = {}

        scored: List[ScoredAtom] = []
        for atom in snapshot:
            score, reasons = self._score(query_tokens, atom, fts_pos.get(atom.atom_id))
            key_bonus = self._key_bonus(query_tokens, atom)
            if key_bonus:
                reasons += ("key_match",)
                score += key_bonus
            if entity is not None and atom.entity == entity:
                score += 0.8
                reasons += ("entity_match",)
            if slot is not None and atom.slot == slot:
                score += 0.8
                reasons += ("slot_match",)
            if score > 0.0:
                scored.append(ScoredAtom(atom, score, reasons + ("as_of",)))
        scored.sort(key=lambda s: s.score, reverse=True)
        return RetrievalResult(query, scored[: top_k or len(scored)], [])

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def retrieve_by_key(
        self, entity: str, slot: str, *, namespace: str = "default"
    ) -> Tuple[Optional[Atom], List[Dict[str, Any]]]:
        """Direct (entity, slot) lookup: active atom + abstention notices."""
        active = self.store.get_active(entity, slot, namespace=namespace)
        notices: List[Dict[str, Any]] = []
        if active is None:
            for atom in self.store.versions(entity, slot, namespace=namespace):
                if atom.status in ("retracted", "anchor_drifted") and atom.valid_to_turn is not None:
                    notices.append({
                        "type": atom.status,
                        "entity": entity,
                        "slot": slot,
                        "atom_id": atom.atom_id,
                        "valid_from_turn": atom.valid_from_turn,
                        "valid_to_turn": atom.valid_to_turn,
                        "anchor_path": atom.anchor_path,
                    })
        return active, notices
