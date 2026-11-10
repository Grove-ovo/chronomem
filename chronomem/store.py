"""M1 — MVCC atomic state storage engine for ChronoMem.

Implements the multi-version ``memory_atoms`` table on SQLite with:

* full lifecycle fields (``status`` / ``valid_from_turn`` / ``valid_to_turn``)
* bidirectional version chain (``supersedes`` / ``superseded_by``)
* provenance and code-anchor fields (``source_turn_id`` / ``anchor_path`` /
  ``anchor_sha256``)

Core database invariant (PRD §3.1): for any ``(namespace, entity, slot)``
there is **at most one** record with ``status == 'active'`` at any time.
The invariant is enforced physically by a partial unique index so that even
raw SQL writes cannot violate it.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

__all__ = ["Atom", "MemoryStore", "STATUSES"]

STATUSES = ("active", "stale", "retracted", "anchor_drifted")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_atoms (
    atom_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace       TEXT    NOT NULL DEFAULT 'default',
    entity          TEXT    NOT NULL,
    slot            TEXT    NOT NULL,
    value_json      TEXT    NOT NULL,
    summary_text    TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL
                    CHECK (status IN ('active', 'stale', 'retracted', 'anchor_drifted')),
    valid_from_turn INTEGER NOT NULL,
    valid_to_turn   INTEGER,
    superseded_by   INTEGER REFERENCES memory_atoms(atom_id),
    supersedes      INTEGER REFERENCES memory_atoms(atom_id),
    source_turn_id  TEXT,
    anchor_path     TEXT,
    anchor_sha256   TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_atoms_key
    ON memory_atoms (namespace, entity, slot);
CREATE INDEX IF NOT EXISTS idx_atoms_status
    ON memory_atoms (status);
CREATE INDEX IF NOT EXISTS idx_atoms_supersedes
    ON memory_atoms (supersedes);

-- PRD §3.1 invariant, enforced by the database itself:
-- at most ONE 'active' record per (namespace, entity, slot) at any time.
CREATE UNIQUE INDEX IF NOT EXISTS uq_one_active_per_key
    ON memory_atoms (namespace, entity, slot) WHERE status = 'active';
"""


@dataclass(frozen=True)
class Atom:
    """One immutable version of a memory atom."""

    atom_id: int
    namespace: str
    entity: str
    slot: str
    value_json: str
    summary_text: str
    status: str
    valid_from_turn: int
    valid_to_turn: Optional[int]
    superseded_by: Optional[int]
    supersedes: Optional[int]
    source_turn_id: Optional[str] = None
    anchor_path: Optional[str] = None
    anchor_sha256: Optional[str] = None
    created_at: Optional[str] = None

    @property
    def value(self) -> Any:
        """Parsed ``value_json`` (dict / list / scalar)."""
        return json.loads(self.value_json)

    @property
    def key(self) -> Tuple[str, str, str]:
        return (self.namespace, self.entity, self.slot)

    @property
    def is_live_at(self) -> bool:
        """Whether this atom is currently the live (active) version."""
        return self.status == "active"

    def valid_at_turn(self, turn_id: int) -> bool:
        """MVCC visibility rule: ``valid_from <= t AND (valid_to IS NULL OR valid_to > t)``."""
        if turn_id < self.valid_from_turn:
            return False
        return self.valid_to_turn is None or self.valid_to_turn > turn_id

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "atom_id": self.atom_id,
            "namespace": self.namespace,
            "entity": self.entity,
            "slot": self.slot,
            "value_json": self.value_json,
            "summary_text": self.summary_text,
            "status": self.status,
            "valid_from_turn": self.valid_from_turn,
            "valid_to_turn": self.valid_to_turn,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "source_turn_id": self.source_turn_id,
            "anchor_path": self.anchor_path,
            "anchor_sha256": self.anchor_sha256,
            "created_at": self.created_at,
        }
        return d


class MemoryStore:
    """Thread-safe MVCC memory-atom store backed by a single SQLite file."""

    def __init__(self, db_path: str = "chronomem.sqlite3") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ #
    # low-level helpers
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Explicit transaction: all-or-nothing multi-statement transitions."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    @staticmethod
    def _row_to_atom(row: sqlite3.Row) -> Atom:
        return Atom(
            atom_id=row["atom_id"],
            namespace=row["namespace"],
            entity=row["entity"],
            slot=row["slot"],
            value_json=row["value_json"],
            summary_text=row["summary_text"],
            status=row["status"],
            valid_from_turn=row["valid_from_turn"],
            valid_to_turn=row["valid_to_turn"],
            superseded_by=row["superseded_by"],
            supersedes=row["supersedes"],
            source_turn_id=row["source_turn_id"],
            anchor_path=row["anchor_path"],
            anchor_sha256=row["anchor_sha256"],
            created_at=row["created_at"],
        )

    # ------------------------------------------------------------------ #
    # writes
    # ------------------------------------------------------------------ #
    def insert_atom(
        self,
        entity: str,
        slot: str,
        value: Any,
        turn_id: int,
        *,
        namespace: str = "default",
        summary_text: str = "",
        status: str = "active",
        supersedes: Optional[int] = None,
        source_turn_id: Optional[str] = None,
        anchor_path: Optional[str] = None,
        anchor_sha256: Optional[str] = None,
    ) -> Atom:
        """Insert a new atom version.

        Raises ``sqlite3.IntegrityError`` if inserting a second ``active``
        version for the same key (invariant enforced by partial unique index).
        """
        if status not in STATUSES:
            raise ValueError(f"invalid status: {status!r}")
        value_json = json.dumps(value, ensure_ascii=False)
        with self._transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO memory_atoms
                    (namespace, entity, slot, value_json, summary_text, status,
                     valid_from_turn, valid_to_turn, superseded_by, supersedes,
                     source_turn_id, anchor_path, anchor_sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    namespace,
                    entity,
                    slot,
                    value_json,
                    summary_text,
                    status,
                    int(turn_id),
                    supersedes,
                    source_turn_id,
                    anchor_path,
                    anchor_sha256,
                ),
            )
            return self.get_atom(cur.lastrowid)  # type: ignore[arg-type]

    def deactivate(
        self,
        atom_id: int,
        *,
        new_status: str,
        valid_to_turn: int,
        superseded_by: Optional[int] = None,
    ) -> Optional[Atom]:
        """End the validity window of the *currently active* atom.

        Only an ``active`` atom may be transitioned; returns the updated atom
        or ``None`` when ``atom_id`` does not exist / is not active.
        """
        if new_status not in STATUSES or new_status == "active":
            raise ValueError(f"deactivate target status must not be 'active': {new_status!r}")
        with self._transaction() as conn:
            cur = conn.execute(
                "SELECT * FROM memory_atoms WHERE atom_id = ? AND status = 'active'",
                (atom_id,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            conn.execute(
                """
                UPDATE memory_atoms
                   SET status = ?, valid_to_turn = ?, superseded_by = ?
                 WHERE atom_id = ?
                """,
                (new_status, int(valid_to_turn), superseded_by, atom_id),
            )
            return self.get_atom(atom_id)

    def supersede(
        self,
        entity: str,
        slot: str,
        value: Any,
        turn_id: int,
        *,
        namespace: str = "default",
        summary_text: str = "",
        source_turn_id: Optional[str] = None,
        anchor_path: Optional[str] = None,
        anchor_sha256: Optional[str] = None,
    ) -> Tuple[Optional[Atom], Atom]:
        """Atomic MVCC SUPERSEDE transition (PRD §3.2 state 2).

        Inside a **single transaction**:

        1. the current ``active`` record (if any) is closed:
           ``status='stale'``, ``valid_to_turn=turn_id``;
        2. a new ``active`` record is inserted with ``supersedes=old_id``;
        3. the old record's ``superseded_by`` is back-filled with ``new_id``.

        If no active record exists this degenerates to ADD. Returns
        ``(old_atom_or_None, new_atom)``.
        """
        value_json = json.dumps(value, ensure_ascii=False)
        with self._transaction() as conn:
            old_row = conn.execute(
                """
                SELECT * FROM memory_atoms
                 WHERE namespace = ? AND entity = ? AND slot = ? AND status = 'active'
                """,
                (namespace, entity, slot),
            ).fetchone()
            old_id: Optional[int] = old_row["atom_id"] if old_row is not None else None

            if old_id is not None:
                conn.execute(
                    """
                    UPDATE memory_atoms
                       SET status = 'stale', valid_to_turn = ?
                     WHERE atom_id = ?
                    """,
                    (int(turn_id), old_id),
                )

            cur = conn.execute(
                """
                INSERT INTO memory_atoms
                    (namespace, entity, slot, value_json, summary_text, status,
                     valid_from_turn, valid_to_turn, superseded_by, supersedes,
                     source_turn_id, anchor_path, anchor_sha256)
                VALUES (?, ?, ?, ?, ?, 'active', ?, NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    namespace,
                    entity,
                    slot,
                    value_json,
                    summary_text,
                    int(turn_id),
                    old_id,
                    source_turn_id,
                    anchor_path,
                    anchor_sha256,
                ),
            )
            new_id = int(cur.lastrowid)

            if old_id is not None:
                conn.execute(
                    "UPDATE memory_atoms SET superseded_by = ? WHERE atom_id = ?",
                    (new_id, old_id),
                )

            old_atom = self.get_atom(old_id) if old_id is not None else None
            return old_atom, self.get_atom(new_id)  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    # reads
    # ------------------------------------------------------------------ #
    def get_atom(self, atom_id: int) -> Optional[Atom]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memory_atoms WHERE atom_id = ?", (atom_id,)
            ).fetchone()
        return self._row_to_atom(row) if row is not None else None

    def get_active(
        self, entity: str, slot: str, *, namespace: str = "default"
    ) -> Optional[Atom]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM memory_atoms
                 WHERE namespace = ? AND entity = ? AND slot = ? AND status = 'active'
                """,
                (namespace, entity, slot),
            ).fetchone()
        return self._row_to_atom(row) if row is not None else None

    def versions(
        self, entity: str, slot: str, *, namespace: str = "default"
    ) -> List[Atom]:
        """All versions of a key ordered along the version chain."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memory_atoms
                 WHERE namespace = ? AND entity = ? AND slot = ?
                 ORDER BY valid_from_turn ASC, atom_id ASC
                """,
                (namespace, entity, slot),
            ).fetchall()
        return [self._row_to_atom(r) for r in rows]

    def snapshot_as_of(
        self,
        turn_id: int,
        *,
        namespace: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> List[Atom]:
        """World state at historical ``turn_id`` (MVCC visibility windows)."""
        sql = """
              SELECT * FROM memory_atoms
               WHERE valid_from_turn <= :t
                 AND (valid_to_turn IS NULL OR valid_to_turn > :t)
            """
        params: Dict[str, Any] = {"t": int(turn_id)}
        if namespace is not None:
            sql += " AND namespace = :ns"
            params["ns"] = namespace
        if entity is not None:
            sql += " AND entity = :ent"
            params["ent"] = entity
        sql += " ORDER BY entity, slot, valid_from_turn"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_atom(r) for r in rows]

    def all_atoms(self, *, namespace: Optional[str] = None) -> List[Atom]:
        if namespace is None:
            sql = "SELECT * FROM memory_atoms ORDER BY atom_id"
            params: Tuple[Any, ...] = ()
        else:
            sql = "SELECT * FROM memory_atoms WHERE namespace = ? ORDER BY atom_id"
            params = (namespace,)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_atom(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM memory_atoms").fetchone()[0])

    # ------------------------------------------------------------------ #
    # invariant & chain verification
    # ------------------------------------------------------------------ #
    def verify_invariants(self) -> List[str]:
        """Return a list of human-readable invariant violations (empty == OK).

        Checks:
        1. at most one ``active`` record per ``(namespace, entity, slot)``;
        2. ``active`` records have an open validity window (``valid_to_turn IS NULL``);
        3. non-active records have a closed validity window;
        4. ``supersedes``/``superseded_by`` are mutually consistent;
        5. stale versions form one contiguous chain per key
           (successor begins exactly where predecessor ends).
        """
        problems: List[str] = []
        atoms = self.all_atoms()
        by_key: Dict[Tuple[str, str, str], List[Atom]] = {}
        for a in atoms:
            by_key.setdefault(a.key, []).append(a)
            # rule 2/3: window closure matches status
            if a.status == "active" and a.valid_to_turn is not None:
                problems.append(
                    f"atom {a.atom_id}: active but valid_to_turn={a.valid_to_turn}"
                )
            if a.status != "active" and a.valid_to_turn is None:
                problems.append(f"atom {a.atom_id}: {a.status} but valid_to_turn is NULL")

        for key, group in by_key.items():
            actives = [a for a in group if a.status == "active"]
            if len(actives) > 1:
                problems.append(
                    f"key {key}: {len(actives)} active atoms "
                    f"{[a.atom_id for a in actives]}"
                )
            by_id = {a.atom_id: a for a in group}
            for a in group:
                # rule 4: bidirectional chain consistency
                if a.supersedes is not None:
                    prev = by_id.get(a.supersedes)
                    if prev is None:
                        problems.append(
                            f"atom {a.atom_id}: supersedes missing atom {a.supersedes}"
                        )
                    elif prev.superseded_by != a.atom_id:
                        problems.append(
                            f"atom {a.atom_id}: supersedes {prev.atom_id} but "
                            f"{prev.atom_id}.superseded_by={prev.superseded_by}"
                        )
                    elif prev.status != "stale":
                        problems.append(
                            f"atom {a.atom_id}: supersedes {prev.atom_id} whose "
                            f"status is {prev.status} (expected stale)"
                        )
                    elif prev.valid_to_turn != a.valid_from_turn:
                        problems.append(
                            f"atom {a.atom_id}: valid_from_turn={a.valid_from_turn} "
                            f"but predecessor {prev.atom_id} closed at "
                            f"valid_to_turn={prev.valid_to_turn}"
                        )
                if a.superseded_by is not None:
                    nxt = by_id.get(a.superseded_by)
                    if nxt is None:
                        problems.append(
                            f"atom {a.atom_id}: superseded_by missing atom "
                            f"{a.superseded_by}"
                        )
                    elif nxt.supersedes != a.atom_id:
                        problems.append(
                            f"atom {a.atom_id}: superseded_by {nxt.atom_id} but "
                            f"{nxt.atom_id}.supersedes={nxt.supersedes}"
                        )
                # rule 4b: only stale atoms may be superseded
                if a.status != "stale" and a.superseded_by is not None:
                    problems.append(
                        f"atom {a.atom_id}: status={a.status} must not have "
                        f"superseded_by set"
                    )
        return problems
