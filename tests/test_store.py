"""Unit tests for M1 — MVCC storage engine invariants (PRD §3.1)."""

from __future__ import annotations

import json
import random
import sqlite3
import threading

import pytest

from chronomem.store import Atom, MemoryStore


def make_store(tmp_path) -> MemoryStore:
    return MemoryStore(str(tmp_path / "mem.sqlite3"))


# --------------------------------------------------------------------- #
# basic CRUD
# --------------------------------------------------------------------- #
def test_insert_and_get_active(tmp_path):
    store = make_store(tmp_path)
    atom = store.insert_atom("user", "db", "PostgreSQL", turn_id=1)
    assert isinstance(atom, Atom)
    assert atom.status == "active"
    assert atom.valid_from_turn == 1
    assert atom.valid_to_turn is None
    assert atom.value == "PostgreSQL"
    got = store.get_active("user", "db")
    assert got is not None and got.atom_id == atom.atom_id


def test_value_json_roundtrip_structured(tmp_path):
    store = make_store(tmp_path)
    value = {"host": "db.internal", "timeout": 30, "retries": [1, 2, 5]}
    atom = store.insert_atom("deploy", "config", value, turn_id=3)
    assert store.get_atom(atom.atom_id).value == value
    assert json.loads(atom.value_json) == value


def test_persistence_across_reopen(tmp_path):
    path = str(tmp_path / "mem.sqlite3")
    s1 = MemoryStore(path)
    a = s1.insert_atom("user", "lang", "中文回复", turn_id=2, source_turn_id="s1-t2")
    s1.close()
    s2 = MemoryStore(path)
    got = s2.get_atom(a.atom_id)
    assert got.value == "中文回复"
    assert got.source_turn_id == "s1-t2"


# --------------------------------------------------------------------- #
# the core single-active invariant
# --------------------------------------------------------------------- #
def test_single_active_invariant_enforced_by_index(tmp_path):
    """A second 'active' insert for the same key must be rejected by the DB."""
    store = make_store(tmp_path)
    store.insert_atom("user", "db", "MySQL", turn_id=1)
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_atom("user", "db", "PostgreSQL", turn_id=2)
    # ...but a *different* key or namespace is fine
    store.insert_atom("user", "lang", "PostgreSQL", turn_id=2)
    store.insert_atom("user", "db", "MySQL", turn_id=2, namespace="other-ns")


def test_supersede_atomic_transition(tmp_path):
    store = make_store(tmp_path)
    old = store.insert_atom("user", "db", "MySQL", turn_id=1)
    prev, new = store.supersede("user", "db", "PostgreSQL", turn_id=2)
    assert prev.atom_id == old.atom_id
    assert prev.status == "stale" and prev.valid_to_turn == 2
    assert prev.superseded_by == new.atom_id
    assert new.supersedes == old.atom_id
    assert new.status == "active" and new.valid_from_turn == 2
    assert store.get_active("user", "db").atom_id == new.atom_id
    assert store.verify_invariants() == []


def test_supersede_without_active_degenerates_to_add(tmp_path):
    store = make_store(tmp_path)
    prev, new = store.supersede("user", "lang", "Python", turn_id=1)
    assert prev is None
    assert new.status == "active" and new.supersedes is None


def test_deactivate_is_idempotent_guard(tmp_path):
    """Deactivating an already-stale atom is a no-op returning None."""
    store = make_store(tmp_path)
    a = store.insert_atom("user", "db", "MySQL", turn_id=1)
    store.deactivate(a.atom_id, new_status="retracted", valid_to_turn=2)
    assert store.deactivate(a.atom_id, new_status="stale", valid_to_turn=3) is None
    assert store.get_atom(a.atom_id).status == "retracted"


# --------------------------------------------------------------------- #
# version chain & visibility windows
# --------------------------------------------------------------------- #
def test_version_chain_ordering(tmp_path):
    store = make_store(tmp_path)
    v1 = store.insert_atom("user", "editor", "vim", turn_id=1)
    _, v2 = store.supersede("user", "editor", "emacs", turn_id=4)
    _, v3 = store.supersede("user", "editor", "neovim", turn_id=7)
    # re-fetch: v1/v2 were closed by later supersedes after they were returned
    v1 = store.get_atom(v1.atom_id)
    v2 = store.get_atom(v2.atom_id)

    versions = store.versions("user", "editor")
    assert [a.atom_id for a in versions] == [v1.atom_id, v2.atom_id, v3.atom_id]
    assert [a.status for a in versions] == ["stale", "stale", "active"]
    # bidirectional chain
    assert v2.supersedes == v1.atom_id and v1.superseded_by == v2.atom_id
    assert v3.supersedes == v2.atom_id and v2.superseded_by == v3.atom_id


def test_snapshot_as_of_visibility_windows(tmp_path):
    store = make_store(tmp_path)
    v1 = store.insert_atom("user", "editor", "vim", turn_id=1)
    store.deactivate(v1.atom_id, new_status="stale", valid_to_turn=4, superseded_by=None)
    v2 = store.insert_atom("user", "editor", "emacs", turn_id=4, supersedes=v1.atom_id)

    # at turn 2 the world still used vim
    snap2 = [a for a in store.snapshot_as_of(2) if a.key == ("default", "user", "editor")]
    assert [a.value for a in snap2] == ["vim"]
    # at turn 5 emacs is live
    snap5 = [a for a in store.snapshot_as_of(5) if a.key == ("default", "user", "editor")]
    assert [a.value for a in snap5] == ["emacs"]
    # boundary: valid_to_turn is exclusive — at turn 4 vim is already gone
    snap4 = [a for a in store.snapshot_as_of(4) if a.key == ("default", "user", "editor")]
    assert [a.value for a in snap4] == ["emacs"]
    # nothing existed before turn 1
    assert [a for a in store.snapshot_as_of(0) if a.key == ("default", "user", "editor")] == []


def test_retract_and_drift_statuses(tmp_path):
    store = make_store(tmp_path)
    a = store.insert_atom("user", "refunded", True, turn_id=1)
    store.deactivate(a.atom_id, new_status="retracted", valid_to_turn=3)
    b = store.insert_atom("deploy", "cfg", {"timeout": 5}, turn_id=2)
    store.deactivate(b.atom_id, new_status="anchor_drifted", valid_to_turn=5)
    assert store.get_atom(a.atom_id).status == "retracted"
    assert store.get_atom(b.atom_id).status == "anchor_drifted"
    assert store.verify_invariants() == []
    # retracted atom is invisible in the *current* active set...
    assert store.get_active("user", "refunded") is None
    # ...and in the as-of snapshot after its window closed
    assert all(x.key != ("default", "user", "refunded") for x in store.snapshot_as_of(4))


# --------------------------------------------------------------------- #
# randomized stress: invariant must survive arbitrary op sequences
# --------------------------------------------------------------------- #
def test_random_op_sequence_preserves_invariants(tmp_path):
    """Seeded random ADD/SUPERSEDE/RETRACT/DRIFT storm — invariant always holds."""
    rng = random.Random(20261110)
    store = make_store(tmp_path)
    turn = 0
    live_ids: dict = {}

    for _ in range(500):
        turn += 1
        ns = f"ns{rng.randint(0, 2)}"
        ent, slot = f"ent{rng.randint(0, 9)}", f"slot{rng.randint(0, 5)}"
        key = (ns, ent, slot)
        op = rng.choices(["add", "overwrite", "retract", "drift"],
                         weights=[4, 4, 1, 1])[0]
        live = live_ids.get(key)

        if live is None:
            atom = store.insert_atom(ent, slot, f"val{turn}", turn_id=turn,
                                     namespace=ns)
        elif op == "retract":
            store.deactivate(live, new_status="retracted", valid_to_turn=turn)
            atom = None
        elif op == "drift":
            store.deactivate(live, new_status="anchor_drifted", valid_to_turn=turn)
            atom = None
        else:  # add/overwrite on a live key -> atomic supersede
            _, atom = store.supersede(ent, slot, f"val{turn}", turn_id=turn,
                                      namespace=ns)

        live_ids[key] = atom.atom_id if atom is not None else None
        # invariant checked after EVERY operation
        assert store.verify_invariants() == []

    # DB-level re-check: partial unique index agrees with verify_invariants
    for key in {a.key for a in store.all_atoms()}:
        nns, nent, nslot = key
        n_active = len([x for x in store.versions(nent, nslot, namespace=nns)
                        if x.status == "active"])
        assert n_active <= 1


def test_chain_integrity_after_heavy_overwrites(tmp_path):
    store = make_store(tmp_path)
    for i in range(60):
        store.supersede("proj", "branch", f"feat/{i}", turn_id=i + 1)
    chain = store.versions("proj", "branch")
    assert len(chain) == 60
    for prev, nxt in zip(chain, chain[1:]):
        assert prev.superseded_by == nxt.atom_id
        assert nxt.supersedes == prev.atom_id
        assert prev.valid_to_turn == nxt.valid_from_turn
    assert chain[-1].status == "active"
    assert store.verify_invariants() == []


# --------------------------------------------------------------------- #
# thread-safety: concurrent writers must never break the invariant
# --------------------------------------------------------------------- #
def test_concurrent_writers_keep_single_active(tmp_path):
    """N threads run atomic supersede on the same key concurrently."""
    store = make_store(tmp_path)
    errors: list = []

    def writer(tid: int) -> None:
        try:
            for i in range(20):
                turn = tid * 100 + i + 1
                store.supersede("shared", "lock", f"t{tid}-{i}", turn_id=turn)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert store.verify_invariants() == []
    chain = store.versions("shared", "lock")
    assert len(chain) == 6 * 20
    actives = [a for a in chain if a.status == "active"]
    assert len(actives) == 1
    # every version is bidirectionally linked (verify_invariants checked this
    # for all atoms, including the interleaved-turn concurrent chain)
    linked = {a.supersedes for a in chain} | {a.superseded_by for a in chain}
    assert all(a.atom_id in linked or a.supersedes is None for a in chain)
