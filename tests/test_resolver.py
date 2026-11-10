"""Unit tests for M2 — 4-state conflict resolution state machine (PRD §3.2)."""

from __future__ import annotations

import pytest

from chronomem.resolver import ConflictResolver, Resolution, deep_merge
from chronomem.store import MemoryStore


@pytest.fixture()
def resolver(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.sqlite3"))
    yield ConflictResolver(store)
    store.close()


# --------------------------------------------------------------------- #
# ADD
# --------------------------------------------------------------------- #
def test_add_creates_first_active(resolver):
    result = resolver.write("user", "db", "MySQL", turn_id=1)
    assert result.operation == Resolution.ADD
    assert result.new_atom.status == "active"
    assert result.new_atom.valid_from_turn == 1
    assert result.new_atom.supersedes is None
    assert result.old_atom is None


# --------------------------------------------------------------------- #
# SUPERSEDE
# --------------------------------------------------------------------- #
def test_supersede_scalar_overwrite(resolver):
    resolver.write("user", "db", "MySQL", turn_id=1)
    result = resolver.write("user", "db", "PostgreSQL", turn_id=4)
    assert result.operation == Resolution.SUPERSEDE
    old, new = result.old_atom, result.new_atom
    assert old.status == "stale" and old.valid_to_turn == 4
    assert old.superseded_by == new.atom_id
    assert new.status == "active" and new.supersedes == old.atom_id
    assert new.valid_from_turn == 4
    assert resolver.store.verify_invariants() == []


def test_supersede_dict_schema_replacement(resolver):
    resolver.write("deploy", "topology", {"mode": "single", "replicas": 1}, turn_id=1)
    result = resolver.write(
        "deploy", "topology", {"mode": "cluster", "shards": 8, "quorum": 5}, turn_id=3
    )
    assert result.operation == Resolution.SUPERSEDE
    assert result.new_atom.value == {"mode": "cluster", "shards": 8, "quorum": 5}


def test_supersede_list_replacement(resolver):
    resolver.write("user", "langs", ["zh", "en"], turn_id=1)
    result = resolver.write("user", "langs", ["zh", "en", "ja"], turn_id=2)
    assert result.operation == Resolution.SUPERSEDE
    assert result.new_atom.value == ["zh", "en", "ja"]


# --------------------------------------------------------------------- #
# MERGE_PARTIAL
# --------------------------------------------------------------------- #
def test_merge_partial_preserves_unmentioned_fields(resolver):
    full = {"host": "db.internal", "port": 5432, "timeout": 5, "retries": 3}
    resolver.write("svc", "retry_config", full, turn_id=1)
    result = resolver.write("svc", "retry_config", {"timeout": 30}, turn_id=6)
    assert result.operation == Resolution.MERGE_PARTIAL
    assert result.new_atom.value == {
        "host": "db.internal", "port": 5432, "timeout": 30, "retries": 3
    }
    # version chain stays traceable
    old, new = result.old_atom, result.new_atom
    assert old.status == "stale" and old.superseded_by == new.atom_id
    assert new.supersedes == old.atom_id
    assert resolver.store.verify_invariants() == []


def test_merge_partial_nested_deep_merge(resolver):
    resolver.write(
        "svc", "config",
        {"db": {"host": "a", "port": 1}, "cache": {"ttl": 60}}, turn_id=1,
    )
    result = resolver.write("svc", "config", {"db": {"host": "b"}}, turn_id=2)
    assert result.operation == Resolution.MERGE_PARTIAL
    assert result.new_atom.value == {
        "db": {"host": "b", "port": 1}, "cache": {"ttl": 60}
    }
    assert result.merged_value == result.new_atom.value


def test_merge_partial_keeps_history_snapshots_intact(resolver):
    resolver.write("deploy", "cfg", {"region": "cn", "replicas": 2}, turn_id=1)
    resolver.write("deploy", "cfg", {"replicas": 5}, turn_id=2)
    resolver.write("deploy", "cfg", {"region": "us"}, turn_id=3)
    versions = resolver.store.versions("deploy", "cfg")
    assert [v.value for v in versions] == [
        {"region": "cn", "replicas": 2},
        {"region": "cn", "replicas": 5},
        {"region": "us", "replicas": 5},
    ]
    assert versions[-1].status == "active"


# --------------------------------------------------------------------- #
# RETRACT
# --------------------------------------------------------------------- #
def test_retract_marks_active_retracted(resolver):
    resolver.write("user", "refunded", True, turn_id=1)
    result = resolver.retract("user", "refunded", turn_id=9)
    assert result.operation == Resolution.RETRACT
    assert result.old_atom.status == "retracted"
    assert result.old_atom.valid_to_turn == 9
    assert result.new_atom is None
    assert resolver.store.get_active("user", "refunded") is None
    assert resolver.store.verify_invariants() == []


def test_retract_without_active_is_noop(resolver):
    result = resolver.retract("ghost", "slot", turn_id=3)
    assert result.operation == Resolution.RETRACT
    assert result.old_atom is None and result.new_atom is None
    assert resolver.store.count() == 0


def test_retract_then_readd(resolver):
    resolver.write("user", "plan", "pro", turn_id=1)
    resolver.retract("user", "plan", turn_id=2)
    result = resolver.write("user", "plan", "team", turn_id=3)
    assert result.operation == Resolution.ADD
    versions = resolver.store.versions("user", "plan")
    assert [v.status for v in versions] == ["retracted", "active"]
    assert resolver.store.verify_invariants() == []


# --------------------------------------------------------------------- #
# long evolution sequences keep the invariant
# --------------------------------------------------------------------- #
def test_mixed_sequence_preserves_invariant_and_chain(resolver):
    resolver.write("user", "editor", "vim", turn_id=1)
    resolver.write("user", "editor", "emacs", turn_id=2)         # SUPERSEDE
    resolver.write("user", "editor", {"name": "neovim"}, turn_id=3)  # SUPERSEDE (dict vs str)
    resolver.retract("user", "editor", turn_id=4)                # RETRACT
    resolver.write("user", "editor", "helix", turn_id=5)         # ADD
    resolver.write("user", "editor", "helix", turn_id=6)         # SUPERSEDE (same value, new turn)

    versions = resolver.store.versions("user", "editor")
    assert [v.status for v in versions] == ["stale", "stale", "retracted", "stale", "active"]
    assert [v.valid_from_turn for v in versions] == [1, 2, 3, 5, 6]
    assert resolver.store.verify_invariants() == []

    ops = resolver.store
    # every stale version closed exactly where its successor began
    for prev, nxt in zip(versions, versions[1:]):
        if nxt.supersedes is not None:
            assert nxt.supersedes == prev.atom_id
            assert prev.valid_to_turn == nxt.valid_from_turn


def test_namespaces_are_independent(resolver):
    resolver.write("user", "db", "MySQL", turn_id=1)
    r = resolver.write("user", "db", "Oracle", turn_id=2)  # default ns
    assert r.operation == Resolution.SUPERSEDE
    store = resolver.store
    other = ConflictResolver(store, namespace="project-x")
    r2 = other.write("user", "db", "PostgreSQL", turn_id=2)
    assert r2.operation == Resolution.ADD  # isolated namespace
    assert store.verify_invariants() == []


# --------------------------------------------------------------------- #
# deep_merge unit behavior
# --------------------------------------------------------------------- #
def test_deep_merge_replaces_lists_wholesale():
    assert deep_merge({"tags": [1, 2]}, {"tags": [3]}) == {"tags": [3]}


def test_deep_merge_scalar_overrides():
    assert deep_merge({"a": 1, "b": 2}, {"b": 9, "c": 3}) == {"a": 1, "b": 9, "c": 3}


def test_deep_merge_deeply_nested():
    old = {"a": {"b": {"c": 1, "d": 2}}}
    assert deep_merge(old, {"a": {"b": {"c": 10}}}) == {"a": {"b": {"c": 10, "d": 2}}}


def test_deep_merge_patch_wins_on_type_conflict():
    assert deep_merge({"a": {"x": 1}}, {"a": [1, 2]}) == {"a": [1, 2]}
    assert deep_merge([1, 2], {"x": 1}) == {"x": 1}
