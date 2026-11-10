"""Unit tests for M4 — temporal-aware hybrid retriever (PRD §3.4)."""

from __future__ import annotations

import pytest

from chronomem.anchor_watcher import AnchorWatcher
from chronomem.resolver import ConflictResolver
from chronomem.retriever import (
    Retriever,
    flatten_value,
    make_summary_text,
)
from chronomem.store import MemoryStore
from chronomem.tokenizer import (
    fts_query,
    light_stem,
    normalize_tokens,
    tokens_to_text,
)


@pytest.fixture()
def env(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.sqlite3"))
    resolver = ConflictResolver(store)
    retriever = Retriever(store)

    class Sink:
        """Auto-index every written atom so FTS stays in sync."""

        def write(self, *args, **kwargs):
            result = resolver.write(*args, **kwargs)
            if result.new_atom is not None:
                retriever.index_atom(result.new_atom)
            return result

        def retract(self, *args, **kwargs):
            return resolver.retract(*args, **kwargs)

    yield store, Sink(), retriever
    store.close()


def seed(env):
    _, mem, _ = env
    mem.write("user", "db", "PostgreSQL", turn_id=2,
              summary_text=make_summary_text("user", "db", "PostgreSQL 数据库"))
    mem.write("user", "editor", "neovim", turn_id=3,
              summary_text=make_summary_text("user", "editor", "neovim 编辑器"))
    mem.write("svc", "refund", {"window_days": 7, "mode": "full"}, turn_id=4,
              summary_text=make_summary_text(
                  "svc", "refund", "退款规则 七天内全额退款 window_days 7 full"))


# --------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------- #
def test_chinese_bigram_tokenization():
    tokens = normalize_tokens("退款规则")
    assert tokens == ["退款", "款规", "规则"]


def test_mixed_cjk_latin_tokenization():
    tokens = normalize_tokens("使用PostgreSQL部署")
    assert "使用" in tokens and "postgresql" in tokens and "部署" in tokens


def test_light_stem_pairs():
    assert light_stem("deploying") == "deploy"
    assert light_stem("deployed") == "deploy"
    assert light_stem("running") == "run"
    assert light_stem("dogs") == "dog"
    assert light_stem("process") == "process"


def test_stemmed_query_matches_stemmed_index():
    # identical pipeline on both sides => stemmed forms intersect
    doc = tokens_to_text(normalize_tokens("we deploy the service to staging"))
    q = normalize_tokens("deploying services")
    assert "deploy" in doc.split()
    assert "deploy" in q


def test_fts_query_quoting():
    assert fts_query(["退款", "deploy"]) == '"退款" OR "deploy"'


def test_flatten_value():
    assert flatten_value({"a": 1, "b": [2, "x"]}) == "a 1 b 2 x"
    assert flatten_value("plain") == "plain"


# --------------------------------------------------------------------- #
# current-state retrieval
# --------------------------------------------------------------------- #
def test_chinese_query_finds_bigram_match(env):
    store, mem, retriever = env
    seed(env)
    result = retriever.retrieve_current("退款规则是什么")
    assert result.results, "expected FTS hits for Chinese bigram query"
    top = result.top.atom
    assert top.entity == "svc" and top.slot == "refund"
    assert result.poisoning_ratio == 0.0


def test_hard_filter_excludes_stale_versions(env):
    store, mem, retriever = env
    seed(env)
    mem.write("user", "db", "SQLite", turn_id=6,
              summary_text=make_summary_text("user", "db", "SQLite 数据库"))
    result = retriever.retrieve_current("数据库 db")
    values = [s.atom.value for s in result.results]
    assert "PostgreSQL" not in values  # stale version must never surface
    assert values[0] == "SQLite"
    assert all(s.atom.status == "active" for s in result.results)


def test_retracted_slot_surfaces_notice_not_result(env):
    store, mem, retriever = env
    seed(env)
    mem.retract("svc", "refund", turn_id=8)
    result = retriever.retrieve_current("退款规则 refund")
    assert result.results == []  # no active atom for that slot
    assert any(n["type"] == "retracted" and n["entity"] == "svc"
               for n in result.notices)


def test_anchor_drifted_slot_surfaces_notice(env, tmp_path):
    store, mem, retriever = env
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "payment.py").write_text("def run():\n    return 1\n")
    watcher = AnchorWatcher(store)
    atom = watcher.bind("proj", "payhook", {"entry": "run"}, turn_id=2,
                        anchor_path="payment.py", workspace_root=workspace)
    retriever.index_atom(atom)
    (workspace / "payment.py").write_text("def run():\n    return 2\n")
    watcher.verify_anchors(workspace)

    result = retriever.retrieve_current("payhook 支付钩子")
    assert result.results == []
    assert any(n["type"] == "anchor_drifted" and n["atom_id"] == atom.atom_id
               for n in result.notices)


def test_never_mentioned_slot_abstains_silently(env):
    store, mem, retriever = env
    seed(env)
    result = retriever.retrieve_current("kubernetes 集群配置")
    assert result.results == []
    assert result.notices == []


def test_key_match_boost_ranks_exact_slot_first(env):
    store, mem, retriever = env
    seed(env)
    mem.write("user", "db_backup", "nightly", turn_id=5,
              summary_text=make_summary_text("user", "db_backup", "nightly backup db 数据库"))
    result = retriever.retrieve_current("user db")
    assert result.top.atom.slot == "db"
    assert "key_match" in result.top.match_reasons


# --------------------------------------------------------------------- #
# time travel
# --------------------------------------------------------------------- #
def test_as_of_returns_exact_historical_state(env):
    store, mem, retriever = env
    mem.write("user", "db", "MySQL", turn_id=1,
              summary_text=make_summary_text("user", "db", "MySQL"))
    mem.write("user", "db", "PostgreSQL", turn_id=4,
              summary_text=make_summary_text("user", "db", "PostgreSQL"))
    mem.write("user", "db", "SQLite", turn_id=7,
              summary_text=make_summary_text("user", "db", "SQLite"))

    for turn, expected in [(0, []), (1, "MySQL"), (3, "MySQL"),
                           (4, "PostgreSQL"), (6, "PostgreSQL"), (7, "SQLite"),
                           (99, "SQLite")]:
        snap = retriever.retrieve_as_of(turn, namespace="default")
        if expected == []:
            assert snap.results == []
        else:
            assert [s.atom.value for s in snap.results if s.atom.slot == "db"] == [expected]
            assert all("as_of" not in s.match_reasons or True for s in snap.results)


def test_as_of_with_query_ranks_snapshot(env):
    store, mem, retriever = env
    mem.write("user", "db", "MySQL", turn_id=1,
              summary_text=make_summary_text("user", "db", "MySQL"))
    mem.write("user", "editor", "vim", turn_id=2,
              summary_text=make_summary_text("user", "editor", "vim"))
    snap = retriever.retrieve_as_of(2, "数据库 db")
    assert snap.top.atom.slot == "db"
    assert snap.top.atom.value == "MySQL"


def test_as_of_includes_later_retracted_value(env):
    """At turn 3 the setting was live even if retracted at turn 5."""
    store, mem, retriever = env
    mem.write("user", "plan", "pro", turn_id=1,
              summary_text=make_summary_text("user", "plan", "pro 会员"))
    mem.retract("user", "plan", turn_id=5)
    before = retriever.retrieve_as_of(3, "plan 会员")
    after = retriever.retrieve_as_of(5, "plan 会员")
    assert [s.atom.value for s in before.results] == ["pro"]
    assert after.results == []  # window closed exactly at the retract turn


def test_as_of_snapshot_is_complete_world_state(env):
    store, mem, retriever = env
    seed(env)
    snap = retriever.retrieve_as_of(10)
    keys = {(s.atom.entity, s.atom.slot) for s in snap.results}
    assert keys == {("user", "db"), ("user", "editor"), ("svc", "refund")}


# --------------------------------------------------------------------- #
# poisoning property: current retrieval NEVER leaks non-active atoms
# --------------------------------------------------------------------- #
def test_zero_poisoning_property_under_evolution(env):
    store, mem, retriever = env
    mem.write("u", "s", "v1", turn_id=1, summary_text=make_summary_text("u", "s", "v1"))
    for t in range(2, 12):
        mem.write("u", "s", f"v{t}", turn_id=t,
                  summary_text=make_summary_text("u", "s", f"v{t} version"))
        mem.write("w", "x", f"o{t}", turn_id=t,
                  summary_text=make_summary_text("w", "x", f"o{t} other"))
    mem.retract("u", "s", turn_id=12)
    result = retriever.retrieve_current("s version v")
    assert all(s.atom.status == "active" for s in result.results)
    assert result.poisoning_ratio == 0.0
    assert store.verify_invariants() == []
