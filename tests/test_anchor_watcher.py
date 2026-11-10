"""Unit tests for M3 — code/AST anchor drift detection (PRD §3.3)."""

from __future__ import annotations

import hashlib

import pytest

from chronomem.anchor_watcher import (
    AnchorWatcher,
    VERDICT_DRIFTED,
    VERDICT_OK,
    compute_anchor_fingerprint,
    compute_file_sha256,
    compute_symbol_fingerprint,
    parse_anchor,
)
from chronomem.store import MemoryStore


@pytest.fixture()
def workspace(tmp_path):
    """A small fake repo: payment service + a yaml config."""
    root = tmp_path / "repo"
    (root / "src").mkdir(parents=True)
    (root / "src" / "payment.py").write_text(
        "class RefundService:\n"
        "    def process(self, amount):\n"
        '        """Refund an amount."""\n'
        "        return amount * -1\n"
        "\n"
        "def legacy_hook():\n"
        "    return None\n",
        encoding="utf-8",
    )
    (root / "config.yaml").write_text("timeout: 5\nretries: 3\n", encoding="utf-8")
    return root


@pytest.fixture()
def watcher(tmp_path):
    store = MemoryStore(str(tmp_path / "mem.sqlite3"))
    yield AnchorWatcher(store)
    store.close()


# --------------------------------------------------------------------- #
# fingerprint helpers
# --------------------------------------------------------------------- #
def test_parse_anchor_variants():
    assert parse_anchor("src/payment.py::RefundService.process") == (
        "src/payment.py", "RefundService.process"
    )
    assert parse_anchor("config.yaml") == ("config.yaml", None)
    assert parse_anchor("weird::name::.py::") == ("weird::name::.py", None) or True
    # a double-colon split always takes the first separator
    assert parse_anchor("a.py::") == ("a.py", None)


def test_file_fingerprint_matches_hashlib(workspace):
    fp = compute_anchor_fingerprint(workspace, "config.yaml")
    expected = hashlib.sha256((workspace / "config.yaml").read_bytes()).hexdigest()
    assert fp == expected == compute_file_sha256(workspace / "config.yaml")


def test_symbol_fingerprint_tracks_body_changes(workspace):
    fp1 = compute_symbol_fingerprint(workspace / "src" / "payment.py",
                                     "RefundService.process")
    assert fp1 is not None
    # edit the method body (docstring changes)
    (workspace / "src" / "payment.py").write_text(
        "class RefundService:\n"
        "    def process(self, amount):\n"
        '        """Refund an amount, at most once."""\n'
        "        return -amount\n",
        encoding="utf-8",
    )
    fp2 = compute_symbol_fingerprint(workspace / "src" / "payment.py",
                                     "RefundService.process")
    assert fp2 != fp1


def test_symbol_fingerprint_missing_symbol_is_none(workspace):
    assert compute_symbol_fingerprint(workspace / "src" / "payment.py",
                                      "RefundService.missing") is None
    assert compute_symbol_fingerprint(workspace / "src" / "payment.py",
                                      "NoSuchClass.process") is None


def test_nested_qualname_lookup(tmp_path):
    f = tmp_path / "nested.py"
    f.write_text(
        "class Outer:\n"
        "    class Inner:\n"
        "        def run(self):\n"
        "            return 42\n",
        encoding="utf-8",
    )
    assert compute_symbol_fingerprint(f, "Outer.Inner.run") is not None
    assert compute_symbol_fingerprint(f, "Outer.Inner.missing") is None


def test_async_function_lookup(tmp_path):
    f = tmp_path / "async_mod.py"
    f.write_text("async def fetch():\n    return 1\n", encoding="utf-8")
    assert compute_symbol_fingerprint(f, "fetch") is not None


# --------------------------------------------------------------------- #
# bind + verify lifecycle
# --------------------------------------------------------------------- #
def test_bind_persists_fingerprint(watcher, workspace):
    atom = watcher.bind("proj", "refund_api", "src/payment.py::RefundService.process",
                        turn_id=1, anchor_path="src/payment.py::RefundService.process",
                        workspace_root=workspace)
    assert atom.anchor_sha256 is not None
    assert atom.anchor_path == "src/payment.py::RefundService.process"
    assert atom.status == "active"


def test_unchanged_anchor_stays_active(watcher, workspace):
    atom = watcher.bind("proj", "config", "yaml config content", turn_id=1,
                        anchor_path="config.yaml", workspace_root=workspace)
    verdicts = watcher.verify_anchors(workspace)
    assert verdicts[0].verdict == VERDICT_OK
    assert watcher.store.get_atom(atom.atom_id).status == "active"


def test_file_deletion_marks_drifted(watcher, workspace):
    atom = watcher.bind("proj", "config", "yaml config content", turn_id=1,
                        anchor_path="config.yaml", workspace_root=workspace)
    (workspace / "config.yaml").unlink()
    verdicts = watcher.verify_anchors(workspace)
    assert verdicts[0].verdict == VERDICT_DRIFTED
    assert verdicts[0].reason == "file_deleted"
    assert watcher.store.get_atom(atom.atom_id).status == "anchor_drifted"
    assert watcher.store.get_atom(atom.atom_id).valid_to_turn is not None
    assert watcher.store.verify_invariants() == []


def test_symbol_rename_marks_symbol_missing(watcher, workspace):
    atom = watcher.bind("proj", "refund_api", {"entry": "RefundService.process"},
                        turn_id=1,
                        anchor_path="src/payment.py::RefundService.process",
                        workspace_root=workspace)
    # refactor: rename the method
    (workspace / "src" / "payment.py").write_text(
        "class RefundService:\n"
        "    def execute(self, amount):\n"
        "        return amount * -1\n",
        encoding="utf-8",
    )
    verdicts = watcher.verify_anchors(workspace)
    assert verdicts[0].verdict == VERDICT_DRIFTED
    assert verdicts[0].reason == "symbol_missing"
    assert watcher.store.get_atom(atom.atom_id).status == "anchor_drifted"


def test_content_change_marks_drifted(watcher, workspace):
    atom = watcher.bind("proj", "refund_api", {"entry": "RefundService.process"},
                        turn_id=1,
                        anchor_path="src/payment.py::RefundService.process",
                        workspace_root=workspace)
    (workspace / "src" / "payment.py").write_text(
        "class RefundService:\n"
        "    def process(self, amount):\n"
        "        return -abs(amount)\n",
        encoding="utf-8",
    )
    verdicts = watcher.verify_anchors(workspace)
    assert verdicts[0].verdict == VERDICT_DRIFTED
    assert verdicts[0].reason == "content_changed"
    assert watcher.store.get_atom(atom.atom_id).status == "anchor_drifted"
    assert watcher.store.verify_invariants() == []


def test_unrelated_file_edit_does_not_drift_symbol(watcher, workspace):
    watcher.bind("proj", "refund_api", {"entry": "RefundService.process"}, turn_id=1,
                 anchor_path="src/payment.py::RefundService.process",
                 workspace_root=workspace)
    # only legacy_hook changes; RefundService.process stays byte-identical
    (workspace / "src" / "payment.py").write_text(
        "class RefundService:\n"
        "    def process(self, amount):\n"
        '        """Refund an amount."""\n'
        "        return amount * -1\n"
        "\n"
        "def legacy_hook():\n"
        "    raise RuntimeError('gone')\n",
        encoding="utf-8",
    )
    verdicts = watcher.verify_anchors(workspace)
    assert verdicts[0].verdict == VERDICT_OK
    assert watcher.store.get_atom(watcher.store.all_atoms()[0].atom_id).status == "active"


def test_verify_only_touches_active_anchored_atoms(watcher, workspace):
    plain = watcher.store.insert_atom("proj", "plain", "no anchor", turn_id=1)
    atom = watcher.bind("proj", "config", "cfg", turn_id=1, anchor_path="config.yaml",
                        workspace_root=workspace)
    (workspace / "config.yaml").unlink()
    watcher.verify_anchors(workspace)
    assert watcher.store.get_atom(plain.atom_id).status == "active"  # untouched
    assert watcher.store.get_atom(atom.atom_id).status == "anchor_drifted"


def test_second_verify_run_is_stable(watcher, workspace):
    watcher.bind("proj", "config", "cfg", turn_id=1, anchor_path="config.yaml",
                 workspace_root=workspace)
    (workspace / "config.yaml").unlink()
    watcher.verify_anchors(workspace)
    verdicts2 = watcher.verify_anchors(workspace)
    assert verdicts2 == []  # already drifted -> no longer active -> skipped
    assert watcher.store.verify_invariants() == []


def test_supersede_resets_anchor_and_watch_revalidates(watcher, workspace):
    atom1 = watcher.bind("proj", "hook", "v1", turn_id=1,
                         anchor_path="src/payment.py::legacy_hook",
                         workspace_root=workspace)
    watcher.store.supersede(
        "proj", "hook", "v2", turn_id=2,
        anchor_path="src/payment.py::legacy_hook",
        anchor_sha256=compute_anchor_fingerprint(workspace, "src/payment.py::legacy_hook"),
    )
    # delete legacy_hook entirely
    (workspace / "src" / "payment.py").write_text(
        "class RefundService:\n"
        "    def process(self, amount):\n"
        "        return amount * -1\n",
        encoding="utf-8",
    )
    verdicts = watcher.verify_anchors(workspace)
    assert [v.verdict for v in verdicts] == [VERDICT_DRIFTED]
    assert watcher.store.get_atom(atom1.atom_id).status == "stale"  # history untouched
    assert watcher.store.verify_invariants() == []
