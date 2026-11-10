"""End-to-end tests for the CLI and visualizer deliverables."""

from __future__ import annotations

import json

import pytest

from chronomem.cli import build_demo_store, build_parser, main
from chronomem.store import MemoryStore
from chronomem.visualizer import collect_visualization_data, render_visualizer


@pytest.fixture()
def db(tmp_path):
    return str(tmp_path / "cli.db")


def run_cli(argv):
    return main(argv)


def test_add_supersede_retract_flow(db, capsys):
    assert run_cli(["--db", db, "add", "--entity", "user", "--slot", "db",
                    "--value", '"MySQL"', "--turn", "1"]) == 0
    assert run_cli(["--db", db, "add", "--entity", "user", "--slot", "db",
                    "--value", '"PostgreSQL"', "--turn", "2"]) == 0
    out = capsys.readouterr().out
    assert "[SUPERSEDE]" in out
    assert run_cli(["--db", db, "retract", "--entity", "user", "--slot", "db",
                    "--turn", "3"]) == 0
    assert "[RETRACT]" in capsys.readouterr().out

    store = MemoryStore(db)
    assert store.get_active("user", "db") is None
    assert store.verify_invariants() == []
    store.close()


def test_merge_partial_via_cli(db, capsys):
    run_cli(["--db", db, "add", "--entity", "svc", "--slot", "cfg",
             "--value", '{"timeout": 5, "retries": 3}', "--turn", "1"])
    run_cli(["--db", db, "add", "--entity", "svc", "--slot", "cfg",
             "--value", '{"timeout": 30}', "--turn", "2"])
    out = capsys.readouterr().out
    assert "[MERGE_PARTIAL]" in out
    assert '"retries": 3' in out.replace(" ", "") or '"retries":3' in out.replace(" ", "")


def test_retrieve_and_as_of_cli(db, capsys):
    run_cli(["--db", db, "add", "--entity", "user", "--slot", "db",
             "--value", '"MySQL"', "--turn", "1"])
    run_cli(["--db", db, "add", "--entity", "user", "--slot", "db",
             "--value", '"PostgreSQL"', "--turn", "4"])
    run_cli(["--db", db, "retrieve", "user db", "--entity", "user", "--slot", "db"])
    out = capsys.readouterr().out
    assert "PostgreSQL" in out
    assert "poisoning_ratio: 0.000" in out

    run_cli(["--db", db, "as-of", "2", "user db"])
    assert "MySQL" in capsys.readouterr().out


def test_inspect_cli(db, capsys):
    run_cli(["--db", db, "add", "--entity", "user", "--slot", "db",
             "--value", '"MySQL"', "--turn", "1"])
    run_cli(["--db", db, "inspect"])
    out = capsys.readouterr().out
    assert "[default] user · db" in out
    assert "superseded_by" not in out  # single version, no chain links yet


def test_visualize_demo(tmp_path):
    out = tmp_path / "vis" / "index.html"
    assert run_cli(["visualize", "--demo", "--out", str(out)]) == 0
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>")
    assert "ChronoMem" in html and "const DATA =" in html
    # embedded data parses as JSON inside the script tag
    payload = html.split("const DATA =", 1)[1].split(";\n", 1)[0]
    data = json.loads(payload)
    assert data["n_atoms"] > 0 and data["lanes"]
    statuses = {v["status"] for lane in data["lanes"] for v in lane["versions"]}
    assert {"active", "stale", "retracted", "anchor_drifted"} <= statuses, \
        "demo data should showcase all four lifecycle states"


def test_build_demo_store_invariants(tmp_path):
    store = build_demo_store(str(tmp_path / "demo.db"), str(tmp_path / "ws"))
    assert store.verify_invariants() == []
    active = [a for a in store.all_atoms() if a.status == "active"]
    keys = [(a.namespace, a.entity, a.slot) for a in active]
    assert len(keys) == len(set(keys))
    store.close()


def test_collect_visualization_data_shape(tmp_path):
    store = MemoryStore(str(tmp_path / "x.db"))
    store.insert_atom("u", "s", {"k": 1}, turn_id=2)
    data = collect_visualization_data(store)
    assert data["n_atoms"] == 1
    lane = data["lanes"][0]
    assert lane["key"] == "default · u · s"
    assert lane["versions"][0]["open"] is True
    assert lane["versions"][0]["value"] == {"k": 1}
    store.close()


def test_cli_parser_has_all_subcommands():
    p = build_parser()
    choices = set(p._subparsers._group_actions[0].choices.keys())
    assert {"add", "retract", "retrieve", "as-of", "verify-anchors",
            "inspect", "bench", "visualize"} <= choices
