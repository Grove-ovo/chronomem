"""Benchmark orchestration: run all four systems over ChronoBench-100."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from benchmarks.dataset import ABSTAIN, build_chronobench_100
from benchmarks.metrics import evaluate_records
from benchmarks.systems import (
    ChronoMemSystem,
    FullContext,
    NaiveVectorRAG,
    RollingSummary,
    count_tokens,
)

__all__ = ["run_benchmark", "apply_workspace_event"]

SYSTEM_FACTORIES = {
    "ChronoMem": lambda workdir: ChronoMemSystem(workdir),
    "Naive Vector RAG": lambda workdir: NaiveVectorRAG(),
    "Rolling Summary": lambda workdir: RollingSummary(),
    "Full-Context": lambda workdir: FullContext(),
}


# --------------------------------------------------------------------- #
# workspace (code anchor) management
# --------------------------------------------------------------------- #
def setup_workspace(sample: Dict[str, Any], root: Path) -> None:
    if sample["category"] != "anchor_drift":
        return
    main = root / sample["service_path"]
    main.parent.mkdir(parents=True, exist_ok=True)
    main.write_text(sample["service_template"], encoding="utf-8")
    stats = root / sample["stats_path"]
    stats.parent.mkdir(parents=True, exist_ok=True)
    stats.write_text(sample["stats_template"], encoding="utf-8")


def apply_workspace_event(root: Path, event: Dict[str, Any]) -> None:
    path = root / event["path"]
    if event["type"] == "delete_file":
        path.unlink(missing_ok=True)
        return
    text = path.read_text(encoding="utf-8")
    if event["type"] == "rename_symbol":
        text = text.replace(f"def {event['old']}(", f"def {event['new']}(")
    elif event["type"] == "delete_symbol":
        import ast

        tree = ast.parse(text)
        for node in ast.walk(tree):
            if (isinstance(node, ast.ClassDef) and node.name == event["cls"]):
                for child in node.body:
                    if (isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                            and child.name == event["symbol"]):
                        lines = text.splitlines(keepends=True)
                        start = child.lineno - 1
                        end = child.end_lineno
                        # also strip a preceding decorator/docstring gap
                        text = "".join(lines[:start] + lines[end:])
                        break
        # collapse resulting double blank lines
        text = text.replace("\n\n\n\n", "\n\n")
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------- #
# ground-truth evolution for the poisoning audit
# --------------------------------------------------------------------- #
@dataclass
class KeyEvolution:
    statements: List[Dict[str, Any]]          # value statements, turn-ordered
    retract_turn: Optional[int]
    drift_turn: Optional[int]                 # refactor turn for anchored keys
    anchor_path: Optional[str]

    def state_at(self, turn_id: int) -> Any:
        state: Any = None
        for s in self.statements:
            if s["turn_id"] > turn_id:
                break
            if s["op"] == "retract":
                state = None
            elif isinstance(s["value"], dict) and isinstance(state, dict):
                state.update(s["value"])
            elif isinstance(s["value"], dict):
                state = dict(s["value"])
            else:
                state = s["value"]
        return state


def build_evolution(sample: Dict[str, Any]) -> Dict[Any, KeyEvolution]:
    by_key: Dict[Any, List[Dict[str, Any]]] = {}
    anchors: Dict[Any, str] = {}
    for sess in sample["sessions"]:
        for t in sess["turns"]:
            st = t.get("statement")
            if not st:
                continue
            key = (st["entity"], st["slot"])
            by_key.setdefault(key, []).append({
                "turn_id": t["turn_id"], "op": st["op"], "value": st["value"],
            })
            if st.get("anchor_path"):
                anchors.setdefault(key, st["anchor_path"])

    refactor_turns = {ev["path"]: ev["turn_id"]
                      for ev in sample.get("workspace_events", [])}
    evolution: Dict[Any, KeyEvolution] = {}
    for key, stmts in by_key.items():
        stmts.sort(key=lambda s: s["turn_id"])
        retract_turn = next((s["turn_id"] for s in stmts if s["op"] == "retract"), None)
        anchor = anchors.get(key)
        drift_turn = None
        if anchor:
            for path, turn in refactor_turns.items():
                if anchor.split("::")[0] == path:
                    drift_turn = turn
                    break
        evolution[key] = KeyEvolution(stmts, retract_turn, drift_turn, anchor)
    return evolution


def _is_poisoned(item: Dict[str, Any], evo: KeyEvolution, query_turn: int) -> bool:
    """True when the item carries information that is no longer valid at
    ``query_turn`` (superseded / retracted / anchor-drifted)."""
    if evo.drift_turn is not None and query_turn >= evo.drift_turn:
        return True  # every record of an anchor-drifted key is poisoned
    if evo.retract_turn is not None and query_turn >= evo.retract_turn:
        return True  # records of a retracted setting are poisoned
    current = evo.state_at(query_turn)
    value = item["value"]
    if current is None:
        return value is not None
    if isinstance(current, dict) and isinstance(value, dict):
        return any(v != current.get(f) for f, v in value.items())
    return value != current


# --------------------------------------------------------------------- #
# main runner
# --------------------------------------------------------------------- #
def run_benchmark(
    samples: Optional[List[Dict[str, Any]]] = None,
    seed: int = 20261111,
) -> Dict[str, Any]:
    samples = samples if samples is not None else build_chronobench_100(seed)
    records: List[Dict[str, Any]] = []

    for sample in samples:
        with tempfile.TemporaryDirectory(prefix="chronobench-") as td:
            root = Path(td)
            setup_workspace(sample, root)

            systems = {name: factory(str(root)) for name, factory in SYSTEM_FACTORIES.items()}
            if "ChronoMem" in systems:
                systems["ChronoMem"].workspace = str(root)

            # ingest the full stream in turn order
            turns = [t for sess in sample["sessions"] for t in sess["turns"]]
            prev_session = None
            for turn in turns:
                if prev_session is not None and turn["session_id"] != prev_session:
                    for system in systems.values():
                        system.end_session(prev_session)
                prev_session = turn["session_id"]
                event = turn.get("workspace_event")
                if event:
                    apply_workspace_event(root, event)
                for system in systems.values():
                    system.ingest_turn(turn, str(root))
            if prev_session is not None:
                for system in systems.values():
                    system.end_session(prev_session)

            evolution = build_evolution(sample)

            for query in sample["queries"]:
                for name, system in systems.items():
                    ans = system.answer(query, str(root))
                    evo = evolution.get((query["entity"], query["slot"]))
                    poisoned = 0
                    if evo is not None and query["type"] == "current":
                        for item in ans.context:
                            if _is_poisoned(item, evo, query["turn_id"]):
                                poisoned += 1
                    elif evo is None:
                        # never-mentioned keys: all context items are noise
                        poisoned = 0
                    records.append({
                        "sample_id": sample["sample_id"],
                        "category": sample["category"],
                        "query_id": query["query_id"],
                        "query_type": query["type"],
                        "system": name,
                        "expected": query["expected"],
                        "expected_reason": query.get("expected_reason"),
                        "expected_slots": query.get("expected_slots"),
                        "abstained": ans.abstained,
                        "reason": ans.reason,
                        "value": ans.value,
                        "slots": ans.slots,
                        "context_poisoned": poisoned,
                        "context_total": len(ans.context),
                        "context_tokens": ans.context_tokens,
                        "as_of_turn": query.get("as_of_turn"),
                    })

            for system in systems.values():
                close = getattr(system, "close", None)
                if callable(close):
                    close()

    metrics = evaluate_records(records)
    return {"records": records, "metrics": metrics, "n_samples": len(samples)}
