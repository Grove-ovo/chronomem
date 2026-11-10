"""ChronoBench-100 dataset + baseline systems test suite (PRD §3.5)."""

from __future__ import annotations

import json

import pytest

from benchmarks.dataset import (
    ABSTAIN,
    CATEGORIES,
    PER_CATEGORY,
    build_chronobench_100,
    dataset_summary,
)
from benchmarks.metrics import evaluate_records, slot_prf1
from benchmarks.runner import run_benchmark
from benchmarks.systems import count_tokens


# --------------------------------------------------------------------- #
# dataset structure & determinism
# --------------------------------------------------------------------- #
def test_dataset_has_100_samples_across_5_categories():
    samples = build_chronobench_100()
    assert len(samples) == 100
    summary = dataset_summary(samples)
    assert summary["by_category"] == {c: PER_CATEGORY for c in CATEGORIES}
    assert summary["queries"] >= 100


def test_dataset_generation_is_deterministic():
    a = json.dumps(build_chronobench_100(), sort_keys=True, ensure_ascii=False)
    b = json.dumps(build_chronobench_100(), sort_keys=True, ensure_ascii=False)
    assert a == b


def test_reversal_samples_have_3_to_5_reversals():
    samples = build_chronobench_100()
    revs = [s for s in samples if s["category"] == "reversal"]
    assert all(3 <= s["n_reversals"] <= 5 for s in revs)


def test_partial_update_final_config_is_initial_plus_updates():
    samples = build_chronobench_100()
    partials = [s for s in samples if s["category"] == "partial_update"]
    for s in partials:
        changed = {k: v for k, v in s["final_config"].items()
                   if s["initial_config"].get(k) != v}
        assert 2 <= len(changed) <= 4
        # all untouched fields preserved
        for k, v in s["initial_config"].items():
            if k not in changed:
                assert s["final_config"][k] == v


def test_multi_session_streams_across_categories():
    samples = build_chronobench_100()
    for s in samples:
        assert s["n_sessions"] >= 2
        turn_ids = [t["turn_id"] for sess in s["sessions"] for t in sess["turns"]]
        assert turn_ids == sorted(turn_ids)


def test_retraction_queries_carry_abstain_ground_truth():
    samples = build_chronobench_100()
    for s in (x for x in samples if x["category"] == "retraction"):
        kinds = {q["expected_reason"] for q in s["queries"]}
        assert kinds == {"retracted", "no_evidence"}
        assert all(q["expected"] == ABSTAIN for q in s["queries"])


def test_anchor_samples_have_workspace_and_drift_query():
    samples = build_chronobench_100()
    for s in (x for x in samples if x["category"] == "anchor_drift"):
        assert s["workspace_events"], "anchor samples need a refactor event"
        q1 = s["queries"][0]
        assert q1["expected"] == ABSTAIN and q1["expected_reason"] == "anchor_drifted"
        q2 = s["queries"][1]
        assert q2["expected"] != ABSTAIN  # control query stays answerable


# --------------------------------------------------------------------- #
# token metering
# --------------------------------------------------------------------- #
def test_count_tokens_is_sane():
    assert count_tokens("hello world") >= 2
    assert count_tokens("退款规则") >= 3  # bigram-ish meter, never zero
    assert count_tokens("") == 0


# --------------------------------------------------------------------- #
# metric units
# --------------------------------------------------------------------- #
def test_slot_prf1_perfect_and_degraded():
    gold = {"a": 1, "b": 2, "c": 3}
    p, r, f1 = slot_prf1({"a": 1, "b": 2, "c": 3}, gold)
    assert (p, r, f1) == (1.0, 1.0, 1.0)
    # one missing + one wrong value
    p, r, f1 = slot_prf1({"a": 1, "b": 9}, gold)
    assert p == 0.5 and r == 1 / 3 and 0 < f1 < 0.6
    p, r, f1 = slot_prf1(None, gold)
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_evaluate_records_aggregates():
    recs = [
        {"system": "X", "category": "reversal", "query_type": "current",
         "expected": "PG", "expected_reason": None, "abstained": False,
         "reason": None, "value": "PG", "slots": None,
         "context_poisoned": 0, "context_total": 2, "context_tokens": 40},
        {"system": "X", "category": "reversal", "query_type": "current",
         "expected": "PG", "expected_reason": None, "abstained": False,
         "reason": None, "value": "MySQL", "slots": None,
         "context_poisoned": 2, "context_total": 2, "context_tokens": 40},
        {"system": "X", "category": "partial_update", "query_type": "current",
         "expected": None, "expected_reason": None, "abstained": False,
         "reason": None, "value": None, "slots": {"a": 1, "b": 2, "c": 3},
         "expected_slots": {"a": 1, "b": 2, "c": 3},
         "context_poisoned": 0, "context_total": 1, "context_tokens": 30},
        {"system": "X", "category": "retraction", "query_type": "current",
         "expected": ABSTAIN, "expected_reason": "retracted", "abstained": True,
         "reason": "retracted", "value": None, "slots": None,
         "context_poisoned": 0, "context_total": 0, "context_tokens": 0},
    ]
    out = evaluate_records(recs)["per_system"]["X"]
    assert out["Temporal_State_Acc"] == 0.5
    assert out["Stale_Poisoning_Rate"] == pytest.approx(2 / 5)
    assert out["Slot_Merge_F1"] == 1.0
    assert out["Abstention_Acc"] == 1.0


# --------------------------------------------------------------------- #
# end-to-end pipeline on a small subset
# --------------------------------------------------------------------- #
def _subset(n=3):
    samples = build_chronobench_100()
    by_cat = {}
    for s in samples:
        by_cat.setdefault(s["category"], []).append(s)
    return [s for cat in CATEGORIES for s in by_cat[cat][:n]]


def test_runner_end_to_end_on_subset():
    result = run_benchmark(samples=_subset())
    assert result["n_samples"] == 15
    per_system = result["metrics"]["per_system"]
    assert set(per_system) == {"ChronoMem", "Naive Vector RAG",
                               "Rolling Summary", "Full-Context"}
    # every query answered by every system (3+3 reversal/partial ×1,
    # retraction/asof/anchor ×2 each → 3+3+6+6+6 = 24)
    assert all(m["n_queries"] == 24 for m in per_system.values())

    chrono = per_system["ChronoMem"]
    assert chrono["Stale_Poisoning_Rate"] == 0.0, (
        "ChronoMem must never leak stale records into context")
    assert chrono["Temporal_State_Acc"] == 1.0
    assert chrono["Slot_Merge_F1"] == 1.0
    assert chrono["Abstention_Acc"] == 1.0
    assert chrono["As_Of_Acc"] == 1.0


def test_chronomem_outperforms_naive_rag_on_subset():
    result = run_benchmark(samples=_subset())
    per_system = result["metrics"]["per_system"]
    chrono, naive = per_system["ChronoMem"], per_system["Naive Vector RAG"]
    assert chrono["Temporal_State_Acc"] > naive["Temporal_State_Acc"]
    assert chrono["Stale_Poisoning_Rate"] < naive["Stale_Poisoning_Rate"]
    assert chrono["Abstention_Acc"] > naive["Abstention_Acc"]


def test_full_context_has_high_poisoning():
    result = run_benchmark(samples=_subset())
    per_system = result["metrics"]["per_system"]
    # full-context keeps every historical statement in play
    assert per_system["Full-Context"]["Stale_Poisoning_Rate"] > 0.4
