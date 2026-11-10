"""Deterministic metric computation for ChronoBench-100 (PRD §2).

Four headline metrics:

* ``Temporal_State_Acc``   — strict latest-state adoption on overwritten slots
* ``Stale_Poisoning_Rate`` — share of retrieved context items that carry
                              stale / retracted / anchor-drifted information
* ``Slot_Merge_F1``        — field-level P/R/F1 on partial-update final states
* ``Abstention_Acc``       — correct refusal on retracted / no-evidence /
                              anchor-drifted queries

Plus an auxiliary ``As_Of_Acc`` (time-travel Q&A) and context-token costs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from benchmarks.dataset import ABSTAIN

__all__ = ["slot_prf1", "evaluate_records", "HEADLINE_METRICS"]

HEADLINE_METRICS = ["Temporal_State_Acc", "Stale_Poisoning_Rate",
                    "Slot_Merge_F1", "Abstention_Acc"]


def slot_prf1(pred: Optional[Dict[str, Any]],
              gold: Dict[str, Any]) -> Tuple[float, float, float]:
    """Field-level precision/recall/F1 (PRD §2 metric 3).

    A predicted field counts toward precision; it is a true positive only
    when its value equals the ground truth (wrong values also count as
    misses for recall).
    """
    gold_fields = set(gold.keys())
    pred = pred or {}
    pred_fields = set(pred.keys())
    tp = sum(1 for f in gold_fields & pred_fields if pred[f] == gold[f])
    p = tp / len(pred_fields) if pred_fields else 0.0
    r = tp / len(gold_fields) if gold_fields else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1


def _value_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a == b
    return a == b


def evaluate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metric values from per-query evaluation records.

    Each record: {system, category, query_type, expected, expected_reason,
    expected_slots, abstained, reason, value, slots,
    context_poisoned: int, context_total: int}
    """
    per_system: Dict[str, Dict[str, Any]] = {}

    systems = sorted({r["system"] for r in records})
    for system in systems:
        rs = [r for r in records if r["system"] == system]

        # 1. Temporal_State_Acc — reversal-category current-state queries
        tsa_q = [r for r in rs
                 if r["category"] == "reversal" and r["query_type"] == "current"]
        tsa_ok = sum(1 for r in tsa_q
                     if not r["abstained"] and _value_equal(r["value"], r["expected"]))
        temporal_acc = tsa_ok / len(tsa_q) if tsa_q else None

        # 2. Stale_Poisoning_Rate — over all current-state queries with context
        pois_q = [r for r in rs if r["query_type"] == "current"]
        total_items = sum(r["context_total"] for r in pois_q)
        poisoned_items = sum(r["context_poisoned"] for r in pois_q)
        poisoning = poisoned_items / total_items if total_items else None

        # 3. Slot_Merge_F1 — partial_update category
        f1s = []
        for r in rs:
            if r["category"] == "partial_update" and r["expected_slots"]:
                _, _, f1 = slot_prf1(r["slots"], r["expected_slots"])
                f1s.append(f1)
        slot_f1 = sum(f1s) / len(f1s) if f1s else None

        # 4. Abstention_Acc — denominator = 拒答类查询 (expected == ABSTAIN):
        #    retracted / no-evidence / anchor-drifted traps
        abst_q = [r for r in rs if r["expected"] == ABSTAIN]
        strict_ok = sum(1 for r in abst_q
                        if r["abstained"] and r["reason"] == r["expected_reason"])
        lenient_ok = sum(1 for r in abst_q if r["abstained"])
        abstention_strict = strict_ok / len(abst_q) if abst_q else None
        abstention_lenient = lenient_ok / len(abst_q) if abst_q else None

        # guardrail: false refusals on answerable queries
        answerable = [r for r in rs if r["expected"] != ABSTAIN]
        false_abstention = (sum(1 for r in answerable if r["abstained"])
                            / len(answerable)) if answerable else None

        # auxiliary: As_Of_Acc
        asof_q = [r for r in rs if r["query_type"] == "as_of"]
        asof_ok = sum(1 for r in asof_q
                      if not r["abstained"] and _value_equal(r["value"], r["expected"]))
        asof_acc = asof_ok / len(asof_q) if asof_q else None

        # context token cost per query (all query types)
        ctx_tokens = [r["context_tokens"] for r in rs]
        avg_ctx_tokens = sum(ctx_tokens) / len(ctx_tokens) if ctx_tokens else 0.0

        per_system[system] = {
            "Temporal_State_Acc": temporal_acc,
            "Stale_Poisoning_Rate": poisoning,
            "Slot_Merge_F1": slot_f1,
            "Abstention_Acc": abstention_strict,
            "Abstention_Acc_lenient": abstention_lenient,
            "False_Abstention_Rate": false_abstention,
            "As_Of_Acc": asof_acc,
            "Avg_Context_Tokens_Per_Query": avg_ctx_tokens,
            "n_queries": len(rs),
            "n_poisoned_items": poisoned_items,
            "n_context_items": total_items,
        }

    # per-category accuracy breakdown for the report
    by_cat: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cat in ("reversal", "partial_update", "retraction", "as_of", "anchor_drift"):
        by_cat[cat] = {}
        for system in systems:
            rs = [r for r in records
                  if r["system"] == system and r["category"] == cat]
            if not rs:
                continue
            if cat == "partial_update":
                f1s = [slot_prf1(r["slots"], r["expected_slots"])[2]
                       for r in rs if r["expected_slots"]]
                by_cat[cat][system] = {"Slot_Merge_F1": sum(f1s) / len(f1s)}
            elif cat == "as_of":
                ok = sum(1 for r in rs
                         if not r["abstained"] and _value_equal(r["value"], r["expected"]))
                by_cat[cat][system] = {"As_Of_Acc": ok / len(rs)}
            elif cat == "anchor_drift":
                ok = sum(1 for r in rs
                         if r["abstained"] == (r["expected"] == "__ABSTAIN__")
                         and (not r["abstained"] or r["reason"] == r["expected_reason"]))
                by_cat[cat][system] = {"Abstention_Acc": ok / len(rs)}
            else:  # reversal / retraction
                ok = sum(1 for r in rs if (
                    (r["expected"] == "__ABSTAIN__" and r["abstained"]
                     and r["reason"] == r["expected_reason"])
                    or (r["expected"] != "__ABSTAIN__" and not r["abstained"]
                        and _value_equal(r["value"], r["expected"]))))
                by_cat[cat][system] = {"Accuracy": ok / len(rs)}

    return {"per_system": per_system, "per_category": by_cat}
