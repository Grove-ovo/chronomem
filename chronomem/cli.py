"""ChronoMem command-line interface.

Subcommands: add / retract / retrieve / as-of / verify-anchors / inspect /
bench / visualize.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from chronomem.anchor_watcher import AnchorWatcher, compute_anchor_fingerprint
from chronomem.resolver import ConflictResolver, Resolution
from chronomem.retriever import Retriever, make_summary_text
from chronomem.store import MemoryStore
from chronomem.visualizer import render_visualizer

__all__ = ["main", "build_demo_store"]


def _parse_value(raw: str) -> Any:
    """JSON first (dict/list/number/bool), plain string otherwise."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


def _open_store(db_path: str) -> MemoryStore:
    return MemoryStore(db_path)


# --------------------------------------------------------------------- #
# demo store builder (for `visualize --demo`)
# --------------------------------------------------------------------- #
def build_demo_store(db_path: str, workdir: str) -> MemoryStore:
    """Build a narrative demo store: reversals + partial merge + retract +
    anchor drift, so the timeline shows all four lifecycle states."""
    store = _open_store(db_path)
    resolver = ConflictResolver(store, namespace="demo")
    retriever = Retriever(store)
    watcher = AnchorWatcher(store)
    workspace = Path(workdir)
    (workspace / "src").mkdir(parents=True, exist_ok=True)
    service = workspace / "src" / "payment.py"
    service.write_text(
        "class RefundService:\n"
        "    def process(self, amount):\n"
        '        """Refund an amount."""\n'
        "        return -amount\n",
        encoding="utf-8",
    )

    def index(res):
        if res.new_atom is not None:
            retriever.index_atom(res.new_atom)

    index(resolver.write("user_alice", "database", "MySQL", 1))
    index(resolver.write("user_alice", "database", "PostgreSQL", 4))       # SUPERSEDE
    index(resolver.write("user_alice", "database", "SQLite", 7))           # SUPERSEDE
    index(resolver.write("user_alice", "editor", "vim", 2))
    index(resolver.write("deploy", "retry_config",
                         {"max_attempts": 3, "backoff_s": 1, "timeout_s": 15,
                          "strategy": "exponential"}, 3))                 # ADD (dict)
    index(resolver.write("deploy", "retry_config", {"timeout_s": 30}, 5))  # MERGE_PARTIAL
    index(resolver.write("deploy", "retry_config", {"strategy": "linear"}, 6))  # MERGE_PARTIAL
    resolver.retract("user_alice", "editor", 8)                            # RETRACT
    index(resolver.write("user_alice", "editor", "neovim", 9))             # re-ADD

    # anchor-bound engineering memory + a later refactor that drifts it
    fp = compute_anchor_fingerprint(workspace, "src/payment.py::RefundService.process")
    res = resolver.write("svc_payment", "refund_entry",
                         {"module": "src/payment.py", "entry": "RefundService.process"},
                         10, anchor_path="src/payment.py::RefundService.process",
                         anchor_sha256=fp)
    index(res)
    service.write_text(
        "class RefundService:\n"
        "    def process_v2(self, amount):\n"
        "        return -amount\n",
        encoding="utf-8",
    )
    watcher.verify_anchors(workspace, namespace="demo")                    # → anchor_drifted
    return store


# --------------------------------------------------------------------- #
# subcommand handlers
# --------------------------------------------------------------------- #
def cmd_add(args) -> int:
    store = _open_store(args.db)
    resolver = ConflictResolver(store, namespace=args.namespace)
    watcher = AnchorWatcher(store)
    retriever = Retriever(store)
    anchor_sha = None
    if args.anchor:
        anchor_sha = compute_anchor_fingerprint(args.workspace, args.anchor)
    result = resolver.write(
        args.entity, args.slot, _parse_value(args.value), args.turn,
        source_turn_id=args.source_turn_id, anchor_path=args.anchor,
        anchor_sha256=anchor_sha)
    if result.new_atom is not None:
        retriever.index_atom(result.new_atom)
    op = result.operation.value
    detail = ""
    if result.operation == Resolution.MERGE_PARTIAL:
        detail = f" → merged snapshot: {json.dumps(result.new_atom.value, ensure_ascii=False)}"
    print(f"[{op}] {args.namespace}/{args.entity}/{args.slot} atom #{result.new_atom.atom_id}{detail}")
    violations = store.verify_invariants()
    print("invariant_violations:", violations or "none")
    store.close()
    return 0


def cmd_retract(args) -> int:
    store = _open_store(args.db)
    resolver = ConflictResolver(store, namespace=args.namespace)
    result = resolver.retract(args.entity, args.slot, args.turn)
    if result.old_atom is None:
        print(f"[RETRACT] no active atom for {args.namespace}/{args.entity}/{args.slot} — nothing to retract")
    else:
        print(f"[RETRACT] atom #{result.old_atom.atom_id} → retracted at turn {args.turn}")
    print("invariant_violations:", store.verify_invariants() or "none")
    store.close()
    return 0


def cmd_retrieve(args) -> int:
    store = _open_store(args.db)
    retriever = Retriever(store)
    result = retriever.retrieve_current(
        args.query, namespace=args.namespace, top_k=args.top_k,
        entity=args.entity, slot=args.slot)
    if not result.results:
        print("no active records matched.")
    for s in result.results:
        a = s.atom
        print(f"  #{a.atom_id} [{a.status}] {a.entity}/{a.slot} = "
              f"{json.dumps(a.value, ensure_ascii=False)} "
              f"(turn {a.valid_from_turn}→{a.valid_to_turn or '∞'}, "
              f"score {s.score:.3f}, {','.join(s.match_reasons)})")
    for n in result.notices:
        print(f"  ⚠ notice [{n['type']}] {n['entity']}/{n['slot']} "
              f"(atom #{n['atom_id']}) — 该槽位不可作答，应拒答/重新核实")
    print(f"poisoning_ratio: {result.poisoning_ratio:.3f}")
    store.close()
    return 0


def cmd_as_of(args) -> int:
    store = _open_store(args.db)
    retriever = Retriever(store)
    result = retriever.retrieve_as_of(args.turn, args.query, namespace=args.namespace)
    if not result.results:
        print(f"world state at turn {args.turn}: empty.")
    for s in result.results:
        a = s.atom
        print(f"  #{a.atom_id} {a.entity}/{a.slot} = "
              f"{json.dumps(a.value, ensure_ascii=False)} "
              f"(live at turn {args.turn}: {a.valid_from_turn}→{a.valid_to_turn or '∞'})")
    store.close()
    return 0


def cmd_verify_anchors(args) -> int:
    store = _open_store(args.db)
    watcher = AnchorWatcher(store)
    verdicts = watcher.verify_anchors(args.workspace, namespace=args.namespace)
    if not verdicts:
        print("no active anchor-bound atoms found.")
    for v in verdicts:
        mark = "OK " if v.verdict == "ok" else "DRIFTED"
        reason = f" ({v.reason})" if v.reason else ""
        print(f"  [{mark}] atom #{v.atom_id} ← {v.anchor_path}{reason}")
    print("invariant_violations:", store.verify_invariants() or "none")
    store.close()
    return 0


def cmd_inspect(args) -> int:
    store = _open_store(args.db)
    atoms = store.all_atoms(namespace=args.namespace)
    if not atoms:
        print("store is empty.")
    by_key: dict = {}
    for a in atoms:
        by_key.setdefault((a.namespace, a.entity, a.slot), []).append(a)
    print(f"{len(atoms)} atoms across {len(by_key)} version chains "
          f"(invariant violations: {store.verify_invariants() or 'none'})\n")
    for (ns, ent, slot), versions in sorted(by_key.items()):
        print(f"[{ns}] {ent} · {slot}")
        for v in versions:
            sup = f" supersedes #{v.supersedes}" if v.supersedes else ""
            sup2 = f" superseded_by #{v.superseded_by}" if v.superseded_by else ""
            print(f"  #{v.atom_id:<4} {v.status:<14} t{v.valid_from_turn}→"
                  f"{v.valid_to_turn if v.valid_to_turn is not None else '∞':<4} "
                  f"{json.dumps(v.value, ensure_ascii=False)[:70]}{sup}{sup2}")
        print()
    store.close()
    return 0


def cmd_bench(args) -> int:
    from benchmarks.dataset import build_chronobench_100
    from benchmarks.runner import run_benchmark
    from benchmarks.report import write_reports

    samples = build_chronobench_100(args.seed)
    if args.samples and args.samples < 100:
        per_cat = max(1, args.samples // 5)
        by_cat: dict = {}
        for s in samples:
            by_cat.setdefault(s["category"], []).append(s)
        samples = [s for cat in by_cat for s in by_cat[cat][:per_cat]]
        print(f"running subset benchmark: {len(samples)} samples...")
    else:
        print(f"running full ChronoBench-100 ({len(samples)} samples, "
              f"{sum(len(s['queries']) for s in samples)} queries)...")
    result = run_benchmark(samples=samples, seed=args.seed)
    paths = write_reports(result, out_dir=args.out)
    print("\n=== 四大核心指标 ===")
    for name, m in result["metrics"]["per_system"].items():
        print(f"  {name:<18} TSA={m['Temporal_State_Acc']:.3f}  "
              f"POISON={m['Stale_Poisoning_Rate']:.3f}  F1={m['Slot_Merge_F1']:.3f}  "
              f"ABST={m['Abstention_Acc']:.3f}  ASOF={m['As_Of_Acc']:.3f}")
    print(f"\nreport: {paths['markdown']}")
    print(f"raw:    {paths['json']}")
    return 0


def cmd_visualize(args) -> int:
    if args.demo:
        with tempfile.TemporaryDirectory(prefix="chronomem-demo-") as td:
            store = build_demo_store(td + "/demo.db", td)
            out = render_visualizer(store, args.out)
            n = store.count()
            store.close()
    else:
        store = _open_store(args.db)
        out = render_visualizer(store, args.out)
        n = store.count()
        store.close()
    print(f"visualizer written: {out} ({n} atoms embedded)")
    return 0


# --------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chronomem",
        description="ChronoMem — 抗过期污染的时效状态与代码锚点 AI 长期记忆引擎")
    p.add_argument("--db", default="chronomem.db", help="SQLite 数据库路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("add", help="写入/更新一条记忆（四态消解）")
    sp.add_argument("--entity", required=True)
    sp.add_argument("--slot", required=True)
    sp.add_argument("--value", required=True, help="JSON 值或纯文本")
    sp.add_argument("--turn", type=int, default=1)
    sp.add_argument("--namespace", default="default")
    sp.add_argument("--anchor", default=None, help="锚点，如 src/x.py::Class.method")
    sp.add_argument("--workspace", default=".", help="锚点解析的工作区根目录")
    sp.add_argument("--source-turn-id", dest="source_turn_id", default=None)
    sp.set_defaults(func=cmd_add)

    sp = sub.add_parser("retract", help="显式撤销一条记忆")
    sp.add_argument("--entity", required=True)
    sp.add_argument("--slot", required=True)
    sp.add_argument("--turn", type=int, default=1)
    sp.add_argument("--namespace", default="default")
    sp.set_defaults(func=cmd_retract)

    sp = sub.add_parser("retrieve", help="当前态零污染混合检索")
    sp.add_argument("query")
    sp.add_argument("--namespace", default="default")
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--entity", default=None)
    sp.add_argument("--slot", default=None)
    sp.set_defaults(func=cmd_retrieve)

    sp = sub.add_parser("as-of", help="时间旅行：还原历史轮次世界状态")
    sp.add_argument("turn", type=int)
    sp.add_argument("query", nargs="?", default="")
    sp.add_argument("--namespace", default="default")
    sp.set_defaults(func=cmd_as_of)

    sp = sub.add_parser("verify-anchors", help="扫描工作区，检测锚点漂移")
    sp.add_argument("workspace")
    sp.add_argument("--namespace", default="default")
    sp.set_defaults(func=cmd_verify_anchors)

    sp = sub.add_parser("inspect", help="检视全部记忆版本链")
    sp.add_argument("--namespace", default=None)
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("bench", help="运行 ChronoBench-100 四基线评测")
    sp.add_argument("--samples", type=int, default=100,
                    help="样本数（100 或 5 的倍数子集）")
    sp.add_argument("--seed", type=int, default=20261111)
    sp.add_argument("--out", default="reports")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("visualize", help="生成记忆版本链时间轴网页")
    sp.add_argument("--out", default="visualizer/index.html")
    sp.add_argument("--demo", action="store_true", help="使用内置演示数据")
    sp.set_defaults(func=cmd_visualize)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
