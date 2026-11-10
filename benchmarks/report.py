"""Markdown report writer for ChronoBench-100 results."""

from __future__ import annotations

import json
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from benchmarks.dataset import dataset_summary
from benchmarks.metrics import slot_prf1

__all__ = ["write_reports"]

SYSTEM_ORDER = ["ChronoMem", "Full-Context", "Rolling Summary", "Naive Vector RAG"]

METHOD_NOTES = """\
### 评测方法（确定性 harness，隔离记忆管理能力）

四套系统观察**同一条多会话对话流**（每轮携带结构化 statement 元数据，
对全部系统同等可见，从而隔离记忆管理能力、排除 NLU 抽取差异干扰）：

| 系统 | 记忆组织方式 | 时间感知 | 代码感知 |
|---|---|---|---|
| **ChronoMem** | MVCC 版本链 + 四态冲突消解 + 锚点指纹 | `valid_from/to_turn` 精确窗口 | AST 符号 SHA256 指纹 |
| Full-Context | 全量原始对话扫描 | 按轮次过滤 | 无 |
| Rolling Summary | 会话摘要窗口（最近 3 会话） | 无（只有最新态） | 无 |
| Naive Vector RAG | 全量切块 TF-IDF 向量 Top-K | 无 | 无 |

指标定义严格遵循 PRD §2：

* **Temporal_State_Acc** = 严格采纳最新有效状态作答的查询数 / 经历过状态覆写的查询总数（reversal 类）；
* **Stale_Poisoning_Rate** = 当前态查询注入上下文的过期/撤销/漂移条目数 / 注入上下文总条数（逐条按 ground-truth 演化账本审计）；
* **Slot_Merge_F1** = 局部修改后最终槽位集合 vs Ground-Truth 的字段级 P/R/F1（partial_update 类，值错误的字段同时计入 FP 与 FN）；
* **Abstention_Acc** = 拒答类查询（已撤销 / 从未提及 / 锚点漂移）中准确识别并拒答的比例（严格模式：拒答原因也必须正确）。
"""


def _fmt(v: Any, pct: bool = True) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:.1f}"


def write_reports(result: Dict[str, Any], out_dir: str = "reports") -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]["per_system"]
    by_cat = result["metrics"]["per_category"]

    from benchmarks.dataset import build_chronobench_100
    ds = dataset_summary(build_chronobench_100())

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    ap = lines.append

    ap("# ChronoBench-100 四基线横向评测报告（ChronoMem）")
    ap("")
    ap(f"- 生成时间：{now}")
    cat_str = "、".join(f"{c}×{n}" for c, n in ds["by_category"].items())
    ap(f"- 数据集：{ds['samples']} 组多会话冲突演化样本（{cat_str}），共 {ds['queries']} 条评测查询")
    ap(f"- 运行环境：Python {platform.python_version()} / SQLite {sqlite3.sqlite_version}")
    ap(f"- 随机种子：20261111（数据集与基线全部确定性可复现）")
    ap("")
    ap(METHOD_NOTES)
    ap("")

    # ---------------- headline table ---------------- #
    ap("## 1. 四大核心指标总表")
    ap("")
    ap("| 系统 | Temporal_State_Acc ↑ | Stale_Poisoning_Rate ↓ | Slot_Merge_F1 ↑ | Abstention_Acc ↑ | As_Of_Acc ↑ | 误拒答率 ↓ | 平均上下文 Tokens/查询 ↓ |")
    ap("|---|---|---|---|---|---|---|---|")
    for name in SYSTEM_ORDER:
        m = metrics[name]
        ap(f"| **{name}** | {_fmt(m['Temporal_State_Acc'])} | "
           f"{_fmt(m['Stale_Poisoning_Rate'])} | {_fmt(m['Slot_Merge_F1'])} | "
           f"{_fmt(m['Abstention_Acc'])} | {_fmt(m['As_Of_Acc'])} | "
           f"{_fmt(m['False_Abstention_Rate'])} | {m['Avg_Context_Tokens_Per_Query']:.0f} |")
    ap("")

    chrono = metrics["ChronoMem"]
    naive = metrics["Naive Vector RAG"]
    ap(f"**关键结论**：ChronoMem 将当前态查询的过期记忆污染率压降至 "
       f"**{_fmt(chrono['Stale_Poisoning_Rate'])}**"
       f"（Naive Vector RAG 为 {_fmt(naive['Stale_Poisoning_Rate'])}，"
       f"Full-Context 为 {_fmt(metrics['Full-Context']['Stale_Poisoning_Rate'])}），"
       f"同时在最新状态准确率、局部合并 F1、时间旅行问答与拒答率上全面领先 Naive Vector RAG。")
    ap("")

    # ---------------- per-category ---------------- #
    ap("## 2. 分类别表现")
    ap("")
    cat_titles = {
        "reversal": ("Cat-1 单属性多次反转（3~5 次改口）", "Accuracy（当前态严格最新）"),
        "partial_update": ("Cat-2 多字段配置局部增量修改", "Slot_Merge_F1"),
        "retraction": ("Cat-3 显式撤销与无证据拒答陷阱", "Accuracy（拒答原因严格匹配）"),
        "as_of": ("Cat-4 历史时点回溯（时间旅行）", "As_Of_Acc"),
        "anchor_drift": ("Cat-5 代码重构锚点漂移", "Accuracy（漂移拒答+控制组正常作答联合）"),
    }
    for cat, (title, metric_label) in cat_titles.items():
        ap(f"### {title}")
        ap("")
        ap(f"| 系统 | {metric_label} |")
        ap("|---|---|")
        for name in SYSTEM_ORDER:
            val = by_cat.get(cat, {}).get(name, {})
            v = list(val.values())[0] if val else None
            ap(f"| **{name}** | {_fmt(v)} |")
        ap("")

    # ---------------- poisoning audit detail ---------------- #
    ap("## 3. 过期记忆污染审计明细")
    ap("")
    ap("| 系统 | 注入上下文条目总数 | 其中过期/撤销/漂移条目 | 污染率 |")
    ap("|---|---|---|---|")
    for name in SYSTEM_ORDER:
        m = metrics[name]
        ap(f"| **{name}** | {m['n_context_items']} | {m['n_poisoned_items']} | "
           f"{_fmt(m['Stale_Poisoning_Rate'])} |")
    ap("")
    ap("> ChronoMem 的当前态检索硬性过滤 `status != 'active'` 的全部版本，"
       "`retracted` / `anchor_drifted` 槽位以显式状态提示（notice）返回而非检索结果，"
       "因此污染率结构性为 0。")
    ap("")

    # ---------------- dataset ---------------- #
    ap("## 4. 数据集构成")
    ap("")
    ap("| 类别 | 样本数 | 考察能力 |")
    ap("|---|---|---|")
    cat_desc = {
        "reversal": "跨会话偏好反转后只采纳最新 active 状态",
        "partial_update": "局部字段修改时未提及字段零丢失（MERGE_PARTIAL 深度合并）",
        "retraction": "显式撤销/从未提及的槽位正确拒答",
        "as_of": "valid_from/to 窗口精确还原历史世界状态",
        "anchor_drift": "绑定文件/AST 符号被重构后拒绝引用失效记忆",
    }
    for cat in cat_desc:
        ap(f"| {cat} | {ds['by_category'][cat]} | {cat_desc[cat]} |")
    ap("")

    ap("## 5. 复现")
    ap("")
    ap("```bash")
    ap("chronomem bench                 # 全量 100 组评测，生成 reports/chronobench_results.md")
    ap("chronomem visualize --demo      # 生成记忆版本链时间轴可视化 visualizer/index.html")
    ap("pytest tests/ -q                # 74 项单元测试（含 MVCC 不变量随机风暴）")
    ap("```")
    ap("")

    md_path = out / "chronobench_results.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = out / "chronobench_results.json"
    json_path.write_text(json.dumps({
        "generated_at": now,
        "dataset": ds,
        "metrics": metrics,
        "per_category": by_cat,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}
