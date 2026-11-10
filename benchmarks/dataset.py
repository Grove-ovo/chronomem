"""ChronoBench-100 — cross-session conflict evolution benchmark (PRD §3.5).

100 multi-session conflict-evolution samples across 5 scenario categories
(20 each):

1. ``reversal``      — single attribute reversed 3~5 times across sessions
2. ``partial_update``— multi-field config updated incrementally, one field at a time
3. ``retraction``    — explicit cancellation + never-mentioned abstention traps
4. ``as_of``         — historical point-in-time (time travel) questions
5. ``anchor_drift``  — code refactors invalidating anchor-bound memories

Every sample is a full conversation stream (turns with structured statement
metadata), optional workspace events (code refactors), and evaluation queries
with ground-truth answers. Generation is fully deterministic (fixed seed).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["build_chronobench_100", "CATEGORIES", "ABSTAIN"]

ABSTAIN = "__ABSTAIN__"

CATEGORIES = ["reversal", "partial_update", "retraction", "as_of", "anchor_drift"]
PER_CATEGORY = 20

# --------------------------------------------------------------------- #
# value pools
# --------------------------------------------------------------------- #
SLOT_POOL = [
    ("database", "数据库", ["MySQL", "PostgreSQL", "SQLite", "MongoDB", "OceanBase"]),
    ("language", "主力语言", ["Python", "Go", "Rust", "TypeScript", "Java"]),
    ("editor", "编辑器", ["vim", "neovim", "vscode", "emacs", "jetbrains"]),
    ("cloud", "部署云厂商", ["AWS", "阿里云", "GCP", "Azure", "腾讯云"]),
    ("framework", "前端框架", ["react", "vue", "svelte", "angular", "nextjs"]),
]

CONFIG_TEMPLATES = {
    "deploy_config": {
        "zh": "部署配置",
        "fields": {
            "region": ["cn-east-1", "cn-north-2", "us-west-1", "eu-central-1"],
            "replicas": [2, 3, 5, 8],
            "timeout_s": [5, 10, 30, 60],
            "image_tag": ["v1.0", "v1.2", "v2.0", "v2.1"],
            "log_level": ["info", "debug", "warn", "error"],
        },
        "field_zh": {
            "region": "部署区域", "replicas": "副本数", "timeout_s": "超时时间",
            "image_tag": "镜像版本", "log_level": "日志级别",
        },
    },
    "retry_config": {
        "zh": "重试配置",
        "fields": {
            "max_attempts": [3, 5, 8, 10],
            "backoff_s": [1, 2, 5, 10],
            "timeout_s": [15, 30, 45, 90],
            "jitter": [True, False],
            "strategy": ["exponential", "linear", "fixed"],
        },
        "field_zh": {
            "max_attempts": "最大尝试次数", "backoff_s": "退避间隔",
            "timeout_s": "单次超时", "jitter": "抖动开关", "strategy": "退避策略",
        },
    },
    "api_config": {
        "zh": "网关配置",
        "fields": {
            "base_url": ["api.example.com", "gw.example.cn", "api2.example.net"],
            "version": ["v1", "v2", "v3"],
            "timeout_ms": [800, 1500, 3000, 5000],
            "retries": [0, 1, 2, 3],
            "auth": ["token", "oauth2", "basic", "hmac"],
        },
        "field_zh": {
            "base_url": "网关地址", "version": "接口版本", "timeout_ms": "超时毫秒数",
            "retries": "重试次数", "auth": "鉴权方式",
        },
    },
}

USER_NAMES = ["alice", "bob", "cindy", "dave", "erina", "frank", "gina", "hank"]
SVC_NAMES = ["orders", "payment", "inventory", "shipping", "notify",
             "search", "billing", "auth"]


# --------------------------------------------------------------------- #
# stream builders
# --------------------------------------------------------------------- #
def _statement(turn_id: int, session_id: int, text: str, *, entity: str,
               slot: str, value: Any = None, op: str = "set",
               anchor_path: Optional[str] = None) -> Dict[str, Any]:
    st: Dict[str, Any] = {
        "turn_id": turn_id, "session_id": session_id, "speaker": "user",
        "text": text,
        "statement": {"entity": entity, "slot": slot, "value": value, "op": op},
    }
    if anchor_path:
        st["statement"]["anchor_path"] = anchor_path
    return st


def _filler_turn(turn_id: int, session_id: int, rng: random.Random) -> Dict[str, Any]:
    fillers = [
        "帮我看看今天的日程安排。", "这句话翻译成英文怎么说？",
        "顺便记一下，明早九点提醒我站会。", "刚才说的方案总结一下吧。",
        "这个问题我们先跳过。", "把这段日志贴给我看看。",
    ]
    return {"turn_id": turn_id, "session_id": session_id,
            "speaker": "user", "text": rng.choice(fillers)}


def _split_sessions(turns: List[Dict[str, Any]], n_sessions: int,
                    rng: random.Random) -> None:
    """Assign session ids to turns, roughly evenly, in order."""
    if len(turns) <= n_sessions:
        for i, t in enumerate(turns):
            t["session_id"] = i + 1
        return
    cuts = sorted(rng.sample(range(1, len(turns)), min(n_sessions - 1, len(turns) - 1)))
    sid, prev = 1, 0
    for c in cuts + [len(turns)]:
        for t in turns[prev:c]:
            t["session_id"] = sid
        prev = c
        sid += 1


def _pack(turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_session: Dict[int, List[Dict[str, Any]]] = {}
    for t in turns:
        by_session.setdefault(t["session_id"], []).append(t)
    return [{"session_id": s, "turns": ts} for s, ts in sorted(by_session.items())]


# --------------------------------------------------------------------- #
# category 1: reversal
# --------------------------------------------------------------------- #
def gen_reversal(idx: int, rng: random.Random) -> Dict[str, Any]:
    slot, slot_zh, pool = rng.choice(SLOT_POOL)
    user = rng.choice(USER_NAMES)
    entity = f"user_{user}_{idx % 3 + 1}"
    n_reversals = rng.randint(3, 5)
    values = rng.sample(pool, n_reversals)

    turns: List[Dict[str, Any]] = []
    turn = 2
    for i, value in enumerate(values):
        if i > 0:
            turns.append(_filler_turn(turn, 1, rng)); turn += 1
        text = rng.choice([
            f"从现在起我的{slot_zh}换成 {value}，以后都用这个。",
            f"改主意了：{slot_zh}以后统一用 {value}。",
            f"把我的{slot_zh}设置更新为 {value}，覆盖之前的设定。",
        ])
        turns.append(_statement(turn, 1, text, entity=entity, slot=slot, value=value))
        turn += rng.randint(2, 4)

    turns.append(_filler_turn(turn, 1, rng)); turn += 1
    q_turn = turn
    query = {
        "query_id": "q_current", "type": "current", "turn_id": q_turn,
        "text": f"我现在用的{slot_zh}到底是什么？",
        "entity": entity, "slot": slot, "expected": values[-1],
        "expected_reason": None,
    }
    return {
        "sample_id": f"reversal-{idx:02d}", "category": "reversal",
        "entity": entity, "slot": slot, "n_reversals": n_reversals,
        "sessions": _pack(turns), "workspace_events": [], "queries": [query],
    }


# --------------------------------------------------------------------- #
# category 2: partial_update
# --------------------------------------------------------------------- #
def gen_partial_update(idx: int, rng: random.Random) -> Dict[str, Any]:
    name = rng.choice(list(CONFIG_TEMPLATES))
    tpl = CONFIG_TEMPLATES[name]
    svc = rng.choice(SVC_NAMES)
    entity = f"svc_{svc}_{idx % 3 + 1}"

    fields = dict(tpl["fields"])
    initial = {f: rng.choice(vals) for f, vals in fields.items()}

    turns: List[Dict[str, Any]] = []
    turn = 2
    init_text = (f"记录一下，{entity} 的{tpl['zh']}如下："
                 + "，".join(f"{tpl['field_zh'][f]}={initial[f]}" for f in initial)
                 + "。以后没提到的字段都保持这个值。")
    turns.append(_statement(turn, 1, init_text, entity=entity, slot=name,
                            value=dict(initial)))
    turn += 2

    final = dict(initial)
    n_updates = rng.randint(2, 4)
    update_fields = rng.sample(list(fields), n_updates)
    for f in update_fields:
        choices = [v for v in fields[f] if v != final[f]]
        new_val = rng.choice(choices)
        final[f] = new_val
        turns.append(_filler_turn(turn, 1, rng)); turn += 1
        text = rng.choice([
            f"{tpl['zh']}只改一项：{tpl['field_zh'][f]}改成 {new_val}，其余字段全部保持不变。",
            f"把{tpl['field_zh'][f]}调整为 {new_val}，其他配置原样保留。",
        ])
        turns.append(_statement(turn, 1, text, entity=entity, slot=name,
                                value={f: new_val}, op="partial"))
        turn += 2

    turns.append(_filler_turn(turn, 1, rng)); turn += 1
    query = {
        "query_id": "q_full_config", "type": "current", "turn_id": turn,
        "text": f"目前 {entity} 的{tpl['zh']}完整配置是什么？逐项列出。",
        "entity": entity, "slot": name, "expected": None,
        "expected_slots": final, "expected_reason": None,
    }
    return {
        "sample_id": f"partial-{idx:02d}", "category": "partial_update",
        "entity": entity, "slot": name, "initial_config": initial,
        "final_config": final,
        "sessions": _pack(turns), "workspace_events": [], "queries": [query],
    }


# --------------------------------------------------------------------- #
# category 3: retraction + no-evidence traps
# --------------------------------------------------------------------- #
def gen_retraction(idx: int, rng: random.Random) -> Dict[str, Any]:
    slot, slot_zh, pool = rng.choice(SLOT_POOL)
    user = rng.choice(USER_NAMES)
    entity = f"user_{user}_{idx % 3 + 1}"
    value = rng.choice(pool)
    ghost_slot, ghost_zh, _ = rng.choice(
        [s for s in SLOT_POOL if s[0] != slot])

    turns: List[Dict[str, Any]] = []
    turn = 2
    turns.append(_statement(turn, 1, f"我的{slot_zh}设定为 {value}。",
                            entity=entity, slot=slot, value=value))
    turn += rng.randint(2, 3)
    turns.append(_filler_turn(turn, 1, rng)); turn += 1
    turn += 1
    retract_text = rng.choice([
        f"把{slot_zh}这个设定取消掉，已经不作数了。",
        f"撤销之前关于{slot_zh}的设定，以后没有这个规则了。",
        f"那个{slot_zh}设定作废，别再引用它。",
    ])
    turns.append(_statement(turn, 1, retract_text, entity=entity, slot=slot,
                            value=None, op="retract"))
    turn += rng.randint(2, 3)

    q1_turn = turn
    q1 = {
        "query_id": "q_retracted", "type": "current", "turn_id": q1_turn,
        "text": f"现在我的{slot_zh}设定是什么？",
        "entity": entity, "slot": slot, "expected": ABSTAIN,
        "expected_reason": "retracted",
    }
    q2 = {
        "query_id": "q_no_evidence", "type": "current", "turn_id": q1_turn,
        "text": f"另外我的{ghost_zh}设置是多少？",
        "entity": entity, "slot": ghost_slot, "expected": ABSTAIN,
        "expected_reason": "no_evidence",
    }
    return {
        "sample_id": f"retraction-{idx:02d}", "category": "retraction",
        "entity": entity, "slot": slot,
        "sessions": _pack(turns), "workspace_events": [], "queries": [q1, q2],
    }


# --------------------------------------------------------------------- #
# category 4: as_of time travel
# --------------------------------------------------------------------- #
def gen_as_of(idx: int, rng: random.Random) -> Dict[str, Any]:
    slot, slot_zh, pool = rng.choice(SLOT_POOL)
    user = rng.choice(USER_NAMES)
    entity = f"user_{user}_tt{idx % 3 + 1}"
    values = rng.sample(pool, 3)

    turns: List[Dict[str, Any]] = []
    change_turns: List[int] = []
    turn = 2
    for i, value in enumerate(values):
        turns.append(_statement(turn, 1,
                                f"我的{slot_zh}从现在开始用 {value}。",
                                entity=entity, slot=slot, value=value))
        change_turns.append(turn)
        turn += rng.randint(3, 5)

    # two historical pins: before 2nd change, and at/before 3rd change
    pin1 = change_turns[1] - 1          # strictly before the 2nd change
    pin2 = change_turns[2] - 1          # strictly before the 3rd change
    q1 = {
        "query_id": "q_asof_1", "type": "as_of", "turn_id": turn,
        "as_of_turn": pin1,
        "text": f"在第 {change_turns[1]} 轮修改之前，我的{slot_zh}是什么？",
        "entity": entity, "slot": slot, "expected": values[0],
        "expected_reason": None,
    }
    q2 = {
        "query_id": "q_asof_2", "type": "as_of", "turn_id": turn,
        "as_of_turn": pin2,
        "text": f"回溯一下：在第 {change_turns[2]} 轮改动前，{slot_zh}用的是哪个？",
        "entity": entity, "slot": slot, "expected": values[1],
        "expected_reason": None,
    }
    return {
        "sample_id": f"asof-{idx:02d}", "category": "as_of",
        "entity": entity, "slot": slot, "change_turns": change_turns,
        "sessions": _pack(turns), "workspace_events": [], "queries": [q1, q2],
    }


# --------------------------------------------------------------------- #
# category 5: anchor drift
# --------------------------------------------------------------------- #
SERVICE_TEMPLATE = """\
class {cls}:
    def {entry}(self, payload):
        '''Entry point bound to ChronoMem memory.'''
        return payload

    def {alt}(self, payload):
        return payload
"""

STATS_TEMPLATE = """\
class Stats:
    def render(self, rows):
        return rows
"""

REF_ACTS = ["rename_entry", "delete_file", "delete_entry"]


def gen_anchor_drift(idx: int, rng: random.Random) -> Dict[str, Any]:
    svc = SVC_NAMES[idx % len(SVC_NAMES)]
    cls = f"{svc.capitalize()}Service"
    entry = "process"
    alt = "validate"
    # two separate files so that deleting the service file never invalidates
    # the control anchor (Stats lives in its own module)
    path = f"src/{svc}_service.py"
    stats_path = f"src/{svc}_stats.py"
    anchor = f"{path}::{cls}.{entry}"
    report_anchor = f"{stats_path}::Stats.render"

    entity = f"svc_{svc}"
    value = {"module": path, "entry": f"{cls}.{entry}"}
    ctrl_value = {"module": stats_path, "entry": "Stats.render"}

    turns: List[Dict[str, Any]] = []
    turn = 2
    turns.append(_statement(
        turn, 1,
        f"{entity} 服务的处理入口绑定到 {anchor}，记住这个结构。",
        entity=entity, slot="process_entry", value=value, op="set",
        anchor_path=anchor))
    turn += 2
    turns.append(_statement(
        turn, 1,
        f"另外，报表入口在 {report_anchor}，这个不要动。",
        entity=entity, slot="report_entry", value=ctrl_value, op="set",
        anchor_path=report_anchor))
    turn += rng.randint(2, 3)

    act = REF_ACTS[idx % len(REF_ACTS)]
    new_entry = f"{entry}_v2"
    if act == "rename_entry":
        event = {"type": "rename_symbol", "turn_id": turn, "path": path,
                 "cls": cls, "old": entry, "new": new_entry}
        prose = f"（重构）把 {cls}.{entry} 重命名为 {cls}.{new_entry}。"
    elif act == "delete_file":
        event = {"type": "delete_file", "turn_id": turn, "path": path}
        prose = f"（重构）整个 {path} 文件被移除了。"
    else:
        event = {"type": "delete_symbol", "turn_id": turn, "path": path,
                 "cls": cls, "symbol": entry}
        prose = f"（重构）{cls}.{entry} 方法被删除了。"
    turns.append({"turn_id": turn, "session_id": 2, "speaker": "user",
                  "text": prose, "workspace_event": event})
    turn += rng.randint(2, 3)

    q1 = {
        "query_id": "q_drifted_entry", "type": "current", "turn_id": turn,
        "text": f"现在 {entity} 服务的处理入口是什么？",
        "entity": entity, "slot": "process_entry", "expected": ABSTAIN,
        "expected_reason": "anchor_drifted",
    }
    q2 = {
        "query_id": "q_control_entry", "type": "current", "turn_id": turn,
        "text": f"那 {entity} 的报表入口是什么？",
        "entity": entity, "slot": "report_entry", "expected": ctrl_value,
        "expected_reason": None,
    }
    return {
        "sample_id": f"anchor-{idx:02d}", "category": "anchor_drift",
        "entity": entity, "slot": "process_entry", "refactor": act,
        "service_template": SERVICE_TEMPLATE.format(cls=cls, entry=entry, alt=alt),
        "stats_template": STATS_TEMPLATE,
        "service_path": path, "stats_path": stats_path,
        "sessions": _pack(turns), "workspace_events": [event], "queries": [q1, q2],
    }


# --------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------- #
_GENERATORS = {
    "reversal": gen_reversal,
    "partial_update": gen_partial_update,
    "retraction": gen_retraction,
    "as_of": gen_as_of,
    "anchor_drift": gen_anchor_drift,
}

# cross-session distribution per category (min, max).  partial_update uses
# 4~5 sessions so a rolling-summary window of 3 loses the initial full config.
SESSION_COUNTS = {
    "reversal": (2, 4),
    "partial_update": (4, 5),
    "retraction": (2, 3),
    "as_of": (2, 3),
    "anchor_drift": (3, 3),
}


def build_chronobench_100(seed: int = 20261111) -> List[Dict[str, Any]]:
    """Deterministically generate the full 100-sample benchmark."""
    samples: List[Dict[str, Any]] = []
    for cat in CATEGORIES:
        for i in range(1, PER_CATEGORY + 1):
            # per-sample seed keeps categories independent of ordering
            rng = random.Random(f"{seed}:{cat}:{i}")
            sample = _GENERATORS[cat](i, rng)
            turns = [t for sess in sample["sessions"] for t in sess["turns"]]
            n_sessions = rng.randint(*SESSION_COUNTS[cat])
            _split_sessions(turns, n_sessions, rng)
            sample["sessions"] = _pack(turns)
            sample["n_sessions"] = len(sample["sessions"])
            samples.append(sample)
    return samples


def dataset_summary(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cat: Dict[str, int] = {}
    q_total = 0
    for s in samples:
        by_cat[s["category"]] = by_cat.get(s["category"], 0) + 1
        q_total += len(s["queries"])
    return {"samples": len(samples), "queries": q_total, "by_category": by_cat}


if __name__ == "__main__":  # pragma: no cover
    data = build_chronobench_100()
    print(json.dumps(dataset_summary(data), ensure_ascii=False))
