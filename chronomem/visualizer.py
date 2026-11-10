"""Memory version-chain timeline visualizer (PRD §5 deliverable).

Generates a fully self-contained ``visualizer/index.html`` (no external
assets, no build step) that renders every ``(namespace, entity, slot)``
version chain as a horizontal timeline lane:

* segments colored by lifecycle status (active / stale / retracted /
  anchor_drifted), spanning ``valid_from_turn → valid_to_turn``;
* supersedes arrows along each lane;
* hover tooltips with the atom's value JSON and anchor path;
* namespace filter + entity search + status legend.

Data is embedded as JSON so the page works offline from ``file://``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from chronomem.store import MemoryStore

__all__ = ["collect_visualization_data", "render_visualizer", "HTML_TEMPLATE"]


def collect_visualization_data(store: MemoryStore) -> Dict[str, Any]:
    atoms = store.all_atoms()
    max_turn = store.max_turn()
    by_key: Dict[Any, List[Dict[str, Any]]] = {}
    for a in atoms:
        by_key.setdefault((a.namespace, a.entity, a.slot), []).append({
            "atom_id": a.atom_id,
            "namespace": a.namespace,
            "entity": a.entity,
            "slot": a.slot,
            "status": a.status,
            "from": a.valid_from_turn,
            "to": a.valid_to_turn if a.valid_to_turn is not None else max_turn + 2,
            "open": a.valid_to_turn is None,
            "supersedes": a.supersedes,
            "superseded_by": a.superseded_by,
            "value": a.value,
            "anchor_path": a.anchor_path,
            "anchor_sha256": (a.anchor_sha256 or "")[:12],
        })
    lanes = [
        {
            "key": f"{ns} · {ent} · {slot}",
            "namespace": ns, "entity": ent, "slot": slot,
            "versions": sorted(vs, key=lambda v: v["from"]),
        }
        for (ns, ent, slot), vs in sorted(by_key.items())
    ]
    return {"max_turn": max_turn, "n_atoms": len(atoms), "lanes": lanes}


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ChronoMem — 记忆版本链时间轴</title>
<style>
  :root {
    --bg: #0f1420; --panel: #171e2e; --panel2: #1d2638; --text: #dbe4f5;
    --muted: #8292b0; --accent: #5b9dff; --line: #2a3550;
    --active: #34c782; --stale: #5a6b8f; --retracted: #e5655f; --drifted: #e5a34f;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.5 "SF Pro Text", "PingFang SC", "Segoe UI", sans-serif; }
  header { padding: 22px 28px 12px; border-bottom: 1px solid var(--line); }
  h1 { margin: 0 0 4px; font-size: 20px; letter-spacing: .3px; }
  h1 .brand { color: var(--accent); }
  .sub { color: var(--muted); font-size: 12.5px; }
  .controls { display: flex; gap: 10px; margin-top: 12px; flex-wrap: wrap; align-items: center; }
  .controls input, .controls select {
    background: var(--panel2); color: var(--text); border: 1px solid var(--line);
    border-radius: 8px; padding: 6px 10px; font-size: 13px; outline: none; }
  .legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted); }
  .legend b { display: inline-block; width: 10px; height: 10px; border-radius: 3px;
              margin-right: 5px; vertical-align: -1px; }
  main { padding: 18px 28px 60px; }
  .stats { color: var(--muted); font-size: 12px; margin: 6px 0 14px; }
  .lane { background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
          padding: 12px 16px 8px; margin-bottom: 14px; }
  .lane-head { display: flex; justify-content: space-between; align-items: baseline;
               margin-bottom: 6px; }
  .lane-key { font-weight: 600; letter-spacing: .2px; }
  .lane-key .ns { color: var(--muted); font-weight: 400; font-size: 12px; margin-right: 8px; }
  .lane-meta { color: var(--muted); font-size: 11.5px; }
  svg { display: block; width: 100%; height: 74px; }
  .bar { rx: 7; ry: 7; cursor: pointer; transition: opacity .15s; }
  .bar:hover { opacity: .82; }
  .active { fill: var(--active); } .stale { fill: var(--stale); }
  .retracted { fill: var(--retracted); } .anchor_drifted { fill: var(--drifted); }
  .bar-label { font: 10.5px "SF Mono", Menlo, monospace; fill: #0d1117; font-weight: 700;
               pointer-events: none; }
  .open-arrow { stroke: var(--active); stroke-width: 2; fill: none; opacity: .8; }
  .axis line { stroke: var(--line); stroke-width: 1; }
  .axis text { font: 10px "SF Mono", Menlo, monospace; fill: var(--muted); }
  .tip { position: fixed; background: #0b0f18ee; border: 1px solid var(--accent);
         border-radius: 8px; padding: 9px 12px; font-size: 12px; max-width: 430px;
         pointer-events: none; display: none; z-index: 10; box-shadow: 0 8px 26px #0009; }
  .tip .v { color: var(--accent); font-family: "SF Mono", Menlo, monospace; }
  .tip .row { margin: 2px 0; }
  .empty { color: var(--muted); padding: 40px; text-align: center; }
</style>
</head>
<body>
<header>
  <h1><span class="brand">ChronoMem</span> · 记忆版本链时间轴可视化器</h1>
  <div class="sub">MVCC 多版本状态链 · 四态冲突消解 (ADD / SUPERSEDE / MERGE_PARTIAL / RETRACT) · 代码锚点漂移</div>
  <div class="controls">
    <input id="search" type="search" placeholder="搜索 entity / slot / namespace…" style="min-width: 260px">
    <select id="ns-filter"><option value="">全部 namespace</option></select>
    <div class="legend">
      <span><b class="active"></b>active</span>
      <span><b class="stale"></b>stale</span>
      <span><b class="retracted"></b>retracted</span>
      <span><b class="anchor_drifted"></b>anchor_drifted</span>
    </div>
  </div>
</header>
<main>
  <div class="stats" id="stats"></div>
  <div id="lanes"></div>
  <div class="empty" id="empty" style="display:none">没有匹配的记忆链</div>
</main>
<div class="tip" id="tip"></div>
<script>
const DATA = __DATA__;
const lanesEl = document.getElementById('lanes');
const tip = document.getElementById('tip');
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// namespace filter options
const nsSet = [...new Set(DATA.lanes.map(l => l.namespace))];
const nsSel = document.getElementById('ns-filter');
nsSet.forEach(ns => { const o = document.createElement('option'); o.value = ns; o.textContent = ns; nsSel.appendChild(o); });

document.getElementById('stats').textContent =
  `${DATA.n_atoms} 个记忆原子版本 · ${DATA.lanes.length} 条 (namespace, entity, slot) 版本链 · 轮次轴 0 → ${DATA.max_turn + 2}`;

function fmtValue(v) {
  const s = JSON.stringify(v);
  return s.length > 120 ? s.slice(0, 117) + '…' : s;
}

function render() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  const ns = nsSel.value;
  lanesEl.innerHTML = '';
  let shown = 0;
  const span = DATA.max_turn + 2;
  const W = Math.max(760, lanesEl.clientWidth || 900), H = 74;
  const left = 8, right = 8, width = W - left - right;
  const x = t => left + (t / span) * width;

  for (const lane of DATA.lanes) {
    const hay = lane.key.toLowerCase();
    if (ns && lane.namespace !== ns) continue;
    if (q && !(hay.includes(q) || lane.versions.some(v => JSON.stringify(v.value).toLowerCase().includes(q)))) continue;
    shown++;
    const div = document.createElement('div');
    div.className = 'lane';
    const act = lane.versions.filter(v => v.status === 'active').length;
    div.innerHTML = `
      <div class="lane-head">
        <div class="lane-key"><span class="ns">${esc(lane.namespace)}</span>${esc(lane.entity)} · ${esc(lane.slot)}</div>
        <div class="lane-meta">${lane.versions.length} 个版本 · ${act} active</div>
      </div>`;
    let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">`;
    // axis
    const ticks = 5;
    svg += `<g class="axis">`;
    for (let i = 0; i <= ticks; i++) {
      const t = Math.round(span * i / ticks);
      svg += `<line x1="${x(t)}" y1="10" x2="${x(t)}" y2="${H - 16}"></line>` +
             `<text x="${x(t)}" y="${H - 3}" text-anchor="middle">${t}</text>`;
    }
    svg += `</g>`;
    const rowY = 26, barH = 24;
    lane.versions.forEach((v, i) => {
      const x1 = x(v.from), x2 = Math.max(x(v.to), x(v.from) + 14);
      const y = rowY + (i % 2) * 0; // single row lane; offsets via opacity
      svg += `<rect class="bar ${v.status}" x="${x1.toFixed(1)}" y="${rowY}" width="${(x2 - x1).toFixed(1)}" height="${barH}" data-i="${i}" data-lane="${esc(lane.key)}"></rect>`;
      const mid = (x1 + x2) / 2;
      svg += `<text class="bar-label" x="${mid.toFixed(1)}" y="${rowY + 16}" text-anchor="middle">#${v.atom_id}</text>`;
      if (v.supersedes != null) {
        svg += `<line x1="${x1.toFixed(1)}" y1="${rowY}" x2="${x1.toFixed(1)}" y2="${rowY - 6}" stroke="#5b9dff" stroke-width="1.5"></line>`;
      }
      if (v.open) {
        svg += `<path class="open-arrow" d="M ${(x2).toFixed(1)} ${rowY + barH / 2} h 12 m -6 -5 l 6 5 l -6 5"></path>`;
      }
    });
    svg += `</svg>`;
    div.innerHTML += svg;
    lanesEl.appendChild(div);
  }
  document.getElementById('empty').style.display = shown ? 'none' : 'block';

  // tooltips
  lanesEl.querySelectorAll('.bar').forEach(bar => {
    bar.addEventListener('mousemove', e => {
      const lane = DATA.lanes.find(l => l.key === bar.dataset.lane);
      const v = lane.versions[+bar.dataset.i];
      tip.innerHTML = `
        <div class="row"><b>#${v.atom_id}</b> · ${esc(v.status)} · 轮次 ${v.from} → ${v.open ? '∞ (active)' : v.to}</div>
        <div class="row v">${esc(fmtValue(v.value))}</div>
        ${v.anchor_path ? `<div class="row">⚓ ${esc(v.anchor_path)} <span style="color:#8292b0">sha:${esc(v.anchor_sha256)}</span></div>` : ''}
        ${v.supersedes != null ? `<div class="row" style="color:#8292b0">supersedes #${v.supersedes}${v.superseded_by ? ' · superseded_by #' + v.superseded_by : ''}</div>` : ''}`;
      tip.style.display = 'block';
      tip.style.left = Math.min(e.clientX + 14, innerWidth - 460) + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
    });
    bar.addEventListener('mouseleave', () => tip.style.display = 'none');
  });
}

document.getElementById('search').addEventListener('input', render);
nsSel.addEventListener('change', render);
addEventListener('resize', render);
render();
</script>
</body>
</html>
"""


def render_visualizer(store: MemoryStore, out_path: str) -> str:
    """Write the self-contained timeline HTML for the given store."""
    data = collect_visualization_data(store)
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path)
