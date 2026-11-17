"""Self-contained research dashboard (reports/dashboard.html): data embedded as JSON, vanilla JS + SVG charts.

Sections: upcoming prints (live forecasts ranked by predicted beat vs guidance), model leaderboard (dev / holdout),
company explorer (forecast history vs reported vs guidance, revenue and gross margin), earnings calendar.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .paths import REPORTS


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return None if pd.isna(o) else str(o)[:19]
    if o is pd.NaT:
        return None
    return o


def build(data: dict, path=None) -> str:
    path = path or REPORTS / "dashboard.html"
    payload = json.dumps(_clean(data), separators=(",", ":"))
    html = TEMPLATE.replace("__DATA__", payload.replace("</", "<\\/"))
    path.write_text(html, encoding="utf-8")
    return str(path)


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semis Earnings Engine</title>
<style>
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#006300;--bad:#d03b3b;--wash:rgba(42,120,214,.08)}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;
--s3:#199e70;--good:#0ca30c;--bad:#e66767;--wash:rgba(57,135,229,.12)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#0ca30c;--bad:#e66767;
--wash:rgba(57,135,229,.12)}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px;font-weight:600}
h2{font-size:16px;margin:32px 0 4px;font-weight:600}
.sub{color:var(--ink2);margin:0 0 16px}
.note{color:var(--muted);font-size:12.5px;margin:6px 0 0}
.filters{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:16px 0}
.filters label{color:var(--ink2);font-size:13px;display:flex;gap:6px;align-items:center}
select,button{font:inherit;color:var(--ink);background:var(--surface);border:1px solid var(--border);border-radius:6px;padding:5px 8px}
button.on{border-color:var(--s1);box-shadow:inset 0 0 0 1px var(--s1)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:8px 0 4px}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile .l{color:var(--ink2);font-size:12.5px}.tile .v{font-size:24px;font-weight:600;margin-top:2px}
.tile .d{color:var(--muted);font-size:12px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px;margin-top:10px;overflow:hidden}
.tw{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
th{color:var(--ink2);font-weight:600;position:sticky;top:0;background:var(--surface)}
td:first-child,th:first-child{text-align:left}
td.l,th.l{text-align:left}
tbody tr:hover{background:var(--wash)}
.num{font-variant-numeric:tabular-nums}
.pos{color:var(--good)}.neg{color:var(--bad)}
.tag{display:inline-block;font-size:11.5px;border:1px solid var(--border);border-radius:999px;padding:1px 8px;color:var(--ink2)}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12.5px;color:var(--ink2);margin:4px 0 6px}
.legend i{display:inline-block;width:16px;height:2px;vertical-align:middle;margin-right:6px;border-radius:1px}
.chart{position:relative}
.chart svg{display:block;width:100%;height:auto}
.tip{position:absolute;pointer-events:none;background:var(--surface);border:1px solid var(--border);border-radius:8px;
padding:8px 10px;font-size:12.5px;box-shadow:0 4px 14px rgba(0,0,0,.12);display:none;min-width:170px;z-index:2}
.tip .h{color:var(--ink2);margin-bottom:4px}.tip .r{display:flex;align-items:center;gap:6px;justify-content:space-between}
.tip .r b{font-variant-numeric:tabular-nums}.tip .k{display:inline-block;width:12px;height:2px;margin-right:6px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:820px){.grid2{grid-template-columns:1fr}}
details summary{cursor:pointer;color:var(--ink2);font-size:13px;margin-top:8px}
</style>
</head>
<body>
<main>
<h1>Semis earnings engine</h1>
<p class="sub" id="subtitle"></p>
<div class="kpis" id="kpis"></div>

<h2>Upcoming prints: model vs management guidance</h2>
<p class="note">Next fiscal quarter for each active company, ranked by the model's predicted revenue beat vs the guidance midpoint.
"Awaiting release" = the quarter has ended and results are not out yet. In-progress quarters are forecasts made before the quarter has even ended.
The track record behind these numbers is in the leaderboard below: read them with it.</p>
<div class="card tw"><table id="live"></table></div>

<h2>Model leaderboard (walk-forward, out of sample)</h2>
<div class="filters">
<label>Period <select id="period"><option value="holdout">Holdout 2021-2026</option><option value="dev">Development 2013-2020</option></select></label>
<label>Experiment <select id="exp"></select></label>
</div>
<p class="note" id="expnote"></p>
<div class="card tw"><table id="board"></table></div>

<h2>Company explorer</h2>
<div class="filters"><label>Company <select id="tk"></select></label><span class="tag" id="tkinfo"></span></div>
<div class="grid2">
<div class="card"><div class="legend" id="lg1"></div><div class="chart" id="c1"></div>
<p class="note">Each forecast is made at the last market close before that quarter's release, using only data public at the time.</p></div>
<div class="card"><div class="legend" id="lg2"></div><div class="chart" id="c2"></div>
<p class="note">GAAP gross margin from XBRL; guidance shown only where the company guided on a GAAP basis.</p></div>
</div>
<details><summary>Table view</summary><div class="card tw"><table id="tktab"></table></div></details>

<h2>Earnings calendar (latest releases)</h2>
<p class="note">Release time = EDGAR acceptance of the 8-K Item 2.02 (UTC converted to New York). BMO before 09:30, AMC after 16:00.</p>
<div class="card tw"><table id="cal"></table></div>

<p class="note" id="foot"></p>
</main>
<script>
const D = __DATA__;
const $ = s => document.querySelector(s);
const el = (t, a = {}, txt) => { const e = document.createElement(t); for (const k in a) e.setAttribute(k, a[k]); if (txt != null) e.textContent = txt; return e; };
const fmtB = v => v == null ? "–" : (Math.abs(v) >= 1e9 ? (v / 1e9).toFixed(2) + "B" : (v / 1e6).toFixed(0) + "M");
const pct = (v, d = 1) => v == null ? "–" : (v > 0 ? "+" : "") + v.toFixed(d) + "%";
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

$("#subtitle").textContent = D.meta.subtitle;
$("#foot").textContent = D.meta.footer;

// KPI tiles
for (const k of D.kpis) { const t = el("div", {class: "tile"}); t.append(el("div", {class: "l"}, k.label), el("div", {class: "v"}, k.value), el("div", {class: "d"}, k.detail)); $("#kpis").append(t); }

// table helper (text only, never innerHTML for data)
function fillTable(tab, cols, rows) {
  tab.replaceChildren();
  const th = el("thead"), tr = el("tr");
  cols.forEach(c => tr.append(el("th", {class: c.l ? "l" : ""}, c.h)));
  th.append(tr); tab.append(th);
  const tb = el("tbody");
  rows.forEach(r => { const row = el("tr"); cols.forEach(c => { const v = c.f ? c.f(r) : r[c.k]; const td = el("td", {class: (c.l ? "l " : "num ") + (c.cls ? c.cls(r) : "")}, v == null ? "–" : v); row.append(td); }); tb.append(row); });
  tab.append(tb);
}

// live
fillTable($("#live"), [
  {h: "Company", k: "ticker", l: 1}, {h: "Status", k: "status_label", l: 1}, {h: "Est. release", k: "est_release", l: 1},
  {h: "Guidance mid", f: r => fmtB(r.guid_mid)}, {h: "Model revenue", f: r => fmtB(r.model_rev)},
  {h: "Predicted beat", f: r => pct(r.pred_beat_pct), cls: r => r.pred_beat_pct > 0 ? "pos" : "neg"},
  {h: "Company's avg past beat", f: r => pct(r.hist_beat_pct)}, {h: "Model GM", f: r => r.model_gm == null ? "–" : r.model_gm.toFixed(1) + "%"},
  {h: "GAAP GM guide", f: r => r.guid_gm == null ? "–" : r.guid_gm.toFixed(1) + "%"}], D.live);

// leaderboard
const exps = Object.keys(D.board);
exps.forEach(x => $("#exp").append(el("option", {value: x}, D.boardLabels[x])));
function drawBoard() {
  const x = $("#exp").value, p = $("#period").value, rows = (D.board[x] || {})[p] || [];
  $("#expnote").textContent = D.boardNotes[x] || "";
  const rev = x.startsWith("rev");
  const cols = [{h: "Model", k: "label", l: 1}, {h: "n", k: "n"}];
  if (rev) cols.push({h: "MAPE %", f: r => r["MAPE_%"]?.toFixed(2)}, {h: "Median APE %", f: r => r["MdAPE_%"]?.toFixed(2)}, {h: "RMSE (log %)", f: r => r["RMSE_log_%"]?.toFixed(2)});
  else cols.push({h: "MAE pp", f: r => r["MAE_pp"]?.toFixed(2)}, {h: "Median AE pp", f: r => r["MdAE_pp"]?.toFixed(2)}, {h: "RMSE pp", f: r => r["RMSE_pp"]?.toFixed(2)});
  cols.push({h: "Direction hit %", f: r => r["dir_acc_%"]?.toFixed(1)});
  if (x === "rev_guided") cols.push({h: "Beat/miss hit %", f: r => r["beat_dir_acc_%"]?.toFixed(1)}, {h: "IC (beat)", f: r => r["IC_beat"]?.toFixed(3)});
  else cols.push({h: "IC", f: r => r["IC"]?.toFixed(3)});
  cols.push({h: "vs reference", f: r => r["MAE_vs_ref_%"] == null ? "ref" : pct(r["MAE_vs_ref_%"])}, {h: "DM t (abs)", f: r => r["DM_t_abs"]?.toFixed(2)});
  fillTable($("#board"), cols, rows);
}
$("#exp").onchange = drawBoard; $("#period").onchange = drawBoard; drawBoard();

// line chart with crosshair tooltip
function lineChart(host, series, opt) {
  host.replaceChildren();
  const W = 560, H = 300, m = {l: 54, r: 14, t: 10, b: 30};
  const xs = [...new Set(series.flatMap(s => s.pts.map(p => p[0])))].sort();
  if (!xs.length) { host.append(el("p", {class: "note"}, "No data")); return; }
  const vals = series.flatMap(s => s.pts.map(p => p[1])).filter(v => v != null);
  const vmin = Math.min(...vals);
  let lo = vmin, hi = Math.max(...vals); const pad = (hi - lo) * 0.08 || 1; lo -= pad; hi += pad;
  if (opt.zero && vmin >= 0) lo = 0;
  const X = i => m.l + (W - m.l - m.r) * (xs.length === 1 ? 0.5 : i / (xs.length - 1));
  const Y = v => m.t + (H - m.t - m.b) * (1 - (v - lo) / (hi - lo));
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const mk = (t, a) => { const e = document.createElementNS(ns, t); for (const k in a) e.setAttribute(k, a[k]); return e; };
  for (let i = 0; i <= 4; i++) { const v = lo + (hi - lo) * i / 4, y = Y(v);
    svg.append(mk("line", {x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: css("--grid"), "stroke-width": 1}));
    const t = mk("text", {x: m.l - 6, y: y + 4, "text-anchor": "end", "font-size": 11, fill: css("--muted")}); t.textContent = opt.fmt(v); svg.append(t); }
  const step = Math.max(1, Math.ceil(xs.length / 7));
  xs.forEach((x, i) => { if (i % step === 0) { const t = mk("text", {x: X(i), y: H - 8, "text-anchor": "middle", "font-size": 11, fill: css("--muted")}); t.textContent = x.slice(0, 7); svg.append(t); } });
  series.forEach(s => {
    const pts = s.pts.filter(p => p[1] != null).map(p => [X(xs.indexOf(p[0])), Y(p[1])]);
    if (s.dots) pts.forEach(([x, y]) => svg.append(mk("circle", {cx: x, cy: y, r: 4, fill: css(s.color), stroke: css("--surface"), "stroke-width": 2})));
    else if (pts.length) svg.append(mk("path", {d: "M" + pts.map(p => p.join(",")).join("L"), fill: "none", stroke: css(s.color), "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round"}));
  });
  const cross = mk("line", {y1: m.t, y2: H - m.b, stroke: css("--axis"), "stroke-width": 1, visibility: "hidden"}); svg.append(cross);
  const hit = mk("rect", {x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent"}); svg.append(hit);
  host.append(svg);
  const tip = el("div", {class: "tip"}); host.append(tip);
  const show = ev => {
    const r = svg.getBoundingClientRect(), px = (ev.clientX - r.left) * W / r.width;
    let i = Math.round((px - m.l) / ((W - m.l - m.r) / Math.max(1, xs.length - 1))); i = Math.max(0, Math.min(xs.length - 1, i));
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("visibility", "visible");
    tip.replaceChildren(el("div", {class: "h"}, xs[i]));
    series.forEach(s => { const p = s.pts.find(q => q[0] === xs[i]); const row = el("div", {class: "r"}); const k = el("span");
      const key = el("i", {class: "k"}); key.style.background = css(s.color); k.append(key, document.createTextNode(s.name));
      row.append(k, el("b", {}, p && p[1] != null ? opt.fmt(p[1]) : "–")); tip.append(row); });
    tip.style.display = "block"; const hx = (X(i) / W) * r.width; tip.style.left = Math.min(hx + 12, r.width - 190) + "px"; tip.style.top = "8px";
  };
  hit.addEventListener("pointermove", show); hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); });
}
function legend(host, series) { host.replaceChildren(); series.forEach(s => { const sp = el("span"); const i = el("i"); i.style.background = css(s.color); if (s.dots) { i.style.height = "8px"; i.style.width = "8px"; i.style.borderRadius = "50%"; } sp.append(i, document.createTextNode(s.name)); host.append(sp); }); }

const tks = Object.keys(D.companies).sort();
tks.forEach(t => $("#tk").append(el("option", {value: t}, t + " · " + D.companies[t].name)));
$("#tk").value = tks.includes("NVDA") ? "NVDA" : tks[0];
function drawCompany() {
  const t = $("#tk").value, c = D.companies[t], q = c.quarters;
  $("#tkinfo").textContent = c.subindustry + (c.status === "active" ? "" : " · " + c.status);
  const s1 = [{name: "Reported revenue", color: "--ink2", pts: q.map(r => [r.pe, r.rev])},
              {name: "Guidance midpoint", color: "--s2", dots: true, pts: q.map(r => [r.pe, r.guid])},
              {name: c.revModelLabel, color: "--s1", dots: true, pts: q.map(r => [r.pe, r.rev_hat])}];
  legend($("#lg1"), s1); lineChart($("#c1"), s1, {fmt: v => fmtB(v), zero: true});
  const s2 = [{name: "Reported GAAP GM", color: "--ink2", pts: q.map(r => [r.pe, r.gm])},
              {name: "GAAP GM guidance", color: "--s2", dots: true, pts: q.map(r => [r.pe, r.gm_guid])},
              {name: c.gmModelLabel, color: "--s1", dots: true, pts: q.map(r => [r.pe, r.gm_hat])}];
  legend($("#lg2"), s2); lineChart($("#c2"), s2, {fmt: v => v.toFixed(1) + "%"});
  fillTable($("#tktab"), [{h: "Quarter end", k: "pe", l: 1}, {h: "Reported", f: r => fmtB(r.rev)}, {h: "Guidance mid", f: r => fmtB(r.guid)},
    {h: "Model", f: r => fmtB(r.rev_hat)}, {h: "Beat vs guide", f: r => r.rev && r.guid ? pct(100 * (r.rev / r.guid - 1)) : "–"},
    {h: "Model error", f: r => r.rev && r.rev_hat ? pct(100 * (r.rev_hat / r.rev - 1)) : "–"},
    {h: "GM", f: r => r.gm == null ? "–" : r.gm.toFixed(1)}, {h: "GM model", f: r => r.gm_hat == null ? "–" : r.gm_hat.toFixed(1)}], [...q].reverse());
}
$("#tk").onchange = drawCompany; drawCompany();
const mq = window.matchMedia("(prefers-color-scheme: dark)"); mq.addEventListener?.("change", drawCompany);

fillTable($("#cal"), [{h: "Company", k: "ticker", l: 1}, {h: "Quarter end", k: "pe", l: 1}, {h: "Released (ET)", k: "release_et", l: 1},
  {h: "Session", k: "session", l: 1}, {h: "Days after quarter end", k: "days"}, {h: "Revenue", f: r => fmtB(r.rev)},
  {h: "vs guidance", f: r => r.beat == null ? "–" : pct(r.beat), cls: r => r.beat > 0 ? "pos" : (r.beat < 0 ? "neg" : "")}], D.calendar);
</script>
</body>
</html>
"""
