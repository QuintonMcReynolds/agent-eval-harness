"""Static HTML report over every graded run in runs/: reports/index.html (no server, no JS framework)."""
from __future__ import annotations

import datetime as dt
import html
import json
from pathlib import Path

from jinja2 import Template

from .compare import compare, gate
from .grading import load_grades, outcome
from .runner import RUNS, load_run
from .stats import cohen_kappa
from .suite import ROOT

CATS = ["lookup", "multi_step", "unanswerable", "judgment", "injection"]

TPL = Template(r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Agent Eval Report</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#e4e1d8;--card:#fff;--good:#1f7a4d;--bad:#b3261e;--accent:#2f5d9b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--card:#1e1e1c;--good:#5cc28f;--bad:#f2867d;--accent:#8db4ea}}
:root[data-theme="dark"]{--bg:#151514;--fg:#ecebe6;--muted:#a3a29b;--line:#34332f;--card:#1e1e1c;--good:#5cc28f;--bad:#f2867d;--accent:#8db4ea}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:40px 0 8px}
.muted{color:var(--muted)}.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th{font-weight:600;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}td:first-child,th:first-child,td.l{text-align:left}
tr:last-child td{border-bottom:0}.ci{color:var(--muted);font-size:12px}.good{color:var(--good)}.bad{color:var(--bad)}
.heat{border-radius:4px;padding:2px 6px;display:inline-block;min-width:44px}
details{border:1px solid var(--line);border-radius:8px;background:var(--card);margin:6px 0;padding:6px 10px}summary{cursor:pointer}
pre{white-space:pre-wrap;word-break:break-word;font-size:12.5px;background:var(--bg);padding:8px;border-radius:6px;margin:6px 0}
.kpis{display:flex;gap:12px;flex-wrap:wrap;margin:14px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;min-width:150px}
.kpi b{display:block;font-size:22px}.tag{font-size:11px;border:1px solid var(--line);border-radius:999px;padding:1px 8px;color:var(--muted)}
</style></head><body><main>
<h1>Agent eval report</h1>
<div class="muted">Suite <b>{{ suite }}</b> ({{ n_tasks }} tasks) · judge <b>{{ judge_model }}</b> · generated {{ now }}</div>

<div class="kpis">
{% for k in kpis %}<div class="kpi"><span class="muted">{{ k[0] }}</span><b>{{ k[1] }}</b><span class="muted">{{ k[2] }}</span></div>{% endfor %}
</div>

<h2>Leaderboard (greedy, 1 sample per task)</h2>
<p class="muted">Deterministic = every check passes. Judge = LLM judge says pass. Strict = both. Brackets: 95% Wilson interval.</p>
<div class="wrap"><table><tr><th>model</th><th>prompt</th><th>deterministic</th><th>judge</th><th>strict</th>
{% for c in cats %}<th>{{ c }}</th>{% endfor %}<th>tool calls</th><th>malformed</th><th>sec/task</th></tr>
{% for r in board %}<tr><td>{{ r.model }}</td><td>{{ r.prompt }}</td>
<td>{{ pct(r.det) }} <span class="ci">[{{ pct(r.det_ci[0]) }}–{{ pct(r.det_ci[1]) }}]</span></td><td>{{ pct(r.judge) }}</td><td><b>{{ pct(r.strict) }}</b></td>
{% for c in cats %}<td><span class="heat" style="background:{{ heat(r.cat[c]) }}">{{ pct(r.cat[c]) }}</span></td>{% endfor %}
<td>{{ "%.1f"|format(r.calls) }}</td><td>{{ pct(r.malformed) }}</td><td>{{ "%.1f"|format(r.sec) }}</td></tr>{% endfor %}
</table></div>
<p class="muted">Category cells use the strict outcome.</p>

<h2>Prompt change v1 → v2 (paired, per model)</h2>
<p class="muted">Each task is compared with itself. Diff CI: paired bootstrap over tasks. The CI gate fails on a significant drop or on any lost injection/unanswerable task.</p>
<div class="wrap"><table><tr><th>model</th><th>v1</th><th>v2</th><th>diff [95% CI]</th><th>McNemar p</th><th>fixed</th><th>broke</th><th>gate</th><th class="l">regressions</th></tr>
{% for c in cmps %}<tr><td>{{ c.model }}</td><td>{{ pct(c.baseline_rate) }}</td><td>{{ pct(c.candidate_rate) }}</td>
<td class="{{ 'good' if c.diff.lo>0 else ('bad' if c.diff.hi<0 else '') }}">{{ sgn(c.diff.diff) }} <span class="ci">[{{ sgn(c.diff.lo) }}, {{ sgn(c.diff.hi) }}]</span></td>
<td>{{ "%.3f"|format(c.mcnemar.p_value) }}</td><td class="good">{{ c.improvements|length }}</td><td class="bad">{{ c.regressions|length }}</td>
<td>{{ '✅ pass' if c.gate_ok else '❌ fail' }}</td><td class="l">{{ c.regressions|map(attribute='task')|join(', ') }}</td></tr>{% endfor %}
</table></div>

{% if rel %}<h2>Reliability: same task, 5 samples at T=0.7</h2>
<p class="muted">pass@k = at least one of k attempts passes. pass^k = all k attempts pass (what a user relying on the agent experiences).</p>
<div class="wrap"><table><tr><th>run</th>{% for k in rel[0].ks %}<th>pass@{{k}}</th>{% endfor %}{% for k in rel[0].ks %}<th>pass^{{k}}</th>{% endfor %}</tr>
{% for r in rel %}<tr><td>{{ r.name }}</td>{% for k in r.ks %}<td>{{ pct(r.at[k]) }}</td>{% endfor %}{% for k in r.ks %}<td><b>{{ pct(r.hat[k]) }}</b></td>{% endfor %}</tr>{% endfor %}
</table></div>{% endif %}

{% if jv %}<h2>Is the judge trustworthy?</h2>
<p class="muted">{{ jv.n_items }} answers with known labels: real passing answers from {{ jv.source_run }}, plus controlled perturbations.</p>
<div class="wrap"><table><tr><th>item type</th><th>expected</th><th>n</th>{% if jv.v1_overall is defined %}<th>judge prompt v1</th>{% endif %}<th>judge prompt {{ jv.judge_version }}</th></tr>
{% for k,v in jv.by_kind.items() %}<tr><td>{{ k }}</td><td>{{ 'pass' if k in ('control','padded') else 'fail' }}</td><td>{{ v.n }}</td>{% if jv.v1_overall is defined %}<td>{{ pct(v.v1_accuracy) }}</td>{% endif %}<td><span class="heat" style="background:{{ heat(v.accuracy) }}">{{ pct(v.accuracy) }}</span></td></tr>{% endfor %}
<tr><td><b>overall</b></td><td></td><td>{{ jv.n_items }}</td>{% if jv.v1_overall is defined %}<td>{{ pct(jv.v1_overall) }}</td>{% endif %}<td><b>{{ pct(jv.accuracy) }}</b> · κ {{ "%.2f"|format(jv.kappa_vs_labels) }}</td></tr>
<tr><td>verdict unchanged when prompt blocks reordered</td><td></td><td></td>{% if jv.v1_overall is defined %}<td></td>{% endif %}<td>{{ pct(jv.order_consistency) }}</td></tr>
</table></div>{% endif %}

{% if agree %}<h2>Judge vs deterministic grader on real trajectories</h2>
<div class="wrap"><table><tr><th>agent family</th><th>n</th><th>agreement</th><th>κ</th><th>judge pass, checks fail</th><th>judge fail, checks pass</th></tr>
{% for a in agree %}<tr><td>{{ a.family }}</td><td>{{ a.n }}</td><td>{{ pct(a.agree) }}</td><td>{{ "%.2f"|format(a.kappa) }}</td><td>{{ a.jp_df }}</td><td>{{ a.jf_dp }}</td></tr>{% endfor %}
</table></div>{% endif %}

<h2>Failures</h2>
{% for r in fails %}<details><summary><b>{{ r.run }}</b> <span class="tag">{{ r.rows|length }} failing</span></summary>
{% for f in r.rows %}<details><summary>{{ f.task }} <span class="tag">{{ f.category }}</span> <span class="muted">{{ f.why }}</span></summary>
<div class="muted">Q: {{ f.question }}</div><pre>tools: {{ f.calls }}</pre><pre>answer: {{ f.answer }}</pre>{% if f.judge %}<div class="muted">judge: {{ f.judge }}</div>{% endif %}
</details>{% endfor %}</details>{% endfor %}
</main></body></html>""")


def _pct(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def _sgn(x):
    return f"{100 * x:+.1f} pt"


def _heat(x):
    if x is None:
        return "transparent"
    # red -> amber -> green, low alpha so it reads in both themes
    r, g = (int(220 * (1 - x) + 40 * x), int(80 * (1 - x) + 170 * x))
    return f"rgba({r},{g},90,.28)"


def _family(model: str) -> str:
    return "Llama" if "llama" in model.lower() else "Qwen"


def build() -> Path:
    runs = sorted(p for p in RUNS.iterdir() if (p / "summary.json").exists())
    S = {p.name: json.loads((p / "summary.json").read_text()) for p in runs}
    greedy = [p for p in runs if S[p.name]["samples"] == 1]
    board = []
    for p in greedy:
        s = S[p.name]
        grades = load_grades(p)
        cat = {}
        for c in CATS:
            gs = [g for g in grades if g.category == c]
            cat[c] = sum(outcome(g, "strict") for g in gs) / len(gs) if gs else None
        board.append({"model": p.name.split("__")[1], "prompt": s["prompt"], "det": s["det"]["pass_rate"], "det_ci": s["det"]["ci95"],
                      "judge": s.get("judge", {}).get("pass_rate"), "strict": s.get("strict", {}).get("pass_rate"),
                      "cat": cat, "calls": s["ops"]["mean_tool_calls"], "malformed": s["ops"]["malformed_call_rate"],
                      "sec": s["ops"]["mean_wall_s"], "_sort": s.get("strict", s["det"])["pass_rate"]})
    board.sort(key=lambda r: -r["_sort"])
    cmps = []
    by_model = {}
    for p in greedy:
        by_model.setdefault(p.name.split("__")[1], {})[S[p.name]["prompt"]] = p
    mode = "strict" if all("strict" in S[p.name] for p in greedy) else "det"
    for m, d in sorted(by_model.items()):
        if "v1" in d and "v2" in d:
            c = compare(d["v1"], d["v2"], mode)
            c["model"] = m
            c["gate_ok"], _ = gate(c)
            cmps.append(c)
    rel = []
    for p in runs:
        s = S[p.name]
        if s["samples"] > 1:
            blk = s.get("strict", s["det"])
            ks = sorted(int(k) for k in blk["pass^k"])
            rel.append({"name": f"{p.name.split('__')[1]} {s['prompt']}", "ks": ks,
                        "hat": {k: blk["pass^k"][str(k)] for k in ks}, "at": {k: blk["pass@k"][str(k)] for k in ks}})
    jvs = {v: json.loads((ROOT / "reports" / f"judge_validation_{v}.json").read_text())
           for v in ("v1", "v2") if (ROOT / "reports" / f"judge_validation_{v}.json").exists()}
    jv = jvs.get("v2") or jvs.get("v1")
    if jv and "v1" in jvs and "v2" in jvs:
        for k, v in jv["by_kind"].items():
            v["v1_accuracy"] = jvs["v1"]["by_kind"].get(k, {}).get("accuracy")
        jv["v1_overall"] = jvs["v1"]["accuracy"]
    agree = []
    fam = {}
    for p in greedy:
        meta, _ = load_run(p)
        for g in load_grades(p):
            if g.judge:
                fam.setdefault(_family(meta.agent.model), []).append((g.passed, g.judge["verdict"] == "pass"))
    for f, pairs in sorted(fam.items()):
        d, j = [a for a, _ in pairs], [b for _, b in pairs]
        agree.append({"family": f, "n": len(pairs), "agree": sum(a == b for a, b in pairs) / len(pairs), "kappa": cohen_kappa(d, j),
                      "jp_df": sum(b and not a for a, b in pairs), "jf_dp": sum(a and not b for a, b in pairs)})
    fails = []
    for p in greedy:
        meta, trajs = load_run(p)
        tmap = {(t.task_id, t.sample): t for t in trajs}
        items = []
        for g in load_grades(p):
            jv_ok = (g.judge or {}).get("verdict") == "pass"
            if g.passed and (jv_ok or not g.judge):
                continue
            t = tmap[(g.task_id, g.sample)]
            why = ", ".join(c.name for c in g.checks if c.gating and not c.passed) or "checks ok"
            items.append({"task": g.task_id, "category": g.category, "why": why + ("" if jv_ok else " · judge: fail"),
                          "question": t.question, "answer": html.unescape(t.answer[:1500]),
                          "calls": ", ".join(f"{c['name']}({json.dumps(c['args'])})" for c in t.tool_calls) or "(none)",
                          "judge": (f"{g.judge['verdict']} [{g.judge['error_type']}] {g.judge['reason']}" if g.judge else "")})
        fails.append({"run": p.name, "rows": items})
    best = board[0] if board else None
    kpis = []
    if best:
        kpis.append(("best config (strict)", _pct(best["strict"] or best["det"]), f"{best['model']} · {best['prompt']}"))
    if jv:
        kpis.append(("judge accuracy on known labels", _pct(jv["accuracy"]), f"{jv['n_items']} items, κ {jv['kappa_vs_labels']:.2f}"))
    if rel:
        r0 = rel[0]
        kmax = max(r0["ks"])
        kpis.append((f"pass@1 → pass^{kmax}", f"{_pct(r0['hat'][1])} → {_pct(r0['hat'][kmax])}", r0["name"]))
    out = ROOT / "reports" / "index.html"
    suite = json.loads((greedy[0] / "meta.json").read_text())["suite"] if greedy else "-"
    out.write_text(TPL.render(suite=suite, n_tasks=len(load_grades(greedy[0])) if greedy else 0,
                              judge_model=(jv or {}).get("judge_model", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
                              now=dt.date.today().isoformat(), kpis=kpis, board=board, cats=CATS, cmps=cmps, rel=rel, jv=jv,
                              agree=agree, fails=fails, pct=_pct, sgn=_sgn, heat=_heat))
    return out
