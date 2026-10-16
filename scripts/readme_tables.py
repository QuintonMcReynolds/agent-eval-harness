"""Regenerate the README's result tables from runs/ and reports/ (between <!-- gen:NAME --> markers)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.compare import compare, gate  # noqa: E402
from harness.grading import load_grades  # noqa: E402
from harness.runner import RUNS, load_run  # noqa: E402
from harness.stats import cohen_kappa  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = [("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ("Qwen2.5-3B", "Qwen2.5-3B-Instruct"),
          ("Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct"), ("Llama-3.2-3B", "Llama-3.2-3B-Instruct")]


def S(run):
    p = RUNS / run / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f} %"


def rid(m, p, t="T0x1"):
    return f"geoint__{m}__{p}__{t}"


def leaderboard():
    rows = ["| model (MLX 4-bit) | v1 | v2 | v3 | v1 → v3 judge-agreed (strict) | sec/task | malformed calls v1 / v2 / v3 |",
            "|---|---|---|---|---|---|---|"]
    for m, name in MODELS:
        s = {p: S(rid(m, p)) for p in ("v1", "v2", "v3")}
        cell = lambda p: f"{pct(s[p]['det']['pass_rate'])} <sub>[{100*s[p]['det']['ci95'][0]:.0f}–{100*s[p]['det']['ci95'][1]:.0f}]</sub>"
        strict = " → ".join(pct(s[p].get("strict", {}).get("pass_rate")) for p in ("v1", "v3"))
        mal = " / ".join(f"{100*s[p]['ops']['malformed_call_rate']:.0f} %" for p in ("v1", "v2", "v3"))
        rows.append(f"| {name} | {cell('v1')} | {cell('v2')} | {cell('v3')} | {strict} | {s['v1']['ops']['mean_wall_s']:.1f} | {mal} |")
    return "\n".join(rows)


def categories():
    rows = ["| category (n) | " + " | ".join(f"{m} v1 → v3" for m, _ in MODELS[:1] + MODELS[3:]) + " |", "|---|---|---|"]
    cats = {"lookup": 26, "multi_step": 11, "unanswerable": 8, "judgment": 6, "injection": 6}
    for c, n in cats.items():
        cells = []
        for m, _ in MODELS[:1] + MODELS[3:]:
            a, b = S(rid(m, "v1"))["det"]["by_category"][c], S(rid(m, "v3"))["det"]["by_category"][c]
            cells.append(f"{pct(a, 0)} → {pct(b, 0)}")
        rows.append(f"| {c} ({n}) | " + " | ".join(cells) + " |")
    return "\n".join(rows)


COMPARISONS = [
    ("Qwen2.5-3B", "v1", "Qwen2.5-3B", "v2", "Qwen-3B v1 → v2"),
    ("Qwen2.5-7B", "v1", "Qwen2.5-7B", "v2", "Qwen-7B v1 → v2"),
    ("Qwen2.5-7B", "v1", "Qwen2.5-7B", "v3", "Qwen-7B v1 → v3"),
    ("Qwen2.5-3B", "v1", "Qwen2.5-3B", "v3", "Qwen-3B v1 → v3"),
    ("Qwen2.5-1.5B", "v1", "Qwen2.5-1.5B", "v3", "Qwen-1.5B v1 → v3"),
    ("Llama-3.2-3B", "v1", "Llama-3.2-3B", "v3", "Llama-3B v1 → v3"),
    ("Qwen2.5-3B", "v2", "Qwen2.5-3B", "v2+repair", "Qwen-3B v2 → v2 + repair loop"),
]


def comparisons():
    rows = ["| change | pass rate | paired diff [95 % CI] | McNemar p | fixed / broke | gate | blocking reasons |",
            "|---|---|---|---|---|---|---|"]
    for bm, bp, cm, cp, label in COMPARISONS:
        c = compare(RUNS / rid(bm, bp), RUNS / rid(cm, cp), "det")
        ok, why = gate(c)
        d = c["diff"]
        reasons = "; ".join(re.sub(r" \(lost .*\)", "", w) for w in why) or "–"
        rows.append(f"| {label} | {pct(c['baseline_rate'])} → {pct(c['candidate_rate'])} | {100*d['diff']:+.1f} pt [{100*d['lo']:+.1f}, {100*d['hi']:+.1f}] | "
                    f"{c['mcnemar']['p_value']:.3f} | {c['mcnemar']['improvements']} / {c['mcnemar']['regressions']} | {'✅ ships' if ok else '❌ blocked'} | {reasons} |")
    return "\n".join(rows)


def reliability():
    rows = ["| prompt | pass@1 | pass@2 | pass@4 | pass^2 | **pass^4** |", "|---|---|---|---|---|---|"]
    for p in ("v1", "v2"):
        s = S(rid("Qwen2.5-3B", p, "T0.7x5"))["det"]
        rows.append(f"| {p} | {pct(s['pass^k']['1'])} | {pct(s['pass@k']['2'])} | {pct(s['pass@k']['4'])} | {pct(s['pass^k']['2'])} | **{pct(s['pass^k']['4'])}** |")
    c = compare(RUNS / rid("Qwen2.5-3B", "v1", "T0.7x5"), RUNS / rid("Qwen2.5-3B", "v2", "T0.7x5"), "det")
    rows.append("")
    rows.append(f"Paired v1 → v2 over the sampled runs: {100*c['diff']['diff']:+.1f} pt (95 % CI {100*c['diff']['lo']:+.1f} to {100*c['diff']['hi']:+.1f}).")
    return "\n".join(rows)


def judge_validation():
    J = {v: json.loads((ROOT / "reports" / f"judge_validation_{v}.json").read_text())
         for v in ("v1", "v2") if (ROOT / "reports" / f"judge_validation_{v}.json").exists()}
    if not J:
        return "_not run yet_"
    vs = list(J)
    rows = ["| item type (expected verdict) | n | " + " | ".join(f"judge prompt {v}" for v in vs) + " |", "|---|---|" + "---|" * len(vs)]
    for k in J[vs[-1]]["by_kind"]:
        exp = "pass" if k in ("control", "padded") else "fail"
        n = J[vs[-1]]["by_kind"][k]["n"]
        rows.append(f"| {k} ({exp}) | {n} | " + " | ".join(pct(J[v]['by_kind'].get(k, {}).get('accuracy'), 0) for v in vs) + " |")
    rows.append(f"| **overall accuracy** | {J[vs[-1]]['n_items']} | " + " | ".join(f"**{pct(J[v]['accuracy'], 0)}**" for v in vs) + " |")
    rows.append("| errors caught (fail items judged fail) | | " + " | ".join(pct(J[v]['fail_detection_rate'], 0) for v in vs) + " |")
    rows.append("| false alarms (pass items judged fail) | | " + " | ".join(pct(J[v]['false_fail_rate'], 0) for v in vs) + " |")
    rows.append("| Cohen's κ vs labels | | " + " | ".join(f"{J[v]['kappa_vs_labels']:.2f}" for v in vs) + " |")
    rows.append("| same verdict with prompt blocks reordered | | " + " | ".join(pct(J[v]['order_consistency'], 0) for v in vs) + " |")
    return "\n".join(rows)


def agreement():
    fam = {}
    for p in sorted(RUNS.glob("*__T0x1")):
        meta, _ = load_run(p)
        for g in load_grades(p):
            if g.judge:
                f = "Llama (judge is a different family)" if "llama" in meta.agent.model.lower() else "Qwen (same family as judge)"
                fam.setdefault(f, []).append((g.passed, g.judge["verdict"] == "pass"))
    if not fam:
        return "_not run yet_"
    rows = ["| agent family | n | agreement | κ | judge pass / checks fail | judge fail / checks pass |", "|---|---|---|---|---|---|"]
    for f, pr in sorted(fam.items()):
        d, j = [a for a, _ in pr], [b for _, b in pr]
        rows.append(f"| {f} | {len(pr)} | {pct(sum(a == b for a, b in pr) / len(pr), 0)} | {cohen_kappa(d, j):.2f} | "
                    f"{sum(b and not a for a, b in pr)} | {sum(a and not b for a, b in pr)} |")
    return "\n".join(rows)


GEN = {"leaderboard": leaderboard, "categories": categories, "comparisons": comparisons, "reliability": reliability,
       "judge_validation": judge_validation, "agreement": agreement}


def main():
    p = ROOT / "README.md"
    s = p.read_text()
    for name, fn in GEN.items():
        s = re.sub(rf"(<!-- gen:{name} -->\n).*?(<!-- /gen:{name} -->)", lambda m: m.group(1) + fn() + "\n" + m.group(2), s, flags=re.S)
    p.write_text(s)
    print("README tables regenerated")


if __name__ == "__main__":
    main()
