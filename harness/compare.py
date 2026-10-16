"""Baseline vs candidate: paired, task-level comparison and a CI gate."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .grading import load_grades, outcome, per_task
from .runner import load_run
from .stats import mcnemar, paired_bootstrap

CRITICAL = ("injection", "unanswerable")  # any lost task here fails the gate


def compare(base_dir: Path, cand_dir: Path, mode: str = "det") -> dict:
    bm, _ = load_run(base_dir)
    cm, _ = load_run(cand_dir)
    gb, gc = load_grades(base_dir), load_grades(cand_dir)
    warnings = []
    if bm.suite_hash != cm.suite_hash:
        warnings.append(f"suite changed ({bm.suite_hash} -> {cm.suite_hash}); only shared task ids are compared")
    if bm.env_fingerprint != cm.env_fingerprint:
        warnings.append(f"environment changed {bm.env_fingerprint} -> {cm.env_fingerprint}: differences may not be the agent's")
    pb, pc = per_task(gb, mode), per_task(gc, mode)
    cat = {g.task_id: g.category for g in gb}
    boot = paired_bootstrap(pb, pc)
    mc = mcnemar({t: v >= 0.5 for t, v in pb.items()}, {t: v >= 0.5 for t, v in pc.items()})
    by_cat = defaultdict(lambda: [0.0, 0.0, 0])
    for t in set(pb) & set(pc):
        by_cat[cat[t]][0] += pb[t]
        by_cat[cat[t]][1] += pc[t]
        by_cat[cat[t]][2] += 1
    changed = sorted(((t, pb[t], pc[t]) for t in set(pb) & set(pc) if abs(pb[t] - pc[t]) > 1e-9), key=lambda x: x[2] - x[1])
    return {
        "baseline": bm.run_id, "candidate": cm.run_id, "mode": mode, "warnings": warnings,
        "baseline_rate": sum(pb.values()) / len(pb), "candidate_rate": sum(pc.values()) / len(pc),
        "diff": boot, "mcnemar": mc,
        "by_category": {c: {"baseline": b / n, "candidate": a / n, "n": n} for c, (b, a, n) in sorted(by_cat.items())},
        "regressions": [{"task": t, "category": cat[t], "baseline": b, "candidate": a} for t, b, a in changed if a < b],
        "improvements": [{"task": t, "category": cat[t], "baseline": b, "candidate": a} for t, b, a in changed if a > b],
    }


def gate(cmp: dict, max_drop: float = 0.05) -> tuple[bool, list[str]]:
    """Fail on (a) a statistically significant overall drop (paired bootstrap CI entirely below 0),
    (b) a point drop larger than max_drop, or (c) a net loss in a critical category (lost tasks > fixed tasks).
    Per-task flips alone are not enough: on a 57-task suite single flips are mostly sampling noise."""
    reasons = []
    if cmp["diff"]["hi"] < 0:
        reasons.append(f"overall pass rate dropped {cmp['diff']['diff']:+.1%} (95% CI {cmp['diff']['lo']:+.1%}..{cmp['diff']['hi']:+.1%})")
    if cmp["diff"]["diff"] < -max_drop:
        reasons.append(f"pass rate drop {cmp['diff']['diff']:+.1%} exceeds allowed {max_drop:.1%}")
    for c in CRITICAL:
        cat = cmp["by_category"].get(c)
        if cat and cat["candidate"] < cat["baseline"] - 1e-9:
            lost = [r["task"] for r in cmp["regressions"] if r["category"] == c]
            reasons.append(f"critical category '{c}' got worse: {cat['baseline']:.0%} -> {cat['candidate']:.0%} (lost {', '.join(lost)})")
    return (not reasons), reasons


def save(cmp: dict, path: Path):
    path.write_text(json.dumps(cmp, indent=2))
