"""Grade a recorded run: deterministic checks always, LLM judge optionally. Writes runs/<id>/grades.jsonl + summary.json."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .graders.deterministic import grade
from .runner import load_run
from .schema import Grade
from .stats import pass_at_k, pass_hat_k, wilson
from .suite import load_suite


def grade_run(run_dir: Path, judge=None) -> list[Grade]:
    meta, trajs = load_run(run_dir)
    if not trajs:
        raise ValueError(f"{run_dir.name}: no trajectories recorded yet")
    tasks = {t.id: t for t in load_suite(meta.suite).tasks}
    # without a judge, keep any verdicts already recorded so a deterministic re-grade doesn't erase them
    prev = {(g.task_id, g.sample): g.judge for g in load_grades(run_dir)} if (run_dir / "grades.jsonl").exists() else {}
    out = []
    for tr in trajs:
        g = grade(tasks[tr.task_id], tr)
        g.judge = judge(tasks[tr.task_id], tr) if judge is not None else prev.get((tr.task_id, tr.sample))
        out.append(g)
    (run_dir / "grades.jsonl").write_text("".join(g.model_dump_json() + "\n" for g in out))
    (run_dir / "summary.json").write_text(json.dumps(summarize(meta, trajs, out), indent=2))
    return out


def load_grades(run_dir: Path) -> list[Grade]:
    return [Grade(**json.loads(x)) for x in (run_dir / "grades.jsonl").read_text().splitlines() if x.strip()]


# Categories where judge validation showed the judge can't be trusted: on injection tasks both judge prompts
# were themselves swayed by the injected instruction (reports/judge_validation_*.json). There the deterministic
# checks decide alone in "strict" mode.
JUDGE_UNTRUSTED = {"injection"}


def outcome(g: Grade, mode: str) -> bool:
    if mode == "det":
        return g.passed
    jp = bool(g.judge) and g.judge.get("verdict") == "pass"
    if mode == "judge":
        return jp
    if mode == "strict":
        return g.passed and (jp or g.category in JUDGE_UNTRUSTED)
    raise ValueError(mode)


def per_task(grades: list[Grade], mode: str) -> dict[str, float]:
    """Mean pass over samples, per task."""
    acc = defaultdict(list)
    for g in grades:
        acc[g.task_id].append(outcome(g, mode))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def summarize(meta, trajs, grades: list[Grade]) -> dict:
    has_judge = any(g.judge for g in grades)
    modes = ["det"] + (["judge", "strict"] if has_judge else [])
    S = {"run_id": meta.run_id, "model": meta.agent.model, "prompt": meta.agent.prompt_version,
         "temperature": meta.agent.temperature, "samples": meta.samples, "n_trajectories": len(trajs)}
    for m in modes:
        pt = per_task(grades, m)
        k = sum(round(v * meta.samples) for v in pt.values())
        n = len(pt) * meta.samples
        lo, hi = wilson(k, n)
        cats = defaultdict(list)
        for g in grades:
            cats[g.category].append(outcome(g, m))
        S[m] = {"pass_rate": k / n, "ci95": [lo, hi],
                "by_category": {c: sum(v) / len(v) for c, v in sorted(cats.items())}}
        if meta.samples > 1:
            counts = defaultdict(lambda: [0, 0])
            for g in grades:
                counts[g.task_id][0] += 1
                counts[g.task_id][1] += outcome(g, m)
            ks = [kk for kk in (1, 2, 4, 8) if kk <= meta.samples]
            S[m]["pass^k"] = {kk: sum(pass_hat_k(n_, c_, kk) for n_, c_ in counts.values()) / len(counts) for kk in ks}
            S[m]["pass@k"] = {kk: sum(pass_at_k(n_, c_, kk) for n_, c_ in counts.values()) / len(counts) for kk in ks}
    steps = [s for t in trajs for s in t.steps]
    S["ops"] = {
        "mean_tool_calls": sum(len(t.tool_calls) for t in trajs) / len(trajs),
        "malformed_call_rate": sum(s.malformed_calls for s in steps) / max(1, sum(len(s.tool_calls) + s.malformed_calls for s in steps if s.kind == "llm")),
        "tool_error_rate": sum(s.error for s in steps if s.kind == "tool") / max(1, sum(s.kind == "tool" for s in steps)),
        "mean_wall_s": sum(t.wall_s for t in trajs) / len(trajs),
        "p95_wall_s": sorted(t.wall_s for t in trajs)[int(0.95 * (len(trajs) - 1))],
        "mean_completion_tokens": sum(s.completion_tokens for s in steps) / len(trajs),
        "stop_reasons": {r: sum(t.stop_reason == r for t in trajs) for r in ("answer", "max_steps", "error")},
    }
    return S
