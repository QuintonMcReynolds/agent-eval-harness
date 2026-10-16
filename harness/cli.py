"""harness run | grade | compare | gate | report | validate-judge | export-otel"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .runner import RUNS


def _run_dir(x: str) -> Path:
    p = Path(x)
    return p if p.exists() else RUNS / x


def main(argv=None):
    ap = argparse.ArgumentParser(prog="harness")
    sp = ap.add_subparsers(dest="cmd", required=True)

    r = sp.add_parser("run", help="run an agent config over a suite and record trajectories")
    r.add_argument("--suite", default="geoint")
    r.add_argument("--model", required=True)
    r.add_argument("--prompt", default="v1")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--samples", type=int, default=1)
    r.add_argument("--tasks", nargs="*")
    r.add_argument("--repair", action="store_true", help="feed malformed tool calls back to the model")

    g = sp.add_parser("grade", help="deterministic grading (+ --judge for the LLM judge)")
    g.add_argument("runs", nargs="+")
    g.add_argument("--judge", action="store_true")
    g.add_argument("--judge-version", default="v2", choices=["v1", "v2"])

    c = sp.add_parser("compare", help="paired comparison of two graded runs")
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--mode", default="det", choices=["det", "judge", "strict"])

    gt = sp.add_parser("gate", help="exit 1 if the candidate regresses vs baseline")
    gt.add_argument("baseline")
    gt.add_argument("candidate")
    gt.add_argument("--mode", default="det", choices=["det", "judge", "strict"])
    gt.add_argument("--max-drop", type=float, default=0.05)

    sp.add_parser("report", help="write reports/index.html from every graded run")
    vj = sp.add_parser("validate-judge", help="measure judge accuracy on perturbed answers with known labels")
    vj.add_argument("--source", default=None)
    vj.add_argument("--judge-version", default="v2", choices=["v1", "v2"])

    o = sp.add_parser("export-otel", help="write a run's spans as OTLP/JSON")
    o.add_argument("run")
    a = ap.parse_args(argv)

    if a.cmd == "run":
        from .runner import run
        from .schema import AgentConfig
        out = run(AgentConfig(model=a.model, prompt_version=a.prompt, temperature=a.temperature, repair_malformed=a.repair), a.suite, a.samples,
                  set(a.tasks) if a.tasks else None)
        print(out)
    elif a.cmd == "grade":
        from .grading import grade_run
        judge = None
        if a.judge:
            from .graders.judge import Judge
            judge = Judge(version=a.judge_version)
        for x in a.runs:
            grade_run(_run_dir(x), judge)
            s = json.loads((_run_dir(x) / "summary.json").read_text())
            print(s["run_id"], {m: round(s[m]["pass_rate"], 3) for m in ("det", "judge", "strict") if m in s})
    elif a.cmd in ("compare", "gate"):
        from .compare import compare, gate
        cmp = compare(_run_dir(a.baseline), _run_dir(a.candidate), a.mode)
        if a.cmd == "compare":
            print(json.dumps(cmp, indent=2))
        else:
            ok, reasons = gate(cmp, a.max_drop)
            for w in cmp["warnings"]:
                print("WARNING:", w)
            print(f"{cmp['baseline']} -> {cmp['candidate']}: {cmp['baseline_rate']:.1%} -> {cmp['candidate_rate']:.1%} "
                  f"(diff {cmp['diff']['diff']:+.1%}, 95% CI {cmp['diff']['lo']:+.1%}..{cmp['diff']['hi']:+.1%}; "
                  f"McNemar p={cmp['mcnemar']['p_value']:.3f})")
            print("GATE PASS" if ok else "GATE FAIL")
            for r_ in reasons:
                print("  -", r_)
            sys.exit(0 if ok else 1)
    elif a.cmd == "report":
        from .report import build
        print(build())
    elif a.cmd == "validate-judge":
        from .judge_validation import main as jv
        jv(a.source, a.judge_version)
    elif a.cmd == "export-otel":
        from .otel import export
        print(export(_run_dir(a.run)))


if __name__ == "__main__":
    main()
