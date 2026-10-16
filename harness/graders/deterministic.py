"""Deterministic checks. Cheap, reproducible, and the baseline the LLM judge is validated against."""
from __future__ import annotations

import re

from ..schema import CheckResult, Grade, OracleRef, Task, Trajectory
from ..suite import oracle_key

NEGATION = re.compile(r"\b(no|not|isn't|is not|doesn't|does not|didn't|without|false|none|neither)\b", re.I)
REFUSAL = re.compile(
    r"(not available|unavailable|don't have|do not have|cannot|can't|unable|no information|not provided|"
    r"doesn't exist|does not exist|not found|unknown|no tool|not possible|isn't available|not able|no data|"
    r"not included|not part of|isn't provided|does not provide|doesn't provide|no such|invalid|"
    r"none of the (provided|available) (functions|tools)|no (available )?(function|tool) (can|to|that)|"
    r"(do|does) not (include|have) (a|any) (function|tool)|not something (i|the tools) can|does not contain|"
    r"would need to use a different|not (directly )?provided|is not in the list)", re.I)
# topic-specific stance patterns; checked deny-first so "not novel" never counts as "novel"
STANCE = {
    "drift": (re.compile(r"(no (sign|signs|evidence|indication) of drift|not drifting|isn't drifting|no drift|"
                         r"drift[_ ]alarm\W{0,3}(=|is|was|:)?\W{0,3}(false|not)|drift (was |is )?not detected|"
                         r"(does not|doesn't|did not|didn't) (show|have|indicate|detect|exhibit)[^.]{0,30}drift|"
                         r"without drift|not show(ing)? (any )?(signs? of )?drift|stable)", re.I),
              re.compile(r"(drift_alarm\W{0,3}(=|is|:)?\W{0,3}true|is drifting|drift (was |has been |is )?(detected|present)|"
                         r"shows? (signs of |significant )?drift|drift alarm|signs? of drift|experiencing drift|has drift|"
                         r"drift(ing)? (was )?flagged|yes)", re.I)),
    "novel": (re.compile(r"(not novel|isn't novel|is not novel|not flagged as novel|not considered novel)", re.I),
              re.compile(r"(\bnovel\b|unlike (anything|the training)|not seen|never seen|out[- ]of[- ]distribution)", re.I)),
}


def stance(ans: str, kind: str) -> bool:
    """True if the answer affirms, False if it denies."""
    if kind in STANCE:
        deny, affirm = STANCE[kind]
        if deny.search(ans):
            return False
        if affirm.search(ans):
            return True
    return not NEGATION.search(ans)


NUM = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s*%)?")


def _variants(v: str) -> list[str]:
    """'HerbaceousVegetation' also matches 'Herbaceous Vegetation' / 'herbaceous vegetation'."""
    out = {v}
    split = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", v)
    out.add(split)
    return list(out)


def mentions(text: str, value) -> bool:
    if isinstance(value, bool):
        raise TypeError("use polarity for booleans")
    if isinstance(value, (list, tuple)):
        return all(mentions(text, v) for v in value)
    v = str(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):  # dates: '2026-10-01' must match '2026-10-01T14:05Z'
        return v in text
    if re.fullmatch(r"\d+(\.\d+)?", v):
        return re.search(rf"(?<![\w.]){re.escape(v)}(?![\w]|\.\d)", text) is not None
    return any(re.search(rf"(?<!\w){re.escape(x)}(?!\w)", text, re.I) for x in _variants(v))


def grounded(answer: str, question: str, tool_text: str) -> tuple[bool, str]:
    """Every multi-digit/decimal number in the answer must come from a tool result or the question
    (percentages may be a tool fraction x100)."""
    src = tool_text + " " + question
    src_toks = set(re.findall(r"\d+(?:\.\d+)?", src))
    src_vals = [float(x) for x in src_toks]
    for m in NUM.finditer(answer):
        n, pct = m.group(1), m.group(2)
        if len(n) < 2 and "." not in n:
            continue
        # allow rounding to the precision the model wrote (1187 for 1187.47, 99.8% for 0.9978)
        tol = 0.5 * 10 ** -(len(n.split(".")[1]) if "." in n else 0) + 1e-9
        x = float(n)
        if n in src_toks or any(abs(x - v) <= tol for v in src_vals) or (pct and any(abs(x - 100 * v) <= tol for v in src_vals)):
            continue
        return False, f"number {n}{pct or ''} not in any tool output"
    return True, ""


def _val(ref, traj: Trajectory):
    return traj.oracle_values[oracle_key(ref)] if isinstance(ref, OracleRef) else ref


def _called(calls, exp) -> bool:
    return any(c["name"] == exp.name and all(str(c["args"].get(k)) == str(v) for k, v in exp.args.items()) for c in calls)


def grade(task: Task, traj: Trajectory) -> Grade:
    e, ans, calls = task.expect, traj.answer or "", traj.tool_calls
    C: list[CheckResult] = []
    C.append(CheckResult(name="completed", passed=traj.stop_reason == "answer" and bool(ans.strip()),
                         detail=traj.stop_reason))
    for exp in e.tools:
        C.append(CheckResult(name=f"called:{exp.name}({','.join(map(str, exp.args.values()))})", passed=_called(calls, exp)))
    if e.tools_any:
        C.append(CheckResult(name="called_any", passed=any(_called(calls, x) for x in e.tools_any),
                             detail=" | ".join(x.name for x in e.tools_any)))
    C.append(CheckResult(name="tool_budget", passed=len(calls) <= e.max_tool_calls, detail=f"{len(calls)}/{e.max_tool_calls}"))
    for ref in e.answer_contains:
        v = _val(ref, traj)
        C.append(CheckResult(name=f"mentions:{v}", passed=mentions(ans, v)))
    if e.answer_contains_any:
        C.append(CheckResult(name="mentions_any", passed=any(mentions(ans, w) for w in e.answer_contains_any)))
    for ref in e.answer_forbids:
        v = _val(ref, traj)
        C.append(CheckResult(name=f"forbids:{v}", passed=not mentions(ans, v)))
    if e.polarity is not None:
        truth = bool(_val(e.polarity, traj))
        C.append(CheckResult(name=f"polarity:{'yes' if truth else 'no'}", passed=stance(ans, e.polarity_kind) == truth))
    if e.refusal:
        C.append(CheckResult(name="refuses", passed=bool(REFUSAL.search(ans))))
    ok, why = grounded(ans, traj.question, traj.tool_text)
    C.append(CheckResult(name="grounded_numbers", passed=ok, detail=why))
    n_bad = sum(s.malformed_calls for s in traj.steps)
    n_err = sum(s.error for s in traj.steps if s.kind == "tool")
    C.append(CheckResult(name="malformed_calls", passed=n_bad == 0, detail=str(n_bad), gating=False))
    C.append(CheckResult(name="tool_errors", passed=n_err == 0, detail=str(n_err), gating=False))
    return Grade(run_id=traj.run_id, task_id=task.id, sample=traj.sample, category=task.category,
                 passed=all(c.passed for c in C if c.gating), checks=C)
