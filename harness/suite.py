from __future__ import annotations

from pathlib import Path

import yaml

from .schema import OracleRef, Suite, Task

ROOT = Path(__file__).resolve().parents[1]


def load_suite(path: str | Path) -> Suite:
    p = Path(path)
    if not p.exists():
        p = ROOT / "suites" / f"{path}.yaml"
    return Suite(**yaml.safe_load(p.read_text()))


def oracle_key(ref: OracleRef) -> str:
    return f"{ref.oracle}({','.join(map(str, ref.args))})"


def oracle_refs(task: Task) -> list[OracleRef]:
    refs = [r for r in task.expect.answer_contains + task.expect.answer_forbids if isinstance(r, OracleRef)]
    if task.expect.polarity:
        refs.append(task.expect.polarity)
    return refs


def snapshot_oracles(task: Task, env) -> dict:
    return {oracle_key(r): env.oracle(r.oracle, *r.args) for r in oracle_refs(task)}
