"""The release decisions recorded in the README, re-derived from the committed trajectories on every CI run."""
import pytest

from harness.compare import compare, gate
from harness.runner import RUNS

DECISIONS = [
    # (baseline, candidate, ship?)
    ("geoint__Qwen2.5-3B__v1__T0x1", "geoint__Qwen2.5-3B__v2__T0x1", False),       # significant drop, malformed JSON
    ("geoint__Qwen2.5-7B__v1__T0x1", "geoint__Qwen2.5-7B__v2__T0x1", False),       # injection regressions
    ("geoint__Qwen2.5-7B__v1__T0x1", "geoint__Qwen2.5-7B__v3__T0x1", False),       # +5 pt overall, but loses an injection task
    ("geoint__Llama-3.2-3B__v1__T0x1", "geoint__Llama-3.2-3B__v3__T0x1", True),    # +21 pt, significant
    ("geoint__Qwen2.5-3B__v2__T0x1", "geoint__Qwen2.5-3B__v2+repair__T0x1", True),  # repair loop recovers malformed calls
]


@pytest.mark.parametrize("base,cand,ship", DECISIONS, ids=[d[1] for d in DECISIONS])
def test_gate_decision(base, cand, ship):
    if not ((RUNS / base / "grades.jsonl").exists() and (RUNS / cand / "grades.jsonl").exists()):
        pytest.skip("runs not present")
    ok, reasons = gate(compare(RUNS / base, RUNS / cand, mode="det"))
    assert ok == ship, reasons
