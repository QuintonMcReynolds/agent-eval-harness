"""Committed runs must re-grade to the same deterministic verdicts: catches silent grader drift in CI."""
import pytest

from harness.graders.deterministic import grade
from harness.grading import load_grades
from harness.runner import RUNS, load_run
from harness.suite import load_suite

RUN_DIRS = sorted(p for p in RUNS.glob("*") if (p / "grades.jsonl").exists())


@pytest.mark.parametrize("run_dir", RUN_DIRS, ids=lambda p: p.name)
def test_regrade_matches_committed(run_dir):
    meta, trajs = load_run(run_dir)
    tasks = {t.id: t for t in load_suite(meta.suite).tasks}
    committed = {(g.task_id, g.sample): g.passed for g in load_grades(run_dir)}
    fresh = {(t.task_id, t.sample): grade(tasks[t.task_id], t).passed for t in trajs}
    assert fresh == committed
