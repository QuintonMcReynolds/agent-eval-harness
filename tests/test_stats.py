import math

from harness import stats


def test_pass_hat_k_and_at_k():
    assert stats.pass_hat_k(5, 5, 3) == 1.0
    assert stats.pass_hat_k(5, 2, 3) == 0.0
    assert math.isclose(stats.pass_hat_k(4, 2, 2), 1 / 6)
    assert math.isclose(stats.pass_at_k(4, 2, 2), 5 / 6)


def test_mcnemar_counts_and_symmetry():
    base = {"a": True, "b": True, "c": False, "d": True}
    cand = {"a": True, "b": False, "c": True, "d": False}
    r = stats.mcnemar(base, cand)
    assert (r["regressions"], r["improvements"]) == (2, 1)
    assert stats.mcnemar(base, base)["p_value"] == 1.0


def test_bootstrap_ci_contains_diff():
    base = {str(i): 1.0 for i in range(40)}
    cand = {str(i): (0.0 if i < 10 else 1.0) for i in range(40)}
    r = stats.paired_bootstrap(base, cand)
    assert math.isclose(r["diff"], -0.25) and r["lo"] <= -0.25 <= r["hi"] and r["hi"] < 0


def test_kappa():
    assert stats.cohen_kappa([True, False] * 5, [True, False] * 5) == 1.0
    assert abs(stats.cohen_kappa([True, True, False, False], [True, False, True, False])) < 1e-9
    lo, hi = stats.wilson(9, 10)
    assert lo < 0.9 < hi
