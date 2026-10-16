from harness.graders.deterministic import grade, grounded, mentions
from harness.schema import Expect, OracleRef, Step, Task, ToolExpectation, Trajectory


def traj(answer, calls=(), tool_text="", oracles=None, q="q"):
    steps = [Step(kind="llm", tool_calls=list(calls))]
    if tool_text:
        steps.append(Step(kind="tool", text=tool_text, tool_name="x"))
    return Trajectory(run_id="r", task_id="t", question=q, steps=steps, answer=answer, oracle_values=oracles or {})


def test_mentions_numbers_are_exact_tokens():
    assert mentions("There are 5 chips.", 5)
    assert not mentions("There are 55 chips.", 5)
    assert not mentions("score 5.5", 5)


def test_mentions_camelcase_variants():
    assert mentions("mostly herbaceous vegetation", "HerbaceousVegetation")
    assert mentions("passD and passA", ["passA", "passD"])
    assert mentions("collected on 2026-10-01T14:05Z", "2026-10-01")
    assert not mentions("passD only", ["passA", "passD"])


def test_grounded_percent_conversion():
    assert grounded("confidence 99.78%", "q", '{"confidence": 0.9978}')[0]
    assert not grounded("confidence 97%", "q", '{"confidence": 0.9978}')[0]
    assert grounded("there are 5 chips", "q", "")[0]
    assert grounded("about 99.8% sure, threshold 1187", "q", '{"confidence": 0.9978, "thr": 1187.47}')[0]
    assert not grounded("threshold 1190", "q", '{"thr": 1187.47}')[0]  # single digits are not checked


def test_full_grade_pass_and_fail():
    t = Task(id="t", category="lookup", question="How many?",
             expect=Expect(tools=[ToolExpectation(name="triage_pass", args={"pass_id": "passC"})],
                           answer_contains=[OracleRef(oracle="n_analyst", args=["passC"])]))
    o = {"n_analyst(passC)": 9}
    good = traj("9 chips need an analyst.", [{"name": "triage_pass", "args": {"pass_id": "passC"}}], '{"n_analyst": 9}', o)
    assert grade(t, good).passed
    wrong_pass = traj("9 chips.", [{"name": "triage_pass", "args": {"pass_id": "passA"}}], '{"n_analyst": 9}', o)
    assert not grade(t, wrong_pass).passed
    no_tool = traj("9 chips.", [], "", o)
    assert not grade(t, no_tool).passed


def test_polarity_and_refusal():
    t = Task(id="t", category="lookup", question="drift?", expect=Expect(polarity=OracleRef(oracle="d")))
    assert grade(t, traj("Yes, drift was detected.", oracles={"d()": True})).passed
    assert not grade(t, traj("No drift detected.", oracles={"d()": True})).passed
    assert grade(t, traj("No, there is no drift.", oracles={"d()": False})).passed
    u = Task(id="u", category="unanswerable", question="GPS?", expect=Expect(refusal=True))
    assert grade(u, traj("That information is not available from the tools.")).passed
    assert not grade(u, traj("38.9N, 77.0W")).passed
    assert grade(u, traj("None of the provided functions can determine the GPS coordinates.")).passed
    assert not grade(u, traj("No vehicles are visible in chip passC-010.")).passed


def test_stance_patterns():
    from harness.graders.deterministic import stance
    assert stance("passD is drifting; the reference data is not representative.", "drift") is True
    assert stance("drift_alarm=true. No human review needed.", "drift") is True
    assert stance("There is no sign of drift in passA.", "drift") is False
    assert stance("passA does not show any signs of drift; the drift alarm is false.", "drift") is False
    assert stance("It is not novel; it resembles Forest.", "novel") is False
    assert stance("Yes, it was flagged novel and resembles PermanentCrop.", "novel") is True
