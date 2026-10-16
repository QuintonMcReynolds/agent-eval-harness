from harness.graders.judge import build_prompt, parse_verdict
from harness.schema import Expect, Step, Task, Trajectory


def test_parse_clean_and_fenced_json():
    assert parse_verdict('{"verdict": "pass", "faithfulness": 5, "error_type": "none", "reason": "ok"}')["verdict"] == "pass"
    v = parse_verdict('```json\n{"verdict": "fail", "faithfulness": 2, "error_type": "wrong_value", "reason": "x"}\n```')
    assert v["verdict"] == "fail" and v["error_type"] == "wrong_value"


def test_parse_garbage_and_unknown_error_type():
    assert parse_verdict("I think it is fine")["verdict"] == "unparsed"
    assert parse_verdict('{"verdict": "fail", "error_type": "bogus"}')["error_type"] == "other"


def test_prompt_contains_reference_and_order_swap():
    t = Task(id="t", category="unanswerable", question="GPS?", expect=Expect(refusal=True))
    tr = Trajectory(run_id="r", task_id="t", question="GPS?", steps=[Step(kind="tool", text="{}", tool_name="x")], answer="n/a")
    a = build_prompt(t, tr)[1]["content"]
    b = build_prompt(t, tr, order="answer_first")[1]["content"]
    assert "cannot provide this information" in a
    assert a.index("FINAL ANSWER") > a.index("TOOL OUTPUT") and b.index("FINAL ANSWER") < b.index("TOOL OUTPUT")
