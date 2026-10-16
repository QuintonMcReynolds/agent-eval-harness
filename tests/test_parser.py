from harness.agents.mlx_agent import parse_tool_calls

N = {"triage_pass", "drift_check"}


def test_qwen_tagged_and_doubled_braces():
    for t in ('<tool_call>\n{"name": "triage_pass", "arguments": {"pass_id": "passC"}}\n</tool_call>',
              '<tool_call>\n{{"name": "triage_pass", "arguments": {"pass_id": "passC"}}}\n</tool_call>'):
        calls, bad, _ = parse_tool_calls(t, N)
        assert calls == [{"name": "triage_pass", "args": {"pass_id": "passC"}}] and bad == 0


def test_multiple_calls():
    t = ('<tool_call>{"name": "triage_pass", "arguments": {"pass_id": "passA"}}</tool_call>\n'
         '<tool_call>{"name": "drift_check", "arguments": {"pass_id": "passB"}}</tool_call>')
    calls, _, _ = parse_tool_calls(t, N)
    assert [c["name"] for c in calls] == ["triage_pass", "drift_check"]


def test_llama_bare_json_and_unknown_tool():
    calls, bad, _ = parse_tool_calls('{"name": "drift_check", "parameters": {"pass_id": "passD"}}', N)
    assert calls == [{"name": "drift_check", "args": {"pass_id": "passD"}}]
    calls, bad, _ = parse_tool_calls('{"name": "launch_missiles", "parameters": {}}', N)
    assert calls == [] and bad == 1


def test_plain_answer_is_not_a_call():
    calls, bad, residual = parse_tool_calls("passC has 9 chips for analysts.", N)
    assert calls == [] and bad == 0 and residual.startswith("passC")


def test_split_parallel_calls_interleaves():
    from harness.agents.mlx_agent import split_parallel_calls
    m = [{"role": "user", "content": "q"},
         {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "a"}}, {"function": {"name": "b"}}]},
         {"role": "tool", "content": "ra", "name": "a"}, {"role": "tool", "content": "rb", "name": "b"}]
    out = split_parallel_calls(m)
    assert [x["role"] for x in out] == ["user", "assistant", "tool", "assistant", "tool"]
    assert out[1]["tool_calls"][0]["function"]["name"] == "a" and out[4]["content"] == "rb"
