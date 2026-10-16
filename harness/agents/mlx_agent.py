"""A generic MCP tool-calling agent: LangGraph loop + an MLX-served open-weights model.

Handles the two tool-call dialects we evaluate:
  Qwen2.5  -> <tool_call>{"name": ..., "arguments": {...}}</tool_call>
  Llama3.2 -> {"name": ..., "parameters": {...}}            (bare JSON, possibly several separated by ';')
"""
from __future__ import annotations

import json
import re
import time
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from mcp import ClientSession

from ..schema import AgentConfig, Span, Step

TAGGED = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _json_objects(text: str):
    """Yield top-level JSON objects found anywhere in text (brace matching, string aware)."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            esc = (ch == "\\") and not esc
            if ch == '"' and not esc:
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def _loads_lenient(blob: str):
    """json.loads, but also accept the doubled braces Qwen2.5's own chat template shows as the example format."""
    for cand in (blob, blob[1:-1] if blob.startswith("{{") and blob.endswith("}}") else None):
        if cand is None:
            continue
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    return None


def parse_tool_calls(text: str, tool_names: set[str]) -> tuple[list[dict], int, str]:
    """Returns (calls, n_malformed, residual_text)."""
    calls, bad = [], 0
    blobs = TAGGED.findall(text)
    residual = TAGGED.sub("", text).strip()
    if not blobs:
        stripped = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        # bare-JSON dialect: only treat as tool calls if the message is (mostly) JSON naming a known tool
        if stripped.startswith("{") or stripped.startswith("<|python_tag|>"):
            blobs = list(_json_objects(stripped))
            residual = ""
    for b in blobs:
        j = _loads_lenient(b)
        if not isinstance(j, dict):
            bad += 1
            continue
        name = j.get("name")
        args = j.get("arguments", j.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                bad += 1
                continue
        if name not in tool_names or not isinstance(args, dict):
            bad += 1
            continue
        calls.append({"name": name, "args": args})
    if not calls and blobs and not residual:
        residual = text.strip()
    return calls, bad, residual


def split_parallel_calls(messages: list[dict]) -> list[dict]:
    """[assistant(c1,c2), tool(r1), tool(r2)] -> [assistant(c1), tool(r1), assistant(c2), tool(r2)]."""
    out, i = [], 0
    while i < len(messages):
        m = messages[i]
        calls = m.get("tool_calls") or []
        if m["role"] == "assistant" and len(calls) > 1:
            results = []
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "tool":
                results.append(messages[j])
                j += 1
            for k, c in enumerate(calls):
                out.append({**m, "tool_calls": [c]})
                if k < len(results):
                    out.append(results[k])
            out.extend(results[len(calls):])
            i = j
        else:
            out.append(m)
            i += 1
    return out


class MLXModel:
    _cache: dict[str, tuple] = {}

    def __init__(self, model_id: str):
        from mlx_lm import load

        if model_id not in self._cache:
            self._cache[model_id] = load(model_id)
        self.model, self.tok = self._cache[model_id]
        self.model_id = model_id

    def generate(self, messages, tools, temperature: float, max_new_tokens: int, seed: int = 0):
        import mlx.core as mx
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        mx.random.seed(seed)
        try:
            prompt = self.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
        except Exception as e:  # noqa: BLE001
            if "single tool" not in str(e):
                raise
            # Llama 3.x templates allow one call per assistant turn: replay parallel calls as sequential turns
            prompt = self.tok.apply_chat_template(split_parallel_calls(messages), tools=tools, add_generation_prompt=True,
                                                  tokenize=False)
        n_prompt = len(self.tok.encode(prompt))
        out = generate(self.model, self.tok, prompt=prompt, max_tokens=max_new_tokens,
                       sampler=make_sampler(temp=temperature), verbose=False)
        return out.strip(), n_prompt, len(self.tok.encode(out))


class State(TypedDict):
    messages: Annotated[list, add_messages]
    steps: list
    spans: list
    n_llm: int
    n_repair: int


REPAIR_MSG = ("Your last message tried to call a tool but it was not valid JSON, so nothing was executed. "
              "Re-send the call exactly as <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call> with valid JSON, "
              "or answer directly if no tool is needed.")
MAX_REPAIRS = 2


def _chat(m) -> dict:
    role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}[m.type]
    d = {"role": role, "content": m.content}
    if role == "assistant" and getattr(m, "tool_calls", None):
        d["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c["args"]}} for c in m.tool_calls]
    if role == "tool":
        d["name"] = m.name
    return d


def build_agent(cfg: AgentConfig, system_prompt: str, session: ClientSession, tool_schemas: list[dict], seed: int = 0):
    from langchain_core.messages import AIMessage, ToolMessage

    llm = MLXModel(cfg.model)
    names = {t["function"]["name"] for t in tool_schemas}
    root = uuid.uuid4().hex[:16]

    async def agent_node(state: State):  # async: MLX's GPU stream lives on the event-loop thread
        t0 = time.time()
        text, n_in, n_out = llm.generate([_chat(m) for m in state["messages"]], tool_schemas, cfg.temperature,
                                         cfg.max_new_tokens, seed=seed + state["n_llm"])
        t1 = time.time()
        calls, bad, residual = parse_tool_calls(text, names)
        calls = [{**c, "id": f"call_{state['n_llm']}_{i}"} for i, c in enumerate(calls)]
        step = Step(kind="llm", text=text, tool_calls=[{"name": c["name"], "args": c["args"]} for c in calls],
                    malformed_calls=bad, prompt_tokens=n_in, completion_tokens=n_out, latency_s=t1 - t0)
        span = Span(span_id=uuid.uuid4().hex[:16], parent_id=root, name="llm.generate", start=t0, end=t1,
                    attributes={"gen_ai.request.model": cfg.model, "gen_ai.usage.input_tokens": n_in,
                                "gen_ai.usage.output_tokens": n_out, "tool_calls": len(calls), "malformed": bad})
        msg = AIMessage(content="" if calls else residual, tool_calls=calls)
        return {"messages": [msg], "steps": state["steps"] + [step], "spans": state["spans"] + [span],
                "n_llm": state["n_llm"] + 1}

    async def tool_node(state: State):
        msgs, steps, spans = [], list(state["steps"]), list(state["spans"])
        for c in state["messages"][-1].tool_calls:
            t0 = time.time()
            err = False
            try:
                r = await session.call_tool(c["name"], c["args"])
                txt = "\n".join(getattr(b, "text", "") for b in r.content)
                err = bool(r.is_error)
            except Exception as e:  # noqa: BLE001 - surfaced to the model like a real tool error
                txt, err = f"ERROR: {e}", True
            t1 = time.time()
            msgs.append(ToolMessage(content=txt, tool_call_id=c["id"], name=c["name"]))
            steps.append(Step(kind="tool", text=txt, tool_name=c["name"], tool_args=c["args"], error=err, latency_s=t1 - t0))
            spans.append(Span(span_id=uuid.uuid4().hex[:16], parent_id=root, name=f"tool.{c['name']}", start=t0, end=t1,
                              attributes={"tool.args": json.dumps(c["args"]), "error": err}))
        return {"messages": msgs, "steps": steps, "spans": spans}

    def repair_node(state: State):
        from langchain_core.messages import HumanMessage
        return {"messages": [HumanMessage(REPAIR_MSG)], "n_repair": state.get("n_repair", 0) + 1}

    def route(state: State):
        if getattr(state["messages"][-1], "tool_calls", None) and state["n_llm"] < cfg.max_steps:
            return "tools"
        last = state["steps"][-1] if state["steps"] else None
        if (cfg.repair_malformed and last is not None and last.kind == "llm" and last.malformed_calls and not last.tool_calls
                and state.get("n_repair", 0) < MAX_REPAIRS and state["n_llm"] < cfg.max_steps):
            return "repair"
        return END

    g = StateGraph(State)
    g.add_node("agent", agent_node)
    g.add_node("tools", tool_node)
    g.add_node("repair", repair_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "repair": "repair", END: END})
    g.add_edge("repair", "agent")
    g.add_edge("tools", "agent")
    return g.compile(), root
