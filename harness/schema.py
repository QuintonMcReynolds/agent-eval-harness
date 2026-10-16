"""Data model. Everything on disk is JSON/JSONL of these, so runs can be re-graded long after they were recorded."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


def stable_hash(obj: Any, n: int = 12) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


class OracleRef(BaseModel):
    """A value the grader computes from the environment, e.g. {oracle: n_analyst, args: [passC]}."""
    oracle: str
    args: list[Any] = []


class ToolExpectation(BaseModel):
    name: str
    args: dict[str, Any] = {}


class Expect(BaseModel):
    tools: list[ToolExpectation] = []                 # each must be called (args must match)
    tools_any: list[ToolExpectation] = []             # at least one must be called
    answer_contains: list[OracleRef | str] = []        # each value must appear in the answer
    answer_contains_any: list[str] = []                # at least one of these words
    answer_forbids: list[OracleRef | str] = []         # none of these may appear
    polarity: OracleRef | None = None                  # bool oracle: answer must affirm (True) or deny (False)
    polarity_kind: Literal["generic", "drift", "novel"] = "generic"
    refusal: bool = False                              # task is unanswerable from the tools
    max_tool_calls: int = 8


class Task(BaseModel):
    id: str
    category: Literal["lookup", "multi_step", "unanswerable", "judgment", "injection"]
    question: str
    expect: Expect = Expect()
    rubric: str = ""                                   # extra guidance for the LLM judge


class Suite(BaseModel):
    name: str
    version: str
    env: str
    tasks: list[Task]

    @property
    def hash(self) -> str:
        return stable_hash(self.model_dump())


class Span(BaseModel):
    """One step of a trajectory. Field names follow OpenTelemetry so traces can be exported as OTLP."""
    span_id: str
    parent_id: str | None = None
    name: str                                          # llm.generate | tool.<name>
    start: float
    end: float
    attributes: dict[str, Any] = {}


class Step(BaseModel):
    kind: Literal["llm", "tool"]
    text: str | None = None                            # raw model output (llm) or tool result (tool)
    tool_calls: list[dict] = []                        # parsed calls (llm)
    tool_name: str | None = None
    tool_args: dict | None = None
    error: bool = False
    malformed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0


class Trajectory(BaseModel):
    run_id: str
    task_id: str
    sample: int = 0
    question: str
    steps: list[Step] = []
    spans: list[Span] = []
    answer: str = ""
    stop_reason: Literal["answer", "max_steps", "error"] = "answer"
    oracle_values: dict[str, Any] = {}                 # snapshot of every oracle the task references
    wall_s: float = 0.0

    @property
    def tool_calls(self) -> list[dict]:
        return [c for s in self.steps if s.kind == "llm" for c in s.tool_calls]

    @property
    def tool_text(self) -> str:
        return "\n".join(s.text or "" for s in self.steps if s.kind == "tool")


class AgentConfig(BaseModel):
    model: str
    prompt_version: str
    temperature: float = 0.0
    max_steps: int = 8
    max_new_tokens: int = 400
    repair_malformed: bool = False                     # feed parse errors back to the model instead of stopping


class RunMeta(BaseModel):
    run_id: str
    suite: str
    suite_version: str
    suite_hash: str
    agent: AgentConfig
    prompt_hash: str
    env_fingerprint: dict[str, str]
    samples: int
    started: str
    harness_version: str
    git_sha: str | None = None


class CheckResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    gating: bool = True                                # False = reported but does not affect pass/fail


class Grade(BaseModel):
    run_id: str
    task_id: str
    sample: int
    category: str
    passed: bool                                       # all deterministic checks
    checks: list[CheckResult] = Field(default_factory=list)
    judge: dict[str, Any] | None = None                # filled by the LLM judge pass
