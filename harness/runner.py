"""Run one agent configuration over a suite and record trajectories.

runs/<run_id>/meta.json            RunMeta
runs/<run_id>/trajectories.jsonl   one Trajectory per (task, sample)
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import subprocess
import time
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import stdio_client

from . import __version__
from .agents.mlx_agent import build_agent
from .envs import ENVS
from .schema import AgentConfig, RunMeta, Trajectory, stable_hash
from .suite import ROOT, load_suite, snapshot_oracles

RUNS = ROOT / "runs"


def prompt_text(version: str) -> str:
    return (ROOT / "prompts" / f"{version}.txt").read_text().strip()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def make_run_id(cfg: AgentConfig, suite_name: str, samples: int) -> str:
    short = cfg.model.split("/")[-1].replace("-Instruct", "").replace("-4bit", "")
    return f"{suite_name}__{short}__{cfg.prompt_version}{'+repair' if cfg.repair_malformed else ''}__T{cfg.temperature:g}x{samples}"


async def _run(cfg: AgentConfig, suite_name: str, samples: int, task_filter: set[str] | None, out_dir: Path | None):
    from langchain_core.messages import HumanMessage, SystemMessage

    suite = load_suite(suite_name)
    env = ENVS[suite.env]()
    system = prompt_text(cfg.prompt_version)
    run_id = make_run_id(cfg, suite.name, samples)
    out = out_dir or RUNS / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta = RunMeta(run_id=run_id, suite=suite.name, suite_version=suite.version, suite_hash=suite.hash, agent=cfg,
                   prompt_hash=stable_hash(system), env_fingerprint=env.fingerprint(), samples=samples,
                   started=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), harness_version=__version__,
                   git_sha=git_sha())
    (out / "meta.json").write_text(meta.model_dump_json(indent=2))
    tasks = [t for t in suite.tasks if not task_filter or t.id in task_filter]
    oracles = {t.id: snapshot_oracles(t, env) for t in tasks}

    errlog = open(out / "server.log", "w")
    async with stdio_client(env.server_params(), errlog=errlog) as (r, w), ClientSession(r, w) as session:
        await session.initialize()
        tools = (await session.list_tools()).tools
        schemas = [{"type": "function", "function": {"name": t.name, "description": t.description or "",
                                                     "parameters": t.input_schema}} for t in tools]
        with open(out / "trajectories.jsonl", "w") as f:
            for t in tasks:
                for s in range(samples):
                    graph, _ = build_agent(cfg, system, session, schemas, seed=1000 * s)
                    t0 = time.time()
                    stop = "answer"
                    try:
                        st = await graph.ainvoke({"messages": [SystemMessage(system), HumanMessage(t.question)],
                                                  "steps": [], "spans": [], "n_llm": 0, "n_repair": 0},
                                                 {"recursion_limit": 4 * cfg.max_steps})
                        last = st["messages"][-1]
                        if getattr(last, "tool_calls", None):
                            stop, answer = "max_steps", ""
                        else:
                            answer = last.content
                        steps, spans = st["steps"], st["spans"]
                    except Exception as e:  # noqa: BLE001
                        stop, answer, steps, spans = "error", f"[harness error] {e}", [], []
                    tr = Trajectory(run_id=run_id, task_id=t.id, sample=s, question=t.question, steps=steps, spans=spans,
                                    answer=answer, stop_reason=stop, oracle_values=oracles[t.id], wall_s=time.time() - t0)
                    f.write(tr.model_dump_json() + "\n")
                    f.flush()
                    print(f"[{run_id}] {t.id}#{s} {stop} {tr.wall_s:.1f}s :: {answer[:90]!r}", flush=True)
    return out


def run(cfg: AgentConfig, suite_name: str, samples: int = 1, tasks: set[str] | None = None, out_dir: Path | None = None) -> Path:
    return asyncio.run(_run(cfg, suite_name, samples, tasks, out_dir))


def load_run(run_dir: Path) -> tuple[RunMeta, list[Trajectory]]:
    meta = RunMeta(**json.loads((run_dir / "meta.json").read_text()))
    trajs = [Trajectory(**json.loads(line)) for line in (run_dir / "trajectories.jsonl").read_text().splitlines() if line.strip()]
    return meta, trajs
