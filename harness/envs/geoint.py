"""Environment adapter for the geoint-triage agent (github.com/QuintonMcReynolds/geoint-triage).

Provides (1) how to start its MCP server, (2) oracles: ground truth computed straight from the engine,
bypassing the agent, and (3) a fingerprint of the model/feed so a changed environment is never mistaken for a
changed agent.
"""
from __future__ import annotations

import hashlib
import os
import sys
from functools import cached_property
from pathlib import Path

from mcp import StdioServerParameters

GEOINT_ROOT = Path(os.environ.get("GEOINT_ROOT", "/Volumes/CuetipDev/geoint-triage"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.exists() else "missing"


class GeointEnv:
    name = "geoint"

    def __init__(self, root: Path = GEOINT_ROOT):
        self.root = root
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    @cached_property
    def engine(self):
        from geoint.engine import TriageEngine
        return TriageEngine(art=self.root / "artifacts", feed=self.root / "samples" / "feed")

    def server_params(self) -> StdioServerParameters:
        return StdioServerParameters(command=sys.executable, args=["-m", "geoint.mcp_server"], cwd=str(self.root),
                                     env={**os.environ, "PYTHONPATH": str(self.root)})

    def fingerprint(self) -> dict[str, str]:
        feed = self.root / "samples" / "feed"
        h = hashlib.sha256()
        for f in sorted(feed.rglob("*.png")):
            h.update(f.name.encode())
            h.update(f.read_bytes())
        return {"model": _sha(self.root / "artifacts" / "chipnet.onnx"), "policy": _sha(self.root / "artifacts" / "policy.json"),
                "feed": h.hexdigest()[:12]}

    # ---------------- oracles ----------------
    @cached_property
    def _tri(self):
        return {p["pass_id"]: self.engine.triage_pass(p["pass_id"]) for p in self.engine.passes()}

    @cached_property
    def _drift(self):
        return {p: self.engine.drift(self.engine.load_pass(p)[1]) for p in self._tri}

    def oracle(self, name: str, *args):
        return getattr(self, f"o_{name}")(*args)

    def o_passes(self):
        return sorted(self._tri)

    def o_collected(self, p):
        return next(x["collected"] for x in self.engine.passes() if x["pass_id"] == p)

    def o_n_chips(self, p):
        return self._tri[p]["n_chips"]

    def o_n_analyst(self, p):
        return self._tri[p]["n_analyst"]

    def o_n_auto(self, p):
        return self._tri[p]["n_auto"]

    def o_n_novel(self, p):
        return self._tri[p]["n_novel"]

    def o_top_class(self, p):
        return next(iter(self._tri[p]["auto_label_counts"]))

    def o_total_analyst(self):
        return sum(t["n_analyst"] for t in self._tri.values())

    def o_most_novel_pass(self):
        return max(self._tri, key=lambda p: self._tri[p]["n_novel"])

    def o_most_analyst_pass(self):
        return max(self._tri, key=lambda p: self._tri[p]["n_analyst"])

    def o_drift_alarm(self, p):
        return self._drift[p]["drift_alarm"]

    def o_drifting_passes(self):
        return sorted(p for p, d in self._drift.items() if d["drift_alarm"])

    def o_chip_label(self, cid):
        return self.engine.classify(self.engine.load_chip(cid)[None], [cid])[0]["label"]

    def o_chip_route(self, cid):
        return self.engine.classify(self.engine.load_chip(cid)[None], [cid])[0]["route"]

    def o_chip_is_novel(self, cid):
        return self.engine.classify(self.engine.load_chip(cid)[None], [cid])[0]["reason"] == "novel"

    def o_neighbor_vote(self, cid):
        return self.engine.similar(self.engine.load_chip(cid), k=5)["neighbor_vote"]

    def o_novel_chips(self, p):
        return [q["chip_id"] for q in self._tri[p]["analyst_queue"] if q["reason"] == "novel"]

    def o_auto_count(self, p, cls):
        return self._tri[p]["auto_label_counts"].get(cls, 0)
