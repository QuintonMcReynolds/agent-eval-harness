"""Export recorded spans as OTLP/JSON (load into Jaeger, Grafana Tempo, Arize Phoenix, etc.)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .runner import load_run


def _attr(k, v):
    if isinstance(v, bool):
        return {"key": k, "value": {"boolValue": v}}
    if isinstance(v, int):
        return {"key": k, "value": {"intValue": str(v)}}
    if isinstance(v, float):
        return {"key": k, "value": {"doubleValue": v}}
    return {"key": k, "value": {"stringValue": str(v)}}


def export(run_dir: Path) -> Path:
    meta, trajs = load_run(run_dir)
    spans = []
    for tr in trajs:
        trace_id = hashlib.sha256(f"{tr.run_id}/{tr.task_id}/{tr.sample}".encode()).hexdigest()[:32]
        if not tr.spans:
            continue
        root_id = tr.spans[0].parent_id
        t0, t1 = min(s.start for s in tr.spans), max(s.end for s in tr.spans)
        spans.append({"traceId": trace_id, "spanId": root_id, "name": "agent.run",
                      "startTimeUnixNano": str(int(t0 * 1e9)), "endTimeUnixNano": str(int(t1 * 1e9)),
                      "attributes": [_attr("task.id", tr.task_id), _attr("run.id", tr.run_id), _attr("sample", tr.sample),
                                     _attr("stop_reason", tr.stop_reason)]})
        for s in tr.spans:
            spans.append({"traceId": trace_id, "spanId": s.span_id, "parentSpanId": s.parent_id, "name": s.name,
                          "startTimeUnixNano": str(int(s.start * 1e9)), "endTimeUnixNano": str(int(s.end * 1e9)),
                          "attributes": [_attr(k, v) for k, v in s.attributes.items()]})
    doc = {"resourceSpans": [{"resource": {"attributes": [_attr("service.name", "agent-eval-harness"),
                                                          _attr("gen_ai.request.model", meta.agent.model),
                                                          _attr("prompt.version", meta.agent.prompt_version)]},
                              "scopeSpans": [{"scope": {"name": "harness"}, "spans": spans}]}]}
    out = run_dir / "otel.json"
    out.write_text(json.dumps(doc))
    return out
