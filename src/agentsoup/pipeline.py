"""Turn a Python file of @step functions into a pipeline with external run tracking."""
import importlib.util
import inspect
import itertools
import json
import os
import secrets
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pydantic

_seq = itertools.count()


def step(fn=None, *, name: str | None = None, order: int | None = None):
    """Mark a function as a pipeline step. Stacks on plain functions, @llm, or @agent
    (put @step outermost). Steps run in definition order unless order= is given."""

    def deco(f):
        f.__agentsoup_step__ = {"name": name or f.__name__, "order": order, "seq": next(_seq)}
        return f

    return deco(fn) if fn is not None else deco


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id(source) -> str:
    return f"{Path(source).stem}-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"


def _post_webhook(url: str, payload: dict, wait: bool = False):
    def _send():
        try:
            req = urllib.request.Request(
                url, data=json.dumps(payload, default=str).encode(), headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=3)
        except Exception:
            pass  # tracking must never fail the run

    if wait:  # terminal events: don't let process exit drop the POST
        _send()
    else:
        threading.Thread(target=_send, daemon=True).start()


def _store_output(step_state: dict, value):
    if isinstance(value, pydantic.BaseModel):
        step_state["output_json"] = value.model_dump(mode="json")
        return
    try:
        json.dumps(value)
        step_state["output_json"] = value
    except (TypeError, ValueError):
        step_state["output_repr"] = repr(value)[:10_000]


def _call_step(step_name: str, fn, prev_value, outputs: dict):
    """Bind a step's parameters: a param named after an earlier step gets that
    step's output; one remaining required param gets the previous step's output
    (the pipeline input, for the first step)."""
    kwargs = {}
    unbound = []
    for pname, p in inspect.signature(fn).parameters.items():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if pname in outputs:
            kwargs[pname] = outputs[pname]
        elif p.default is inspect.Parameter.empty:
            unbound.append(pname)
    if len(unbound) > 1:
        raise ValueError(
            f"Step '{step_name}' has multiple unbound parameters {unbound}; "
            "name them after earlier steps or give them defaults"
        )
    if unbound:
        kwargs[unbound[0]] = prev_value
    return fn(**kwargs)


class PipelineResult(NamedTuple):
    run_id: str
    status: str
    output: object
    state_path: str


class Pipeline:
    def __init__(self, steps, source: str = "<module>"):
        self.steps = steps  # list of (meta_dict, fn)
        self.source = source

    @classmethod
    def from_module(cls, module) -> "Pipeline":
        found = [
            (obj.__agentsoup_step__, obj)
            for obj in vars(module).values()
            if callable(obj) and hasattr(obj, "__agentsoup_step__")
        ]
        found.sort(key=lambda mf: (mf[0]["order"] if mf[0]["order"] is not None else float("inf"), mf[0]["seq"]))
        return cls(found, getattr(module, "__file__", None) or module.__name__)

    @classmethod
    def from_file(cls, path) -> "Pipeline":
        path = os.path.abspath(path)
        spec = importlib.util.spec_from_file_location(f"_agentsoup_pipeline_{Path(path).stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pipeline = cls.from_module(module)
        pipeline.source = path
        return pipeline

    def run(
        self,
        input=None,
        *,
        run_id: str | None = None,
        state_dir=".agentsoup/runs",
        webhook_url: str | None = None,
        on_event=None,
    ) -> PipelineResult:
        if not self.steps:
            raise ValueError(f"No @step functions found in {self.source}")
        run_id = run_id or new_run_id(self.source)
        state_path = Path(state_dir) / f"{run_id}.json"
        state = {
            "run_id": run_id,
            "pipeline_file": self.source,
            "status": "running",
            "created_at": _now(),
            "updated_at": _now(),
            "input_repr": None if input is None else repr(input)[:1_000],
            "steps": [
                {
                    "name": meta["name"], "index": i, "status": "pending", "started_at": None,
                    "finished_at": None, "duration_s": None, "output_json": None,
                    "output_repr": None, "error": None,
                }
                for i, (meta, _) in enumerate(self.steps)
            ],
        }

        def save():
            state["updated_at"] = _now()
            state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = Path(str(state_path) + ".tmp")
            tmp.write_text(json.dumps(state, indent=2, default=str))
            os.replace(tmp, state_path)  # atomic: readers never see a partial doc

        def emit(event: str, step_state: dict | None = None):
            if on_event:
                on_event(event, step_state)
            if webhook_url:
                _post_webhook(webhook_url, {
                    "event": event, "run_id": run_id, "pipeline": self.source,
                    "timestamp": _now(), "step": step_state, "run_status": state["status"],
                }, wait=event in ("run_finished", "run_failed"))

        save()
        emit("run_started")
        value = input
        outputs = {}  # step name -> output, for name-based parameter wiring
        for i, (meta, fn) in enumerate(self.steps):
            step_state = state["steps"][i]
            step_state.update(status="running", started_at=_now())
            started = time.monotonic()
            save()
            emit("step_started", step_state)
            try:
                value = _call_step(meta["name"], fn, value, outputs)
                outputs[meta["name"]] = value
            except Exception as e:
                step_state.update(
                    status="failed", finished_at=_now(),
                    duration_s=round(time.monotonic() - started, 3), error=f"{type(e).__name__}: {e}",
                )
                state["status"] = "failed"
                save()
                emit("step_failed", step_state)
                emit("run_failed")
                raise
            step_state.update(
                status="done", finished_at=_now(), duration_s=round(time.monotonic() - started, 3)
            )
            _store_output(step_state, value)
            save()
            emit("step_finished", step_state)
        state["status"] = "done"
        save()
        emit("run_finished")
        return PipelineResult(run_id, "done", value, str(state_path))


def load_run(path) -> dict:
    """Read a run's state file (from any process, at any time)."""
    return json.loads(Path(path).read_text())


def list_runs(state_dir=".agentsoup/runs") -> list[dict]:
    """All runs in a state dir, oldest first."""
    d = Path(state_dir)
    if not d.is_dir():
        return []
    return sorted(
        (json.loads(p.read_text()) for p in d.glob("*.json")),
        key=lambda s: s.get("created_at", ""),
    )
