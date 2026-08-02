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
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait as _wait
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import pydantic

_seq = itertools.count()


def step(
    fn=None,
    *,
    name: str | None = None,
    order: int | None = None,
    description: str | None = None,
    fan_out: bool = False,
):
    """Mark a function as a pipeline step. Stacks on plain functions, @llm, or @agent
    (put @step outermost). Steps run in definition order unless order= is given.
    description= (or the docstring) is recorded in the run state file.
    fan_out=True runs the step once per item of its input list, in parallel,
    and its output becomes the list of results."""

    def deco(f):
        f.__agentsoup_step__ = {
            "name": name or f.__name__,
            "order": order,
            "description": description or inspect.getdoc(f),
            "fan_out": fan_out,
            "seq": next(_seq),
        }
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


def _params_of(fn) -> dict:
    return {
        n: p
        for n, p in inspect.signature(fn).parameters.items()
        if p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


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
        max_workers: int | None = None,
    ) -> PipelineResult:
        """Execute the pipeline as a dependency graph.

        A step whose parameters all name earlier steps depends only on those
        steps and runs as soon as they finish — independent steps run in
        parallel. A step with an unbound required parameter (or no parameters)
        depends on the step defined just before it, preserving plain linear
        chaining. Progress is written atomically to a JSON state file and
        optionally POSTed to a webhook.
        """
        if not self.steps:
            raise ValueError(f"No @step functions found in {self.source}")
        names = [meta["name"] for meta, _ in self.steps]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate step names: {sorted(n for n in names if names.count(n) > 1)}")
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
                    "name": meta["name"], "description": meta.get("description"),
                    "index": i, "status": "pending", "started_at": None,
                    "finished_at": None, "duration_s": None, "output_json": None,
                    "output_repr": None, "error": None,
                }
                for i, (meta, _) in enumerate(self.steps)
            ],
        }
        lock = threading.Lock()

        def save():
            with lock:
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

        # Dependency graph: named params -> those steps; unbound/no params -> previous step.
        deps = []
        for i, (meta, fn) in enumerate(self.steps):
            params = _params_of(fn)
            named = {n for n in params if n in names[:i]}
            has_unbound = any(
                n not in named and p.default is inspect.Parameter.empty for n, p in params.items()
            )
            d = set(named)
            if i > 0 and (has_unbound or not named):
                d.add(names[i - 1])
            deps.append(d)

        outputs = {}

        def execute(i: int):
            meta, fn = self.steps[i]
            step_state = state["steps"][i]
            step_state.update(status="running", started_at=_now())
            started = time.monotonic()
            save()
            emit("step_started", step_state)
            try:
                params = _params_of(fn)
                kwargs, unbound = {}, []
                for n, p in params.items():
                    if n in names[:i]:
                        kwargs[n] = outputs[n]
                    elif p.default is inspect.Parameter.empty:
                        unbound.append(n)
                if len(unbound) > 1:
                    raise ValueError(
                        f"Step '{meta['name']}' has multiple unbound parameters {unbound}; "
                        "name them after earlier steps or give them defaults"
                    )
                if unbound:
                    kwargs[unbound[0]] = input if i == 0 else outputs[names[i - 1]]
                if meta.get("fan_out"):
                    if not kwargs:
                        raise ValueError(f"fan_out step '{meta['name']}' needs an input parameter")
                    fan_param = unbound[0] if unbound else next(iter(params))
                    items = list(kwargs[fan_param])
                    with ThreadPoolExecutor(max_workers=max_workers) as fan_pool:
                        value = list(fan_pool.map(lambda item: fn(**{**kwargs, fan_param: item}), items))
                else:
                    value = fn(**kwargs)
            except Exception as e:
                step_state.update(
                    status="failed", finished_at=_now(),
                    duration_s=round(time.monotonic() - started, 3), error=f"{type(e).__name__}: {e}",
                )
                state["status"] = "failed"
                save()
                emit("step_failed", step_state)
                raise
            step_state.update(
                status="done", finished_at=_now(), duration_s=round(time.monotonic() - started, 3)
            )
            _store_output(step_state, value)
            save()
            emit("step_finished", step_state)
            return value

        save()
        emit("run_started")
        failure = None
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            pending = set(range(len(self.steps)))
            running = {}
            while pending or running:
                if failure is None:
                    ready = [i for i in sorted(pending) if deps[i] <= set(outputs)]
                    for i in ready:
                        pending.discard(i)
                        running[pool.submit(execute, i)] = i
                if not running:
                    break  # a failure stopped scheduling; nothing left in flight
                completed, _ = _wait(running, return_when=FIRST_COMPLETED)
                for future in completed:
                    i = running.pop(future)
                    try:
                        outputs[names[i]] = future.result()
                    except Exception as e:
                        failure = failure or e
        if failure is not None:
            emit("run_failed")
            raise failure
        state["status"] = "done"
        save()
        emit("run_finished")
        return PipelineResult(run_id, "done", outputs[names[-1]], str(state_path))


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
