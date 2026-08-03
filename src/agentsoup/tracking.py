"""Tracking: every @llm/@agent call inside a `with track():` block emits plain
JSON-serializable event dicts to sinks. A sink is any callable taking one dict —
`print` works, `events.append` works, and the library ships three:
state_file() (atomic live-readable JSON snapshot), webhook() (POST per event),
and otel() (one OpenTelemetry span per call).

The library owns the correctness: the active tracker is context-local (nested
blocks restore the outer; threads and .map workers stay isolated), sink calls
are serialized under one lock so naive sinks are safe under parallel fan-out,
and a sink that raises is disabled with a logged warning — tracking can never
fail the run."""
import contextlib
import contextvars
import json
import logging
import os
import secrets
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pydantic

_current: contextvars.ContextVar = contextvars.ContextVar("agentsoup_track", default=None)
_lock = threading.Lock()
_logger = logging.getLogger("agentsoup")


def current_tracker() -> "_Tracker | None":
    """The active tracker in this context, if any (used by the decorators)."""
    return _current.get()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _jsonable(value):
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)[:10_000]


class _Tracker:
    def __init__(self, sinks, run_id: str):
        self.run_id = run_id
        self.sinks = list(sinks)
        self._next_id = 0

    def emit(self, event: str, **fields):
        e = {"event": event, "run_id": self.run_id, "time": _now(), **fields}
        with _lock:
            for i, sink in enumerate(self.sinks):
                if sink is None:
                    continue
                try:
                    sink(e)
                except Exception:
                    self.sinks[i] = None  # tracking must never fail the run
                    _logger.warning("agentsoup sink disabled after error: %r", sink, exc_info=True)

    def call_started(self, name: str) -> int:
        with _lock:
            call_id = self._next_id
            self._next_id += 1
        self.emit("call_started", call_id=call_id, name=name)
        return call_id

    def call_finished(self, call_id: int, name: str, duration_s: float, output, usage: dict):
        self.emit("call_finished", call_id=call_id, name=name, duration_s=duration_s,
                  output=_jsonable(output), usage=usage)

    def call_failed(self, call_id: int, name: str, duration_s: float, error: BaseException, usage: dict):
        self.emit("call_failed", call_id=call_id, name=name, duration_s=duration_s,
                  error=f"{type(error).__name__}: {error}", usage=usage)


@contextlib.contextmanager
def track(*sinks, run_id: str | None = None):
    """`with track(*sinks) as run_id:` — emit every decorated call in the block
    as event dicts to the sinks. No sinks given -> state_file() by default."""
    run_id = run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
    tracker = _Tracker(sinks or (state_file(),), run_id)
    token = _current.set(tracker)
    tracker.emit("run_started")
    try:
        yield run_id
    except BaseException as e:
        tracker.emit("run_failed", error=f"{type(e).__name__}: {e}")
        raise
    else:
        tracker.emit("run_finished")
    finally:
        _current.reset(token)  # restore any outer run, don't clear it


def state_file(state_dir=".agentsoup/runs"):
    """Sink: maintain <state_dir>/<run_id>.json — a full snapshot, atomically
    rewritten after every event (statuses, timings, outputs, errors, per-call
    usage, run totals) — readable live from any other process."""
    states: dict = {}

    def _call(state, call_id):
        while len(state["calls"]) <= call_id:
            state["calls"].append({
                "name": None, "status": "pending", "started_at": None, "finished_at": None,
                "duration_s": None, "output": None, "error": None, "usage": None,
            })
        return state["calls"][call_id]

    def sink(e: dict):
        state = states.setdefault(e["run_id"], {
            "run_id": e["run_id"], "status": "running", "created_at": e["time"],
            "updated_at": e["time"],
            "usage": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
            "calls": [],
        })
        event = e["event"]
        if event == "call_started":
            _call(state, e["call_id"]).update(name=e["name"], status="running", started_at=e["time"])
        elif event in ("call_finished", "call_failed"):
            _call(state, e["call_id"]).update(
                name=e["name"], status="done" if event == "call_finished" else "failed",
                finished_at=e["time"], duration_s=e["duration_s"],
                output=e.get("output"), error=e.get("error"), usage=e.get("usage"),
            )
            totals = state["usage"]
            for key, value in (e.get("usage") or {}).items():
                totals[key] = round(totals.get(key, 0) + value, 6)
        elif event == "run_finished":
            state["status"] = "done"
        elif event == "run_failed":
            state["status"] = "failed"
        state["updated_at"] = e["time"]
        path = Path(state_dir) / f"{e['run_id']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(state, indent=2, default=str))
        os.replace(tmp, path)  # atomic: readers never see a partial doc

    return sink


def webhook(url: str, headers: dict | None = None, timeout: float = 3.0):
    """Sink: POST each event as JSON. Fire-and-forget on a daemon thread, except
    run_finished/run_failed which block so process exit can't drop them."""

    def sink(e: dict):
        def send():
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(e, default=str).encode(),
                    headers={"Content-Type": "application/json", **(headers or {})},
                )
                urllib.request.urlopen(req, timeout=timeout)
            except Exception:
                pass  # delivery is best-effort

        if e["event"] in ("run_finished", "run_failed"):
            send()
        else:
            threading.Thread(target=send, daemon=True).start()

    return sink


def otel(tracer=None):  # noqa: D103 — attached below as track.otel
    """Sink: one OpenTelemetry span per call, with usage/cost as attributes.
    Uses the globally configured tracer provider unless tracer= is given.
    Requires opentelemetry-api (pip install agentsoup[otel])."""
    if tracer is None:
        from opentelemetry import trace

        tracer = trace.get_tracer("agentsoup")
    spans: dict = {}

    def sink(e: dict):
        event = e["event"]
        if event == "call_started":
            spans[e["call_id"]] = tracer.start_span(
                e["name"], attributes={"agentsoup.run_id": e["run_id"], "agentsoup.call_id": e["call_id"]}
            )
        elif event in ("call_finished", "call_failed"):
            span = spans.pop(e["call_id"], None)
            if span is None:
                return
            span.set_attribute("agentsoup.duration_s", e["duration_s"])
            for key, value in (e.get("usage") or {}).items():
                span.set_attribute(f"agentsoup.{key}", value)
            if event == "call_failed":
                span.set_attribute("agentsoup.error", e["error"])
                try:
                    from opentelemetry.trace import Status, StatusCode

                    span.set_status(Status(StatusCode.ERROR, e["error"]))
                except Exception:
                    pass
            span.end()

    return sink


# the sinks hang off track itself: track.state_file(), track.webhook(), track.otel()
track.state_file = state_file
track.webhook = webhook
track.otel = otel
