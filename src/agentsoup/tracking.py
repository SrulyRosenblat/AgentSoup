"""Track every @llm/@agent call inside a `with track():` block — a run's progress
is written atomically to a JSON state file readable from any other process, and
optionally POSTed to a webhook. A pipeline is just a Python function; this is
how you watch one from elsewhere.

The active run is context-local (contextvars), so concurrent runs in different
threads/tasks don't interfere, and tracking I/O can never fail the tracked run."""
import contextvars
import json
import logging
import os
import secrets
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pydantic

_current: contextvars.ContextVar = contextvars.ContextVar("agentsoup_run", default=None)
_lock = threading.Lock()
_logger = logging.getLogger("agentsoup")


def current_run() -> "Run | None":
    """The active tracked Run in this context, if any (used by the decorators)."""
    return _current.get()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _jsonable(value):
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump(mode="json")
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)[:10_000]


class Run:
    """Context manager: while active, every @llm/@agent call is recorded."""

    def __init__(self, run_id=None, state_dir=".agentsoup/runs", webhook_url=None):
        self.run_id = run_id or f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
        self.state_path = Path(state_dir) / f"{self.run_id}.json"
        self.webhook_url = webhook_url
        self.state = {
            "run_id": self.run_id, "status": "running",
            "created_at": _now(), "updated_at": _now(),
            "usage": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0},
            "calls": [],
        }
        self._token = None
        self._broken = False

    def _save(self):
        if self._broken:
            return
        try:
            with _lock:
                self.state["updated_at"] = _now()
                self.state_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = Path(str(self.state_path) + ".tmp")
                tmp.write_text(json.dumps(self.state, indent=2, default=str))
                os.replace(tmp, self.state_path)  # atomic: readers never see a partial doc
        except Exception:
            self._broken = True  # tracking must never fail the run
            _logger.warning("agentsoup tracking disabled: cannot write %s", self.state_path, exc_info=True)

    def _emit(self, event: str, call: dict | None = None):
        if self.webhook_url:
            _post_webhook(self.webhook_url, {
                "event": event, "run_id": self.run_id, "timestamp": _now(),
                "call": call, "run_status": self.state["status"],
            }, wait=event in ("run_finished", "run_failed"))

    def _add_usage(self, stats):
        if stats:
            totals = self.state["usage"]
            for key, value in stats.items():
                totals[key] = round(totals.get(key, 0) + value, 6)

    def _start(self, name: str) -> dict:
        call = {
            "name": name, "status": "running", "started_at": _now(),
            "finished_at": None, "duration_s": None, "output": None,
            "error": None, "usage": None,
        }
        with _lock:
            call["index"] = len(self.state["calls"])
            self.state["calls"].append(call)
        self._save()
        self._emit("call_started", call)
        return call

    def _finish(self, call: dict, started: float, output, stats=None):
        with _lock:
            call.update(status="done", finished_at=_now(),
                        duration_s=round(time.monotonic() - started, 3),
                        output=_jsonable(output), usage=stats)
            self._add_usage(stats)
        self._save()
        self._emit("call_finished", call)

    def _fail(self, call: dict, started: float, error: BaseException, stats=None):
        with _lock:
            call.update(status="failed", finished_at=_now(),
                        duration_s=round(time.monotonic() - started, 3),
                        error=f"{type(error).__name__}: {error}", usage=stats)
            self._add_usage(stats)
        self._save()
        self._emit("call_failed", call)

    def __enter__(self) -> "Run":
        self._save()  # before publishing: a broken state_dir only disables tracking
        self._emit("run_started")
        self._token = _current.set(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _current.reset(self._token)  # restore any outer run, don't clear it
            self._token = None
        self.state["status"] = "failed" if exc_type else "done"
        self._save()
        self._emit("run_failed" if exc_type else "run_finished")
        return False


def track(run_id=None, state_dir=".agentsoup/runs", webhook_url=None) -> Run:
    """`with track() as run:` — record every decorated call made inside the block."""
    return Run(run_id=run_id, state_dir=state_dir, webhook_url=webhook_url)


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
