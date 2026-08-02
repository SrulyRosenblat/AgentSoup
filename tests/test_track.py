import json
import threading

import pytest

from agentsoup import list_runs, llm, load_run, track
from tests.conftest import text_response


def _content_fake(monkeypatch, reply):
    import litellm

    def completion(**kwargs):
        return text_response(reply(kwargs["messages"][0]["content"][0]["text"]))

    monkeypatch.setattr(litellm, "completion", completion)


def test_track_records_calls_in_a_plain_function_pipeline(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t.upper())

    @llm(model="m")
    def outline(topic: str) -> str:
        return f"outline {topic}"

    @llm(model="m")
    def draft(outline_text: str) -> str:
        return f"draft {outline_text}"

    def pipeline(topic):  # a pipeline is just a function
        return draft(outline(topic))

    with track(run_id="r1", state_dir=tmp_path) as run:
        result = pipeline("fish")

    assert result == "DRAFT OUTLINE FISH"
    state = load_run(run.state_path)
    assert state["status"] == "done"
    assert [(c["name"], c["status"]) for c in state["calls"]] == [("outline", "done"), ("draft", "done")]
    assert state["calls"][0]["output"] == "OUTLINE FISH"
    assert state["calls"][0]["duration_s"] is not None


def test_track_records_parallel_map_calls(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t.upper())

    @llm(model="m")
    def shout(w: str) -> str:
        return w

    with track(run_id="r1", state_dir=tmp_path) as run:
        assert shout.map(["a", "b", "c"]) == ["A", "B", "C"]

    state = load_run(run.state_path)
    assert len(state["calls"]) == 3
    assert {c["output"] for c in state["calls"]} == {"A", "B", "C"}
    assert all(c["status"] == "done" for c in state["calls"])


def test_track_marks_failures_and_reraises(monkeypatch, tmp_path):
    import litellm

    def explode(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(litellm, "completion", explode)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with pytest.raises(RuntimeError, match="provider down"):
        with track(run_id="r1", state_dir=tmp_path):
            ask("hi")

    state = load_run(tmp_path / "r1.json")
    assert state["status"] == "failed"
    assert state["calls"][0]["status"] == "failed"
    assert "provider down" in state["calls"][0]["error"]


def test_untracked_calls_write_nothing(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    assert ask("hi") == "hi"
    assert list_runs(tmp_path) == []


def test_track_webhook_events(monkeypatch, tmp_path):
    events = []
    import agentsoup.tracking as tr

    monkeypatch.setattr(tr, "_post_webhook", lambda url, payload, wait=False: events.append(payload))
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(run_id="r1", state_dir=tmp_path, webhook_url="http://x/hook"):
        ask("hi")

    assert [e["event"] for e in events] == ["run_started", "call_started", "call_finished", "run_finished"]
    assert events[2]["call"]["name"] == "ask"
    assert events[-1]["run_status"] == "done"


def test_state_readable_from_elsewhere_mid_run(monkeypatch, tmp_path):
    """A concurrent reader sees the running call while the LLM call is in flight."""
    observed = {}
    import litellm

    def completion(**kwargs):
        observed["mid"] = load_run(tmp_path / "r1.json")
        return text_response("ok")

    monkeypatch.setattr(litellm, "completion", completion)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(run_id="r1", state_dir=tmp_path):
        ask("hi")

    assert observed["mid"]["status"] == "running"
    assert observed["mid"]["calls"][0]["status"] == "running"
    assert not list(tmp_path.glob("*.tmp"))


def test_list_runs_ordering(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(run_id="a", state_dir=tmp_path):
        ask("1")
    with track(run_id="b", state_dir=tmp_path):
        ask("2")
    assert [r["run_id"] for r in list_runs(tmp_path)] == ["a", "b"]


def test_concurrent_tracks_in_threads_are_isolated(monkeypatch, tmp_path):
    """C1: two threads with their own track() blocks must not cross-record."""
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    def worker(run_id, n):
        with track(run_id=run_id, state_dir=tmp_path):
            for i in range(n):
                ask(f"{run_id}-{i}")

    t1 = threading.Thread(target=worker, args=("ra", 3))
    t2 = threading.Thread(target=worker, args=("rb", 3))
    t1.start(); t2.start(); t1.join(); t2.join()

    ra, rb = load_run(tmp_path / "ra.json"), load_run(tmp_path / "rb.json")
    assert len(ra["calls"]) == 3 and len(rb["calls"]) == 3
    assert all(c["output"].startswith("ra-") for c in ra["calls"])
    assert all(c["output"].startswith("rb-") for c in rb["calls"])


def test_nested_track_restores_outer(monkeypatch, tmp_path):
    """C1: closing an inner track() must restore the outer run, not clear it."""
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(run_id="outer", state_dir=tmp_path):
        with track(run_id="inner", state_dir=tmp_path):
            ask("in")
        ask("out")  # must land in the outer run

    outer = load_run(tmp_path / "outer.json")
    inner = load_run(tmp_path / "inner.json")
    assert [c["output"] for c in inner["calls"]] == ["in"]
    assert [c["output"] for c in outer["calls"]] == ["out"]


def test_map_calls_land_in_the_callers_run(monkeypatch, tmp_path):
    """C1: the caller's run must propagate into .map worker threads."""
    _content_fake(monkeypatch, lambda t: t.upper())

    @llm(model="m")
    def shout(w: str) -> str:
        return w

    with track(run_id="r1", state_dir=tmp_path):
        shout.map(["a", "b"])
    assert len(load_run(tmp_path / "r1.json")["calls"]) == 2


def test_unwritable_state_dir_never_breaks_the_run(monkeypatch, tmp_path):
    """C2/C3: tracking I/O failures disable tracking, nothing else."""
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    blocker = tmp_path / "file"
    blocker.write_text("not a dir")  # state_dir parent is a file -> mkdir fails

    with track(run_id="r1", state_dir=blocker / "runs"):
        assert ask("hi") == "hi"          # run proceeds
    assert ask("after") == "after"        # and later untracked calls are unaffected


def test_usage_recorded_per_call_and_totalled(monkeypatch, tmp_path):
    import litellm
    from types import SimpleNamespace

    def completion(**kwargs):
        r = text_response("ok")
        r.usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return r

    monkeypatch.setattr(litellm, "completion", completion)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(run_id="r1", state_dir=tmp_path):
        ask("a")
        ask("b")

    state = load_run(tmp_path / "r1.json")
    assert state["calls"][0]["usage"] == {
        "llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.0}
    assert state["usage"] == {
        "llm_calls": 2, "prompt_tokens": 20, "completion_tokens": 10, "cost_usd": 0.0}


def test_untracked_calls_skip_usage_accounting(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    assert ask("hi") == "hi"  # no run active: stats path disabled, no error
