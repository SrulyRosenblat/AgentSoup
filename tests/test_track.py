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
