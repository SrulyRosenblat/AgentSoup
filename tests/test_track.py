import json
import threading
from pathlib import Path

import pytest

from agentsoup import llm, track
from tests.conftest import text_response


def _content_fake(monkeypatch, reply):
    import litellm

    def completion(**kwargs):
        return text_response(reply(kwargs["messages"][0]["content"][0]["text"]))

    monkeypatch.setattr(litellm, "completion", completion)


def read_state(state_dir, run_id) -> dict:
    return json.loads((Path(state_dir) / f"{run_id}.json").read_text())


def test_state_file_records_a_plain_function_pipeline(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t.upper())

    @llm(model="m")
    def outline(topic: str) -> str:
        return f"outline {topic}"

    @llm(model="m")
    def draft(outline_text: str) -> str:
        return f"draft {outline_text}"

    def pipeline(topic):  # a pipeline is just a function
        return draft(outline(topic))

    with track(track.state_file(tmp_path), run_id="r1"):
        result = pipeline("fish")

    assert result == "DRAFT OUTLINE FISH"
    state = read_state(tmp_path, "r1")
    assert state["status"] == "done"
    assert [(c["name"], c["status"]) for c in state["calls"]] == [("outline", "done"), ("draft", "done")]
    assert state["calls"][0]["output"] == "OUTLINE FISH"
    assert state["calls"][0]["duration_s"] is not None


def test_event_sequence_and_call_ids(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t.upper())
    events = []

    @llm(model="m")
    def shout(w: str) -> str:
        return w

    with track(events.append, run_id="r1"):
        assert shout.map(["a", "b", "c"]) == ["A", "B", "C"]

    names = [e["event"] for e in events]
    assert names[0] == "run_started" and names[-1] == "run_finished"
    starts = [e for e in events if e["event"] == "call_started"]
    finishes = [e for e in events if e["event"] == "call_finished"]
    assert {e["call_id"] for e in starts} == {0, 1, 2}
    assert {e["call_id"] for e in finishes} == {0, 1, 2}  # ids correlate under parallelism
    assert all(e["run_id"] == "r1" for e in events)
    assert {e["output"] for e in finishes} == {"A", "B", "C"}


def test_failure_marks_run_and_reraises(monkeypatch, tmp_path):
    import litellm

    def explode(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(litellm, "completion", explode)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with pytest.raises(RuntimeError, match="provider down"):
        with track(track.state_file(tmp_path), run_id="r1"):
            ask("hi")

    state = read_state(tmp_path, "r1")
    assert state["status"] == "failed"
    assert state["calls"][0]["status"] == "failed"
    assert "provider down" in state["calls"][0]["error"]


def test_run_failed_event_carries_error(monkeypatch):
    _content_fake(monkeypatch, lambda t: t)
    events = []

    with pytest.raises(ValueError):
        with track(events.append):
            raise ValueError("user code broke")

    assert events[-1]["event"] == "run_failed"
    assert "user code broke" in events[-1]["error"]


def test_untracked_calls_emit_nothing(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    assert ask("hi") == "hi"
    assert list(Path(tmp_path).glob("*.json")) == []


def test_default_sink_is_state_file(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t)
    monkeypatch.chdir(tmp_path)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(run_id="r1"):
        ask("hi")
    assert read_state(tmp_path / ".agentsoup/runs", "r1")["status"] == "done"


def test_webhook_sink_posts_events(monkeypatch):
    _content_fake(monkeypatch, lambda t: t)
    sent = []

    class FakeResponse:
        def read(self):
            return b""

    def fake_urlopen(req, timeout=None):
        sent.append((json.loads(req.data), dict(req.headers)))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    # one worker thread keeps event order; terminal events flush before the block exits
    with track(track.webhook("http://x/hook", headers={"Authorization": "Bearer t"}), run_id="r1"):
        ask("hi")

    assert [e["event"] for e, _ in sent] == ["run_started", "call_started", "call_finished", "run_finished"]
    assert sent[2][0]["name"] == "ask"
    assert any(k.lower() == "authorization" for _, h in sent for k in h)


def test_webhook_failure_never_breaks_run(monkeypatch, tmp_path):
    def explode(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(track.webhook("http://down.example"), run_id="r1"):
        assert ask("hi") == "hi"


def test_raising_sink_is_disabled_others_continue(monkeypatch):
    _content_fake(monkeypatch, lambda t: t)
    good = []

    def bad(event):
        raise RuntimeError("sink bug")

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(bad, good.append, run_id="r1"):
        assert ask("a") == "a"
        assert ask("b") == "b"

    assert [e["event"] for e in good] == [
        "run_started", "call_started", "call_finished", "call_started", "call_finished", "run_finished"]


def test_state_readable_from_elsewhere_mid_run(monkeypatch, tmp_path):
    """A concurrent reader sees the running call while the LLM call is in flight."""
    observed = {}
    import litellm

    def completion(**kwargs):
        observed["mid"] = read_state(tmp_path, "r1")
        return text_response("ok")

    monkeypatch.setattr(litellm, "completion", completion)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(track.state_file(tmp_path), run_id="r1"):
        ask("hi")

    assert observed["mid"]["status"] == "running"
    assert observed["mid"]["calls"][0]["status"] == "running"
    assert not list(Path(tmp_path).glob("*.tmp"))  # atomic writes leave no partials


def test_concurrent_tracks_in_threads_are_isolated(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t)
    sink = track.state_file(tmp_path)  # one sink instance shared across runs is fine

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    def worker(run_id, n):
        with track(sink, run_id=run_id):
            for i in range(n):
                ask(f"{run_id}-{i}")

    t1 = threading.Thread(target=worker, args=("ra", 3))
    t2 = threading.Thread(target=worker, args=("rb", 3))
    t1.start(); t2.start(); t1.join(); t2.join()

    ra, rb = read_state(tmp_path, "ra"), read_state(tmp_path, "rb")
    assert len(ra["calls"]) == 3 and len(rb["calls"]) == 3
    assert all(c["output"].startswith("ra-") for c in ra["calls"])
    assert all(c["output"].startswith("rb-") for c in rb["calls"])


def test_nested_track_restores_outer(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(track.state_file(tmp_path), run_id="outer"):
        with track(track.state_file(tmp_path), run_id="inner"):
            ask("in")
        ask("out")  # must land in the outer run

    assert [c["output"] for c in read_state(tmp_path, "inner")["calls"]] == ["in"]
    assert [c["output"] for c in read_state(tmp_path, "outer")["calls"]] == ["out"]


def test_map_calls_land_in_the_callers_run(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t.upper())

    @llm(model="m")
    def shout(w: str) -> str:
        return w

    with track(track.state_file(tmp_path), run_id="r1"):
        shout.map(["a", "b"])
    assert len(read_state(tmp_path, "r1")["calls"]) == 2


def test_unwritable_state_dir_never_breaks_the_run(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    blocker = tmp_path / "file"
    blocker.write_text("not a dir")  # state_dir parent is a file -> mkdir fails

    with track(track.state_file(blocker / "runs"), run_id="r1"):
        assert ask("hi") == "hi"          # sink disabled with a warning, run proceeds
    assert ask("after") == "after"        # later untracked calls unaffected


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

    with track(track.state_file(tmp_path), run_id="r1"):
        ask("a")
        ask("b")

    state = read_state(tmp_path, "r1")
    assert state["calls"][0]["usage"] == {
        "llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5, "cost_usd": 0.0}
    assert state["usage"] == {
        "llm_calls": 2, "prompt_tokens": 20, "completion_tokens": 10, "cost_usd": 0.0}


def test_otel_sink_spans(monkeypatch):
    

    _content_fake(monkeypatch, lambda t: t)
    spans = []

    class FakeSpan:
        def __init__(self, name, attributes):
            self.name, self.attrs, self.ended = name, dict(attributes), False

        def set_attribute(self, k, v):
            self.attrs[k] = v

        def set_status(self, *a, **k):
            pass

        def end(self):
            self.ended = True

    class FakeTracer:
        def start_span(self, name, attributes=None):
            span = FakeSpan(name, attributes or {})
            spans.append(span)
            return span

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(track.otel(tracer=FakeTracer()), run_id="r1"):
        ask("hi")

    [span] = spans
    assert span.name == "ask" and span.ended
    assert span.attrs["agentsoup.run_id"] == "r1"
    assert span.attrs["agentsoup.llm_calls"] == 1
    assert "agentsoup.duration_s" in span.attrs


def test_sink_calling_an_llm_function_does_not_deadlock(monkeypatch, tmp_path):
    """C1: a sink is just a function — including one that itself calls an @llm fn."""
    _content_fake(monkeypatch, lambda t: t)
    seen = []

    @llm(model="m")
    def label(text: str) -> str:
        return text

    @llm(model="m")
    def work(text: str) -> str:
        return text

    def reflective_sink(event):
        if event["event"] == "call_finished" and event["name"] == "work":
            seen.append(label("summarize: " + str(event["output"])))

    with track(reflective_sink, run_id="r1"):
        assert work("hi") == "hi"
    assert seen == ["summarize: hi"]  # completed without hanging


def test_slow_sink_does_not_block_an_unrelated_run(monkeypatch, tmp_path):
    """C2: locks are per sink instance, not global."""
    import time as _t
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    def slow(event):
        _t.sleep(0.3)

    fast_events = []
    durations = {}

    def run_slow():
        with track(slow, run_id="slow"):
            ask("a")

    def run_fast():
        _t.sleep(0.05)  # start while the slow sink is mid-event
        start = _t.monotonic()
        with track(fast_events.append, run_id="fast"):
            ask("b")
        durations["fast"] = _t.monotonic() - start

    t1 = threading.Thread(target=run_slow)
    t2 = threading.Thread(target=run_fast)
    t1.start(); t2.start(); t1.join(); t2.join()
    assert durations["fast"] < 0.25  # not serialized behind the slow sink
    assert len(fast_events) == 4


def test_state_file_evicts_finished_runs_and_never_resumes(monkeypatch, tmp_path):
    """I3: no unbounded memory; reusing a run_id starts fresh."""
    _content_fake(monkeypatch, lambda t: t)
    sink = track.state_file(tmp_path)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    for _ in range(2):
        with track(sink, run_id="dup"):
            ask("hi")

    state = read_state(tmp_path, "dup")
    assert len(state["calls"]) == 1                  # not merged with the earlier run
    assert state["usage"]["llm_calls"] == 1
    assert sink.__closure__ is not None
    retained = [c for c in sink.__closure__ if isinstance(c.cell_contents, dict)]
    assert all(not c.cell_contents for c in retained)  # states dict emptied


def test_keyboard_interrupt_marks_call_failed(monkeypatch, tmp_path):
    """I2: BaseException must still close out the call."""
    import litellm

    def interrupt(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(litellm, "completion", interrupt)
    events = []

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with pytest.raises(KeyboardInterrupt):
        with track(events.append, run_id="r1"):
            ask("hi")

    assert [e["event"] for e in events] == ["run_started", "call_started", "call_failed", "run_failed"]


def test_otel_spans_isolated_per_run(monkeypatch):
    """I1: call_ids restart per run, so spans must key on (run_id, call_id)."""
    _content_fake(monkeypatch, lambda t: t)
    spans = []

    class FakeSpan:
        def __init__(self, name, attrs):
            self.name, self.attrs, self.ended = name, dict(attrs), False

        def set_attribute(self, k, v):
            self.attrs[k] = v

        def set_status(self, *a, **k):
            pass

        def end(self):
            self.ended = True

    class FakeTracer:
        def start_span(self, name, attributes=None):
            s = FakeSpan(name, attributes or {})
            spans.append(s)
            return s

    sink = track.otel(tracer=FakeTracer())

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    def worker(run_id):
        with track(sink, run_id=run_id):
            ask(run_id)

    t1 = threading.Thread(target=worker, args=("ra",))
    t2 = threading.Thread(target=worker, args=("rb",))
    t1.start(); t2.start(); t1.join(); t2.join()

    assert len(spans) == 2
    assert all(s.ended for s in spans)                       # no leaked spans
    assert {s.attrs["agentsoup.run_id"] for s in spans} == {"ra", "rb"}


def test_complete_response_output_records_parsed_value(monkeypatch, tmp_path):
    """I5: don't dump a 10KB repr of the raw completion into the state file."""
    from agentsoup import CompleteResponse

    _content_fake(monkeypatch, lambda t: t)
    events = []

    @llm(model="m")
    def ask(q: str) -> CompleteResponse[str]:
        return q

    with track(events.append, run_id="r1"):
        ask("hi")

    finished = [e for e in events if e["event"] == "call_finished"][0]
    assert finished["output"] == "hi"


def test_run_id_is_sanitized_for_the_filename(monkeypatch, tmp_path):
    _content_fake(monkeypatch, lambda t: t)

    @llm(model="m")
    def ask(q: str) -> str:
        return q

    with track(track.state_file(tmp_path), run_id="a/b:c"):
        ask("hi")
    assert (Path(tmp_path) / "a_b_c.json").exists()
