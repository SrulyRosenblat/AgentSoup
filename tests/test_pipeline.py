import json
import threading

import pytest

from agentsoup import Pipeline, list_runs, load_run, step
from agentsoup.cli import main


def write_pipeline(tmp_path, body):
    p = tmp_path / "steps.py"
    p.write_text(body)
    return str(p)


BASIC = """
from agentsoup import step

@step
def double(x):
    return int(x) * 2

@step
def stringify(x):
    return {"result": x}
"""


def test_from_file_runs_and_chains(tmp_path):
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, BASIC))
    result = pipeline.run(input=5, state_dir=tmp_path / "runs")
    assert result.status == "done" and result.output == {"result": 10}
    state = load_run(result.state_path)
    assert state["status"] == "done"
    assert [s["status"] for s in state["steps"]] == ["done", "done"]
    assert state["steps"][1]["output_json"] == {"result": 10}
    assert state["steps"][0]["duration_s"] is not None


def test_explicit_order_overrides_definition_order(tmp_path):
    src = """
from agentsoup import step

@step(order=2)
def second(x):
    return x + ["second"]

@step(order=1)
def first(x=None):
    return ["first"]
"""
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, src))
    result = pipeline.run(input=None, state_dir=tmp_path / "runs")
    assert result.output == ["first", "second"]


def test_zero_param_first_step_gets_no_input(tmp_path):
    src = """
from agentsoup import step

@step
def start():
    return "seed"

@step
def grow(x):
    return x + "!"
"""
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, src))
    assert pipeline.run(state_dir=tmp_path / "runs").output == "seed!"


def test_failure_marks_run_and_reraises(tmp_path):
    src = """
from agentsoup import step

@step
def ok(x=None):
    return 1

@step
def bad(x):
    raise RuntimeError("kaput")
"""
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, src))
    with pytest.raises(RuntimeError, match="kaput"):
        pipeline.run(state_dir=tmp_path / "runs", run_id="r1")
    state = load_run(tmp_path / "runs" / "r1.json")
    assert state["status"] == "failed"
    assert state["steps"][1]["status"] == "failed"
    assert "kaput" in state["steps"][1]["error"]
    assert state["steps"][0]["status"] == "done"


def test_state_readable_from_elsewhere_mid_run(tmp_path):
    """A concurrent reader sees a complete, current state file while the run is going."""
    observed = {}
    state_dir = tmp_path / "runs"

    src = """
from agentsoup import step
import checker

@step
def one(x=None):
    checker.check("during_step_one")
    return 1

@step
def two(x):
    return x + 1
"""
    import sys
    import types

    checker = types.ModuleType("checker")

    def check(label):
        observed[label] = load_run(state_dir / "r1.json")

    checker.check = check
    sys.modules["checker"] = checker
    try:
        pipeline = Pipeline.from_file(write_pipeline(tmp_path, src))
        pipeline.run(state_dir=state_dir, run_id="r1")
    finally:
        del sys.modules["checker"]

    mid = observed["during_step_one"]
    assert mid["status"] == "running"
    assert mid["steps"][0]["status"] == "running"
    assert not list(state_dir.glob("*.tmp"))  # atomic writes leave no partials


def test_webhook_events(tmp_path, monkeypatch):
    events = []

    class FakeResponse:
        def read(self):
            return b""

    def fake_urlopen(req, timeout=None):
        events.append(json.loads(req.data))
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    # webhook posts happen on daemon threads; make them synchronous for the test
    monkeypatch.setattr(
        "agentsoup.pipeline.threading.Thread",
        lambda target, daemon: type("T", (), {"start": staticmethod(target)})(),
    )

    pipeline = Pipeline.from_file(write_pipeline(tmp_path, BASIC))
    pipeline.run(input=1, state_dir=tmp_path / "runs", webhook_url="http://track.example/hook")
    names = [e["event"] for e in events]
    assert names == [
        "run_started", "step_started", "step_finished",
        "step_started", "step_finished", "run_finished",
    ]
    assert events[2]["step"]["name"] == "double"
    assert events[-1]["run_status"] == "done"


def test_webhook_failure_never_breaks_run(tmp_path, monkeypatch):
    def explode(req, timeout=None):
        raise OSError("network down")

    monkeypatch.setattr("urllib.request.urlopen", explode)
    monkeypatch.setattr(
        "agentsoup.pipeline.threading.Thread",
        lambda target, daemon: type("T", (), {"start": staticmethod(target)})(),
    )
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, BASIC))
    result = pipeline.run(input=1, state_dir=tmp_path / "runs", webhook_url="http://down.example")
    assert result.status == "done"


def test_list_runs(tmp_path):
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, BASIC))
    pipeline.run(input=1, state_dir=tmp_path / "runs", run_id="a")
    pipeline.run(input=2, state_dir=tmp_path / "runs", run_id="b")
    assert [r["run_id"] for r in list_runs(tmp_path / "runs")] == ["a", "b"]
    assert list_runs(tmp_path / "missing") == []


def test_cli_run_and_status(tmp_path, capsys):
    path = write_pipeline(tmp_path, BASIC)
    state_dir = str(tmp_path / "runs")

    assert main(["run", path, "--input", "3", "--run-id", "cli-run", "--state-dir", state_dir]) == 0
    out = capsys.readouterr().out
    assert "cli-run" in out and "done. output: {'result': 6}" in out

    assert main(["status", "--state-dir", state_dir]) == 0
    assert "cli-run  done  2/2 steps" in capsys.readouterr().out

    assert main(["status", "cli-run", "--state-dir", state_dir]) == 0
    out = capsys.readouterr().out
    assert "1. double: done" in out and "2. stringify: done" in out

    assert main(["status", "cli-run", "--state-dir", state_dir, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "cli-run"


def test_cli_run_failure_exit_code(tmp_path, capsys):
    src = """
from agentsoup import step

@step
def bad(x=None):
    raise ValueError("broken")
"""
    path = write_pipeline(tmp_path, src)
    assert main(["run", path, "--state-dir", str(tmp_path / "runs")]) == 1
    assert "failed: ValueError: broken" in capsys.readouterr().out


def test_steps_receive_earlier_outputs_by_name(tmp_path):
    src = """
from agentsoup import step

@step
def seed(x=None):
    return 2

@step
def grow(seed):
    return seed * 10

@step
def report(x, seed):
    return {"seed": seed, "grown": x}
"""
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, src))
    result = pipeline.run(state_dir=tmp_path / "runs")
    # `seed` params got seed's output; `x` (unmatched) got the previous step's output
    assert result.output == {"seed": 2, "grown": 20}


def test_named_wiring_respects_step_rename(tmp_path):
    src = """
from agentsoup import step

@step(name="load")
def whatever(x=None):
    return "data"

@step
def use(load):
    return load + "!"
"""
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, src))
    assert pipeline.run(state_dir=tmp_path / "runs").output == "data!"


def test_multiple_unbound_params_raise(tmp_path):
    src = """
from agentsoup import step

@step
def one(x=None):
    return 1

@step
def bad(a, b):
    return a + b
"""
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, src))
    with pytest.raises(ValueError, match="multiple unbound parameters"):
        pipeline.run(state_dir=tmp_path / "runs", run_id="r1")
    state = load_run(tmp_path / "runs" / "r1.json")
    assert state["status"] == "failed"


def test_step_description_in_state(tmp_path):
    src = """
from agentsoup import step

@step(description="Doubles the input")
def double(x):
    return x * 2

@step
def label(x):
    '''Wrap in a dict.'''
    return {"v": x}

@step
def bare(x):
    return x
"""
    pipeline = Pipeline.from_file(write_pipeline(tmp_path, src))
    result = pipeline.run(input=2, state_dir=tmp_path / "runs")
    steps = load_run(result.state_path)["steps"]
    assert steps[0]["description"] == "Doubles the input"
    assert steps[1]["description"] == "Wrap in a dict."  # docstring fallback
    assert steps[2]["description"] is None
