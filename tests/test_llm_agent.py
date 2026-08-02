import json

import pytest
from pydantic import BaseModel

from agentsoup import AgentMaxTurnsError, CompleteResponse, Tool, agent, llm
from tests.conftest import text_response, tool_call_response


class Answer(BaseModel):
    value: int


def test_llm_plain_text(fake_completion):
    fake_completion.queue.append(text_response("hi there"))

    @llm(model="test-model", temperature=0)
    def greet(name: str) -> str:
        return f"Greet {name}"

    assert greet("Sam") == "hi there"
    call = fake_completion.calls[0]
    assert call["model"] == "test-model" and call["temperature"] == 0
    assert call["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Greet Sam"}]}]
    assert "response_format" not in call


def test_llm_structured(fake_completion):
    fake_completion.queue.append(text_response('{"value": 42}'))

    @llm(model="test-model")
    def compute() -> Answer:
        return "compute"

    assert compute() == Answer(value=42)
    assert fake_completion.calls[0]["response_format"] is Answer


def test_llm_complete_response(fake_completion):
    response = text_response('{"value": 7}')
    fake_completion.queue.append(response)

    @llm(model="test-model")
    def compute() -> CompleteResponse[Answer]:
        return "compute"

    result = compute()
    assert result.parsed_response == Answer(value=7) and result.completion is response


def test_llm_tuple_multimodal_return(fake_completion, tmp_path):
    img = tmp_path / "x.png"
    img.write_bytes(b"png")
    fake_completion.queue.append(text_response("a cat"))

    @llm(model="test-model")
    def caption(path) -> str:
        return "describe:", path

    assert caption(img) == "a cat"
    content = fake_completion.calls[0]["messages"][0]["content"]
    assert [p["type"] for p in content] == ["text", "image_url"]


def test_tool_from_function_schema():
    def get_weather(city: str, units: str = "c") -> str:
        """Look up the weather."""
        return f"{city}:{units}"

    tool = Tool.from_function(get_weather)
    schema = tool.to_openai_schema()
    assert schema["function"]["name"] == "get_weather"
    assert schema["function"]["description"] == "Look up the weather."
    params = schema["function"]["parameters"]
    assert params["required"] == ["city"]
    assert params["properties"]["units"]["default"] == "c"
    assert tool.invoke({"city": "NYC"}) == "NYC:c"


def test_tool_invoke_validates_args():
    def double(n: int) -> int:
        return n * 2

    tool = Tool.from_function(double)
    assert tool.invoke({"n": "3"}) == 6  # coerced
    with pytest.raises(Exception):
        tool.invoke({"n": "not a number"})


def test_agent_tool_loop(fake_completion):
    seen = []

    def get_weather(city: str) -> str:
        """Weather lookup."""
        seen.append(city)
        return "sunny"

    fake_completion.queue.append(tool_call_response([("call_1", "get_weather", {"city": "NYC"})]))
    fake_completion.queue.append(text_response('{"value": 25}'))

    @agent(model="test-model", tools=[get_weather])
    def ask(q: str) -> Answer:
        return q

    assert ask("weather?") == Answer(value=25)
    assert seen == ["NYC"]
    first, second = fake_completion.calls
    assert first["tools"][0]["function"]["name"] == "get_weather"
    transcript = second["messages"]
    assert transcript[1]["tool_calls"][0]["id"] == "call_1"
    assert transcript[2] == {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}


def test_agent_tool_error_fed_back(fake_completion):
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("nope")

    fake_completion.queue.append(tool_call_response([("call_1", "boom", {})]))
    fake_completion.queue.append(text_response("recovered"))

    @agent(model="test-model", tools=[boom])
    def ask(q: str) -> str:
        return q

    assert ask("go") == "recovered"
    tool_msg = fake_completion.calls[1]["messages"][2]
    assert tool_msg["content"].startswith("Error: RuntimeError")


def test_agent_max_turns(fake_completion):
    def noop() -> str:
        """No-op."""
        return "ok"

    for _ in range(3):
        fake_completion.queue.append(tool_call_response([("c", "noop", {})]))

    @agent(model="test-model", tools=[noop], max_turns=3)
    def ask(q: str) -> str:
        return q

    with pytest.raises(AgentMaxTurnsError):
        ask("loop forever")


def test_agent_without_tools_acts_like_llm(fake_completion):
    fake_completion.queue.append(text_response("plain"))

    @agent(model="test-model")
    def ask(q: str) -> str:
        return q

    assert ask("hi") == "plain"
    assert "tools" not in fake_completion.calls[0]


def test_llm_function_is_a_valid_tool(fake_completion):
    @llm(model="inner-model")
    def summarize(text: str) -> str:
        """Summarize text in two sentences."""
        return f"Summarize: {text}"

    tool = Tool.from_function(summarize)
    assert tool.name == "summarize"
    assert tool.description == "Summarize text in two sentences."
    assert tool.to_openai_schema()["function"]["parameters"]["required"] == ["text"]

    fake_completion.queue.append(tool_call_response([("call_1", "summarize", {"text": "long doc"})]))
    fake_completion.queue.append(text_response("inner summary"))  # nested @llm call
    fake_completion.queue.append(text_response("final answer"))

    @agent(model="outer-model", tools=[summarize])
    def research(q: str) -> str:
        return q

    assert research("summarize the doc") == "final answer"
    inner_call = fake_completion.calls[1]
    assert inner_call["model"] == "inner-model"
    assert inner_call["messages"][0]["content"][0]["text"] == "Summarize: long doc"
    assert fake_completion.calls[2]["messages"][2]["content"] == "inner summary"


def test_duplicate_tool_names_rejected(fake_completion):
    def t() -> str:
        """T."""
        return "x"

    @agent(model="m", tools=[t, Tool("t", "dup", {"type": "object", "properties": {}}, lambda a: "y")])
    def ask(q: str) -> str:
        return q

    with pytest.raises(ValueError, match="Duplicate tool names"):
        ask("hi")


def test_with_options_overrides_llm_params(fake_completion):
    fake_completion.queue.append(text_response("base"))
    fake_completion.queue.append(text_response("tuned"))

    @llm(model="base-model", temperature=1)
    def ask(q: str) -> str:
        return q

    assert ask("hi") == "base"
    tuned = ask.with_options(model="fast-model", temperature=0)
    assert tuned("hi") == "tuned"
    base_call, tuned_call = fake_completion.calls
    assert (base_call["model"], base_call["temperature"]) == ("base-model", 1)
    assert (tuned_call["model"], tuned_call["temperature"]) == ("fast-model", 0)


def test_with_options_on_agent_keeps_tools(fake_completion):
    seen = []

    def probe() -> str:
        """Probe."""
        seen.append(1)
        return "ok"

    fake_completion.queue.append(tool_call_response([("c1", "probe", {})]))
    fake_completion.queue.append(text_response("done"))

    @agent(model="base-model", tools=[probe])
    def ask(q: str) -> str:
        return q

    fast = ask.with_options(model="fast-model", max_turns=5)
    assert fast("go") == "done"
    assert seen == [1]
    assert fake_completion.calls[0]["model"] == "fast-model"
    assert fake_completion.calls[0]["tools"][0]["function"]["name"] == "probe"
