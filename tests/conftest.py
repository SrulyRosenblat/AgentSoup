import json
from types import SimpleNamespace

import pytest


def text_response(content):
    """A litellm-completion-shaped response with plain text content."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))]
    )


class ToolCallMessage:
    """Assistant message carrying tool calls; model_dump mimics litellm's Message."""

    def __init__(self, calls):
        self.content = None
        self.tool_calls = [
            SimpleNamespace(
                id=call_id,
                type="function",
                function=SimpleNamespace(name=name, arguments=json.dumps(args)),
            )
            for call_id, name, args in calls
        ]

    def model_dump(self):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in self.tool_calls
            ],
        }


def tool_call_response(calls):
    """A response asking to run tools: calls = [(id, name, args_dict), ...]."""
    return SimpleNamespace(choices=[SimpleNamespace(message=ToolCallMessage(calls))])


@pytest.fixture
def fake_completion(monkeypatch):
    """Patch litellm.completion with a queue of canned responses; records calls."""
    import litellm

    queue = []
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if not queue:
            raise AssertionError("fake_completion queue is empty")
        return queue.pop(0)

    monkeypatch.setattr(litellm, "completion", completion)
    return SimpleNamespace(queue=queue, calls=calls)
