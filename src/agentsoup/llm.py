"""The @llm and @agent decorators: the function's return value is the prompt."""
import functools
import inspect
import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Generic, NamedTuple, TypeVar, get_args, get_origin, get_type_hints

import litellm
import pydantic

from .parts import coerce

T = TypeVar("T")


class CompleteResponse(NamedTuple, Generic[T]):
    """Annotate a return type as CompleteResponse[T] to also get the raw completion."""

    parsed_response: T
    completion: object


class AgentMaxTurnsError(RuntimeError):
    def __init__(self, max_turns: int, messages: list):
        super().__init__(f"Agent did not finish within {max_turns} turns")
        self.max_turns = max_turns
        self.messages = messages


def _split_return_type_hint(hint) -> tuple[type | None, bool]:
    if get_origin(hint) == CompleteResponse:
        return get_args(hint)[0], True
    return hint, False


def _split_return_type(f) -> tuple[type | None, bool]:
    return _split_return_type_hint(get_type_hints(f).get("return"))


def _is_model(t) -> bool:
    return isinstance(t, type) and issubclass(t, pydantic.BaseModel)


def _finalize(response, return_type, wants_complete):
    content = response.choices[0].message.content
    if return_type in (None, str):
        parsed = content
    elif _is_model(return_type):
        parsed = return_type.model_validate_json(content)
    else:  # list[str], dict, dataclass, ... — anything pydantic can adapt
        parsed = pydantic.TypeAdapter(return_type).validate_json(content)
    return CompleteResponse(parsed, response) if wants_complete else parsed


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    invoke: Callable[[dict], Any]

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    @classmethod
    def from_function(cls, fn: Callable) -> "Tool":
        target = inspect.unwrap(fn)  # see through @llm/@agent wrappers
        hints = get_type_hints(target)
        hints.pop("return", None)
        fields = {}
        for name, p in inspect.signature(target).parameters.items():
            default = ... if p.default is inspect.Parameter.empty else p.default
            fields[name] = (hints.get(name, str), default)
        arg_model = pydantic.create_model(f"{target.__name__}_args", **fields)

        def invoke(args: dict):
            validated = arg_model(**args)
            return fn(**{k: getattr(validated, k) for k in arg_model.model_fields})

        return cls(
            name=target.__name__,
            description=inspect.getdoc(target) or target.__name__,
            parameters=arg_model.model_json_schema(),
            invoke=invoke,
        )


def _serialize_result(out) -> str:
    if isinstance(out, str):
        return out
    if isinstance(out, pydantic.BaseModel):
        return out.model_dump_json()
    return json.dumps(out, default=str)


def _split_tools(tools):
    """Sort a mixed tools list into (static Tool objects, MCP server configs).

    Accepts callables (incl. @llm/@agent functions), Tool instances,
    StdioServer/HTTPServer configs, and strings — a URL becomes an HTTPServer,
    any other string is a stdio server command line."""
    from .mcp import HTTPServer, StdioServer

    static, servers = [], []
    for t in tools:
        if isinstance(t, (StdioServer, HTTPServer)):
            servers.append(t)
        elif isinstance(t, str):
            if t.startswith(("http://", "https://")):
                servers.append(HTTPServer(t))
            else:
                command, *args = shlex.split(t)
                servers.append(StdioServer(command, args))
        elif isinstance(t, Tool):
            static.append(t)
        else:
            static.append(Tool.from_function(t))
    return static, servers


def llm(
    model: str = "gpt-4.1",
    tools: tuple = (),
    max_turns: int = 10,
    **llm_kwargs,
):
    """Decorator: the wrapped function's return value becomes the prompt; its
    return type hint picks the output format. With tools= it runs a tool-calling
    loop; tools may be Python functions, other @llm functions, Tool instances,
    MCP server configs (StdioServer/HTTPServer), or strings (URL or command).

    Every decorated function also gets ``.map(items)``: call it once per item,
    in parallel, returning the list of results in order."""

    def deco(f):
        return_type, wants_complete = _split_return_type(f)
        static_tools, mcp_servers = _split_tools(tools)
        extra = dict(llm_kwargs)
        if _is_model(return_type):  # other hints (list[str], ...) are parse-only
            extra["response_format"] = return_type

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if mcp_servers:
                from .mcp import _open_mcp

                ctx = _open_mcp(mcp_servers)
            else:
                ctx = nullcontext([])
            with ctx as mcp_tools:
                all_tools = static_tools + list(mcp_tools)
                tool_map = {t.name: t for t in all_tools}
                if len(tool_map) != len(all_tools):
                    raise ValueError("Duplicate tool names in tools=")
                call_kwargs = dict(extra)
                if all_tools:
                    call_kwargs["tools"] = [t.to_openai_schema() for t in all_tools]

                def run_one(prompt_value):
                    messages = [m.to_openai_format() for m in coerce(prompt_value)]
                    for _ in range(max_turns):
                        response = litellm.completion(model=model, messages=messages, **call_kwargs)
                        msg = response.choices[0].message
                        tool_calls = getattr(msg, "tool_calls", None)
                        if not tool_calls:
                            return _finalize(response, return_type, wants_complete)
                        messages.append(msg.model_dump() if hasattr(msg, "model_dump") else msg)
                        for tc in tool_calls:
                            try:
                                out = tool_map[tc.function.name].invoke(json.loads(tc.function.arguments or "{}"))
                                content = _serialize_result(out)
                            except Exception as e:  # feed errors back to the model
                                content = f"Error: {type(e).__name__}: {e}"
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})
                    raise AgentMaxTurnsError(max_turns, messages)

                return run_one(f(*args, **kwargs))

        def map_(items, **kwargs):
            """Call the function once per item, in parallel; results in order."""
            items = list(items)
            if not items:
                return []
            with ThreadPoolExecutor() as pool:
                return list(pool.map(lambda item: wrapper(item, **kwargs), items))

        wrapper.map = map_
        wrapper.with_options = lambda **overrides: llm(
            **{"model": model, "tools": tools, "max_turns": max_turns, **llm_kwargs, **overrides}
        )(f)
        return wrapper

    return deco


agent = llm  # one decorator; the alias just reads better when tools are involved
