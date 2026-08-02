"""The @llm and @agent decorators: the function's return value is the prompt."""
import contextvars
import functools
import inspect
import json
import random
import shlex
import time
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
    if content is None:
        finish = getattr(response.choices[0], "finish_reason", None)
        raise RuntimeError(f"Model returned no content (finish_reason={finish!r})")
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


# transient provider errors worth retrying (present across litellm versions)
_RETRYABLE = tuple(
    exc for exc in (
        getattr(litellm, name, None)
        for name in ("RateLimitError", "APIConnectionError", "Timeout",
                     "InternalServerError", "ServiceUnavailableError")
    ) if isinstance(exc, type)
)


def _retry_completion(retries: int, **kwargs):
    """litellm.completion with exponential backoff + jitter on transient errors."""
    for attempt in range(retries + 1):
        try:
            return litellm.completion(**kwargs)
        except _RETRYABLE:
            if attempt == retries:
                raise
            time.sleep(min(8.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))


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
                words = shlex.split(t)
                if not words:
                    raise ValueError("Empty string in tools=")
                servers.append(StdioServer(words[0], words[1:]))
        elif isinstance(t, Tool):
            static.append(t)
        else:
            static.append(Tool.from_function(t))
    return static, servers


def _merge_tools(static_tools, labeled_mcp):
    """Merge static tools with (server_label, Tool) pairs from MCP; an MCP tool
    whose name collides with anything already present gets its server label as
    a prefix instead of failing."""
    merged = list(static_tools)
    names = {t.name for t in merged}
    for label, tool in labeled_mcp:
        if tool.name in names:
            tool = Tool(f"{label}__{tool.name}", tool.description, tool.parameters, tool.invoke)
        if tool.name in names:
            raise ValueError(f"Duplicate tool name {tool.name!r} even after server prefixing")
        names.add(tool.name)
        merged.append(tool)
    return merged


def llm(
    model: str = "gpt-4.1",
    tools: tuple = (),
    max_turns: int = 10,
    retries: int = 3,
    **llm_kwargs,
):
    """Decorator: the wrapped function's return value becomes the prompt; its
    return type hint picks the output format. With tools= it runs a tool-calling
    loop; tools may be Python functions, other @llm functions, Tool instances,
    MCP server configs (StdioServer/HTTPServer), or strings (URL or command).

    Every decorated function also gets ``.map(items, max_workers=, return_exceptions=)``:
    call it once per item, in parallel, returning the list of results in order.
    MCP sessions are opened once per outer call (shared across a whole .map).

    Transient provider errors (rate limits, timeouts, connection/5xx) are retried
    ``retries`` times with exponential backoff + jitter (0.5s doubling, 8s cap)."""
    if callable(model):  # bare @llm / @agent without parentheses
        return llm()(model)

    def deco(f):
        return_type, wants_complete = _split_return_type(f)
        static_tools, mcp_servers = _split_tools(tools)
        names = [t.name for t in static_tools]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate tool names: {sorted(n for n in names if names.count(n) > 1)}")
        extra = dict(llm_kwargs)
        if _is_model(return_type):
            extra["response_format"] = return_type
        elif return_type not in (None, str):
            # list[str], dict, dataclass, ... — ask for JSON matching the adapted schema
            extra["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"{f.__name__}_response",
                    "schema": pydantic.TypeAdapter(return_type).json_schema(),
                },
            }

        def _mcp_ctx():
            if mcp_servers:
                from .mcp import _open_mcp

                return _open_mcp(mcp_servers)
            return nullcontext([])

        def _invoke(args, kwargs, labeled_mcp):
            all_tools = _merge_tools(static_tools, labeled_mcp)
            tool_map = {t.name: t for t in all_tools}
            call_kwargs = dict(extra)
            if all_tools:
                call_kwargs["tools"] = [t.to_openai_schema() for t in all_tools]
            messages = [m.to_openai_format() for m in coerce(f(*args, **kwargs))]
            for _ in range(max_turns):
                response = _retry_completion(retries, model=model, messages=messages, **call_kwargs)
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

        def _tracked(args, kwargs, labeled_mcp):
            from .tracking import current_run

            run = current_run()
            if run is None:
                return _invoke(args, kwargs, labeled_mcp)
            call = run._start(f.__name__)
            started = time.monotonic()
            try:
                result = _invoke(args, kwargs, labeled_mcp)
            except Exception as e:
                run._fail(call, started, e)
                raise
            run._finish(call, started, result)
            return result

        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            with _mcp_ctx() as labeled_mcp:
                return _tracked(args, kwargs, labeled_mcp)

        def map_(items, *, max_workers=None, return_exceptions=False, **kwargs):
            """Call the function once per item, in parallel; results in order.
            With return_exceptions=True a failed item yields its exception
            instead of cancelling the rest."""
            items = list(items)
            if not items:
                return []
            # one context copy per item so the caller's track() run propagates
            contexts = [contextvars.copy_context() for _ in items]
            with _mcp_ctx() as labeled_mcp, ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(ctx.run, _tracked, (item,), kwargs, labeled_mcp)
                    for ctx, item in zip(contexts, items)
                ]
                results = []
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as e:
                        if not return_exceptions:
                            raise
                        results.append(e)
                return results

        wrapper.map = map_
        wrapper.with_options = lambda **overrides: llm(
            **{"model": model, "tools": tools, "max_turns": max_turns,
               "retries": retries, **llm_kwargs, **overrides}
        )(f)
        return wrapper

    return deco


agent = llm  # one decorator; the alias just reads better when tools are involved
