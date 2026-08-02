"""MCP client: connect to servers and expose their tools as ordinary Tool objects."""
import asyncio
import concurrent.futures
import json
import threading
from contextlib import AsyncExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .llm import Tool


@dataclass
class StdioServer:
    command: str
    args: list = field(default_factory=list)
    env: dict | None = None
    name: str | None = None


@dataclass
class HTTPServer:
    url: str
    headers: dict | None = None
    name: str | None = None


class _LoopThread:
    """A daemon thread running an event loop; async MCP work happens there."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 60):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)


def _label(server) -> str:
    if server.name:
        return server.name
    if isinstance(server, StdioServer):
        return Path(server.command).stem
    return urlparse(server.url).hostname or "http"


def _result_text(result) -> str:
    parts = []
    for c in result.content:
        text = getattr(c, "text", None)
        parts.append(text if text is not None else json.dumps(c.model_dump(), default=str))
    return "\n".join(parts)


async def _connect(stack: AsyncExitStack, server):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if isinstance(server, StdioServer):
        transport = stdio_client(
            StdioServerParameters(command=server.command, args=list(server.args), env=server.env)
        )
    else:
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        http_client = create_mcp_http_client(headers=server.headers) if server.headers else None
        transport = streamable_http_client(server.url, http_client=http_client)
    read, write, *_ = await stack.enter_async_context(transport)
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


@contextmanager
def _open_mcp(servers):
    """Connect to servers, yield their tools as Tool objects, tear down on exit.

    Setup and teardown of the transports run inside one long-lived coroutine on
    the loop thread — anyio cancel scopes are task-bound, so the AsyncExitStack
    must be entered and closed by the same task.
    """
    if not servers:
        yield []
        return
    lt = _LoopThread()
    ready: concurrent.futures.Future = concurrent.futures.Future()

    async def _lifecycle():
        done = asyncio.Event()
        async with AsyncExitStack() as stack:
            try:
                labeled = []
                for server in servers:
                    session = await _connect(stack, server)
                    for t in (await session.list_tools()).tools:
                        def invoke(args, _s=session, _n=t.name):
                            return _result_text(lt.run(_s.call_tool(_n, args)))

                        labeled.append(
                            (_label(server), Tool(t.name, t.description or "", t.input_schema, invoke))
                        )
                ready.set_result((labeled, done))
            except BaseException as e:
                ready.set_exception(e)
                return
            await done.wait()

    lifecycle = asyncio.run_coroutine_threadsafe(_lifecycle(), lt.loop)
    done = None
    try:
        labeled, done = ready.result(timeout=60)
        names = [t.name for _, t in labeled]
        tools = [
            Tool(f"{label}__{t.name}", t.description, t.parameters, t.invoke)
            if names.count(t.name) > 1
            else t
            for label, t in labeled
        ]
        yield tools
    finally:
        if done is not None:
            lt.loop.call_soon_threadsafe(done.set)
        try:
            lifecycle.result(timeout=10)
        except Exception:
            pass
        lt.stop()
