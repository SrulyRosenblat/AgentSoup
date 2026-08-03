import asyncio
from types import SimpleNamespace

from agentsoup import StdioServer
from agentsoup.mcp import _LoopThread, _open_mcp, _result_text


def test_loop_thread_runs_coroutines_and_stops():
    lt = _LoopThread()

    async def add(a, b):
        await asyncio.sleep(0)
        return a + b

    assert lt.run(add(2, 3)) == 5
    lt.stop()
    assert not lt._thread.is_alive()


def test_result_text_joins_content():
    result = SimpleNamespace(content=[
        SimpleNamespace(text="hello"),
        SimpleNamespace(text="world"),
    ])
    assert _result_text(result) == "hello\nworld"


def test_open_mcp_empty_yields_nothing():
    with _open_mcp(()) as tools:
        assert tools == []


class _StubSession:
    """Mimics mcp.ClientSession for the bridge, without any transport."""

    def __init__(self):
        self.calls = []

    async def initialize(self):
        pass

    async def list_tools(self):
        tool = SimpleNamespace(
            name="echo",
            description="Echo a message.",
            input_schema={"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]},
        )
        return SimpleNamespace(tools=[tool])

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return SimpleNamespace(content=[SimpleNamespace(text=f"echo: {args['msg']}")])


def test_bridge_with_stub_session(monkeypatch):
    stub = _StubSession()

    async def fake_connect(stack, server):
        return stub

    monkeypatch.setattr("agentsoup.mcp._connect", fake_connect)
    with _open_mcp([StdioServer("dummy", name="srv")]) as labeled:
        [(label, tool)] = labeled
        assert label == "srv"
        assert tool.name == "echo"
        assert tool.parameters["required"] == ["msg"]
        assert tool.invoke({"msg": "hi"}) == "echo: hi"
    assert stub.calls == [("echo", {"msg": "hi"})]


def test_merge_prefixes_collisions_with_static_and_mcp_tools():
    from agentsoup import Tool
    from agentsoup.llm import _merge_tools

    static = [Tool("echo", "python echo", {"type": "object", "properties": {}}, lambda a: "py")]
    mcp_pairs = [
        ("a", Tool("echo", "mcp echo", {"type": "object", "properties": {}}, lambda a: "a")),
        ("b", Tool("echo", "mcp echo", {"type": "object", "properties": {}}, lambda a: "b")),
        ("b", Tool("other", "fine", {"type": "object", "properties": {}}, lambda a: "o")),
    ]
    merged = _merge_tools(static, mcp_pairs)
    assert sorted(t.name for t in merged) == ["a__echo", "b__echo", "echo", "other"]
    # invokers stay attached to the right server
    by_name = {t.name: t for t in merged}
    assert by_name["a__echo"].invoke({}) == "a" and by_name["echo"].invoke({}) == "py"


def test_real_stdio_server(tmp_path):
    """End-to-end against a real MCP server subprocess over stdio."""
    server_py = tmp_path / "server.py"
    server_py.write_text(
        "from mcp.server.mcpserver import MCPServer\n"
        "mcp = MCPServer('demo')\n"
        "@mcp.tool()\n"
        "def add(a: int, b: int) -> int:\n"
        "    'Add two numbers.'\n"
        "    return a + b\n"
        "mcp.run()\n"
    )
    import sys

    with _open_mcp([StdioServer(sys.executable, [str(server_py)])]) as labeled:
        add = next(t for _, t in labeled if t.name == "add")
        assert add.description == "Add two numbers."
        assert "5" in add.invoke({"a": 2, "b": 3})
