# 🥣 AgentSoup

**Mix prompts, models, and logic — cook up LLM-powered functions with ease.**

Write a Python function, return the prompt, get back a typed result. AgentSoup turns functions into LLM calls, agents with tools, and trackable pipelines — all with the same tiny interface.

```bash
pip install agentsoup
```

Works with any provider [litellm](https://docs.litellm.ai) supports (OpenAI, Anthropic, Gemini, …) — set the matching API key env var and pass the model name.

## The idea: return the prompt

```python
from agentsoup import llm
from pydantic import BaseModel

class Book(BaseModel):
    title: str
    author: str

@llm(model="gpt-4.1")
def recommend(topic: str) -> Book:
    return f"Recommend one great book about {topic}"

book = recommend("octopuses")   # Book(title='...', author='...')
```

The function body builds the prompt; the return **type hint** picks the output: `-> str` (or none) returns text, a pydantic model returns a parsed instance, and `-> CompleteResponse[T]` returns `(parsed, raw_completion)`.

Transient provider errors (rate limits, timeouts, connection/5xx) are retried automatically with exponential backoff and jitter — `retries=3` by default, tune per function (`retries=5`) or per call site via `with_options`; `retries=0` disables.

Set a default system prompt on the decorator with `system=` — it's prepended to every call, and overridden whenever the return value includes its own `system(...)` message (or via `with_options(system=...)`):

```python
@llm(model="gpt-4.1", system="You are a terse librarian.")
def recommend(topic: str) -> Book:
    return f"Recommend one great book about {topic}"
```

Any decorated function can be re-tuned without re-defining it — `with_options` returns a copy with merged parameters (any litellm kwarg; on agents also `tools`, `mcp_servers`, `max_turns`):

```python
cheap = recommend.with_options(model="gpt-4.1-mini", temperature=0)
cheap("octopuses")
```

## Multimodal: just return the pieces

Strings, `pathlib.Path`s, and media URLs mix freely in a returned tuple — each becomes the right content part automatically (images, video, audio, PDFs; MIME type detected from the file). Local files must be `Path` objects — bare strings are always sent as text, never sniffed for file paths:

```python
from pathlib import Path
from agentsoup import llm, system

@llm(model="gemini/gemini-2.5-flash")
def analyze(img: str, clip: str) -> str:
    return "Compare this photo and video:", Path(img), Path(clip)

@llm(model="gpt-4.1")
def summarize_pdf(url: str) -> str:
    return system("You are terse."), "Summarize:", url   # e.g. https://x.com/doc.pdf
```

Explicit part/message types (`Text`, `Image`, `Video`, `Audio`, `File`, `system(...)`, `user(...)`, `assistant(...)`) are there when you want control — each media class takes a local path or URL.

## Agents: the same decorator, plus tools

There is exactly one decorator — `@agent` is an alias of `@llm`. Add `tools=` and the model can call them in a loop until it has an answer. A tool is any typed Python function with a docstring:

```python
from agentsoup import agent

def get_weather(city: str, units: str = "c") -> str:
    """Look up current weather for a city."""
    ...

@llm(model="gpt-4.1-mini")
def summarize(text: str) -> str:
    """Summarize text in two sentences."""
    return f"Summarize: {text}"

@agent(model="gpt-4.1", tools=[get_weather, summarize], max_turns=10)
def assistant(question: str) -> str:
    return question
```

Note `summarize`: **`@llm` and `@agent` functions are themselves valid tools**, so agents can delegate to sub-agents with zero extra syntax. Tool schemas are generated from signatures and type hints; tool errors are fed back to the model instead of crashing.

## Fan out with `.map()`

Every decorated function has `.map(items)` — one call per item, all in parallel, results in order. Because the function body is plain Python that runs before the LLM call, fan-out-and-summarize fits in one function:

```python
@llm(model="gpt-4.1-mini")
def summarize(chunk: str) -> str:
    return "Summarize:", chunk

@llm(model="gpt-4.1")
def report(chunks) -> str:
    return "Combine these summaries:", summarize.map(chunks)   # parallel fan-out, then one final call
```

`.map()` works the same on agents with tools (MCP sessions are opened once and shared across the whole map), and extra kwargs are forwarded to every call: `summarize.map(chunks, style="terse")`. Control it with `max_workers=` (cap concurrency for rate limits) and `return_exceptions=True` (a failed item yields its exception instead of cancelling the rest).

## MCP servers

MCP servers go in the **same `tools=` list** — as a URL string, a command string, or a config object when you need headers/env:

```python
from agentsoup import agent, HTTPServer

@agent(model="gpt-4.1", tools=[
    get_weather,                                              # python function
    "npx -y @modelcontextprotocol/server-filesystem ./docs",  # stdio MCP server
    "https://mcp.example.com/mcp",                            # HTTP MCP server
    HTTPServer("https://mcp.linear.app/mcp", headers={"Authorization": "Bearer ..."}),
])
def helper(task: str) -> str:
    return "Complete this task using the available tools:", task
```

The model sees MCP tools and Python tools identically; a name collision with one of your Python tools is resolved by prefixing the MCP tool with its server label. Sessions connect when the function is called and tear down when it returns (one shared session for a whole `.map`). Set `timeout=` on a server config for long-running tools.

## Pipelines and subagents are just functions

There is no pipeline framework. A pipeline is a function that calls other functions; a subagent is an `@llm` function called by another (directly in the body, or handed to the model via `tools=`); parallelism is `.map()`:

```python
def write_article(topic):                       # the whole pipeline
    o = outline(topic)
    sections = draft_section.map(plan(o))       # fan out
    return edit(sections)                       # fan in
```

## Chat is a list

No session or context objects — history is a plain list of messages you own. Splat it into the returned tuple; the trailing loose value becomes the new user message:

```python
from agentsoup import llm, user, assistant

@llm(model="gpt-4.1", system="You are a helpful assistant.")
def reply(history, msg: str) -> str:
    return *history, msg

history = []
while (msg := input("> ")):
    answer = reply(history, msg)
    print(answer)
    history += [user(msg), assistant(answer)]
```

The loop above carries answers but not the agent's internal tool activity. To continue with **full context** — reasoning, tool calls, and tool results included — annotate `-> CompleteResponse[...]` and make the returned transcript the next history:

```python
@agent(model="gpt-4.1", tools=[search], system="You are a research assistant.")
def turn(history, msg: str) -> CompleteResponse[str]:
    return *history, msg

history = []
while (msg := input("> ")):
    resp = turn(history, msg)
    print(resp.parsed_response)
    history = resp.messages     # full transcript: the next turn sees every tool call + result
```

Windowing is `history[-20:]`; branching a conversation is copying the list (`turn.map` over variants works too); persistence is `json.dumps([m.to_openai_format() for m in history])` and back via `Message.from_openai_format`. Structured outputs mid-conversation just work — `assistant(some_pydantic_obj)` serializes it as JSON.

## Validation is just a loop

There is no requirements/verifier framework either — a judge is just another `@llm` function, and validate-and-repair is a `for` loop:

```python
class Verdict(BaseModel):
    passed: bool
    feedback: str

@llm(model="gpt-4.1-mini")
def judge(draft: str, rules: str) -> Verdict:
    return f"Check this draft against the rules: {rules}", draft

def reliable_write(topic, rules, budget=3):
    feedback = ""
    for _ in range(budget):
        draft = write(topic, feedback)
        verdict = judge(draft, rules)
        if verdict.passed:
            return draft
        feedback = verdict.feedback
    return draft
```

Deterministic checks are an `if`; a judge panel is `judge.map(...)`; best-of-N is `write.map([topic] * 5)` plus picking the winner; escalation is `write.with_options(model=...)` on the last attempt.

## Tracking: watch any run from anywhere

Wrap any code in `track()` and every `@llm`/`@agent` call inside — nested, parallel, agent-in-agent — is recorded:

```python
from agentsoup import track, load_run, list_runs

with track(webhook_url="https://example.com/hook") as run:
    write_article("octopus intelligence")

print(run.state_path)      # .agentsoup/runs/<run_id>.json
```

The state file is updated atomically after every call (name, status, timings, output, error), so another process can watch progress live with `load_run(path)` / `list_runs(dir)`. The webhook receives `run_started` / `call_started` / `call_finished` / `call_failed` / `run_finished` events as they happen. The active run is context-local, so concurrent runs in different threads stay isolated (and it follows into `.map` workers); tracking I/O failures disable tracking with a logged warning — they never break the run itself.

## API summary

| | |
|---|---|
| `@llm(model, system=, tools=, max_turns=, retries=, **litellm_kwargs)` | the one decorator: return value → prompt, return hint → output type, `tools=` → agent loop |
| `@agent` | alias of `@llm` — reads better when tools are involved |
| `tools=[...]` | functions, `@llm` functions, `Tool` objects, MCP servers (config, URL, or command string) — all in one list |
| `StdioServer` / `HTTPServer` | MCP server configs, for when you need headers/env |
| `track()`, `load_run`, `list_runs` | record every call in a block to a state file + webhook |
| `Text`, `Image`, `Video`, `Audio`, `File`, `system/user/assistant` | explicit content when you want it |
| `CompleteResponse[T]` | also get the raw completion + full transcript (`.messages`) for full-context continuation |
| `fn.map(items, **kwargs)` | call once per item, in parallel; ordered list of results |
| `fn.with_options(**overrides)` | copy of a decorated function with changed parameters |

## Development

```bash
pip install -e ".[dev]"
pytest
```

MIT licensed.
