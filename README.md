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

Any decorated function can be re-tuned without re-defining it — `with_options` returns a copy with merged parameters (any litellm kwarg; on agents also `tools`, `mcp_servers`, `max_turns`):

```python
cheap = recommend.with_options(model="gpt-4.1-mini", temperature=0)
cheap("octopuses")
```

## Multimodal: just return the pieces

Strings, file paths, and URLs mix freely in a returned tuple — each becomes the right content part automatically (images, video, audio, PDFs; MIME type detected from the file):

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

The model sees MCP tools and Python tools identically. Sessions connect when the function is called and tear down when it returns.

## Pipelines: a file of steps, tracked from anywhere

Mark functions in a file with `@step`; they run in order, each step's output feeding the next. Steps can be plain functions, `@llm`, or `@agent` (put `@step` on top):

```python
# steps.py
from agentsoup import step, llm

@step
@llm(model="gpt-4.1")
def outline(topic: str) -> str:
    return f"Write an outline about {topic}"

@step
@llm(model="gpt-4.1")
def draft(outline: str) -> str:
    return "Write a full draft from this outline:", outline
```

Steps can also pull **any earlier step's output by name** — name a parameter after a step and it receives that step's result; one leftover required parameter still gets the previous step's output:

```python
@step
def publish(draft, outline):        # gets draft's AND outline's outputs
    return {"outline": outline, "article": draft}
```

**Named parameters are also the dependency graph.** Steps whose parameters all name earlier steps run as soon as those steps finish — so independent branches run **in parallel**, no extra syntax:

```python
@step
def outline(topic): ...

@step
def facts(outline): ...        # these two only depend on outline,
@step
def quotes(outline): ...       # so they run concurrently

@step
def article(facts, quotes):    # waits for both branches
    ...
```

**Fan out** over a list with `@step(fan_out=True)` — the step runs once per item, in parallel, and its output is the list of results. Perfect for fanning out agents:

```python
@step
@llm(model="gpt-4.1")
def subtopics(topic) -> list:
    return f"List 5 subtopics of {topic} as a JSON array"

@step(fan_out=True)
@agent(model="gpt-4.1", tools=[search])
def research(subtopics):               # called once per subtopic, concurrently
    return "Research this subtopic:", subtopics

@step
def combine(research):                 # gets the list of all agent answers
    ...
```

(`@step(name="...")` renames a step; `@step(order=n)` overrides definition order; `@step(description="...")` — or the docstring — is recorded in the run state; `run(max_workers=n)` caps concurrency.)

Run it from Python or the CLI:

```bash
agentsoup run steps.py --input "octopus intelligence" --webhook https://example.com/hook
agentsoup status                 # list runs:  steps-20260802-101502-a3f9c1  done  2/2 steps ...
agentsoup status <run_id>        # per-step detail (--json for raw state)
```

```python
from agentsoup import Pipeline, load_run, list_runs

result = Pipeline.from_file("steps.py").run(input="octopus intelligence")
state = load_run(result.state_path)          # from any process, any time
```

Every run writes a JSON state file to `.agentsoup/runs/<run_id>.json` — updated atomically after each step with status, timings, outputs, and errors — so completion is trackable from another process. Pass `webhook_url=` (or `--webhook`) to also POST `run_started` / `step_finished` / `run_failed` … events as the run progresses.

## API summary

| | |
|---|---|
| `@llm(model, tools=, max_turns=, **litellm_kwargs)` | the one decorator: return value → prompt, return hint → output type, `tools=` → agent loop |
| `@agent` | alias of `@llm` — reads better when tools are involved |
| `tools=[...]` | functions, `@llm` functions, `Tool` objects, MCP servers (config, URL, or command string) — all in one list |
| `StdioServer` / `HTTPServer` | MCP server configs, for when you need headers/env |
| `@step`, `Pipeline.from_file`, `load_run`, `list_runs` | pipelines + tracking |
| `Text`, `Image`, `Video`, `Audio`, `File`, `system/user/assistant` | explicit content when you want it |
| `CompleteResponse[T]` | also get the raw completion |
| `fn.with_options(**overrides)` | copy of a decorated function with changed parameters |

## Development

```bash
pip install -e ".[dev]"
pytest
```

MIT licensed.
