"""Deep research: plan -> fan out search agents -> sometimes dig deeper -> synthesize.

Run:  OPENAI_API_KEY=... python examples/deep_research.py "octopus intelligence"
Watch from another terminal:  python -c "from agentsoup import load_run; import sys, json; print(json.dumps(load_run(sys.argv[1]), indent=2))" .agentsoup/runs/<run_id>.json
"""
import sys

from pydantic import BaseModel

from agentsoup import agent, llm, track


# ---------- tools (any typed function with a docstring) ----------

def web_search(query: str, max_results: int = 5) -> list:
    """Search the web and return result snippets."""
    import urllib.parse, urllib.request, json
    url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.load(r)
    topics = [t.get("Text") for t in data.get("RelatedTopics", []) if t.get("Text")]
    return topics[:max_results] or [data.get("AbstractText") or "no results"]


# ---------- structured outputs ----------

class Plan(BaseModel):
    questions: list[str]        # the sub-questions worth researching


class Finding(BaseModel):
    question: str
    summary: str
    confidence: float           # 0..1 — how well the sources answered it
    followup_query: str         # what to search next if confidence is low


class Report(BaseModel):
    title: str
    summary: str
    sections: list[str]
    open_questions: list[str]


# ---------- the agents ----------

@llm(model="gpt-4.1-mini")
def plan(topic: str) -> Plan:
    return f"Break '{topic}' into 4-6 crisp research sub-questions."


@agent(model="gpt-4.1", tools=[web_search])
def research(question: str) -> Finding:
    """A search agent: uses tools in a loop until the question is answered."""
    return "Research this question thoroughly using web_search:", question


@agent(model="gpt-4.1", tools=[web_search])
def dig_deeper(finding: Finding) -> Finding:
    return (
        "This finding was low-confidence. Run the follow-up search and improve it:",
        finding,
    )


@llm(model="gpt-4.1")
def synthesize(topic: str, findings: list[Finding]) -> Report:
    return f"Write a research report on '{topic}' from these findings:", findings


# ---------- the pipeline: just a function ----------

def deep_research(topic: str) -> Report:
    questions = plan(topic).questions
    findings = research.map(questions)                     # fan out: one agent per question

    weak = [f for f in findings if f.confidence < 0.6]     # sometimes...
    if weak:
        improved = dig_deeper.map(weak)                    # ...fan out a second round
        strong = [f for f in findings if f.confidence >= 0.6]
        findings = strong + improved

    return synthesize(topic, findings)


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "octopus intelligence"
    with track() as run_id:                                # every agent call recorded
        report = deep_research(topic)
    print(report.title, "\n")
    print(report.summary, "\n")
    for section in report.sections:
        print(" -", section)
    print(f"\nrun state: .agentsoup/runs/{run_id}.json")
