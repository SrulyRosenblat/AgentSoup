"""AgentSoup — mix prompts, models, and logic; cook up LLM-powered functions with ease."""
from .llm import AgentMaxTurnsError, CompleteResponse, Tool, agent, llm
from .mcp import HTTPServer, StdioServer
from .parts import (
    Audio,
    File,
    Image,
    Message,
    MessagePart,
    Text,
    Video,
    assistant,
    coerce,
    system,
    user,
)
from .tracking import list_runs, load_run, track

__version__ = "0.2.0"

__all__ = [
    "llm", "agent", "Tool", "CompleteResponse", "AgentMaxTurnsError",
    "StdioServer", "HTTPServer",
    "MessagePart", "Text", "Image", "Video", "Audio", "File",
    "Message", "system", "user", "assistant", "coerce",
    "track", "load_run", "list_runs",
]
