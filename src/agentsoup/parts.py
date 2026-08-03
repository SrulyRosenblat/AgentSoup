"""Content parts, messages, and coercion of loose values into prompts."""
import base64
import json
import mimetypes
import os
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel


def _is_url(src) -> bool:
    return isinstance(src, str) and src.startswith(("http://", "https://"))


def _data_uri(path, fallback_mime: str) -> str:
    mime = mimetypes.guess_type(str(path))[0] or fallback_mime
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


class MessagePart:
    def __init__(self, **content):
        self.content = content

    def to_openai_format(self) -> dict:
        return self.content

    def __repr__(self):
        return str(self.content)


class Text(MessagePart):
    def __init__(self, text: str):
        super().__init__(type="text", text=text)


class Image(MessagePart):
    """An image from a local path or an http(s) URL."""

    def __init__(self, src):
        url = src if _is_url(src) else _data_uri(src, "image/jpeg")
        super().__init__(type="image_url", image_url={"url": url})


class Video(MessagePart):
    """A video from a local path or URL. Provider support is narrow (Gemini via litellm)."""

    def __init__(self, src):
        if _is_url(src):
            super().__init__(type="image_url", image_url={"url": src})
        else:
            super().__init__(type="file", file={"file_data": _data_uri(src, "video/mp4")})


class Audio(MessagePart):
    """A local audio file, sent in OpenAI input_audio format. URLs are not supported."""

    def __init__(self, src):
        if _is_url(src):
            raise ValueError("Audio does not support URLs; download the file first")
        with open(src, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        fmt = Path(str(src)).suffix.lstrip(".").lower() or "wav"
        super().__init__(type="input_audio", input_audio={"data": data, "format": fmt})


class File(MessagePart):
    """A PDF or other document from a local path or URL."""

    def __init__(self, src, mime: str | None = None):
        if _is_url(src):
            super().__init__(type="file", file={"file_id": src})
        else:
            super().__init__(
                type="file",
                file={
                    "filename": os.path.basename(str(src)),
                    "file_data": _data_uri(src, mime or "application/octet-stream"),
                },
            )


# message fields a provider accepts on the way *in*; everything else a response
# carries (reasoning_content, annotations, provider_specific_fields, ...) is
# dropped when a transcript is replayed as history.
REPLAYABLE_FIELDS = frozenset({"tool_calls", "tool_call_id", "name", "thinking_blocks"})


class Message:
    def __init__(self, role: str, parts, **extra):
        self.role = role
        self.parts = list(parts)
        self.extra = extra  # passthrough: tool_calls, tool_call_id, reasoning_content, ...

    def to_openai_format(self) -> dict:
        content = [p.to_openai_format() for p in self.parts] if self.parts else None
        return {"role": self.role, "content": content, **self.extra}

    @classmethod
    def from_openai_format(cls, d: dict) -> "Message":
        """Rebuild a Message from an OpenAI-format dict (e.g. a transcript or
        persisted history). Response-only fields that providers reject on the
        way back in (reasoning_content, annotations, ...) are dropped."""
        content = d.get("content")
        if isinstance(content, str):
            parts = [Text(content)]
        elif content is None:
            parts = []
        else:
            parts = [MessagePart(**p) for p in content]
        extra = {
            k: v for k, v in d.items()
            if k in REPLAYABLE_FIELDS and v is not None
        }
        return cls(d["role"], parts, **extra)

    def __repr__(self):
        return str(self.to_openai_format())


def system(*items) -> Message:
    return Message("system", [coerce_item(i) for i in items])


def user(*items) -> Message:
    return Message("user", [coerce_item(i) for i in items])


def assistant(*items) -> Message:
    return Message("assistant", [coerce_item(i) for i in items])


_MEDIA_PREFIXES = ("image/", "video/", "audio/")


def _part_for(src, mime: str) -> MessagePart:
    if mime.startswith("image/"):
        return Image(src)
    if mime.startswith("video/"):
        return Video(src)
    if mime.startswith("audio/"):
        return Audio(src)  # raises a clear error for URLs
    return File(src, mime=mime)


def _media_mime(path_like) -> str | None:
    mime = mimetypes.guess_type(str(path_like))[0]
    if mime and (mime.startswith(_MEDIA_PREFIXES) or mime == "application/pdf"):
        return mime
    return None


def coerce_item(item) -> MessagePart:
    """Turn one loose value into a MessagePart."""
    if isinstance(item, MessagePart):
        return item
    if isinstance(item, Path):
        if not item.exists():
            raise ValueError(f"File not found: {item}")
        mime = mimetypes.guess_type(str(item))[0] or "application/octet-stream"
        return _part_for(str(item), mime)
    if isinstance(item, str):
        # Bare strings are always text — use pathlib.Path for local files.
        # (Sniffing os.path.exists on prompt strings would silently upload any
        # local file a user-supplied string happens to name.)
        if _is_url(item):
            mime = _media_mime(urlparse(item).path)
            if mime:
                return _part_for(item, mime)
        return Text(item)
    if isinstance(item, BaseModel):
        return Text(item.model_dump_json())
    if isinstance(item, (dict, list)):
        return Text(json.dumps(item, default=str))
    return Text(str(item))


def coerce(value) -> list[Message]:
    """Turn a decorated function's return value into a list of messages.

    Accepts a str, a Message, a tuple/list mixing Messages, MessageParts, Paths,
    strings, and other loose values — e.g. ``return prompt, image, video``.
    """
    if value is None:
        raise ValueError("Prompt function returned None; return a str, Message, or tuple of content")
    if isinstance(value, Message):
        return [value]
    if isinstance(value, str):
        return [Message("user", [Text(value)])]
    if isinstance(value, Iterator):  # generators etc. — don't stringify the object
        value = tuple(value)
    if not isinstance(value, (list, tuple)):
        value = (value,)
    if not value:
        raise ValueError("Prompt function returned an empty sequence")
    messages, pending = [], []
    for item in value:
        if isinstance(item, Message):
            if pending:
                messages.append(Message("user", pending))
                pending = []
            messages.append(item)
        else:
            pending.append(coerce_item(item))
    if pending:
        messages.append(Message("user", pending))
    return messages
