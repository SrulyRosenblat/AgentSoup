import base64

import pytest
from pydantic import BaseModel

from agentsoup import Audio, File, Image, Message, Text, Video, coerce, system, user
from agentsoup.parts import coerce_item


@pytest.fixture
def png(tmp_path):
    p = tmp_path / "pic.png"
    p.write_bytes(b"\x89PNG fake")
    return p


def part_dicts(messages):
    return [m.to_openai_format() for m in messages]


def test_str_is_a_user_text_message():
    [msg] = coerce("hello")
    assert msg.to_openai_format() == {"role": "user", "content": [{"type": "text", "text": "hello"}]}


def test_single_message_passes_through():
    m = user("hi")
    assert coerce(m) == [m]


def test_tuple_of_loose_values_becomes_one_user_message(png):
    [msg] = coerce(("describe:", png))
    types = [p.content["type"] for p in msg.parts]
    assert msg.role == "user" and types == ["text", "image_url"]


def test_message_inside_tuple_flushes_pending_parts_in_order():
    msgs = coerce(("first", system("be brief"), "second"))
    assert [(m.role, m.parts[0].content["text"]) for m in msgs] == [
        ("user", "first"), ("system", "be brief"), ("user", "second"),
    ]


def test_list_of_messages_passes_through():
    ms = [system("a"), user("b")]
    assert coerce(ms) == ms


def test_none_and_empty_raise():
    with pytest.raises(ValueError):
        coerce(None)
    with pytest.raises(ValueError):
        coerce(())


def test_missing_path_raises(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        coerce_item(tmp_path / "nope.png")


def test_local_image_mime_guessed(png):
    part = coerce_item(png)
    url = part.content["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x89PNG fake"


def test_url_strings_by_extension():
    assert coerce_item("https://x.com/a.png").content["type"] == "image_url"
    assert coerce_item("https://x.com/a.mp4").content["image_url"]["url"] == "https://x.com/a.mp4"
    assert coerce_item("https://x.com/a.pdf").content == {"type": "file", "file": {"file_id": "https://x.com/a.pdf"}}
    assert coerce_item("https://x.com/page").content["type"] == "text"  # no media extension


def test_prose_containing_a_filename_stays_text(png):
    s = f"look at {png} please"
    assert coerce_item(s).content["type"] == "text"


def test_bare_string_paths_are_never_sniffed(png):
    # strings are always text — only Path objects load local files
    assert coerce_item(str(png)).content["type"] == "text"


def test_generator_return_values_coerce(png):
    [msg] = coerce(p for p in ["describe:", png])
    assert [p.content["type"] for p in msg.parts] == ["text", "image_url"]


def test_video_and_audio_local(tmp_path):
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"vid")
    a = tmp_path / "voice.mp3"
    a.write_bytes(b"aud")
    assert coerce_item(v).content["type"] == "file"
    assert coerce_item(v).content["file"]["file_data"].startswith("data:video/mp4;base64,")
    audio = coerce_item(a).content
    assert audio["type"] == "input_audio" and audio["input_audio"]["format"] == "mp3"


def test_audio_url_unsupported():
    with pytest.raises(ValueError):
        Audio("https://x.com/a.mp3")


def test_pdf_filename_is_basename(tmp_path):
    p = tmp_path / "deep" / "doc.pdf"
    p.parent.mkdir()
    p.write_bytes(b"%PDF")
    assert coerce_item(p).content["file"]["filename"] == "doc.pdf"


def test_pydantic_and_dict_become_json_text():
    class M(BaseModel):
        x: int

    assert coerce_item(M(x=1)).content == {"type": "text", "text": '{"x":1}'}
    assert coerce_item({"a": 1}).content == {"type": "text", "text": '{"a": 1}'}


def test_single_loose_value_coerces(png):
    [msg] = coerce(png)
    assert msg.parts[0].content["type"] == "image_url"


def test_from_openai_format_round_trips():
    cases = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "http://x/a.png"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result text"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
         "reasoning_content": "thinking..."},
    ]
    for d in cases:
        m = Message.from_openai_format(d)
        out = m.to_openai_format()
        assert out["role"] == d["role"]
        if isinstance(d["content"], str):
            assert out["content"] == [{"type": "text", "text": d["content"]}]  # semantic round trip
        else:
            assert out["content"] == d["content"]
        for k, v in d.items():
            if k not in ("role", "content"):
                assert out[k] == v


def test_from_openai_format_strips_none_extras():
    m = Message.from_openai_format({"role": "assistant", "content": "hi", "function_call": None, "audio": None})
    assert m.extra == {}
    assert m.to_openai_format() == {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}
