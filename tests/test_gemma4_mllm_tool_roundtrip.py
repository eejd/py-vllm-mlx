# SPDX-License-Identifier: Apache-2.0
"""Regression tests for #82 — the --mllm message builder must preserve the
tool-call/tool-result round-trip across turns for templates (e.g. Gemma) that
don't render ``tool_calls`` / ``role:"tool"`` natively.
"""

from types import SimpleNamespace

from vllm_mlx.models.mllm import (
    _build_mllm_chat_messages,
    _coerce_tool_call,
    _render_gemma_tool_call_args,
    _serialize_tool_messages_gemma_native,
    _template_supports_tool_role,
)


def _assistant_tool_call(name, arguments, *, content=""):
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _build(messages):
    """Run the serializer then the real chat-message builder, as the path does."""
    serialized = _serialize_tool_messages_gemma_native(messages)
    return serialized, _build_mllm_chat_messages(
        serialized, all_image_urls=[], video_frame_counts={}
    )


# --- core round-trip (issue's canonical repro) --------------------------------


def test_multiturn_roundtrip_preserves_call_and_result():
    messages = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "how many movies in radarr?"},
        _assistant_tool_call("radarr_get_movies", '{"search": "Dune"}'),
        {"role": "tool", "tool_call_id": "call_1", "content": "694 movies"},
        {"role": "user", "content": "thanks"},
    ]
    serialized, chat_messages = _build(messages)

    # The assistant tool-call turn must NOT be dropped (the bug): it now carries
    # a tool_code block, so _build_mllm_chat_messages keeps it.
    assistant = [m for m in chat_messages if m["role"] == "assistant"]
    assert len(assistant) == 1
    assert "```tool_code" in assistant[0]["content"]
    assert 'radarr_get_movies(search=' in assistant[0]["content"]

    # The tool result survives as a tool_output block inside a user turn.
    rendered = "\n".join(str(m["content"]) for m in chat_messages)
    assert "```tool_output" in rendered
    assert "694 movies" in rendered

    # No role:"tool" leaks through to the template.
    assert all(m["role"] != "tool" for m in chat_messages)


def test_call_emitted_before_result():
    messages = [
        _assistant_tool_call("arr_status", "{}"),
        {"role": "tool", "tool_call_id": "call_1", "content": "healthy"},
    ]
    serialized, _ = _build(messages)
    roles = [m["role"] for m in serialized]
    assert roles == ["assistant", "user"]
    assert "```tool_code" in serialized[0]["content"]
    assert "```tool_output" in serialized[1]["content"]


# --- argument fidelity --------------------------------------------------------


def test_arg_types_roundtrip_via_repr():
    # repr keeps values ast.literal_eval-able for the gemma4 output parser.
    rendered = _render_gemma_tool_call_args(
        '{"q": "Dune", "limit": 5, "watched": true}'
    )
    assert "q='Dune'" in rendered
    assert "limit=5" in rendered
    assert "watched=True" in rendered


def test_multi_call_assistant_turn():
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": "f", "arguments": '{"a": 1}'}},
            {"function": {"name": "g", "arguments": '{"b": "x"}'}},
        ],
    }
    serialized = _serialize_tool_messages_gemma_native([msg])
    block = serialized[0]["content"]
    assert "f(a=1)" in block
    assert "g(b='x')" in block


def test_assistant_text_preserved_before_block():
    msg = _assistant_tool_call("f", "{}", content="let me check")
    serialized = _serialize_tool_messages_gemma_native([msg])
    content = serialized[0]["content"]
    assert content.startswith("let me check")
    assert "```tool_code" in content


def test_consecutive_tool_results_merged():
    messages = [
        _assistant_tool_call("f", "{}"),
        {"role": "tool", "content": "r1"},
        {"role": "tool", "content": "r2"},
    ]
    serialized = _serialize_tool_messages_gemma_native(messages)
    outputs = [m for m in serialized if "```tool_output" in str(m.get("content", ""))]
    assert len(outputs) == 1
    assert "r1" in outputs[0]["content"] and "r2" in outputs[0]["content"]


# --- pydantic-object tool_calls ----------------------------------------------


class _FakePydantic:
    """Minimal stand-in for a pydantic v2 model (exposes model_dump)."""

    def __init__(self, **data):
        self._data = data

    def model_dump(self, exclude_none=False):
        return {
            k: (v.model_dump(exclude_none=exclude_none) if isinstance(v, _FakePydantic) else v)
            for k, v in self._data.items()
            if not (exclude_none and v is None)
        }


def test_pydantic_tool_call_input():
    fn = _FakePydantic(name="radarr_get_movies", arguments='{"search": "Dune"}')
    call = _FakePydantic(id="call_1", type="function", function=fn)
    coerced = _coerce_tool_call(call)
    assert coerced is not None
    name, args = coerced
    assert name == "radarr_get_movies"
    assert "search='Dune'" in args


# --- gating: pass through when template renders tools -------------------------


def test_passthrough_when_template_supports_tools():
    processor = SimpleNamespace(
        chat_template="{% if message.role == 'tool' %}...{% endif %}"
    )
    config = SimpleNamespace(model_type="qwen3")
    assert _template_supports_tool_role(processor, config) is True


def test_gemma_template_needs_serialization():
    # Gemma-style template: only user/model, no tool handling.
    processor = SimpleNamespace(
        chat_template="{% for m in messages %}<start_of_turn>{{ m.role }}{% endfor %}"
    )
    config = SimpleNamespace(model_type="gemma4")
    assert _template_supports_tool_role(processor, config) is False


def test_unknown_template_falls_back_to_model_type():
    processor = SimpleNamespace(chat_template=None, tokenizer=SimpleNamespace(chat_template=None))
    assert _template_supports_tool_role(processor, SimpleNamespace(model_type="gemma4")) is False
    assert _template_supports_tool_role(processor, SimpleNamespace(model_type="qwen3")) is True


def test_non_tool_messages_unchanged():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
    ]
    serialized = _serialize_tool_messages_gemma_native(messages)
    assert serialized == messages
