# SPDX-License-Identifier: Apache-2.0
"""Round-trip the ``mistral`` tool parser against Ministral-3's own chat template.

Scope: a template/parser round-trip *regression* test only. It is **not** proof of
the Ministral-3 serving fix: it builds ``MistralToolParser()`` without a tokenizer
and never loads a model, so it passes on the commit before the dispatch and
vocabulary fixes. Those are guarded by ``test_text_model_dispatch.py`` and
``test_tool_parsers.py``.

The template renders an assistant tool call as
``[TOOL_CALLS]name[ARGS]{json}`` (the format the model is trained to emit). These
tests render two calls through the real template and require the parser to
recover the exact arguments, whole and token-streamed. They need the model's
tokenizer files in the local Hugging Face cache (``HF_HOME``) and skip, with the
reason stated, when they are absent: CI has no model cache, so a green skip there
says nothing about the template.
"""

import glob
import json
import os

import pytest

pytest.importorskip("transformers")

from vllm_mlx.tool_parsers.mistral_tool_parser import MistralToolParser  # noqa: E402

_HUB = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
)
_REPO = "mlx-community/Ministral-3-8B-Instruct-2512-4bit"

_ARGS_RICH = {
    "city": "Paris",
    "units": "metric",
    "days": 3,
    "verbose": True,
    "opts": {"a": [1, None, True], "b": "x'y"},
    "tags": ["a", "b"],
    "ratio": 0.5,
    "note": "it's a\nnew line",
}
_ARGS_PLAIN = {"city": "Tokyo"}

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    }
]


def _snapshot() -> str:
    paths = sorted(
        glob.glob(
            os.path.join(_HUB, f"models--{_REPO.replace('/', '--')}", "snapshots", "*")
        )
    )
    paths = [p for p in paths if os.path.exists(os.path.join(p, "chat_template.jinja"))]
    if not paths:
        pytest.skip(
            f"{_REPO} chat_template.jinja not found under {_HUB}: the Ministral-3 "
            "template round-trip was NOT exercised (set HF_HOME to a cache that "
            "has the model)"
        )
    return paths[-1]


def _rendered_assistant_turn() -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_snapshot())
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "abc123def",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": _ARGS_RICH},
                },
                {
                    "id": "abc123deg",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": _ARGS_PLAIN},
                },
            ],
        },
    ]
    text = tokenizer.apply_chat_template(
        messages, tools=_TOOLS, tokenize=False, add_generation_prompt=False
    )
    # Keep only the assistant turn the model is trained to produce.
    turn = text[text.rfind("[/INST]") + len("[/INST]") :]
    assert turn.startswith("[TOOL_CALLS]get_weather[ARGS]"), turn
    return turn.removesuffix("</s>")


def test_parser_recovers_arguments_the_template_rendered():
    turn = _rendered_assistant_turn()
    result = MistralToolParser().extract_tool_calls(turn, {"tools": _TOOLS})

    assert result.tools_called is True
    assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_weather"]
    assert [json.loads(c["arguments"]) for c in result.tool_calls] == [
        _ARGS_RICH,
        _ARGS_PLAIN,
    ]


def _token_deltas(turn: str) -> list[str]:
    """Split ``turn`` the way the server streams it: one delta per generated token.

    ``[TOOL_CALLS]`` and ``[ARGS]`` are single special tokens, so they always
    arrive whole; cutting the text at arbitrary characters would test a stream
    the model cannot produce.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_snapshot())
    ids = tokenizer.encode(turn, add_special_tokens=False)
    deltas, decoded = [], ""
    for end in range(1, len(ids) + 1):
        text = tokenizer.decode(ids[:end], skip_special_tokens=False)
        deltas.append(text[len(decoded) :])
        decoded = text
    assert "".join(deltas) == turn
    return deltas


def test_streaming_emits_each_rendered_call_once():
    turn = _rendered_assistant_turn()
    parser = MistralToolParser()
    parser.reset()
    calls: dict[int, dict] = {}
    acc = ""
    for delta in _token_deltas(turn):
        previous, acc = acc, acc + delta
        out = parser.extract_tool_calls_streaming(
            previous, acc, delta, request={"tools": _TOOLS}
        )
        for tc in (out or {}).get("tool_calls") or []:
            slot = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
            slot["name"] += tc["function"].get("name") or ""
            slot["arguments"] += tc["function"].get("arguments") or ""

    assert sorted(calls) == [0, 1]
    assert [calls[i]["name"] for i in (0, 1)] == ["get_weather", "get_weather"]
    assert [json.loads(calls[i]["arguments"]) for i in (0, 1)] == [
        _ARGS_RICH,
        _ARGS_PLAIN,
    ]
