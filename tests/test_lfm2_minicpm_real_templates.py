# SPDX-License-Identifier: Apache-2.0
"""Round-trip the parsers against the models' own chat templates.

Hand-written samples only prove a parser reads what its author expected. These tests
render an assistant turn with ``tool_calls`` through the real LFM2.5 / MiniCPM5 chat
template (the format the model was trained to emit) and require the parser to recover
the exact arguments. They need the model's tokenizer files in the local Hugging Face
cache and skip when they are absent (CI has no model cache).
"""

import glob
import json
import os

import pytest

pytest.importorskip("transformers")

from vllm_mlx.tool_parsers.lfm2_tool_parser import Lfm2ToolParser  # noqa: E402
from vllm_mlx.tool_parsers.minicpm_tool_parser import MiniCPMToolParser  # noqa: E402

_HUB = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
)

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
                "properties": {
                    "city": {"type": "string"},
                    "units": {"type": "string"},
                    "days": {"type": "integer"},
                    "verbose": {"type": "boolean"},
                    "opts": {"type": "object"},
                    "tags": {"type": "array"},
                    "ratio": {"type": "number"},
                    "note": {"type": "string"},
                },
            },
        },
    }
]


def _snapshot(repo: str):
    paths = sorted(
        glob.glob(
            os.path.join(_HUB, f"models--{repo.replace('/', '--')}", "snapshots", "*")
        )
    )
    if not paths:
        pytest.skip(f"{repo} is not in the local Hugging Face cache")
    return paths[-1]


def _rendered_assistant_turn(repo: str) -> str:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(_snapshot(repo))
    messages = [
        {"role": "user", "content": "What is the weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": _ARGS_RICH},
                },
                {
                    "id": "2",
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
    return text[text.rfind("assistant") + len("assistant") :]


@pytest.mark.parametrize(
    "repo,parser_cls",
    [
        ("mlx-community/LFM2.5-2.6B-4bit", Lfm2ToolParser),
        ("mlx-community/MiniCPM5-2B-mlx-4Bit", MiniCPMToolParser),
    ],
)
def test_parser_recovers_arguments_the_template_rendered(repo, parser_cls):
    turn = _rendered_assistant_turn(repo)
    result = parser_cls().extract_tool_calls(turn, {"tools": _TOOLS})

    assert result.tools_called is True
    assert [c["name"] for c in result.tool_calls] == ["get_weather", "get_weather"]
    assert [json.loads(c["arguments"]) for c in result.tool_calls] == [
        _ARGS_RICH,
        _ARGS_PLAIN,
    ]


@pytest.mark.parametrize(
    "repo,parser_cls",
    [
        ("mlx-community/LFM2.5-2.6B-4bit", Lfm2ToolParser),
        ("mlx-community/MiniCPM5-2B-mlx-4Bit", MiniCPMToolParser),
    ],
)
@pytest.mark.parametrize("size", [1, 7, 10_000])
def test_streaming_emits_each_rendered_call_once(repo, parser_cls, size):
    turn = _rendered_assistant_turn(repo)
    parser = parser_cls()
    parser.reset()
    calls: dict[int, dict] = {}
    seen: dict[int, int] = {}
    acc = ""
    for i in range(0, len(turn), size):
        delta = turn[i : i + size]
        previous, acc = acc, acc + delta
        out = parser.extract_tool_calls_streaming(
            previous, acc, delta, request={"tools": _TOOLS}
        )
        for tc in (out or {}).get("tool_calls") or []:
            slot = calls.setdefault(tc["index"], {"name": "", "arguments": ""})
            slot["name"] += tc["function"].get("name") or ""
            slot["arguments"] += tc["function"].get("arguments") or ""
            seen[tc["index"]] = seen.get(tc["index"], 0) + 1

    assert sorted(calls) == [0, 1]
    assert all(n == 1 for n in seen.values())
    assert [json.loads(calls[i]["arguments"]) for i in (0, 1)] == [
        _ARGS_RICH,
        _ARGS_PLAIN,
    ]
