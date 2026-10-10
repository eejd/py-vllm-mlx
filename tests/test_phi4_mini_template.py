# SPDX-License-Identifier: Apache-2.0
"""The shipped Phi-4-mini tool template, and the parser against what it renders.

Phi-4-mini's own chat template ignores the OpenAI ``tools=`` argument, so the model
never sees a tool unless the template is replaced (``--chat-template``). These tests
render the shipped template with jinja2 (configured the way transformers configures
it) and, when the model's tokenizer is in the local Hugging Face cache, with the real
tokenizer too; they skip that part in CI, which has no model cache.
"""

import glob
import json
import os
import re

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from vllm_mlx.templates import PHI4_MINI_TOOL_TEMPLATE
from vllm_mlx.tool_parsers.phi4_mini_tool_parser import Phi4MiniToolParser

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
]

# Arguments chosen to break naive handling: a ``]`` in a string, a nested list,
# non-ASCII, and an ``&`` that jinja's own ``tojson`` would HTML-escape.
RICH_ARGS = {
    "city": "Zürich",
    "ids": [1, [2, 3]],
    "note": "see a[0] ] & b",
    "units": {"t": "c"},
}


def _tojson(value, ensure_ascii=False, indent=None, **kwargs):
    # transformers replaces jinja's HTML-escaping ``tojson`` with plain json.dumps.
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def render(messages, tools=None, add_generation_prompt=True, **extra):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = _tojson
    template = env.from_string(PHI4_MINI_TOOL_TEMPLATE.read_text(encoding="utf-8"))
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        **extra,
    )


def _assistant_turns(prompt: str) -> list[str]:
    return re.findall(r"<\|assistant\|>(.*?)<\|end\|>", prompt, re.DOTALL)


def _tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class TestShippedFile:
    def test_keeps_license_header_and_attribution(self):
        text = PHI4_MINI_TOOL_TEMPLATE.read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: Apache-2.0" in text
        assert "Copyright contributors to the vLLM project" in text
        assert "tool_chat_template_phi4_mini.jinja" in text


class TestRenderTools:
    def test_tools_are_rendered_into_the_system_prompt(self):
        prompt = render(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "weather in Paris?"},
            ],
            tools=TOOLS,
        )
        system = prompt.split("<|end|>")[0]
        assert system.startswith("<|system|>")
        assert "Be brief." in system
        assert '"name": "get_weather"' in system
        assert '"required": [\n' in system  # tojson(indent=4)
        assert 'functools[{"name": [function name]' in system
        assert prompt.endswith("<|user|>weather in Paris?<|end|><|assistant|>")

    def test_without_tools_there_is_no_function_calling_prompt(self):
        prompt = render([{"role": "user", "content": "hi"}])
        assert "functools" not in prompt
        assert prompt == (
            "<|system|>\nYou are a helpful assistant.<|end|><|user|>hi<|end|>"
            "<|assistant|>"
        )

    def test_generation_prompt_is_not_added_when_not_requested(self):
        prompt = render(
            [{"role": "user", "content": "hi"}], add_generation_prompt=False
        )
        assert prompt.endswith("<|user|>hi<|end|>")


class TestRenderToolHistory:
    def _history(self, arguments):
        return [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("get_weather", arguments)],
            },
        ]

    @pytest.mark.parametrize("as_string", [False, True])
    def test_assistant_tool_calls_round_trip_through_the_parser(self, as_string):
        arguments = json.dumps(RICH_ARGS) if as_string else RICH_ARGS
        prompt = render(self._history(arguments), tools=TOOLS)
        (turn,) = _assistant_turns(prompt)
        assert turn.startswith("functools[")

        result = Phi4MiniToolParser().extract_tool_calls(turn)
        assert result.tools_called is True
        assert [c["name"] for c in result.tool_calls] == ["get_weather"]
        assert json.loads(result.tool_calls[0]["arguments"]) == RICH_ARGS

    def test_arguments_are_json_not_a_python_repr(self):
        prompt = render(self._history({"ok": True, "none": None}), tools=TOOLS)
        assert '"arguments": {"ok": true, "none": null}' in prompt

    def test_parallel_calls_share_one_list(self):
        messages = [
            {"role": "user", "content": "both?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    _tool_call("get_weather", {"city": "Paris"}, "a"),
                    _tool_call("get_weather", {"city": "Tokyo"}, "b"),
                ],
            },
        ]
        (turn,) = _assistant_turns(render(messages, tools=TOOLS))
        assert turn.count("functools[") == 1
        calls = Phi4MiniToolParser().extract_tool_calls(turn).tool_calls
        assert [json.loads(c["arguments"]) for c in calls] == [
            {"city": "Paris"},
            {"city": "Tokyo"},
        ]

    def test_content_alongside_tool_calls_keeps_both(self):
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "Let me check.",
                "tool_calls": [_tool_call("get_weather", {"city": "Paris"})],
            },
        ]
        (turn,) = _assistant_turns(render(messages, tools=TOOLS))
        result = Phi4MiniToolParser().extract_tool_calls(turn)
        assert result.content == "Let me check."
        assert [c["name"] for c in result.tool_calls] == ["get_weather"]

    def test_tool_role_result_json_object(self):
        prompt = render(
            [
                *self._history({"city": "Paris"}),
                {"role": "tool", "tool_call_id": "call_1", "content": '{"t": 20}'},
            ],
            tools=TOOLS,
        )
        assert '<|tools|>{"result": {"t": 20}}<|end|><|assistant|>' in prompt

    def test_tool_role_result_plain_text_stays_valid_json(self):
        prompt = render(
            [
                *self._history({"city": "Paris"}),
                {"role": "tool", "tool_call_id": "call_1", "content": 'Sunny, "20" C'},
            ],
            tools=TOOLS,
        )
        match = re.search(r"<\|tools\|>(.*?)<\|end\|>", prompt, re.DOTALL)
        assert json.loads(match.group(1)) == {"result": 'Sunny, "20" C'}

    def test_tools_role_is_rendered_like_the_tool_role(self):
        base = self._history({"city": "Paris"})
        a = render([*base, {"role": "tool", "content": "[1, 2]"}], tools=TOOLS)
        b = render([*base, {"role": "tools", "content": "[1, 2]"}], tools=TOOLS)
        assert a == b
        assert '<|tools|>{"result": [1, 2]}<|end|>' in a

    def test_full_tool_loop_ends_ready_for_the_final_answer(self):
        prompt = render(
            [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_tool_call("get_weather", {"city": "Paris"})],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": '{"t": 20}'},
            ],
            tools=TOOLS,
        )
        assert prompt.endswith('"result": {"t": 20}}<|end|><|assistant|>')


def _phi4_snapshot():
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    for repo in ("Phi-4-mini-instruct-4bit", "Phi-4-mini-instruct-6bit"):
        paths = sorted(
            glob.glob(
                os.path.join(
                    hf_home, "hub", f"models--mlx-community--{repo}", "snapshots", "*"
                )
            )
        )
        if paths and os.path.exists(os.path.join(paths[-1], "tokenizer_config.json")):
            return paths[-1]
    return None


@pytest.mark.skipif(_phi4_snapshot() is None, reason="Phi-4-mini not in the HF cache")
class TestRealTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        transformers = pytest.importorskip("transformers")
        return transformers.AutoTokenizer.from_pretrained(_phi4_snapshot())

    @pytest.fixture(scope="class")
    def template(self):
        return PHI4_MINI_TOOL_TEMPLATE.read_text(encoding="utf-8")

    def test_the_models_own_template_never_shows_the_tools(self, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "weather in Paris?"}],
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "get_weather" not in prompt

    def test_shipped_template_shows_the_tools(self, tokenizer, template):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "weather in Paris?"}],
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=template,
        )
        assert "get_weather" in prompt
        assert prompt.endswith("<|user|>weather in Paris?<|end|><|assistant|>")

    def test_assistant_tool_calls_round_trip_with_the_real_tokenizer(
        self, tokenizer, template
    ):
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _tool_call("get_weather", RICH_ARGS, "a"),
                    _tool_call("get_weather", {"city": "Tokyo"}, "b"),
                ],
            },
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=False,
            chat_template=template,
        )
        (turn,) = _assistant_turns(prompt)
        calls = Phi4MiniToolParser().extract_tool_calls(turn).tool_calls
        assert [json.loads(c["arguments"]) for c in calls] == [
            RICH_ARGS,
            {"city": "Tokyo"},
        ]

    def test_markers_survive_tokenization(self, tokenizer):
        # The parser reads decoded text: ``functools[`` and the chat-turn markers must
        # round-trip through the tokenizer unchanged.
        text = 'functools[{"name": "f", "arguments": {"a": [1]}}]'
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(ids) == text
