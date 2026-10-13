# SPDX-License-Identifier: Apache-2.0
"""The shipped Hermes tool template, and the parser against what it renders.

The ChatML template of Hermes-3-Llama-3.2-3B (and its mlx-community conversions) never
reads the OpenAI ``tools=`` argument, so the model is not told that any tool exists.
These tests render the shipped template with jinja2 configured the way transformers
configures it, and with the real tokenizer when it is in the local Hugging Face cache
(skipped in CI, which has no model cache).
"""

import glob
import json
import re

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from vllm_mlx.templates import (
    BUNDLED_TOOL_TEMPLATES,
    HERMES_TOOL_TEMPLATE,
    PHI4_MINI_TOOL_TEMPLATE,
)
from vllm_mlx.tool_parsers import ToolParserManager

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "days": {"type": "integer", "description": "Forecast days"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Labels",
                    },
                },
                "required": ["city"],
            },
        },
    }
]

# The checkpoint's own template, from mlx-community/Hermes-3-Llama-3.2-3B-4bit
# tokenizer_config.json. Without tools the shipped template must render exactly this.
STOCK_TEMPLATE = (
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false "
    "%}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + "
    "'\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _tojson(value, ensure_ascii=False, indent=None, **kwargs):
    # transformers replaces jinja's HTML-escaping ``tojson`` with plain json.dumps.
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def _render(template: str, messages, **variables) -> str:
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = _tojson
    return env.from_string(template).render(messages=messages, **variables)


def render(messages, tools=None, **variables):
    return _render(
        HERMES_TOOL_TEMPLATE.read_text(encoding="utf-8"),
        messages,
        tools=tools,
        **variables,
    )


def _tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class TestShippedFile:
    def test_keeps_license_header_and_attribution(self):
        text = HERMES_TOOL_TEMPLATE.read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: Apache-2.0" in text
        assert "Copyright contributors to the vLLM project" in text
        assert "tool_chat_template_hermes.jinja" in text

    def test_bundled_templates_map_to_registered_parsers(self):
        for parser_name, path in BUNDLED_TOOL_TEMPLATES.items():
            assert path.is_file(), path
            assert ToolParserManager.get_tool_parser(parser_name) is not None
        assert BUNDLED_TOOL_TEMPLATES["hermes"] == HERMES_TOOL_TEMPLATE
        assert BUNDLED_TOOL_TEMPLATES["phi4_mini"] == PHI4_MINI_TOOL_TEMPLATE


# name -> (messages, add_generation_prompt)
_USER = {"role": "user", "content": "Hi there"}
TOOL_LESS_CASES = {
    "plain": ([_USER], True),
    "no_generation_prompt": ([_USER], False),
    "system": ([{"role": "system", "content": "Be brief."}, _USER], True),
    "multi_turn": (
        [
            {"role": "system", "content": "Be brief."},
            _USER,
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "And now?"},
        ],
        True,
    ),
}


class TestToolLessRequestsKeepTheStockLayout:
    """A server-wide --chat-template must not change prompts that carry no tools."""

    @pytest.mark.parametrize("case", sorted(TOOL_LESS_CASES))
    @pytest.mark.parametrize("tools", [None, []])
    def test_renders_exactly_what_the_checkpoints_template_renders(self, case, tools):
        messages, add_generation_prompt = TOOL_LESS_CASES[case]
        stock = _render(
            STOCK_TEMPLATE, messages, add_generation_prompt=add_generation_prompt
        )
        shipped = render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        )
        assert shipped == stock
        assert "<tools>" not in shipped

    def test_default_generation_prompt_matches_when_the_variable_is_absent(self):
        # The stock template defaults add_generation_prompt to False.
        stock = _render(STOCK_TEMPLATE, [_USER])
        shipped = render([_USER])
        assert shipped == stock


class TestRenderTools:
    def test_tools_are_announced_in_a_system_turn_before_the_conversation(self):
        prompt = render(
            [{"role": "system", "content": "Be brief."}, _USER],
            tools=TOOLS,
            add_generation_prompt=True,
        )
        head, _, rest = prompt.partition("<|im_end|>\n")
        assert head.startswith(
            "<|im_start|>system\nYou are a function calling AI model."
        )
        assert "<tools> " in head and " </tools>" in head
        assert '"name": "get_weather"' in head
        assert (
            "get_weather(city: str, days: int, tags: list[str]) - Current weather"
            in head
        )
        # The caller's own system message is kept, as a second system turn.
        assert rest.startswith("<|im_start|>system\nBe brief.<|im_end|>\n")
        assert prompt.endswith(
            "<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n"
        )

    def test_args_entries_of_a_tool_are_listed_one_per_line_under_one_header(self):
        # Upstream's template runs the entries together on one line.
        head = render([_USER], tools=TOOLS).partition("<|im_end|>")[0]
        assert head.count("Args:") == 1
        assert (
            "    Args:\n"
            "        city(str): City name\n"
            "        days(int): Forecast days\n"
            "        tags(list[str]): Labels"
        ) in head

    def test_no_bos_token_is_rendered(self):
        # vllm-mlx tokenizes the prompt with special tokens: the tokenizer adds BOS.
        prompt = render([_USER], tools=TOOLS, bos_token="<|begin_of_text|>")
        assert "<|begin_of_text|>" not in prompt

    def test_arguments_rendered_as_json_not_python_repr(self):
        prompt = render(
            [
                _USER,
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        _tool_call(
                            "get_weather", {"city": "Zürich", "ok": True, "n": None}
                        )
                    ],
                },
            ],
            tools=TOOLS,
        )
        assert (
            '{"arguments": {"city": "Zürich", "ok": true, "n": null}, "name": "get_weather"}'
            in prompt
        )

    def test_string_arguments_pass_through_unchanged(self):
        prompt = render(
            [_USER, {"role": "assistant", "tool_calls": [_tool_call("f", '{"a": 1}')]}],
            tools=TOOLS,
        )
        assert '{"arguments": {"a": 1}, "name": "f"}' in prompt

    def test_assistant_content_survives_next_to_tool_calls(self):
        prompt = render(
            [
                _USER,
                {
                    "role": "assistant",
                    "content": "Checking.",
                    "tool_calls": [_tool_call("f", {"a": 1})],
                },
            ],
            tools=TOOLS,
        )
        assert "<|im_start|>assistant\nChecking.\n<tool_call>\n" in prompt

    @pytest.mark.parametrize("empty", [None, []])
    def test_empty_tool_calls_means_a_plain_assistant_turn(self, empty):
        prompt = render(
            [_USER, {"role": "assistant", "content": "Hello", "tool_calls": empty}],
            tools=TOOLS,
        )
        assert "<|im_start|>assistant\nHello<|im_end|>\n" in prompt
        conversation = prompt.split("</tool_call><|im_end|>\n", 1)[1]
        assert "<tool_call>" not in conversation

    def test_parallel_results_share_one_tool_turn(self):
        prompt = render(
            [
                _USER,
                {
                    "role": "assistant",
                    "tool_calls": [
                        _tool_call("f", {"a": 1}, "c1"),
                        _tool_call("g", {"b": 2}, "c2"),
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "one"},
                {"role": "tool", "tool_call_id": "c2", "content": "two"},
            ],
            tools=TOOLS,
            add_generation_prompt=True,
        )
        assert prompt.endswith(
            "<|im_start|>tool\n<tool_response>\none\n</tool_response>\n"
            "<tool_response>\ntwo\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n"
        )
        assert prompt.count("<|im_start|>tool") == 1

    def test_tool_without_parameters_or_description_renders(self):
        prompt = render(
            [_USER], tools=[{"type": "function", "function": {"name": "ping"}}]
        )
        assert '"name": "ping"' in prompt
        assert '"parameters": {}' in prompt

    def test_flat_tool_shape_renders(self):
        flat = [
            {
                "name": "ping",
                "description": "Ping",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        assert '"name": "ping"' in render([_USER], tools=flat)

    @pytest.mark.parametrize(
        "schema, python_type",
        [
            ({"type": "null"}, "None"),
            ({"type": ["string", "null"]}, "Union[str,None]"),
            ({"type": "any"}, "Any"),
            ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, "Any"),
            ({}, "Any"),
            ({"type": "array"}, "list[Any]"),
        ],
    )
    def test_nullable_and_unknown_property_types_render(self, schema, python_type):
        # A string-valued `type` that is not known used to recurse until RecursionError
        # (HTTP 500) because a str is iterable.
        tool = {
            "type": "function",
            "function": {
                "name": "f",
                "description": "d",
                "parameters": {"type": "object", "properties": {"p": schema}},
            },
        }
        assert f"f(p: {python_type}) - d" in render([_USER], tools=[tool])

    @pytest.mark.parametrize("add_generation_prompt", [True, False])
    def test_generation_prompt_is_honoured(self, add_generation_prompt):
        prompt = render(
            [_USER], tools=TOOLS, add_generation_prompt=add_generation_prompt
        )
        assert prompt.endswith("<|im_start|>assistant\n") is add_generation_prompt


class TestParserReadsWhatTheTemplateTeachesTheModel:
    """History rendered by the template parses back to the same call."""

    def test_rendered_tool_call_round_trips_through_the_hermes_parser(self):
        from vllm_mlx.tool_parsers import HermesToolParser

        arguments = {"city": "Zürich", "days": 3, "tags": ["a", "b]"]}
        prompt = render(
            [
                _USER,
                {
                    "role": "assistant",
                    "tool_calls": [_tool_call("get_weather", arguments)],
                },
            ],
            tools=TOOLS,
        )
        turn = re.search(r"<\|im_start\|>assistant(.*?)<\|im_end\|>", prompt, re.DOTALL)
        parsed = HermesToolParser().extract_tool_calls(turn.group(1))
        assert parsed.tools_called
        assert parsed.tool_calls[0]["name"] == "get_weather"
        assert json.loads(parsed.tool_calls[0]["arguments"]) == arguments

    def test_the_format_the_prompt_asks_for_is_what_the_parser_reads(self):
        from vllm_mlx.tool_parsers import HermesToolParser

        # The sentence at the end of the system turn is the model's instruction.
        assert (
            '<tool_call>\n{"arguments": <args-dict>, "name": <function-name>}\n</tool_call>'
            in render([_USER], tools=TOOLS)
        )
        # The model card's pydantic schema lists `arguments` first.
        assert '"required": ["arguments", "name"]' in render([_USER], tools=TOOLS)
        parsed = HermesToolParser().extract_tool_calls(
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
        )
        assert parsed.tools_called and parsed.tool_calls[0]["name"] == "get_weather"


class TestToolResultGivenAsTextParts:
    """``content: [{"type": "text", ...}]`` on a tool message is valid OpenAI input.

    The template concatenates ``message.content`` as a string, which raised
    ``TypeError`` (HTTP 500) on a list. The server flattens the text parts first, as
    vLLM does for string-format templates.
    """

    PARTS = [
        {"type": "text", "text": "Sunny"},
        {"type": "text", "text": "21C"},
    ]

    def _conversation(self, tool_content):
        return [
            _USER,
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call("get_weather", '{"city": "Paris"}')],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": tool_content},
        ]

    def test_text_parts_render_as_the_tool_response(self):
        from vllm_mlx.api.utils import extract_multimodal_content

        messages, *_ = extract_multimodal_content(
            self._conversation(self.PARTS), preserve_native_format=True
        )
        prompt = render(messages, tools=TOOLS, add_generation_prompt=True)
        assert "<tool_response>\nSunny\n21C\n</tool_response>" in prompt

    def test_pydantic_content_parts_render_too(self):
        from vllm_mlx.api.models import ContentPart, Message
        from vllm_mlx.api.utils import extract_multimodal_content

        conversation = self._conversation([ContentPart(type="text", text="Sunny")])
        conversation = [
            Message(**m) if m["role"] == "tool" else m for m in conversation
        ]
        messages, *_ = extract_multimodal_content(
            conversation, preserve_native_format=True
        )
        assert messages[-1]["content"] == "Sunny"
        assert "<tool_response>\nSunny\n</tool_response>" in render(
            messages, tools=TOOLS
        )

    def test_non_text_parts_are_not_part_of_the_response(self):
        from vllm_mlx.api.utils import extract_multimodal_content

        parts = [
            {"type": "text", "text": "Sunny"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]
        messages, *_ = extract_multimodal_content(
            self._conversation(parts), preserve_native_format=True
        )
        assert messages[-1]["content"] == "Sunny"

    def test_a_template_less_model_gets_the_text_not_a_list_repr(self):
        from vllm_mlx.api.utils import extract_multimodal_content

        messages, *_ = extract_multimodal_content(self._conversation(self.PARTS))
        assert messages[-1] == {
            "role": "user",
            "content": "[Tool Result (call_1)]: Sunny\n21C",
        }

    def test_string_and_empty_results_are_unchanged(self):
        from vllm_mlx.api.utils import extract_multimodal_content

        for content, expected in [("Sunny", "Sunny"), (None, ""), ("", "")]:
            messages, *_ = extract_multimodal_content(
                self._conversation(content), preserve_native_format=True
            )
            assert messages[-1]["content"] == expected


def _hermes_tokenizer_dir():
    paths = glob.glob(
        "/hive-cache/*/models/hub/models--mlx-community--Hermes-3-Llama-3.2-3B-4bit/"
        "snapshots/*"
    ) + glob.glob(
        "~/.cache/huggingface/hub/models--mlx-community--Hermes-3-Llama-3.2-3B-4bit/"
        "snapshots/*"
    )
    return paths[0] if paths else None


@pytest.mark.skipif(
    _hermes_tokenizer_dir() is None, reason="Hermes-3 tokenizer not in the local cache"
)
class TestWithTheRealTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(_hermes_tokenizer_dir())

    def test_the_models_own_template_drops_tools(self, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [_USER], tools=TOOLS, tokenize=False, add_generation_prompt=True
        )
        assert "get_weather" not in prompt

    def test_shipped_template_renders_tools_and_matches_stock_without_them(
        self, tokenizer
    ):
        stock = tokenizer.chat_template
        messages = [{"role": "system", "content": "Be brief."}, _USER]
        before = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenizer.chat_template = HERMES_TOOL_TEMPLATE.read_text(encoding="utf-8")
        try:
            assert (
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                == before
            )
            with_tools = tokenizer.apply_chat_template(
                messages, tools=TOOLS, tokenize=False, add_generation_prompt=True
            )
            assert "get_weather" in with_tools
            # One BOS only: the tokenizer adds it, the template must not.
            ids = tokenizer.encode(with_tools)
            assert ids[:2] != [tokenizer.bos_token_id] * 2
        finally:
            tokenizer.chat_template = stock
