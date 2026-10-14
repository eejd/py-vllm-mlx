# SPDX-License-Identifier: Apache-2.0
"""The shipped Gemma 3 pythonic tool template, and the parser against what it renders.

Gemma 3's own template ignores the OpenAI ``tools=`` argument. These tests render the
shipped template with jinja2 (configured the way transformers configures it) and, when
the model's tokenizer is in the local Hugging Face cache, with the real tokenizer too;
they skip that part in CI, which has no model cache.
"""

import glob
import json
import os
import re

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from vllm_mlx.templates import GEMMA3_PYTHONIC_TOOL_TEMPLATE
from vllm_mlx.tool_parsers.pythonic_tool_parser import PythonicToolParser

BOS = "<bos>"
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
# Arguments chosen to break naive handling: a ``)`` and ``]`` in a string, nesting,
# non-ASCII, booleans, null, a quote and a backslash.
RICH_ARGS = {
    "city": "Zürich",
    "ids": [1, [2, 3]],
    "note": 'see a[0]) ] & "b"\\',
    "flag": True,
    "off": False,
    "none": None,
    "units": {"t": "c"},
    "n": -1.5,
}


def _tojson(value, ensure_ascii=False, indent=None, **kwargs):
    # transformers replaces jinja's HTML-escaping ``tojson`` with plain json.dumps.
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def _raise(message):
    raise ValueError(message)


def render(messages, tools=None, add_generation_prompt=True, **extra):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    template = env.from_string(
        GEMMA3_PYTHONIC_TOOL_TEMPLATE.read_text(encoding="utf-8")
    )
    return template.render(
        messages=messages,
        tools=tools,
        add_generation_prompt=add_generation_prompt,
        bos_token=BOS,
        **extra,
    )


def _model_turns(prompt: str) -> list[str]:
    return re.findall(r"<start_of_turn>model\n(.*?)<end_of_turn>", prompt, re.DOTALL)


def _tool_call(name, arguments, call_id="call_1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _calls(turn):
    result = PythonicToolParser().extract_tool_calls(turn)
    return [(c["name"], json.loads(c["arguments"])) for c in result.tool_calls], (
        result.content
    )


class TestShippedFile:
    def test_keeps_license_header_and_attribution(self):
        text = GEMMA3_PYTHONIC_TOOL_TEMPLATE.read_text(encoding="utf-8")
        assert "SPDX-License-Identifier: Apache-2.0" in text
        assert "Copyright contributors to the vLLM project" in text
        assert "tool_chat_template_gemma3_pythonic.jinja" in text

    def test_ships_in_the_package(self):
        assert GEMMA3_PYTHONIC_TOOL_TEMPLATE.name == (
            "tool_chat_template_gemma3_pythonic.jinja"
        )
        assert GEMMA3_PYTHONIC_TOOL_TEMPLATE.is_file()


class TestWithoutTools:
    """Gemma 3's own format, byte for byte."""

    def test_plain_conversation(self):
        prompt = render(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": " hi "},
                {"role": "assistant", "content": "hello\n"},
                {"role": "user", "content": "bye"},
            ]
        )
        assert prompt == (
            f"{BOS}<start_of_turn>user\nBe brief.\n\nhi<end_of_turn>\n"
            "<start_of_turn>model\nhello<end_of_turn>\n"
            "<start_of_turn>user\nbye<end_of_turn>\n<start_of_turn>model\n"
        )

    def test_exactly_one_bos_and_it_comes_first(self):
        prompt = render([{"role": "user", "content": "hi"}])
        assert prompt.startswith(BOS + "<start_of_turn>user")
        assert prompt.count(BOS) == 1

    def test_no_generation_prompt(self):
        prompt = render(
            [{"role": "user", "content": "hi"}], add_generation_prompt=False
        )
        assert prompt.endswith("hi<end_of_turn>\n")

    def test_images_keep_their_placeholder(self):
        prompt = render(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image"},
                        {"type": "text", "text": "b"},
                    ],
                }
            ]
        )
        assert "<start_of_turn>user\na<start_of_image>b<end_of_turn>" in prompt

    @pytest.mark.parametrize("tools", [None, []])
    def test_no_or_empty_tools_add_no_tool_prompt(self, tools):
        prompt = render([{"role": "user", "content": "hi"}], tools=tools)
        assert "Tools (functions)" not in prompt


class TestRenderTools:
    def test_tools_reach_the_first_user_turn_after_the_system_prompt(self):
        prompt = render(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "weather in Paris?"},
            ],
            tools=TOOLS,
        )
        head, _, tail = prompt.partition("Tools (functions) are available.")
        assert head == f"{BOS}<start_of_turn>user\nBe brief.\n\n"
        assert "Example Format: [func_name1(params_name1=params_value1" in tail
        assert json.dumps(TOOLS, indent=4) in tail
        assert tail.endswith("weather in Paris?<end_of_turn>\n<start_of_turn>model\n")

    def test_tools_are_listed_once_even_over_several_turns(self):
        prompt = render(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
            tools=TOOLS,
        )
        assert prompt.count("get_weather") == 1


class TestHistory:
    def _history(self, arguments, content="", extra_calls=()):
        return [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [_tool_call("get_weather", arguments), *extra_calls],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]

    def test_keyword_arguments_are_comma_separated(self):
        # vLLM's template renders ``f(a=1b="x")`` here (its flag is loop-scoped)
        prompt = render(self._history({"a": 1, "b": "x", "c": [1, 2]}), tools=TOOLS)
        (turn,) = _model_turns(prompt)
        assert turn == 'get_weather(a=1, b="x", c=[1, 2])'.join("[]")

    def test_assistant_tool_calls_round_trip_through_the_parser(self):
        extra = [_tool_call("get_weather", {"city": "Tokyo"}, "call_2")]
        prompt = render(self._history(RICH_ARGS, extra_calls=extra), tools=TOOLS)
        (turn,) = _model_turns(prompt)
        calls, content = _calls(turn)
        assert calls == [("get_weather", RICH_ARGS), ("get_weather", {"city": "Tokyo"})]
        assert content is None

    def test_arguments_are_not_html_escaped_or_ascii_escaped(self):
        prompt = render(self._history({"q": "a & b <c> Zürich"}), tools=TOOLS)
        assert '[get_weather(q="a & b <c> Zürich")]' in prompt

    def test_a_call_without_arguments(self):
        (turn,) = _model_turns(render(self._history({}), tools=TOOLS))
        assert turn == "[get_weather()]"
        assert _calls(turn)[0] == [("get_weather", {})]

    def test_arguments_that_are_still_a_json_string_are_one_value(self):
        # not iterated character by character
        (turn,) = _model_turns(render(self._history('{"city": "Paris"}'), tools=TOOLS))
        assert turn == '[get_weather("{\\"city\\": \\"Paris\\"}")]'

    def test_text_next_to_calls_follows_them_and_parses_as_content(self):
        prompt = render(
            self._history({"city": "Paris"}, content="Checking."), tools=TOOLS
        )
        (turn,) = _model_turns(prompt)
        assert turn == '[get_weather(city="Paris")]Checking.'
        assert _calls(turn) == ([("get_weather", {"city": "Paris"})], "Checking.")

    @pytest.mark.parametrize("calls", [None, []])
    def test_empty_tool_calls_make_a_plain_assistant_message(self, calls):
        prompt = render(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b", "tool_calls": calls},
                {"role": "user", "content": "c"},
            ],
            tools=TOOLS,
        )
        assert _model_turns(prompt) == ["b"]

    def test_tool_results_are_user_turns_in_a_tool_response_envelope(self):
        history = self._history({"city": "Paris"}) + [
            {"role": "tool", "tool_call_id": "call_2", "content": '{"t": 20}'}
        ]
        prompt = render(history, tools=TOOLS)
        assert prompt.endswith(
            "<start_of_turn>user\n<tool_response>\nsunny</tool_response><end_of_turn>\n"
            '<start_of_turn>user\n<tool_response>\n{"t": 20}</tool_response>'
            "<end_of_turn>\n<start_of_turn>model\n"
        )

    def test_a_full_tool_loop_ends_ready_for_the_answer(self):
        prompt = render(
            self._history({"city": "Paris"})
            + [
                {"role": "assistant", "content": "It is sunny."},
                {"role": "user", "content": "thanks"},
            ],
            tools=TOOLS,
        )
        assert prompt.endswith(
            "<start_of_turn>model\nIt is sunny.<end_of_turn>\n"
            "<start_of_turn>user\nthanks<end_of_turn>\n<start_of_turn>model\n"
        )

    def test_roles_must_alternate(self):
        with pytest.raises(ValueError, match="alternate"):
            render(
                [
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                ],
                tools=TOOLS,
            )


def _gemma_snapshot():
    homes = [os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface")]
    for hf_home in filter(None, homes):
        for repo in (
            "gemma-3-4b-it-qat-4bit",
            "gemma-3-4b-it-4bit",
            "gemma-3-1b-it-4bit",
        ):
            paths = sorted(
                glob.glob(
                    os.path.join(
                        hf_home,
                        "hub",
                        f"models--mlx-community--{repo}",
                        "snapshots",
                        "*",
                    )
                )
            )
            if paths and os.path.exists(
                os.path.join(paths[-1], "tokenizer_config.json")
            ):
                return paths[-1]
    return None


@pytest.mark.skipif(_gemma_snapshot() is None, reason="Gemma 3 not in the HF cache")
class TestRealTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        transformers = pytest.importorskip("transformers")
        return transformers.AutoTokenizer.from_pretrained(_gemma_snapshot())

    @pytest.fixture(scope="class")
    def template(self):
        return GEMMA3_PYTHONIC_TOOL_TEMPLATE.read_text(encoding="utf-8")

    def test_the_models_own_template_never_shows_the_tools(self, tokenizer):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "weather in Paris?"}],
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
        )
        assert "get_weather" not in prompt

    @pytest.mark.parametrize("add_generation_prompt", [True, False])
    @pytest.mark.parametrize(
        "messages",
        [
            [{"role": "user", "content": "hi"}],
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": " hi "},
                {"role": "assistant", "content": "hello\n"},
                {"role": "user", "content": "bye"},
            ],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "image"},
                        {"type": "text", "text": "b"},
                    ],
                }
            ],
            [
                {"role": "system", "content": [{"type": "text", "text": "S"}]},
                {"role": "user", "content": "x"},
            ],
        ],
        ids=["user", "system-and-turns", "image", "system-list"],
    )
    def test_without_tools_the_prompt_is_the_models_own(
        self, tokenizer, template, messages, add_generation_prompt
    ):
        kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
        stock = tokenizer.apply_chat_template(messages, **kwargs)
        shipped = tokenizer.apply_chat_template(
            messages, chat_template=template, **kwargs
        )
        assert shipped == stock

    def test_the_real_tokenizer_renders_what_the_jinja_environment_renders(
        self, tokenizer, template
    ):
        messages = [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [_tool_call("get_weather", RICH_ARGS)],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ]
        real = tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=template,
        )
        assert real == render(messages, tools=TOOLS).replace(BOS, tokenizer.bos_token)
        (turn,) = _model_turns(real)
        assert _calls(turn)[0] == [("get_weather", RICH_ARGS)]

    def test_the_prompt_starts_with_one_bos(self, tokenizer, template):
        # mlx-lm and SimpleEngine skip the tokenizer's own BOS when the prompt already
        # starts with one; the MLLM path tokenizes without special tokens.
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "weather in Paris?"}],
            tools=TOOLS,
            tokenize=False,
            add_generation_prompt=True,
            chat_template=template,
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        start_of_turn = tokenizer.convert_tokens_to_ids("<start_of_turn>")
        assert ids[:2] == [tokenizer.bos_token_id, start_of_turn]

    def test_markers_survive_tokenization(self, tokenizer):
        text = "[get_weather(city='Paris', days=[1, 2]), get_time(tz=None)]"
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert tokenizer.decode(ids) == text
