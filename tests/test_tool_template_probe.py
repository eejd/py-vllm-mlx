# SPDX-License-Identifier: Apache-2.0
"""Tool calling needs a chat template that renders ``tools``; say so when it does not.

``--enable-auto-tool-choice`` only parses what the model writes. A template that never
reads ``tools`` (Phi-4-mini's, Hermes-3-3B's, Gemma-3's as converted) leaves the model
unaware of any tool, so every request ends in prose and nothing in the log says why.
"""

import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

import vllm_mlx.server as server
from vllm_mlx.templates import HERMES_TOOL_TEMPLATE, PHI4_MINI_TOOL_TEMPLATE
from vllm_mlx.utils.chat_template_tools import (
    engine_template_renderer,
    template_renders_tools,
    tools_ignored_warning,
)

IGNORES_TOOLS = (
    "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)
RENDERS_TOOLS = (
    "{% if tools %}TOOLS:{% for t in tools %}{{ t.function.name }};{% endfor %}{% endif %}"
    "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


def _tokenizer(template: str):
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tok.chat_template = template
    return tok


def _render(tok):
    return lambda messages, tools=None: tok.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )


class TestProbe:
    def test_a_template_that_ignores_tools(self):
        assert template_renders_tools(_render(_tokenizer(IGNORES_TOOLS))) is False

    def test_a_template_that_renders_tools(self):
        assert template_renders_tools(_render(_tokenizer(RENDERS_TOOLS))) is True

    def test_a_template_that_cannot_be_probed_is_not_evidence(self):
        def broken(messages, tools=None):
            raise ValueError("nope")

        assert template_renders_tools(broken) is None
        assert template_renders_tools(lambda messages, tools=None: ["ids"]) is None

    @pytest.mark.parametrize(
        "path", [HERMES_TOOL_TEMPLATE, PHI4_MINI_TOOL_TEMPLATE], ids=["hermes", "phi4"]
    )
    def test_the_shipped_templates_pass_their_own_probe(self, path):
        tok = _tokenizer(path.read_text(encoding="utf-8"))
        tok.eos_token = "<unk>"  # the phi4 template's tool-less branch reads eos_token
        assert template_renders_tools(_render(tok)) is True


class TestWarningText:
    @pytest.mark.parametrize(
        "parser, path",
        [
            ("hermes", HERMES_TOOL_TEMPLATE),
            ("nous", HERMES_TOOL_TEMPLATE),
            ("phi4_mini", PHI4_MINI_TOOL_TEMPLATE),
            ("phi4_mini_json", PHI4_MINI_TOOL_TEMPLATE),
        ],
    )
    def test_names_the_shipped_template_for_the_parser(self, parser, path):
        text = tools_ignored_warning(parser)
        assert f"--tool-call-parser {parser}" in text
        assert f"--chat-template {path}" in text

    def test_without_a_shipped_template_says_what_to_pass(self):
        text = tools_ignored_warning("generic")
        assert "--chat-template PATH" in text
        assert "tool_chat_template_" in text


class TestEngineRenderer:
    def test_batched_engine_renders_through_its_own_template_applicator(self):
        calls = []

        class Engine:
            def _apply_chat_template(self, messages, tools=None, **kwargs):
                calls.append((messages, tools))
                return "prompt"

        render = engine_template_renderer(Engine())
        assert render([{"role": "user", "content": "x"}], tools=[{"t": 1}]) == "prompt"
        assert calls == [([{"role": "user", "content": "x"}], [{"t": 1}])]

    def test_tokenizer_engine(self):
        engine = SimpleNamespace(tokenizer=_tokenizer(RENDERS_TOOLS))
        assert template_renders_tools(engine_template_renderer(engine)) is True

    def test_a_processor_without_a_template_falls_back_to_its_tokenizer(self):
        # Gemma-3: the processor has no template, the tokenizer inside it does.
        processor = SimpleNamespace(
            chat_template=None,
            apply_chat_template=lambda *a, **k: (_ for _ in ()).throw(ValueError()),
            tokenizer=_tokenizer(IGNORES_TOOLS),
        )
        render = engine_template_renderer(SimpleNamespace(tokenizer=processor))
        assert template_renders_tools(render) is False

    def test_no_template_means_no_renderer(self):
        assert engine_template_renderer(SimpleNamespace(tokenizer=None)) is None
        assert engine_template_renderer(SimpleNamespace(tokenizer=object())) is None


class TestStartupWarning:
    @pytest.fixture
    def tools_on(self, monkeypatch):
        monkeypatch.setattr(server, "_enable_auto_tool_choice", True)
        monkeypatch.setattr(server, "_tool_call_parser", "hermes")

    def _warnings(self, caplog):
        return [
            r.getMessage()
            for r in caplog.records
            if r.levelno == logging.WARNING and r.name == server.logger.name
        ]

    def test_warns_when_the_template_ignores_tools(self, tools_on, caplog):
        engine = SimpleNamespace(tokenizer=_tokenizer(IGNORES_TOOLS))
        with caplog.at_level(logging.WARNING, logger=server.logger.name):
            server._warn_if_template_ignores_tools(engine)
        (message,) = self._warnings(caplog)
        assert "does not render the request's `tools`" in message
        assert str(HERMES_TOOL_TEMPLATE) in message

    def test_silent_when_the_template_renders_tools(self, tools_on, caplog):
        engine = SimpleNamespace(tokenizer=_tokenizer(RENDERS_TOOLS))
        with caplog.at_level(logging.WARNING, logger=server.logger.name):
            server._warn_if_template_ignores_tools(engine)
        assert self._warnings(caplog) == []

    def test_silent_when_the_shipped_template_is_installed(self, tools_on, caplog):
        tok = _tokenizer(HERMES_TOOL_TEMPLATE.read_text(encoding="utf-8"))
        with caplog.at_level(logging.WARNING, logger=server.logger.name):
            server._warn_if_template_ignores_tools(SimpleNamespace(tokenizer=tok))
        assert self._warnings(caplog) == []

    def test_silent_without_auto_tool_choice(self, monkeypatch, caplog):
        monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
        monkeypatch.setattr(server, "_tool_call_parser", "hermes")
        engine = SimpleNamespace(tokenizer=_tokenizer(IGNORES_TOOLS))
        with caplog.at_level(logging.WARNING, logger=server.logger.name):
            server._warn_if_template_ignores_tools(engine)
        assert self._warnings(caplog) == []

    def test_silent_for_harmony_which_renders_without_jinja(self, tools_on, caplog):
        engine = SimpleNamespace(
            tokenizer=_tokenizer(IGNORES_TOOLS), use_harmony_rendering=True
        )
        with caplog.at_level(logging.WARNING, logger=server.logger.name):
            server._warn_if_template_ignores_tools(engine)
        assert self._warnings(caplog) == []

    def test_a_template_that_cannot_be_probed_stays_silent(self, tools_on, caplog):
        class Engine:
            def _apply_chat_template(self, messages, tools=None, **kwargs):
                raise RuntimeError("template raised")

        with caplog.at_level(logging.WARNING, logger=server.logger.name):
            server._warn_if_template_ignores_tools(Engine())
        assert self._warnings(caplog) == []


# Gemma-3's template: user/model/system only, no tool turns, raises on any other order.
GEMMA_LIKE = (
    "{{ bos_token }}{%- for message in messages -%}"
    "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{%- endif -%}"
    "<start_of_turn>{{ 'model' if message['role'] == 'assistant' else message['role'] }}\n"
    "{{ message['content'] | trim }}<end_of_turn>\n"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<start_of_turn>model\n{%- endif -%}"
)
TOOL_CONVERSATION = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "Sunny, 21C"},
]


def _batched_mllm(template: str):
    with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine("org/gemma-like-vlm")
    tokenizer = _tokenizer(template)
    # Gemma-3's processor carries no template of its own; the tokenizer inside does.
    engine._processor = SimpleNamespace(
        tokenizer=tokenizer,
        chat_template=None,
        apply_chat_template=lambda *a, **k: (_ for _ in ()).throw(
            ValueError("Cannot use apply_chat_template ... does not have a chat template.")
        ),
    )
    engine._is_mllm = True
    engine._mllm_instance = SimpleNamespace(config=SimpleNamespace(model_type="gemma3"))
    engine._loaded = True
    return engine


class TestBatchedMultimodalToolTurns:
    """The simple engine serialises tool turns for a template that cannot render them
    (#82); the batched engine did not, so such a conversation was a 500."""

    def test_tool_turns_become_gemma_native_text(self):
        engine = _batched_mllm(GEMMA_LIKE)
        prompt = engine._apply_chat_template(TOOL_CONVERSATION)
        assert "```tool_code\nget_weather(city='Paris')\n```" in prompt
        assert "```tool_output\nSunny, 21C\n```" in prompt
        assert "<start_of_turn>tool" not in prompt
        assert prompt.endswith("<start_of_turn>model")

    def test_a_template_with_tool_support_renders_the_turns_itself(self):
        template = (
            "{% for m in messages %}{{ m['role'] }}:"
            "{% if m.tool_calls %}CALLS {% endif %}{{ m['content'] }};{% endfor %}"
        )
        engine = _batched_mllm(template)
        prompt = engine._apply_chat_template(TOOL_CONVERSATION)
        assert "tool:Sunny, 21C;" in prompt
        assert "tool_code" not in prompt

    def test_text_models_are_untouched(self):
        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
            from vllm_mlx.engine.batched import BatchedEngine

            engine = BatchedEngine("org/text")
        engine._tokenizer = _tokenizer(
            "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
        )
        prompt = engine._apply_chat_template(TOOL_CONVERSATION)
        assert "<tool>Sunny, 21C" in prompt
        assert "tool_code" not in prompt
