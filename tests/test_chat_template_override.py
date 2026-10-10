# SPDX-License-Identifier: Apache-2.0
"""``--chat-template``: the operator's template replaces the one the model ships."""

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from vllm_mlx.utils.chat_template_override import (
    apply_chat_template_override,
    chat_template_arg,
    resolve_chat_template,
)

SHIPPED = (
    "SHIPPED {% for m in messages %}{{ m['role'] }}={{ m['content'] }};{% endfor %}"
)
OVERRIDE = (
    "OVERRIDE {% for m in messages %}{{ m['role'] }}={{ m['content'] }};{% endfor %}"
    "{% if tools %}tools={{ tools | length }}{% endif %}"
)
MESSAGES = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def _hf_tokenizer(template: str = SHIPPED):
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tok.chat_template = template
    return tok


def _wrapped_tokenizer(template: str = SHIPPED):
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    return TokenizerWrapper(_hf_tokenizer(template))


def _render(tokenizer):
    return tokenizer.apply_chat_template(
        MESSAGES, tools=TOOLS, tokenize=False, add_generation_prompt=True
    )


class TestResolve:
    def test_file(self, tmp_path):
        path = tmp_path / "t.jinja"
        path.write_text(OVERRIDE, encoding="utf-8")
        assert resolve_chat_template(str(path)) == OVERRIDE

    def test_inline_single_line_decodes_escapes(self):
        assert resolve_chat_template("a{{ x }}\\nb") == "a{{ x }}\nb"

    def test_inline_non_ascii_text_is_kept(self):
        template = resolve_chat_template('{{ "é" }} {{ "日本" }} {{ "😀" }}')
        assert template == '{{ "é" }} {{ "日本" }} {{ "😀" }}'

    def test_inline_escaped_newline_and_tab_next_to_non_ascii(self):
        assert resolve_chat_template("é{{ x }}\\n日本\\t\\r.") == (
            "é{{ x }}\n日本\t\r."
        )

    def test_inline_escaped_quote_is_left_for_jinja(self):
        template = resolve_chat_template('{{ "a\\"b" }}')
        assert template == '{{ "a\\"b" }}'
        from jinja2 import Environment

        assert Environment().from_string(template).render() == 'a"b'

    def test_inline_double_backslash_then_n_is_not_a_newline(self):
        assert resolve_chat_template("{{ x }}\\\\n") == "{{ x }}\\\\n"

    def test_missing_path_is_an_error_not_an_inline_template(self, tmp_path):
        with pytest.raises(ValueError, match="not an existing file"):
            resolve_chat_template(str(tmp_path / "missing.jinja"))

    def test_invalid_jinja_is_rejected_before_any_model_loads(self, tmp_path):
        path = tmp_path / "bad.jinja"
        path.write_text("{% for m in messages %}no endfor", encoding="utf-8")
        with pytest.raises(ValueError, match="not valid Jinja"):
            resolve_chat_template(str(path))

    @pytest.mark.parametrize("value", ["", "   "])
    def test_empty_is_rejected(self, value):
        with pytest.raises(ValueError, match="empty"):
            resolve_chat_template(value)

    def test_argparse_type_reports_a_usage_error(self):
        import argparse

        with pytest.raises(argparse.ArgumentTypeError, match="--chat-template"):
            chat_template_arg("/no/such/template.jinja")


class TestApply:
    def test_replaces_the_template_on_a_wrapped_tokenizer(self):
        tok = _wrapped_tokenizer()
        assert _render(tok).startswith("SHIPPED")
        apply_chat_template_override(tok, OVERRIDE)
        assert _render(tok) == "OVERRIDE user=hi;tools=1"

    def test_none_keeps_the_models_own_template(self):
        tok = _wrapped_tokenizer()
        apply_chat_template_override(tok, None)
        assert _render(tok).startswith("SHIPPED")

    def test_processor_and_its_tokenizer_both_get_it(self):
        inner = _hf_tokenizer()
        processor = SimpleNamespace(tokenizer=inner, chat_template=SHIPPED)
        apply_chat_template_override(processor, OVERRIDE)
        assert processor.chat_template == OVERRIDE
        assert _render(inner) == "OVERRIDE user=hi;tools=1"


class TestParsers:
    @pytest.mark.parametrize("module", ["vllm_mlx.cli", "vllm_mlx.server"])
    def test_option_resolves_to_template_text(self, module, tmp_path):
        import importlib

        path = tmp_path / "t.jinja"
        path.write_text(OVERRIDE, encoding="utf-8")
        mod = importlib.import_module(module)
        parser = mod.create_parser()
        argv = ["serve", "m"] if module == "vllm_mlx.cli" else ["--model", "m"]
        args = parser.parse_args([*argv, "--chat-template", str(path)])
        assert args.chat_template == OVERRIDE
        assert parser.parse_args(argv).chat_template is None

    def test_cli_rejects_a_missing_template_file(self, capsys):
        from vllm_mlx import cli

        with pytest.raises(SystemExit) as exc:
            cli.create_parser().parse_args(
                ["serve", "m", "--chat-template", "/no/such/t.jinja"]
            )
        assert exc.value.code != 0
        assert "--chat-template" in capsys.readouterr().err

    def test_serve_refuses_chat_template_with_models_config(self, tmp_path, capsys):
        from vllm_mlx import cli

        args = cli.create_parser().parse_args(
            ["serve", "--models-config", "m.yaml", "--chat-template", "{{ x }}\\n"]
        )
        with pytest.raises(SystemExit) as exc:
            cli.serve_command(args)
        assert exc.value.code == 1
        assert "--chat-template cannot be used with --models-config" in (
            capsys.readouterr().out
        )


@pytest.fixture
def stock_tokenizer(monkeypatch):
    """Make the model loader hand back a tokenizer that carries SHIPPED."""
    tokenizer = _wrapped_tokenizer()
    monkeypatch.setattr(
        "vllm_mlx.utils.tokenizer._load_model_with_fallback",
        lambda model_name, tokenizer_config=None: (object(), tokenizer),
    )
    monkeypatch.setattr(
        "vllm_mlx.utils.tokenizer._install_custom_chat_template",
        lambda model_name, tok: tok,
    )
    return tokenizer


class TestEnginesRenderTheOverride:
    def test_simple_engine(self, stock_tokenizer):
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine(
            "org/text-model", force_mllm=False, chat_template=OVERRIDE
        )
        engine.prepare_for_start()
        assert _render(engine._model.tokenizer) == "OVERRIDE user=hi;tools=1"

    def test_simple_engine_without_override_is_untouched(self, stock_tokenizer):
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine("org/text-model", force_mllm=False)
        engine.prepare_for_start()
        assert _render(engine._model.tokenizer).startswith("SHIPPED")

    def test_batched_engine_prompt(self, stock_tokenizer, monkeypatch):
        from vllm_mlx.engine.batched import BatchedEngine

        monkeypatch.setattr(
            BatchedEngine, "_configure_metal_memory_limits", lambda s: None
        )
        engine = BatchedEngine(
            "org/text-model", force_mllm=False, chat_template=OVERRIDE
        )
        engine.prepare_for_start()
        prompt = engine._apply_chat_template(MESSAGES, tools=TOOLS)
        assert prompt == "OVERRIDE user=hi;tools=1"

    def test_lifecycle_spec_reaches_the_engine(self, stock_tokenizer):
        from vllm_mlx.lifecycle import ModelSpec
        from vllm_mlx.server import _build_engine

        for use_batching in (False, True):
            spec = ModelSpec(
                model_key="default",
                model_name="org/text-model",
                use_batching=use_batching,
                chat_template=OVERRIDE,
            )
            engine = _build_engine(spec)
            assert engine._chat_template == OVERRIDE

    def test_multimodal_loader_sets_processor_and_tokenizer(self, monkeypatch):
        import mlx_vlm
        import mlx_vlm.utils

        from vllm_mlx.models.mllm import MLXMultimodalLM

        inner = _hf_tokenizer()
        processor = SimpleNamespace(tokenizer=inner, chat_template=SHIPPED)
        model = SimpleNamespace(config=SimpleNamespace())
        monkeypatch.setattr(mlx_vlm, "load", lambda name: (model, processor))
        monkeypatch.setattr(mlx_vlm.utils, "load_config", lambda name: {})

        lm = MLXMultimodalLM("org/vlm", chat_template=OVERRIDE)
        lm.load()
        assert lm.processor.chat_template == OVERRIDE
        assert _render(lm.get_tokenizer()) == "OVERRIDE user=hi;tools=1"


# --- the template reaches every engine constructor --------------------------------
#
# Each hand-off below is a place where dropping ``chat_template`` would silently
# ignore the option while everything else keeps working.


@pytest.fixture
def clean_server_state(monkeypatch):
    """load_model/serve/main rewrite server globals; restore them afterwards."""
    from vllm_mlx import server

    for name in (
        "_engine",
        "_residency_manager",
        "_default_model_key",
        "_model_name",
        "_model_path",
        "_model_manager",
        "_lazy_load_model",
        "_auto_unload_idle_seconds",
        "_force_mllm_model",
        "_default_max_tokens",
        "_max_request_tokens",
        "_api_key",
        "_default_timeout",
        "_metrics_enabled",
        "_rate_limiter",
        "_enable_auto_tool_choice",
        "_tool_call_parser",
        "_default_temperature",
        "_default_top_p",
        "_default_top_k",
        "_default_min_p",
        "_default_presence_penalty",
        "_default_repetition_penalty",
        "_default_chat_template_kwargs",
        "_max_audio_upload_bytes",
        "_max_tts_input_chars",
        "_embedding_max_length",
        "_embedding_overflow_policy",
        "_reasoning_parser",
        "_reasoning_parser_name",
    ):
        if hasattr(server, name):
            monkeypatch.setattr(server, name, getattr(server, name))
    monkeypatch.setattr(server, "_lifespan_active", False)
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(server, "_residency_manager", None)
    return server


class TestLoadModelPassesTheTemplate:
    def test_simple_engine(self, clean_server_state):
        from unittest.mock import MagicMock, patch

        server = clean_server_state
        with (
            patch.object(
                server, "SimpleEngine", return_value=MagicMock()
            ) as engine_cls,
            patch.object(server, "_detect_native_tool_support", return_value=False),
            patch("vllm_mlx.server.asyncio.new_event_loop", return_value=MagicMock()),
            patch("vllm_mlx.server.asyncio.set_event_loop"),
        ):
            server.load_model("m", use_batching=False, chat_template=OVERRIDE)
            assert engine_cls.call_args.kwargs["chat_template"] == OVERRIDE
            server.load_model("m", use_batching=False)
            assert engine_cls.call_args.kwargs["chat_template"] is None

    def test_batched_engine(self, clean_server_state):
        from unittest.mock import MagicMock, patch

        server = clean_server_state
        with (
            patch.object(
                server, "BatchedEngine", return_value=MagicMock()
            ) as engine_cls,
            patch.object(server, "_detect_native_tool_support", return_value=False),
        ):
            server.load_model("m", use_batching=True, chat_template=OVERRIDE)
            assert engine_cls.call_args.kwargs["chat_template"] == OVERRIDE
            server.load_model("m", use_batching=True)
            assert engine_cls.call_args.kwargs["chat_template"] is None

    def test_lifecycle_residency_spec(self, clean_server_state, monkeypatch):
        server = clean_server_state
        specs = []

        class FakeResidencyManager:
            def __init__(self, *args, **kwargs):
                pass

            def register_model(self, spec):
                specs.append(spec)

        monkeypatch.setattr(server, "ResidencyManager", FakeResidencyManager)
        server.load_model(
            "m",
            auto_unload_idle_seconds=60.0,
            lazy_load_model=True,
            chat_template=OVERRIDE,
        )
        assert [s.chat_template for s in specs] == [OVERRIDE]


class TestEntryPointsPassTheTemplate:
    def test_serve_command(self, clean_server_state, monkeypatch, tmp_path):
        from vllm_mlx import cli
        from vllm_mlx.utils import download

        path = tmp_path / "t.jinja"
        path.write_text(OVERRIDE, encoding="utf-8")
        loaded = {}
        monkeypatch.setattr(
            download, "ensure_model_downloaded", lambda *a, **k: "local-test-model"
        )
        monkeypatch.setattr(
            clean_server_state,
            "load_model",
            lambda *args, **kwargs: loaded.update(kwargs),
        )
        monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)

        args = cli.create_parser().parse_args(
            ["serve", "local-test-model", "--chat-template", str(path)]
        )
        cli.serve_command(args)
        assert loaded["chat_template"] == OVERRIDE

        loaded.clear()
        cli.serve_command(
            cli.create_parser().parse_args(["serve", "local-test-model"])
        )
        assert loaded["chat_template"] is None

    def test_server_main(self, clean_server_state, monkeypatch, tmp_path):
        import sys

        server = clean_server_state
        path = tmp_path / "t.jinja"
        path.write_text(OVERRIDE, encoding="utf-8")
        loaded = {}
        monkeypatch.setattr(
            server, "load_model", lambda *args, **kwargs: loaded.update(kwargs)
        )
        monkeypatch.setattr(server, "load_embedding_model", lambda *a, **k: None)
        monkeypatch.setattr(server.uvicorn, "run", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            sys,
            "argv",
            ["vllm_mlx.server", "--model", "m", "--chat-template", str(path)],
        )
        server.main()
        assert loaded["chat_template"] == OVERRIDE


class TestMultimodalEnginesPassTheTemplate:
    @pytest.fixture
    def recorded_mllm(self, monkeypatch):
        import vllm_mlx.models.mllm as mllm_mod

        constructed = []

        class FakeMLXMultimodalLM:
            def __init__(self, model_name, **kwargs):
                constructed.append(kwargs)
                self.model = object()
                self.processor = object()

            def load(self):
                return None

        monkeypatch.setattr(mllm_mod, "MLXMultimodalLM", FakeMLXMultimodalLM)
        return constructed

    def test_simple_engine(self, recorded_mllm):
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine("org/vlm", force_mllm=True, chat_template=OVERRIDE)
        engine.prepare_for_start()
        assert recorded_mllm[-1]["chat_template"] == OVERRIDE

    def test_batched_engine(self, recorded_mllm, monkeypatch):
        from vllm_mlx.engine.batched import BatchedEngine

        import mlx.core as mx

        # The MLLM branch sets process-wide Metal limits when a GPU is present.
        monkeypatch.setattr(mx.metal, "is_available", lambda: False)
        engine = BatchedEngine("org/vlm", force_mllm=True, chat_template=OVERRIDE)
        engine.prepare_for_start()
        assert recorded_mllm[-1]["chat_template"] == OVERRIDE
