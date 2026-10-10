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
