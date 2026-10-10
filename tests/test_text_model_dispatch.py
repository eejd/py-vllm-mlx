# SPDX-License-Identifier: Apache-2.0
"""Text-model class selection for VLM-derived TextModels.

The dispatch used to match one exact ``model_type`` string and send everything
else to the Qwen3.5 text model. That is a guess, and a wrong guess does not
fail where it is made: it fails deep inside the chosen constructor with an
error naming neither the model nor the class, ``build_text_model`` returns
None, and the engine carries on with ``_text_model=None`` — a route quietly
losing its backend.

Concretely, Gemma 4 reports ``gemma4_text`` on some checkpoints and
``gemma4_unified_text`` on others. The latter reached ``qwen3_5.TextModelArgs``,
which leaves ``num_experts`` as None, and died on ``args.num_experts > 0``.
"""

import logging

import pytest

pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from vllm_mlx.text_model_from_vlm import (  # noqa: E402
    _import_text_model_classes,
    build_text_model,
)


@pytest.mark.parametrize(
    "model_type",
    ["gemma4_text", "gemma4_unified_text", "gemma4_unified", "gemma4"],
)
def test_every_gemma4_variant_gets_the_gemma4_text_model(model_type):
    """A family must cover its variants, not one spelling of it."""
    Model, ModelArgs = _import_text_model_classes(model_type)
    assert Model.__module__ == "mlx_lm.models.gemma4_text", (
        f"{model_type!r} selected {Model.__module__}.{Model.__qualname__}; "
        "a Gemma 4 config passed to another family dies on a field it has no "
        "opinion about"
    )
    assert ModelArgs.__module__ == "mlx_lm.models.gemma4_text"


def test_gemma4_unified_text_config_actually_constructs():
    """The regression, end to end: this config used to raise.

    ``qwen3_5.TextModelArgs.from_dict`` leaves ``num_experts`` as None for a
    Gemma 4 config, and ``qwen3_5.DecoderLayer.__init__`` compares it to 0:
    ``TypeError: '>' not supported between instances of 'NoneType' and 'int'``.
    """
    text_config = {
        "model_type": "gemma4_unified_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
    }
    _, ModelArgs = _import_text_model_classes(text_config["model_type"])
    args = ModelArgs.from_dict(text_config)
    assert getattr(args, "num_hidden_layers", None) == 2


@pytest.mark.parametrize("model_type", ["qwen3_5_text", "qwen3_6_text", "", "unknown"])
def test_unmatched_types_keep_the_generic_fallback(model_type):
    """Unknown families must not start raising — that would be a regression.

    qwen3_5.TextModel handles dense and MoE natively, so it stays the default.
    """
    Model, ModelArgs = _import_text_model_classes(model_type)
    assert Model.__module__ == "mlx_lm.models.qwen3_5"
    assert Model.__qualname__ == "TextModel"
    assert ModelArgs.__qualname__ == "TextModelArgs"


def test_failure_names_the_model_type_and_the_chosen_class(tmp_path, caplog):
    """The old log line was a bare TypeError from someone else's constructor.

    Without the model_type and the class that was picked, the only way to find
    out which family was guessed is to bisect the dispatch by hand.
    """
    (tmp_path / "config.json").write_text(
        '{"text_config": {"model_type": "gemma4_unified_text", '
        '"num_hidden_layers": "not-an-int"}}'
    )

    class _Vlm:
        language_model = object()

    with caplog.at_level(logging.ERROR, logger="vllm_mlx.text_model_from_vlm"):
        assert build_text_model(_Vlm(), tmp_path) is None

    assert caplog.records, "a failed build must be logged"
    message = caplog.records[-1].getMessage()
    assert "gemma4_unified_text" in message, message
    assert "mlx_lm.models.gemma4_text" in message, message
    assert caplog.records[-1].exc_info is not None, "traceback must be preserved"


def test_missing_config_is_not_reported_as_a_build_failure(tmp_path, caplog):
    """No config.json is a "not applicable", not an error worth a traceback."""

    class _Vlm:
        language_model = object()

    with caplog.at_level(logging.ERROR, logger="vllm_mlx.text_model_from_vlm"):
        assert build_text_model(_Vlm(), tmp_path) is None
    assert not caplog.records


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_qwen4_exp_stays_on_mlx_vlm_text_path(tmp_path, caplog, model_type):
    """Qwen4-Exp is not compatible with the generic Qwen3.5 TextModel."""
    (tmp_path / "config.json").write_text(
        '{"text_config": {"model_type": "' + model_type + '"}}'
    )

    class _Vlm:
        language_model = object()

    with caplog.at_level(logging.INFO, logger="vllm_mlx.text_model_from_vlm"):
        assert build_text_model(_Vlm(), tmp_path) is None

    assert "mlx-vlm text path" in caplog.records[-1].getMessage()
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]


# --- Ministral 3 (top-level mistral3, text_config.model_type ministral3) ---------
#
# It used to reach the generic Qwen3.5 fallback. That skeleton builds (hybrid
# layers, qk-norm) and loads under strict=False with nothing to say, then dies on
# the first request: ``[rms_norm] (*weight) must have the same size as the last
# dimension of x but has 128 elements`` (the 128 is Ministral-3's head_dim).

_TINY_MINISTRAL3 = {
    "model_type": "ministral3",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 128,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 4096,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "llama_4_scaling_beta": 0.1,
        "original_max_position_embeddings": 1024,
    },
}


def test_ministral3_gets_the_ministral3_text_model():
    Model, ModelArgs = _import_text_model_classes("ministral3")
    assert Model.__module__ == "mlx_lm.models.ministral3"
    assert ModelArgs.__module__ == "mlx_lm.models.ministral3"


def test_ministral3_text_model_matches_the_loaded_vlm_logits(tmp_path):
    """The extracted TextModel must be the model the vlm loader loaded.

    Ministral-3 keeps ``tie_word_embeddings: false`` at the top level of
    config.json, outside the ``text_config`` the TextModel is built from, and
    mlx-lm's ``ministral3.ModelArgs`` defaults to tied. Dispatch alone therefore
    produced a skeleton with no ``lm_head``: the real head was dropped by
    ``load_weights(strict=False)`` and logits came from the embedding table,
    a model that runs and answers garbage.
    """
    import json

    import mlx.core as mx
    from mlx_lm.models import ministral3

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "mistral3",
                "tie_word_embeddings": False,
                "text_config": _TINY_MINISTRAL3,
            }
        )
    )

    # What mlx_vlm's language model looks like for this checkpoint: an untied
    # ``model.*`` tree plus ``lm_head``, same parameter names as mlx-lm's.
    vlm_language_model = ministral3.Model(
        ministral3.ModelArgs.from_dict(
            {**_TINY_MINISTRAL3, "tie_word_embeddings": False}
        )
    )
    mx.eval(vlm_language_model.parameters())

    class _Vlm:
        language_model = vlm_language_model

    text_model = build_text_model(_Vlm(), tmp_path, enable_mtp=False)

    assert text_model is not None
    assert type(text_model).__module__ == "mlx_lm.models.ministral3"
    assert hasattr(text_model, "lm_head")
    tokens = mx.array([[3, 17, 99, 5, 42, 8]])
    expected, actual = vlm_language_model(tokens), text_model(tokens)
    mx.eval(expected, actual)
    assert mx.allclose(expected, actual).item()


@pytest.mark.parametrize(
    "tied,weight_names,expect_tied",
    [
        (True, ["model.embed_tokens.weight", "lm_head.weight"], False),
        (True, ["model.embed_tokens.weight", "lm_head.scales"], False),
        (False, ["model.embed_tokens.weight"], True),
        (True, ["model.embed_tokens.weight"], True),
        (False, ["model.embed_tokens.weight", "lm_head.weight"], False),
    ],
)
def test_tied_embeddings_follow_the_vlm_weights(tied, weight_names, expect_tied):
    from types import SimpleNamespace

    from vllm_mlx.text_model_from_vlm import _align_tied_embeddings

    args = SimpleNamespace(tie_word_embeddings=tied)
    _align_tied_embeddings(args, [(name, None) for name in weight_names], "x")
    assert args.tie_word_embeddings is expect_tied
