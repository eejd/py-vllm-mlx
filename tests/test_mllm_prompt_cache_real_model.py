# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint check of the Simple-engine MLLM prompt-KV cache.

A request served from the prompt-KV cache must answer exactly like the same
request served cold, through ``generate``, ``chat`` and ``stream_chat``, with
the stored entry left unchanged by the hits. A hit prefills only the last prompt
token on a copy of the stored snapshot; the old code prefilled the whole prompt
again on top of the stored (prompt + output) layers.

Run with a local vision-language checkpoint (Gemma-3 mixes sliding-window and
global layers, which is the case that matters)::

    VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 \\
    VLLM_MLX_REAL_MLLM_MODEL=/path/to/gemma-3-4b-it-qat-4bit \\
    python -m pytest tests/test_mllm_prompt_cache_real_model.py --run-slow
"""

import base64
import io
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

from mlx_vlm.generate import dispatch  # noqa: E402

from vllm_mlx.models.mllm import MLXMultimodalLM  # noqa: E402

pytestmark = pytest.mark.slow

MAX_TOKENS = 24


def _model_path():
    if os.environ.get("VLLM_MLX_REAL_CACHE_MODEL_TESTS") != "1":
        pytest.skip("set VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 for real checkpoints")
    value = os.environ.get("VLLM_MLX_REAL_MLLM_MODEL")
    if not value or not Path(value).joinpath("config.json").is_file():
        pytest.fail("VLLM_MLX_REAL_MLLM_MODEL must name a complete local model")
    return value


def _image_url():
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([30, 30, 200, 200], fill="red")
    draw.ellipse([240, 240, 410, 410], fill="blue")
    draw.polygon([(300, 40), (420, 40), (360, 160)], fill="green")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


QUESTION = "Describe the shapes and colors in this image in one sentence."


@pytest.fixture(scope="module")
def model():
    loaded = MLXMultimodalLM(_model_path(), enable_cache=True)
    loaded.load()
    return loaded


@pytest.fixture
def steps(monkeypatch):
    """Every generate_step call: tokens produced and the cache it started from."""
    calls = []
    real = dispatch.generate_step

    def recording(input_ids, model, pixel_values, mask, **kwargs):
        cache = kwargs.get("prompt_cache")
        call = {
            "fed": int(input_ids.shape[1]),
            "offsets_before": [getattr(layer, "offset", None) for layer in cache],
            "cache": cache,
            "tokens": [],
        }
        calls.append(call)
        for token, logprobs in real(input_ids, model, pixel_values, mask, **kwargs):
            call["tokens"].append(int(token))
            yield token, logprobs

    monkeypatch.setattr(dispatch, "generate_step", recording)
    return calls


def _generate(model, **kw):
    return model.generate(
        QUESTION, images=[_image_url()], max_tokens=MAX_TOKENS, temperature=0.0, **kw
    ).text


def _messages():
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_url()}},
                {"type": "text", "text": QUESTION},
            ],
        }
    ]


def _chat(model, **kw):
    return model.chat(_messages(), max_tokens=MAX_TOKENS, temperature=0.0, **kw).text


def _stream_chat(model, **kw):
    return "".join(
        o.text
        for o in model.stream_chat(
            _messages(), max_tokens=MAX_TOKENS, temperature=0.0, **kw
        )
    )


ROUTES = {"generate": _generate, "chat": _chat, "stream_chat": _stream_chat}


def _entry(model):
    (entry,) = model._cache_manager._cache.values()
    return entry


@pytest.mark.parametrize("route", ROUTES)
def test_a_cache_hit_answers_exactly_like_a_cold_request(model, steps, route):
    ask = ROUTES[route]
    model.clear_cache()
    cold = ask(model, use_cache=False)
    miss = ask(model)
    entry = _entry(model)
    snapshot = [layer.offset for layer in entry.kv_cache]
    prompt_tokens = steps[1]["fed"]
    assert snapshot == [prompt_tokens - 1] * len(snapshot)

    hits = [ask(model) for _ in range(3)]

    assert miss == cold
    assert hits == [cold] * 3
    cold_tokens, miss_tokens, *hit_steps = steps
    assert miss_tokens["tokens"] == cold_tokens["tokens"]
    for hit in hit_steps:
        # Only the last prompt token is prefilled, onto the cached prompt.
        assert hit["fed"] == 1
        assert hit["offsets_before"] == snapshot
        assert hit["tokens"] == cold_tokens["tokens"]
    # The entry still holds the prompt: hits ran on copies.
    assert _entry(model) is entry
    assert [layer.offset for layer in entry.kv_cache] == snapshot
    assert all(
        layer is not stored
        for hit in hit_steps
        for layer in hit["cache"]
        for stored in entry.kv_cache
    )


def test_chat_then_stream_chat_share_one_snapshot(model, steps):
    model.clear_cache()
    cold = _chat(model, use_cache=False)
    assert _chat(model) == cold

    streamed = [_stream_chat(model), _stream_chat(model), _chat(model)]

    assert streamed == [cold] * 3
    cold_step, miss_step, *hit_steps = steps
    assert [step["fed"] for step in hit_steps] == [1, 1, 1]
    assert all(step["tokens"] == cold_step["tokens"] for step in hit_steps)
    snapshot = [layer.offset for layer in _entry(model).kv_cache]
    assert snapshot == [miss_step["fed"] - 1] * len(snapshot)


def test_text_only_requests_are_served_without_the_cache(model, steps):
    model.clear_cache()
    messages = [{"role": "user", "content": "Name three primary colors."}]

    first = model.chat(messages, max_tokens=MAX_TOKENS, temperature=0.0).text
    second = model.chat(messages, max_tokens=MAX_TOKENS, temperature=0.0).text

    assert first == second
    assert len(model._cache_manager) == 0
    assert [step["fed"] for step in steps] == [steps[0]["fed"]] * 2
