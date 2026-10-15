# SPDX-License-Identifier: Apache-2.0
"""Opt-in real-checkpoint check that media caches never serve another request's data.

A request answered with a warm cache must say exactly what the same request
says cold (temperature 0): the pixel/KV entry another picture or another
prompt left behind may not leak into it.

Run against a local vision-language checkpoint (the Gemma-3 and Qwen3-VL
families both matter: Qwen's 3-D rotary positions depend on the picture's
grid)::

    VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 \\
    VLLM_MLX_REAL_MLLM_MODEL=/path/to/gemma-3-4b-it-qat-4bit \\
    python -m pytest tests/test_media_cache_identity_real_model.py -m slow --run-slow
"""

import asyncio
import base64
import hashlib
import io
import itertools
import os
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
Image = pytest.importorskip("PIL.Image")
ImageDraw = pytest.importorskip("PIL.ImageDraw")

from vllm_mlx.engine.batched import BatchedEngine  # noqa: E402
from vllm_mlx.models.mllm import MLXMultimodalLM  # noqa: E402

pytestmark = pytest.mark.slow

MAX_TOKENS = 24
URL = "https://pictures.example.test/latest.png"
COLOR_QUESTION = "What color is this picture? Answer with one word."


def _model_path():
    if os.environ.get("VLLM_MLX_REAL_CACHE_MODEL_TESTS") != "1":
        pytest.skip("set VLLM_MLX_REAL_CACHE_MODEL_TESTS=1 for real checkpoints")
    value = os.environ.get("VLLM_MLX_REAL_MLLM_MODEL")
    if not value or not Path(value).joinpath("config.json").is_file():
        pytest.fail("VLLM_MLX_REAL_MLLM_MODEL must name a complete local model")
    return value


def _png(color, size=(448, 448)):
    image = Image.new("RGB", size, color)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "photo", fill="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _data_url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


RED = _png("red")
BLUE = _png("blue")
# A different aspect ratio: Qwen's patch grid (and so every image position's
# rotary index) changes with it, Gemma resizes every picture to one size.
TALL_GREEN = _png("green", size=(336, 560))


@pytest.fixture(scope="module")
def model():
    loaded = MLXMultimodalLM(_model_path(), enable_cache=True)
    loaded.load()
    return loaded


@pytest.fixture
def served(monkeypatch, tmp_path):
    """The bytes the test URL currently serves."""
    content = {URL: RED}

    def download_image(url, **kwargs):
        path = tmp_path / f"download-{len(list(tmp_path.iterdir()))}.png"
        path.write_bytes(content[url])
        return str(path)

    monkeypatch.setattr("vllm_mlx.models.mllm.download_image", download_image)
    return content


def _messages(url, question=COLOR_QUESTION):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": question},
            ],
        }
    ]


def _generate(model, url, **kw):
    return model.generate(
        COLOR_QUESTION, images=[url], max_tokens=MAX_TOKENS, temperature=0.0, **kw
    ).text


def _chat(model, url, **kw):
    return model.chat(_messages(url), max_tokens=MAX_TOKENS, temperature=0.0, **kw).text


def _stream_chat(model, url, **kw):
    return "".join(
        o.text
        for o in model.stream_chat(
            _messages(url), max_tokens=MAX_TOKENS, temperature=0.0, **kw
        )
    )


ROUTES = {"generate": _generate, "chat": _chat, "stream_chat": _stream_chat}


@pytest.mark.parametrize("route", ROUTES)
def test_a_url_that_serves_another_picture_is_answered_for_that_picture(
    model, served, route
):
    ask = ROUTES[route]
    model.clear_cache()
    cold_red = ask(model, URL, use_cache=False)
    ask(model, URL)  # the red picture's prompt KV is stored
    served[URL] = BLUE
    cold_blue = ask(model, URL, use_cache=False)
    assert cold_red != cold_blue, "the model must tell the two pictures apart"

    warm_blue = ask(model, URL)

    assert warm_blue == cold_blue


@pytest.mark.parametrize("route", ROUTES)
def test_interleaved_pictures_each_get_their_own_answer(model, served, route):
    """Another picture served in between (another patch grid for Qwen-VL, so
    another rotary state on the shared model) leaves the first one's hit exact."""
    ask = ROUTES[route]
    model.clear_cache()
    first = _data_url(RED)
    second = _data_url(TALL_GREEN)
    cold_first = ask(model, first, use_cache=False)
    cold_second = ask(model, second, use_cache=False)

    answers = [ask(model, url) for url in (first, second, first, second, first)]

    assert answers == [cold_first, cold_second, cold_first, cold_second, cold_first]


# --- the batched engine ------------------------------------------------------


@pytest.fixture(scope="module")
def batched():
    path = _model_path()
    loop = asyncio.new_event_loop()
    engine = BatchedEngine(path, force_mllm=True)
    loop.run_until_complete(engine.start())
    yield loop, engine
    loop.run_until_complete(engine.stop())
    loop.close()


def _forget_media(engine):
    # The generator (and its pixel cache) exists once a request has run.
    generator = engine._mllm_scheduler.batch_generator
    if generator is not None:
        generator.vision_cache.clear()


def _chat_prompt(engine, question):
    return engine._apply_chat_template(
        _messages(_data_url(RED), question), None, num_images=1, num_audios=0
    )


def _batched(loop, engine, prompt, **kw):
    out = loop.run_until_complete(
        engine.generate(
            prompt=prompt,
            images=[_data_url(RED)],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            **kw,
        )
    )
    return out.text, out.prompt_tokens


def test_batched_pixel_cache_keeps_the_request_kind(batched):
    """One prompt text as a rendered chat prompt (no BOS is added) and as a raw
    completion (the tokenizer adds its BOS) is two different token sequences."""
    loop, engine = batched
    rendered = _chat_prompt(engine, COLOR_QUESTION)
    bos = getattr(engine._processor.tokenizer, "bos_token", None)
    if not bos:
        pytest.skip("this tokenizer adds no BOS, so the two kinds tokenise alike")
    assert rendered.startswith(bos)
    prompt = rendered[len(bos) :]

    _forget_media(engine)
    cold_raw = _batched(loop, engine, prompt, add_special_tokens=None)
    _forget_media(engine)
    chat_kind = _batched(loop, engine, prompt, add_special_tokens=False)
    warm_raw = _batched(loop, engine, prompt, add_special_tokens=None)

    assert chat_kind[1] == cold_raw[1] - 1  # the BOS is the one token of difference
    assert warm_raw == cold_raw


def _prompts_sharing_a_12_digit_prefix(engine):
    """Two rendered chat prompts (same picture slot, different question) whose
    SHA-256 agree on the first 12 hex digits -- the length of the digest prefix
    the pixel cache used to key the prompt by.  A birthday search over the
    question: about 2**24 candidates, seconds of hashing."""
    marker = "@@"
    template = _chat_prompt(engine, f"What is in image number {marker}?")
    seen = {}
    for i in itertools.count():
        text = template.replace(marker, f"{i}" + " very" * (i % 5))
        other = seen.setdefault(hashlib.sha256(text.encode()).hexdigest()[:12], text)
        if len(other) != len(text):  # same-length texts can tokenise alike
            return other, text


def test_batched_pixel_cache_does_not_answer_a_colliding_prompt(batched):
    loop, engine = batched
    first, second = _prompts_sharing_a_12_digit_prefix(engine)
    _forget_media(engine)
    cold_first = _batched(loop, engine, first, add_special_tokens=False)
    _forget_media(engine)
    cold_second = _batched(loop, engine, second, add_special_tokens=False)
    assert cold_first[1] != cold_second[1]

    _forget_media(engine)
    _batched(loop, engine, first, add_special_tokens=False)
    warm_second = _batched(loop, engine, second, add_special_tokens=False)

    assert warm_second == cold_second
