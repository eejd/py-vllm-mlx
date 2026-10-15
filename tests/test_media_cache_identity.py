# SPDX-License-Identifier: Apache-2.0
"""Media caches are keyed by everything that decides their content.

A cached pixel entry / prompt KV is only valid for the request that produced
it.  The key must therefore pin

* the image *bytes* (a URL string names no content: the same URL can serve a
  different picture a minute later),
* the prompt text and the way it is tokenised (``add_special_tokens``), and
* all of the digest, not a 48/64-bit prefix of it.

The Simple-engine fake below keeps KV rows ``position * 10000 + token`` where an
image placeholder token also carries the identity of the picture behind it, so
a prompt KV computed for another picture changes the answer.
"""

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
vlm_cache = pytest.importorskip("mlx_vlm.models.cache")
PromptCacheState = pytest.importorskip("mlx_vlm.generate").PromptCacheState

from vllm_mlx import mllm_cache  # noqa: E402
from vllm_mlx import vision_embedding_cache as vision_cache  # noqa: E402
from vllm_mlx.mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    MLLMBatchRequest,
)
from vllm_mlx.models.mllm import MLXMultimodalLM  # noqa: E402

BOS = "<bos>"
PLACEHOLDER = 7


# --- batched engine: pixel cache -------------------------------------------


@pytest.fixture
def batch_generator(monkeypatch, tmp_path):
    """A generator whose processor writes down what each request was tokenised as."""
    image = tmp_path / "image.png"
    image.write_bytes(b"picture")
    calls = []

    def prepare_inputs(
        processor, images, audio, prompts, image_token_index, add_special_tokens, **kw
    ):
        calls.append(add_special_tokens)
        ids = [PLACEHOLDER] + [ord(c) for c in prompts]
        if add_special_tokens:
            ids = [2] + ids
        return {
            "input_ids": mx.array([ids]),
            "pixel_values": mx.zeros((1, 1)),
            "attention_mask": mx.ones((1, len(ids))),
        }

    monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", prepare_inputs)
    monkeypatch.setattr("vllm_mlx.models.mllm.process_image_input", lambda image: image)
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.processor = SimpleNamespace(tokenizer=SimpleNamespace(bos_token=BOS))
    generator.model = SimpleNamespace(config=None)
    generator.vision_cache = vision_cache.VisionEmbeddingCache()
    generator._stats = SimpleNamespace(num_images_processed=0, vision_encoding_time=0.0)
    generator.calls = calls
    generator.image = str(image)
    return generator


def preprocess(generator, prompt, add_special_tokens=None):
    request = MLLMBatchRequest(
        uid=1,
        request_id="r",
        prompt=prompt,
        images=[generator.image],
        add_special_tokens=add_special_tokens,
    )
    generator._preprocess_request(request)
    return request.input_ids.reshape(-1).tolist()


def test_batched_pixel_cache_keeps_the_request_kind(batch_generator):
    """The same words as a chat prompt (nothing added) and as a raw completion
    (the tokenizer's BOS added) are different token ids; each request must get
    the ids of its own kind."""
    prompt = "describe it"
    chat_ids = preprocess(batch_generator, prompt, add_special_tokens=False)
    raw_ids = preprocess(batch_generator, prompt, add_special_tokens=None)

    assert chat_ids[0] == PLACEHOLDER
    assert raw_ids == [2] + chat_ids
    assert preprocess(batch_generator, prompt, add_special_tokens=False) == chat_ids
    assert batch_generator.calls == [False, True]  # the repeat was a hit


# Two ordinary prompts whose SHA-256 agree on the first 12 hex digits.
COLLIDING_PROMPTS = (
    "What is in image number 570330?",
    "What is in image number 14123373?",
)


def test_the_colliding_prompts_really_collide_on_the_old_key_prefix():
    first, second = (hashlib.sha256(p.encode()).hexdigest() for p in COLLIDING_PROMPTS)
    assert first != second
    assert first[:12] == second[:12]


def test_batched_pixel_cache_does_not_serve_a_prompt_with_a_colliding_digest(
    batch_generator,
):
    first, second = COLLIDING_PROMPTS
    assert preprocess(batch_generator, first) == [2, PLACEHOLDER] + [
        ord(c) for c in first
    ]
    assert preprocess(batch_generator, second) == [2, PLACEHOLDER] + [
        ord(c) for c in second
    ]


# --- a digest prefix that is shared must not share an entry -----------------


class _PrefixCollidingHashlib:
    """``hashlib`` whose SHA-256 agrees on the first 16 hex digits for all inputs."""

    @staticmethod
    def sha256(data=b""):
        real = hashlib.sha256(data)

        class Digest:
            def hexdigest(self_inner):
                return "0" * 16 + real.hexdigest()[16:]

        return Digest()


@pytest.mark.parametrize("module", [vision_cache, mllm_cache])
def test_inputs_whose_digests_share_a_64_bit_prefix_stay_distinct(
    module, monkeypatch, tmp_path
):
    first, second = tmp_path / "a.png", tmp_path / "b.png"
    first.write_bytes(b"first picture")
    second.write_bytes(b"second picture")
    monkeypatch.setattr(vision_cache, "hashlib", _PrefixCollidingHashlib)
    monkeypatch.setattr(mllm_cache, "hashlib", _PrefixCollidingHashlib)

    assert module.compute_images_hash([str(first)]) != module.compute_images_hash(
        [str(second)]
    )
    cache = vision_cache.VisionEmbeddingCache()
    if module is vision_cache:
        cache.set_pixel_cache([str(first)], "p", mx.zeros((1, 1)), mx.zeros((1, 2)))
        assert cache.get_pixel_cache([str(first)], "another prompt") is None
        assert cache.get_pixel_cache([str(second)], "p") is None
    else:
        manager = mllm_cache.MLLMPrefixCacheManager()
        manager.store_cache([str(first)], "p", [object()], num_tokens=3)
        assert manager.fetch_cache([str(first)], "another prompt") == (None, False)
        assert manager.fetch_cache([str(second)], "p") == (None, False)


# --- remote sources name no content ------------------------------------------

REMOTE = "https://images.example.test/latest.png"


def test_the_vision_cache_refuses_a_remote_url_it_cannot_pin():
    cache = vision_cache.VisionEmbeddingCache()
    cache.set_pixel_cache([REMOTE], "p", mx.zeros((1, 1)), mx.zeros((1, 2)))

    assert cache.get_pixel_cache([REMOTE], "p") is None


def test_the_prefix_manager_refuses_a_remote_url_it_cannot_pin():
    manager = mllm_cache.MLLMPrefixCacheManager()
    manager.store_cache([REMOTE], "p", [object()], num_tokens=3)
    manager.store(
        [REMOTE], "p", "features", [object()], token_ids=[1, 2, 3], num_image_tokens=1
    )

    assert len(manager) == 0
    assert manager.fetch_cache([REMOTE], "p") == (None, False)
    assert manager.fetch([REMOTE], "p", [1, 2, 3]) == (None, 0)


# --- Simple engine: the prompt KV of a URL whose picture changed -------------


def feed(layer, ids, tag):
    first = layer.offset
    rows = [
        [(first + i) * 10000 + t + (1000 * tag if t == PLACEHOLDER else 0), 0]
        for i, t in enumerate(ids)
    ]
    keys = mx.array([[rows]], dtype=mx.float32)
    layer.update_and_fetch(keys, keys)


def seen_rows(layer):
    keys = layer.keys[0, 0, : layer.offset, 0]
    return [(int(v) // 10000, int(v) % 10000) for v in keys.tolist()]


class FakeMlxVlm:
    """mlx-vlm's prompt-cache contract; the answer reads every cached row."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def make_prompt_cache(language_model, max_kv_size=None):
        return [vlm_cache.KVCache()]

    def _run(self, prompt_cache, state, input_ids, pixel_values, max_tokens):
        ids = [int(t) for t in input_ids.flatten().tolist()]
        tag = int(pixel_values.sum().item()) if pixel_values is not None else 0
        cache, feed_ids = prompt_cache, ids
        if state is not None and state.cache is not None:
            prefix = state.find_prefix_length(ids)
            if 0 < prefix < len(ids):
                cache, feed_ids = state.cache, ids[prefix:]
                for layer in cache:
                    layer.trim(max(0, layer.offset - prefix))
        self.calls.append({"fed": feed_ids, "prompt_ids": ids})
        for layer in cache:
            feed(layer, feed_ids, tag)
        answer = 800 + sum(p * t for p, t in seen_rows(cache[0])) % 97
        generated = [answer + j for j in range(max_tokens)]
        for token in generated[:-1]:
            for layer in cache:
                feed(layer, [token], tag)
        if state is not None:
            state.pending = (ids + generated, cache)
        return ids, generated

    def generate(
        self,
        model,
        processor,
        prompt,
        image=None,
        *,
        prompt_cache=None,
        prompt_cache_state=None,
        input_ids,
        pixel_values=None,
        max_tokens=3,
        **kwargs,
    ):
        ids, generated = self._run(
            prompt_cache, prompt_cache_state, input_ids, pixel_values, max_tokens
        )
        if prompt_cache_state is not None:
            prompt_cache_state.update(*prompt_cache_state.pending)
        return SimpleNamespace(
            text=" ".join(map(str, generated)),
            prompt_tokens=len(ids),
            generation_tokens=len(generated),
            finish_reason="length",
        )

    def stream_generate(
        self,
        model,
        processor,
        prompt,
        image=None,
        *,
        prompt_cache=None,
        prompt_cache_state=None,
        input_ids,
        pixel_values=None,
        max_tokens=3,
        **kwargs,
    ):
        ids, generated = self._run(
            prompt_cache, prompt_cache_state, input_ids, pixel_values, max_tokens
        )
        for n, token in enumerate(generated):
            yield SimpleNamespace(
                text=f"{token} ",
                prompt_tokens=len(ids),
                generation_tokens=n + 1,
                finish_reason="length" if n == len(generated) - 1 else None,
            )
        if prompt_cache_state is not None:
            prompt_cache_state.update(*prompt_cache_state.pending)


QUESTION = "what is shown"
PICTURES = {}


def _prepare_inputs(processor, images=None, prompts=None, **kwargs):
    tags = [PICTURES[open(path, "rb").read()] for path in images or []]
    ids = [2] + [PLACEHOLDER] * (4 * len(tags)) + [10 + ord(c) % 90 for c in prompts]
    return {
        "input_ids": mx.array([ids]),
        "pixel_values": mx.array(tags, dtype=mx.float32) if tags else None,
        "attention_mask": mx.ones((1, len(ids))),
    }


@pytest.fixture
def engine(monkeypatch, tmp_path):
    PICTURES.clear()
    PICTURES.update({b"red picture": 1, b"blue picture": 2})
    served = {REMOTE: b"red picture"}
    vlm = FakeMlxVlm()

    def download_image(url, **kwargs):
        path = tmp_path / f"download-{len(list(tmp_path.iterdir()))}.png"
        path.write_bytes(served[url])
        return str(path)

    monkeypatch.setattr("vllm_mlx.models.mllm.download_image", download_image)
    package = ModuleType("mlx_vlm")
    package.generate = vlm.generate
    package.stream_generate = vlm.stream_generate
    generate_module = ModuleType("mlx_vlm.generate")
    generate_module.PromptCacheState = PromptCacheState
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = lambda *a, **k: BOS + a[2]
    prompt_utils.get_chat_template = lambda processor, messages, **k: BOS + " ".join(
        part["text"]
        for m in messages
        for part in (m["content"] if isinstance(m["content"], list) else [])
        if part.get("type") == "text"
    )
    cache = ModuleType("mlx_vlm.models.cache")
    cache.make_prompt_cache = vlm.make_prompt_cache
    models = ModuleType("mlx_vlm.models")
    models.cache = cache
    utils = ModuleType("mlx_vlm.utils")
    utils.prepare_inputs = _prepare_inputs
    for module in (package, generate_module, prompt_utils, models, cache, utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    model = MLXMultimodalLM("fake", enable_cache=True)
    model._loaded = True
    model.model = SimpleNamespace(
        language_model=object(), config=SimpleNamespace(image_token_index=PLACEHOLDER)
    )
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(bos_token=BOS))
    model.config = {}
    return model, vlm, served


CHAT = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": REMOTE}},
            {"type": "text", "text": QUESTION},
        ],
    }
]


def ask_generate(model, **kw):
    return model.generate(
        QUESTION, images=[REMOTE], max_tokens=3, temperature=0.0, **kw
    ).text


def ask_chat(model, **kw):
    return model.chat(CHAT, max_tokens=3, temperature=0.0, **kw).text


def ask_stream(model, **kw):
    return "".join(
        o.text for o in model.stream_chat(CHAT, max_tokens=3, temperature=0.0, **kw)
    )


@pytest.mark.parametrize(
    "ask", [ask_generate, ask_chat, ask_stream], ids=["generate", "chat", "stream_chat"]
)
def test_a_url_that_now_serves_another_picture_is_not_answered_from_the_old_kv(
    engine, ask
):
    model, vlm, served = engine
    ask(model)  # the red picture's prompt KV is stored
    served[REMOTE] = b"blue picture"

    warm = ask(model)
    cold = ask(model, use_cache=False)

    assert warm == cold


@pytest.mark.parametrize(
    "ask", [ask_generate, ask_chat, ask_stream], ids=["generate", "chat", "stream_chat"]
)
def test_the_same_picture_behind_one_url_is_still_a_prompt_cache_hit(engine, ask):
    model, vlm, served = engine
    ask(model)
    cold_fed = vlm.calls[0]["fed"]

    ask(model)

    assert vlm.calls[1]["fed"] == cold_fed[-1:]


def test_the_same_picture_behind_two_urls_shares_one_entry(engine):
    """Content, not the name it is fetched under, identifies the prompt KV."""
    model, vlm, served = engine
    other = "https://cdn.example.test/copy.png"
    served[other] = served[REMOTE]
    cold = model.generate(QUESTION, images=[REMOTE], max_tokens=3, temperature=0.0).text

    hit = model.generate(QUESTION, images=[other], max_tokens=3, temperature=0.0).text

    assert vlm.calls[1]["fed"] == vlm.calls[0]["fed"][-1:]
    assert hit == cold
