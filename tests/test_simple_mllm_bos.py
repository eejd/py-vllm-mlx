# SPDX-License-Identifier: Apache-2.0
"""The Simple-engine MLLM path tokenizes its prompts itself, per request kind.

``mlx_vlm.generate`` picks ``add_special_tokens`` with ``should_add_special_tokens``.
For gemma3/gemma4 that keys on ``processor.chat_template`` being unset, which is
always true of the custom ``Gemma3Processor`` mlx-vlm registers, so a chat prompt
the template already opened with ``<bos>`` reached Gemma-3 as ``[2, 2, 105, ...]``
(text-only and with images alike, ``usage.prompt_tokens`` one too high); every
other model type adds the tokenizer's markers unconditionally.

``MLXMultimodalLM`` now runs ``mlx_vlm.utils.prepare_inputs`` itself and hands
mlx-vlm the finished ``input_ids`` / ``pixel_values`` / ``mask`` plus any extra
processor outputs: chat prompts are tokenized with ``add_special_tokens=False``
(upstream vLLM's chat default; the template owns the BOS), raw ``generate`` /
``stream_generate`` prompts get the tokenizer's BOS unless they already start
with it.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vllm_mlx.models.mllm import MLXMultimodalLM

BOS = "<bos>"
RENDERED = f"{BOS}<start_of_turn>user\nDescribe this image<end_of_turn>\n"
RAW = "Once upon a time"


class FakeTokenizer:
    bos_token = BOS

    def encode(self, text, add_special_tokens=True):
        ids = [2 if w == BOS else 100 + len(w) for w in text.split()]
        return ([2] if add_special_tokens else []) + ids


@pytest.fixture
def boundary(monkeypatch):
    seen = {"prepare_inputs": [], "generate": [], "cache_fetch": []}

    def prepare_inputs(processor, **kwargs):
        seen["prepare_inputs"].append({"processor": processor, **kwargs})
        return {
            "input_ids": [[1, 2, 3]],
            "pixel_values": "pixels",
            "attention_mask": [[1, 1, 1]],
            "image_grid_thw": "grid",
        }

    def generate(model, processor, prompt, image=None, **kwargs):
        seen["generate"].append({"prompt": prompt, "image": image, **kwargs})
        return SimpleNamespace(text="ok", prompt_tokens=3, generation_tokens=1)

    vlm = ModuleType("mlx_vlm")
    vlm.generate = generate
    vlm.stream_generate = lambda *args, **kwargs: iter([generate(*args, **kwargs)])
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = lambda *args, **kwargs: RENDERED
    prompt_utils.get_chat_template = lambda *args, **kwargs: seen.get(
        "rendered", RENDERED
    )
    cache = ModuleType("mlx_vlm.models.cache")
    cache.make_prompt_cache = lambda *args, **kwargs: []
    models = ModuleType("mlx_vlm.models")
    models.cache = cache
    utils = ModuleType("mlx_vlm.utils")
    utils.prepare_inputs = prepare_inputs
    for module in (vlm, prompt_utils, models, cache, utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    wrapper = MLXMultimodalLM("test-model", enable_cache=True)
    wrapper._loaded = True
    wrapper.model = SimpleNamespace(
        language_model=object(), config=SimpleNamespace(image_token_index=7)
    )
    wrapper.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    wrapper.config = {}
    monkeypatch.setattr(wrapper, "_prepare_images", lambda images: ["prepared-image"])

    def fetch(images, prompt, token_ids):
        seen["cache_fetch"].append(token_ids)
        return None, 0

    wrapper._cache_manager = SimpleNamespace(
        fetch=fetch,
        fetch_cache=lambda images, prompt: (None, False),
        store=lambda **kw: None,
    )
    return wrapper, seen


CHAT = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image"},
            {"type": "image_url", "image_url": {"url": "test-image"}},
        ],
    }
]


def _drain(result):
    return list(result) if not hasattr(result, "text") else [result]


@pytest.mark.parametrize("route", ["chat", "stream_chat"])
def test_chat_prompt_is_tokenized_without_special_tokens(boundary, route):
    wrapper, seen = boundary
    _drain(getattr(wrapper, route)(messages=CHAT, max_tokens=4))
    (call,) = seen["prepare_inputs"]
    assert call["add_special_tokens"] is False
    assert call["prompts"] == RENDERED
    assert call["images"] == ["prepared-image"]
    assert call["image_token_index"] == 7


@pytest.mark.parametrize("route", ["chat", "stream_chat"])
def test_mlx_vlm_receives_the_prepared_tensors_and_extras(boundary, route):
    wrapper, seen = boundary
    _drain(getattr(wrapper, route)(messages=CHAT, max_tokens=4))
    (call,) = seen["generate"]
    # mlx-vlm skips its own tokenization (and its BOS heuristic) for these.
    assert call["input_ids"] == [[1, 2, 3]]
    assert call["pixel_values"] == "pixels"
    assert call["mask"] == [[1, 1, 1]]
    assert call["image_grid_thw"] == "grid"
    assert call["image"] == ["prepared-image"]


def test_text_only_chat_prompt_is_prepared_without_images(boundary):
    wrapper, seen = boundary
    wrapper.chat(messages=[{"role": "user", "content": "hello"}], max_tokens=4)
    (call,) = seen["prepare_inputs"]
    assert call["add_special_tokens"] is False
    assert call["images"] is None


def test_chat_prefix_cache_key_uses_the_ids_the_model_sees(boundary):
    wrapper, seen = boundary
    wrapper.chat(messages=CHAT, max_tokens=4)
    (token_ids,) = seen["cache_fetch"]
    assert token_ids == FakeTokenizer().encode(RENDERED, add_special_tokens=False)


@pytest.mark.parametrize("route", ["generate", "stream_generate"])
def test_raw_prompt_gets_the_tokenizer_bos(boundary, route):
    wrapper, seen = boundary
    _drain(getattr(wrapper, route)(prompt=RAW, max_tokens=4))
    (call,) = seen["prepare_inputs"]
    assert call["add_special_tokens"] is True


@pytest.mark.parametrize("route", ["generate", "stream_generate"])
def test_raw_prompt_carrying_its_bos_does_not_get_a_second(boundary, route):
    wrapper, seen = boundary
    _drain(getattr(wrapper, route)(prompt=RENDERED, max_tokens=4))
    (call,) = seen["prepare_inputs"]
    assert call["add_special_tokens"] is False


@pytest.mark.parametrize("route", ["chat", "stream_chat"])
def test_chat_template_without_a_bos_still_owns_the_markers(boundary, route):
    # vLLM tokenizes every rendered chat prompt with add_special_tokens=False,
    # also when the template renders no BOS (the rule alone would add one).
    wrapper, seen = boundary
    seen["rendered"] = "<|im_start|>user\nhello<|im_end|>\n"
    _drain(getattr(wrapper, route)(messages=CHAT, max_tokens=4))
    (call,) = seen["prepare_inputs"]
    assert call["add_special_tokens"] is False


class TestProcessorsThatDropTheFlag:
    """mlx-vlm forwards ``add_special_tokens`` only to a named ``__call__`` param."""

    def _mlx_vlm_prepare_inputs(self, monkeypatch):
        # Mirror mlx_vlm.utils.process_inputs' gating on the processor signature.
        import inspect

        seen = {}

        def prepare_inputs(processor, add_special_tokens=False, **kwargs):
            parameters = inspect.signature(processor.__call__).parameters
            call = {}
            if "add_special_tokens" in parameters:
                call["add_special_tokens"] = add_special_tokens
            seen["processor"] = processor
            seen["tokenizer_kwargs"] = processor(text="x", images=None, **call)
            seen["pad_token"] = processor.pad_token
            return {"input_ids": [[1]]}

        monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", prepare_inputs)
        return seen

    def _gemma_like(self):
        class GemmaLikeProcessor:
            """``__call__(images, text, **kwargs)`` handing kwargs to the tokenizer."""

            tokenizer = FakeTokenizer()
            pad_token = "<pad>"

            def __call__(self, images=None, text=None, **kwargs):
                return kwargs

        return GemmaLikeProcessor()

    @pytest.mark.parametrize("flag", [True, False])
    def test_flag_reaches_a_processor_that_only_takes_kwargs(self, monkeypatch, flag):
        from vllm_mlx.utils.vlm_inputs import prepare_inputs

        seen = self._mlx_vlm_prepare_inputs(monkeypatch)
        prepare_inputs(self._gemma_like(), add_special_tokens=flag)
        assert seen["tokenizer_kwargs"] == {"add_special_tokens": flag}
        assert seen["pad_token"] == "<pad>"  # attribute access still delegates

    def test_processor_with_a_named_parameter_is_passed_through(self, monkeypatch):
        from vllm_mlx.utils.vlm_inputs import prepare_inputs

        class Explicit:
            tokenizer = FakeTokenizer()
            pad_token = "<pad>"

            def __call__(self, images=None, text=None, add_special_tokens=True):
                return {"add_special_tokens": add_special_tokens}

        seen = self._mlx_vlm_prepare_inputs(monkeypatch)
        explicit = Explicit()
        prepare_inputs(explicit, add_special_tokens=False)
        assert seen["tokenizer_kwargs"] == {"add_special_tokens": False}
        assert seen["processor"] is explicit

    def test_simple_mllm_chat_with_images_uses_it(self, boundary, monkeypatch):
        wrapper, seen = boundary
        wrapper.processor = self._gemma_like()
        wrapper.chat(messages=CHAT, max_tokens=4)
        (call,) = seen["prepare_inputs"]
        # The fake mlx_vlm prepare_inputs receives the forwarding wrapper, which
        # is what makes the real one hand the flag to the processor.
        assert call["add_special_tokens"] is False
        assert type(call["processor"]).__name__ == "_SpecialTokensForwarder"

    def test_mllm_batch_generator_uses_it(self, monkeypatch):
        from types import SimpleNamespace

        from vllm_mlx.mllm_batch_generator import (
            MLLMBatchGenerator,
            MLLMBatchRequest,
            MLLMBatchStats,
        )

        seen = {}

        def fake_prepare_inputs(processor, **kwargs):
            seen["processor"] = processor
            seen.update(kwargs)
            import mlx.core as mx

            return {
                "input_ids": mx.array([[1, 2, 3]]),
                "attention_mask": mx.array([[1, 1, 1]]),
            }

        monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", fake_prepare_inputs)
        gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
        gen.processor = self._gemma_like()
        gen.model = SimpleNamespace(config=SimpleNamespace(image_token_index=7))
        gen.vision_cache = SimpleNamespace(
            get_pixel_cache=lambda *a, **k: None, set_pixel_cache=lambda **k: None
        )
        gen._stats = MLLMBatchStats()
        gen._preprocess_request(
            MLLMBatchRequest(
                uid=1, request_id="r", prompt=RENDERED, add_special_tokens=False
            )
        )
        assert seen["add_special_tokens"] is False
        assert type(seen["processor"]).__name__ == "_SpecialTokensForwarder"


@pytest.mark.parametrize("route", ["generate", "stream_generate"])
def test_raw_prompt_flagged_as_rendered_gets_no_bos(boundary, route):
    # What prompt warm-up sends: a rendered prefix, tokenized like a chat prompt.
    wrapper, seen = boundary
    _drain(getattr(wrapper, route)(prompt=RAW, max_tokens=4, add_special_tokens=False))
    (call,) = seen["prepare_inputs"]
    assert call["add_special_tokens"] is False
    # the kind is the wrapper's, not an mlx-vlm generation option
    assert "add_special_tokens" not in seen["generate"][0]


@pytest.mark.parametrize("route", ["generate", "stream_generate"])
def test_mlx_vlm_rendering_follows_the_raw_prompt_rule_not_the_caller_kind(
    boundary, route
):
    # With images the prompt is rendered by mlx-vlm, not by the caller's chat
    # template: it carries no promise about the BOS, so the rule decides.
    wrapper, seen = boundary
    sys.modules["mlx_vlm.prompt_utils"].apply_chat_template = lambda *a, **k: "[INST]x"
    wrapper.processor.apply_chat_template = lambda *a, **k: "[INST]x"
    _drain(
        getattr(wrapper, route)(
            prompt=RAW, images=["img"], max_tokens=4, add_special_tokens=False
        )
    )
    (call,) = seen["prepare_inputs"]
    assert call["prompts"] == "[INST]x"
    assert call["add_special_tokens"] is True


@pytest.mark.anyio
async def test_simple_engine_forwards_the_request_kind_to_the_mllm_model():
    from vllm_mlx.engine.simple import SimpleEngine

    calls = []

    def stream_generate(**kwargs):
        calls.append(kwargs)
        yield "ok"

    engine = SimpleEngine("test-model", force_mllm=True)
    engine._loaded = True
    engine._model = SimpleNamespace(
        stream_generate=stream_generate, tokenizer=FakeTokenizer()
    )
    async for _ in engine.stream_generate(RAW, max_tokens=4, add_special_tokens=False):
        pass
    async for _ in engine.stream_generate(RAW, max_tokens=4):
        pass
    assert [call["add_special_tokens"] for call in calls] == [False, None]
