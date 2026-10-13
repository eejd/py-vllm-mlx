# SPDX-License-Identifier: Apache-2.0
"""Batched image prefill follows mlx-vlm's own calling convention.

Under ``--continuous-batching`` the dense image prefill called the model's
``__call__`` with ``attention_mask=``.  mlx-vlm names that argument ``mask`` and
every family's ``get_input_embeddings`` reads it under that name: Gemma-3
crashed (``expand_dims(None)``), Mistral/Ministral-3 required ``mask``
positionally, and Qwen-VL silently ignored the mask.  Renaming the argument is
not enough for Gemma-3, whose ``__call__`` also hands the language model an
all-ones 4D mask (bidirectional prompt attention), so the prefill runs
``get_input_embeddings`` + the language model exactly as
``mlx_vlm.generate.ar`` does.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    MLLMBatchRequest,
)
from vllm_mlx.vision_embedding_cache import VisionEmbeddingCache  # noqa: E402

SEQ = 6
VOCAB = 5


class _Features:
    """Stands in for mlx_vlm.models.base.InputEmbeddingsFeatures."""

    def __init__(self, inputs_embeds, **extra):
        self.inputs_embeds = inputs_embeds
        self.extra = extra

    def to_dict(self):
        base = {
            "inputs_embeds": self.inputs_embeds,
            "attention_mask_4d": None,
            "visual_pos_masks": None,
            "position_ids": None,
            "rope_deltas": None,
        }
        base.update(self.extra)
        return base


class _Recorder:
    """Records what the model and the language model were called with."""

    def __init__(self, features_extra=None, require_mask=False):
        self.embedding_call = None
        self.language_call = None
        self.features_extra = features_extra or {}
        self.require_mask = require_mask

    def get_input_embeddings(self, input_ids=None, pixel_values=None, **kwargs):
        if self.require_mask and kwargs.get("mask") is None:
            raise TypeError("expand_dims(): incompatible function arguments")
        self.embedding_call = (input_ids, pixel_values, kwargs)
        return _Features(mx.ones((1, input_ids.shape[1], 3)), **self.features_extra)

    def language_model(self, inputs, **kwargs):
        self.language_call = (inputs, kwargs)
        return SimpleNamespace(logits=mx.zeros((1, inputs.shape[1], VOCAB)))


def _generator(recorder):
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    # The model object has no usable __call__: the prefill must not need it.
    generator.model = SimpleNamespace(
        get_input_embeddings=recorder.get_input_embeddings
    )
    generator.language_model = recorder.language_model
    return generator


def _request(**fields):
    request = MLLMBatchRequest(uid=1, request_id="img", prompt="p")
    request.input_ids = mx.arange(SEQ)[None, :]
    request.pixel_values = mx.ones((1, 3, 4, 4))
    request.attention_mask = mx.ones((1, SEQ), dtype=mx.int32)
    for name, value in fields.items():
        setattr(request, name, value)
    return request


def test_mask_reaches_get_input_embeddings_as_mask():
    """The Gemma-3 crash: its embedding step needs ``mask=`` (not None)."""
    recorder = _Recorder(require_mask=True)
    request = _request()
    mask = request.attention_mask

    logits = _generator(recorder)._run_vision_encoding(request, cache=["cache"])

    input_ids, pixel_values, kwargs = recorder.embedding_call
    assert kwargs["mask"] is mask
    assert "attention_mask" not in kwargs
    assert input_ids.shape == (1, SEQ)
    assert pixel_values.shape == (1, 3, 4, 4)
    assert logits.shape == (1, SEQ, VOCAB)


def test_language_model_gets_embeddings_and_cache_but_no_raw_mask():
    """Gemma-3's ``__call__`` fed the LM an all-ones 4D mask; the LM must only
    see what the embedding step returned (``attention_mask_4d`` is ignored by
    its LM, as in mlx-vlm), never the processor's 2D mask."""
    recorder = _Recorder(features_extra={"attention_mask_4d": mx.ones((1, 1, 2, 2))})
    request = _request()
    generator = _generator(recorder)

    generator._run_vision_encoding(request, cache=["layer-cache"])

    inputs, kwargs = recorder.language_call
    assert inputs.shape == (1, SEQ)
    assert kwargs["inputs_embeds"].shape == (1, SEQ, 3)
    assert kwargs["cache"] == ["layer-cache"]
    assert "mask" not in kwargs
    assert "attention_mask" not in kwargs
    assert kwargs["attention_mask_4d"].shape == (1, 1, 2, 2)
    # The pixels are consumed by the embedding step only.
    assert "pixel_values" not in kwargs


def test_processor_outputs_reach_both_embedding_step_and_language_model():
    """Mistral/Ministral-3 ``image_sizes`` and Qwen-VL ``image_grid_thw``."""
    recorder = _Recorder()
    sizes = mx.array([[448, 448]])
    grid = mx.array([[1, 28, 28]])
    request = _request(image_grid_thw=grid, extra_kwargs={"image_sizes": sizes})

    _generator(recorder)._run_vision_encoding(request, cache=None)

    _, _, embed_kwargs = recorder.embedding_call
    assert embed_kwargs["image_sizes"] is sizes
    assert embed_kwargs["image_grid_thw"] is grid
    _, lm_kwargs = recorder.language_call
    assert lm_kwargs["image_sizes"] is sizes
    assert lm_kwargs["image_grid_thw"] is grid


def test_embedding_outputs_are_forwarded_and_unset_ones_dropped():
    """Qwen-VL mRoPE state computed by the embedding step drives the LM."""
    positions = mx.zeros((3, 1, SEQ))
    deltas = mx.array([[7]])
    recorder = _Recorder(
        features_extra={"position_ids": positions, "rope_deltas": deltas}
    )

    _generator(recorder)._run_vision_encoding(_request(), cache=None)

    _, lm_kwargs = recorder.language_call
    assert lm_kwargs["position_ids"] is positions
    assert lm_kwargs["rope_deltas"] is deltas
    for unset in ("visual_pos_masks", "attention_mask_4d"):
        assert unset not in lm_kwargs


def test_one_dimensional_inputs_are_batched():
    recorder = _Recorder(require_mask=True)
    request = _request(
        input_ids=mx.arange(SEQ), attention_mask=mx.ones((SEQ,), dtype=mx.int32)
    )

    _generator(recorder)._run_vision_encoding(request, cache=None)

    input_ids, _, kwargs = recorder.embedding_call
    assert input_ids.shape == (1, SEQ)
    assert kwargs["mask"].shape == (1, SEQ)


def test_request_media_is_released_after_prefill():
    recorder = _Recorder()
    request = _request(
        image_grid_thw=mx.array([[1, 2, 2]]),
        extra_kwargs={"image_sizes": mx.ones((1, 2))},
    )

    _generator(recorder)._run_vision_encoding(request, cache=None)

    assert request.vision_encoded is True
    assert request.pixel_values is None
    assert request.attention_mask is None
    assert request.image_grid_thw is None
    assert request.extra_kwargs == {}


def test_plain_tensor_language_output_is_accepted():
    recorder = _Recorder()
    recorder.language_model = lambda inputs, **kw: mx.zeros((1, inputs.shape[1], VOCAB))
    logits = _generator(recorder)._run_vision_encoding(_request(), cache=None)
    assert logits.shape == (1, SEQ, VOCAB)


# --- pixel cache entries survive the request that produced them -----------------


def test_pixel_cache_entry_keeps_extra_kwargs_after_the_request_releases_them():
    """A repeated image+prompt must still get Mistral-3's ``image_sizes``."""
    cache = VisionEmbeddingCache()
    request = _request(extra_kwargs={"image_sizes": mx.array([[448, 448]])})
    cache.set_pixel_cache(
        images=["a.png"],
        prompt="p",
        pixel_values=request.pixel_values,
        input_ids=request.input_ids,
        attention_mask=request.attention_mask,
        image_grid_thw=None,
        extra_kwargs=request.extra_kwargs,
    )

    MLLMBatchGenerator._release_preprocessed_inputs(request)

    entry = cache.get_pixel_cache(["a.png"], "p")
    assert entry is not None
    assert entry.extra_kwargs["image_sizes"].tolist() == [[448, 448]]


def test_pixel_cache_hit_gets_its_own_extra_kwargs():
    cache = VisionEmbeddingCache()
    sizes = mx.array([[448, 448]])
    cache.set_pixel_cache(
        images=["a.png"],
        prompt="p",
        pixel_values=mx.ones((1, 3, 4, 4)),
        input_ids=mx.arange(SEQ)[None, :],
        extra_kwargs={"image_sizes": sizes},
    )
    entry = cache.get_pixel_cache(["a.png"], "p")
    entry.extra_kwargs.clear()
    assert cache.get_pixel_cache(["a.png"], "p").extra_kwargs == {}


# --- prompt length of media requests --------------------------------------------


def _bare_generator():
    return MLLMBatchGenerator.__new__(MLLMBatchGenerator)


def test_media_request_reports_its_processed_prompt_length():
    generator = _bare_generator()
    request = _request(is_text_only=False)
    generator._record_prompt_tokens(request)

    assert generator.pop_prompt_tokens("img") == SEQ
    assert generator.pop_prompt_tokens("img") is None


def test_text_only_request_leaves_the_scheduler_estimate_alone():
    generator = _bare_generator()
    generator._record_prompt_tokens(_request(is_text_only=True))
    assert generator.pop_prompt_tokens("img") is None


def test_abort_forgets_the_prompt_length():
    generator = _bare_generator()
    generator._prefix_checkpoint_lock = __import__("threading").Lock()
    generator._aborted_request_ids = set()
    generator._request_prefix_checkpoints = {}
    generator._record_prompt_tokens(_request(is_text_only=False))

    generator.abort_prefill("img")

    assert generator.pop_prompt_tokens("img") is None


def test_prompt_length_table_is_bounded():
    generator = _bare_generator()
    for i in range(5000):
        request = _request(is_text_only=False)
        request.request_id = f"r{i}"
        generator._record_prompt_tokens(request)
    assert len(generator._prompt_token_counts) <= 4096


class _Resp:
    def __init__(self, uid, token, finish_reason=None):
        self.uid, self.token, self.finish_reason = uid, token, finish_reason
        self.logprobs = None
        self.prompt_cache = None
        self.from_draft = False
        self.mtp_attempted = False
        self.mtp_attempted_count = 0
        self.specprefill_outcome = None


def _scheduler(actual):
    from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler

    scheduler = MLLMScheduler.__new__(MLLMScheduler)
    tokenizer = MagicMock()
    tokenizer.decode = lambda ids: "x"
    scheduler.processor = SimpleNamespace(tokenizer=tokenizer)
    scheduler.uid_to_request_id = {1: "a"}
    scheduler._detokenizer_pool = {}
    scheduler.num_requests_processed = 0
    scheduler.total_completion_tokens = 0
    scheduler.total_prompt_tokens = 20  # includes the 20-token estimate below
    request = MLLMRequest(request_id="a", prompt="p")
    request.num_prompt_tokens = 20
    scheduler.running = {"a": request}
    scheduler.batch_generator = SimpleNamespace(
        pop_cached_tokens=lambda rid: None,
        pop_prompt_tokens=lambda rid: actual,
    )
    return scheduler, request


def test_usage_reports_the_processed_prompt_length_for_image_requests():
    """Gemma-3 expands one ``<start_of_image>`` into 256 image tokens."""
    scheduler, request = _scheduler(279)

    outputs, _ = scheduler._process_batch_responses([_Resp(1, 5, "stop")])

    assert outputs[0].prompt_tokens == 279
    assert request.num_prompt_tokens == 279
    assert scheduler.total_prompt_tokens == 279


def test_usage_keeps_the_estimate_when_the_generator_has_no_count():
    scheduler, request = _scheduler(None)

    outputs, _ = scheduler._process_batch_responses([_Resp(1, 5, "stop")])

    assert outputs[0].prompt_tokens == 20
    assert scheduler.total_prompt_tokens == 20


# --- the whole prompt batch ------------------------------------------------------


def test_prompt_batch_prefills_a_media_request_and_reports_its_length(monkeypatch):
    """``_process_prompts`` -> ``_run_vision_encoding`` -> first token, with the
    processed prompt length available to the scheduler."""
    import threading
    from contextlib import nullcontext

    from mlx_lm.models.cache import KVCache

    from vllm_mlx.mllm_batch_generator import MLLMBatchStats

    recorder = _Recorder(require_mask=True)

    def language_model(inputs, cache=None, **kwargs):
        recorder.language_call = (inputs, kwargs)
        values = mx.zeros((1, 1, inputs.shape[1], 2))
        cache[0].update_and_fetch(values, values)
        scores = mx.array([0.0, 0.0, 9.0])
        return mx.broadcast_to(scores, (1, inputs.shape[1], 3))

    generator = _generator(recorder)
    generator.language_model = language_model
    generator.max_kv_size = 0
    generator._stats = MLLMBatchStats()
    generator._pending_error_responses = []
    generator._aborted_request_ids = set()
    generator._prefill_progress = {}
    generator._prefix_checkpoint_lock = threading.Lock()
    generator._request_prefix_checkpoints = {}
    generator.prefill_step_size = 512
    generator._think_suffix_len = 0
    generator.prefix_cache = None
    generator.sampler = lambda scores: mx.argmax(scores, axis=-1)
    generator.specprefill_enabled = False
    generator.specprefill_keep_pct = 0.5
    generator.specprefill_backbone_pct = 0.1
    generator._preprocess_request = lambda request: None
    monkeypatch.setattr(mx, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        "mlx_lm.models.cache.make_prompt_cache", lambda *a, **k: [KVCache()]
    )
    request = _request(is_text_only=False, images=["a.png"], temperature=0.0)

    batch = generator._process_prompts([request])

    assert batch.y.tolist() == [2]
    assert recorder.embedding_call[2]["mask"] is not None
    assert generator.pop_prompt_tokens("img") == SEQ
