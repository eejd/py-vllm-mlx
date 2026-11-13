# SPDX-License-Identifier: Apache-2.0
"""Request-cache dtype evidence must come from live generation arrays."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx_lm
import pytest

from vllm_mlx.engine.simple import SimpleEngine


class FixtureCache:
    def __init__(self, dtype):
        self.keys = mx.zeros((1, 1, 1, 4), dtype=dtype)
        self.values = mx.zeros((1, 1, 1, 4), dtype=dtype)


@pytest.mark.parametrize("dtype,name", [(mx.bfloat16, "bfloat16"), (mx.float16, "float16")])
def test_request_cache_dtype_evidence(dtype, name):
    engine = SimpleEngine("test-model")
    engine._observe_request_kv_cache([FixtureCache(dtype)])
    evidence = engine._kv_dtype_observation
    assert evidence["kv_dtype"] == name
    assert evidence["kv_dtype_source"] == "observed-cache-arrays"
    assert evidence["kv_dtype_reason"] is None
    assert evidence["observed_after_generation_requests"] == 1
    assert evidence["kv_dtype_layers"] == [{"layer": 0, "cache_class": "FixtureCache", "key_dtype": name, "value_dtype": name}]


def test_request_cache_dtype_fails_closed():
    engine = SimpleEngine("test-model")
    engine._observe_request_kv_cache([FixtureCache(mx.bfloat16), FixtureCache(mx.float16)])
    assert engine._kv_dtype_observation["kv_dtype"] is None
    assert engine._kv_dtype_observation["kv_dtype_reason"] == "cache arrays have mixed dtypes"
    engine._observe_request_kv_cache([SimpleNamespace(keys=None, values=None)])
    assert engine._kv_dtype_observation["kv_dtype"] is None
    assert engine._kv_dtype_observation["observed_after_generation_requests"] == 2


@pytest.mark.anyio
@pytest.mark.parametrize("dtype,name", [(mx.bfloat16, "bfloat16"), (mx.float16, "float16")])
async def test_mllm_text_generation_observes_request_cache(dtype, name):
    tokenizer = MagicMock()
    tokenizer.bos_token = None
    tokenizer.apply_chat_template.return_value = "prompt"
    tokenizer.encode.return_value = [1, 2, 3]
    cache = [FixtureCache(dtype)]

    def generate(*args, **kwargs):
        assert "prompt_cache" not in kwargs
        yield SimpleNamespace(text="answer", finish_reason="stop")

    with patch("vllm_mlx.engine.simple._bind_worker_generation_streams"), \
         patch.object(mlx_lm, "stream_generate", side_effect=generate):
        engine = SimpleEngine("test-model", force_mllm=True)
        engine._loaded = True
        engine._text_model = MagicMock()
        engine._text_model.mtp = None
        engine._text_tokenizer = tokenizer
        stream = engine._stream_generate_text([{"role": "user", "content": "hello"}], 8, 0.7, 0.9)
        async for _ in stream:
            pass
        assert engine._kv_dtype_observation["kv_dtype"] is None
        assert engine._kv_dtype_observation["observed_after_generation_requests"] == 0
        await engine.stop()


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_simple_text_completion_and_closed_stream_observe_cache(cancel):
    cache = [FixtureCache(mx.bfloat16)]
    model = MagicMock()

    def generate(**kwargs):
        assert kwargs["prompt_cache"] is cache
        yield SimpleNamespace(text="first", prompt_tokens=1, finished=False)
        yield SimpleNamespace(text="last", prompt_tokens=1, finished=True)

    model.stream_generate.side_effect = generate
    with patch("vllm_mlx.engine.simple._bind_worker_generation_streams"), \
         patch("mlx_lm.models.cache.make_prompt_cache", return_value=cache):
        engine = SimpleEngine("test-model")
        engine._loaded = True
        engine._model = model
        stream = engine.stream_generate("hello", max_tokens=8)
        if cancel:
            await stream.__anext__()
            await stream.aclose()
        else:
            async for _ in stream:
                pass
        assert engine._kv_dtype_observation["kv_dtype"] == "bfloat16"
        assert engine._kv_dtype_observation["observed_after_generation_requests"] == 1
        await engine.stop()


@pytest.mark.anyio
async def test_inspection_exposes_observed_evidence(monkeypatch):
    from vllm_mlx import server

    engine = SimpleEngine("test-model")
    engine._loaded = True
    engine._model = MagicMock()
    engine._model.tokenizer.chat_template = "template"
    engine._observe_request_kv_cache([FixtureCache(mx.float16)])
    monkeypatch.setattr(server, "_engine", engine)
    result = await server.inspect_serving(server.InspectionRequest())
    assert result["cache"]["kv_dtype"] == "float16"
    assert result["cache"]["kv_dtype_source"] == "observed-cache-arrays"
    assert result["cache"]["observed_after_generation_requests"] == 1
    assert result["cache"]["kv_dtype_reason"] is None


def test_none_cache_observation_preserves_prior_valid_observation():
    from vllm_mlx.engine import SimpleEngine

    engine = SimpleEngine("fixture")
    prior = {"kv_dtype": "bfloat16", "complete": True}
    engine._kv_dtype_observation = dict(prior)
    engine._observed_after_generation_requests = 5
    engine._observe_request_kv_cache(None)
    assert engine._kv_dtype_observation == prior
    assert engine._observed_after_generation_requests == 5
