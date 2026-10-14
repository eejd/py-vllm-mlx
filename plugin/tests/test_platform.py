# SPDX-License-Identifier: Apache-2.0
"""Tests for MLX Platform."""

import importlib.util
import platform
import sys

import pytest


def test_is_apple_silicon():
    """Test Apple Silicon detection."""
    from vllm_mlx_plugin.plugin import is_mlx_available

    if sys.platform == "darwin" and platform.machine() == "arm64":
        # Should be available on Apple Silicon Mac
        # (assuming MLX is installed)
        try:
            import mlx.core  # noqa: F401

            assert is_mlx_available()
        except ImportError:
            pytest.skip("MLX not installed")
    else:
        # Should not be available on other platforms
        assert not is_mlx_available()


# MLXPlatform subclasses vLLM's Platform and builds on torch, neither of which
# the engine's own CI installs. Skip (rather than error) where they are absent.
needs_vllm_and_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("vllm") is None,
    reason="vllm and torch are not installed",
)


@needs_vllm_and_torch
def test_mlx_platform_properties():
    """Test MLXPlatform class properties."""
    from vllm_mlx_plugin.vllm_platform import MLXPlatform

    platform_obj = MLXPlatform()

    assert platform_obj.device_name == "mlx"
    # vLLM passes this through torch.device(), so the platform shim must expose
    # a torch-recognized device type even though execution is backed by MLX.
    assert platform_obj.device_type == "cpu"
    assert platform_obj.dispatch_key == "CPU"
    assert platform_obj.dist_backend == "gloo"
    assert platform_obj.is_mlx()
    assert not platform_obj.is_cuda()
    assert not platform_obj.is_rocm()


@needs_vllm_and_torch
def test_get_device_name():
    """Test getting device name."""
    from vllm_mlx_plugin.vllm_platform import MLXPlatform

    name = MLXPlatform.get_device_name()
    assert isinstance(name, str)
    assert len(name) > 0


@needs_vllm_and_torch
def test_get_device_memory():
    """Test getting device memory."""
    from vllm_mlx_plugin.vllm_platform import MLXPlatform

    memory = MLXPlatform.get_device_total_memory()
    assert isinstance(memory, int)
    assert memory > 0


@needs_vllm_and_torch
def test_supported_dtypes():
    """Test supported dtypes."""
    import torch
    from vllm_mlx_plugin.vllm_platform import MLXPlatform

    platform_obj = MLXPlatform()
    dtypes = platform_obj.supported_dtypes

    assert torch.float32 in dtypes
    assert torch.float16 in dtypes


def test_plugin_entry_point():
    """Test plugin entry point returns correct class path."""
    from vllm_mlx_plugin.plugin import mlx_platform_plugin

    if sys.platform != "darwin" or platform.machine() != "arm64":
        pytest.skip("Not on Apple Silicon")

    try:
        import mlx.core  # noqa: F401
    except ImportError:
        pytest.skip("MLX not installed")

    result = mlx_platform_plugin()
    assert result == "vllm_mlx_plugin.vllm_platform.MLXPlatform"


def test_device_info():
    """Test getting device info."""
    from vllm_mlx_plugin.plugin import get_mlx_device_info

    info = get_mlx_device_info()

    assert isinstance(info, dict)
    assert "platform" in info
    assert "available" in info
    assert "chip_name" in info
    assert "memory_gb" in info


@needs_vllm_and_torch
def test_get_attn_backend_cls_accepts_the_selectors_call():
    """vLLM's selector passes ``attn_selector_config=`` and ``num_heads=`` by name."""
    import torch
    from vllm.v1.attention.selector import AttentionSelectorConfig
    from vllm_mlx_plugin.vllm_platform import MLXPlatform

    config = AttentionSelectorConfig(
        head_size=64, dtype=torch.float16, kv_cache_dtype=None, block_size=16
    )
    assert (
        MLXPlatform.get_attn_backend_cls(None, attn_selector_config=config, num_heads=8)
        == "vllm_mlx_plugin.attention.MLXAttentionBackend"
    )


@needs_vllm_and_torch
@pytest.mark.parametrize(
    "parameter, kwargs",
    [
        ("logprobs", dict(logprobs=1)),
        ("logprobs", dict(logprobs=0)),
        ("prompt_logprobs", dict(prompt_logprobs=1)),
        ("seed", dict(seed=1)),
        ("repetition_penalty", dict(repetition_penalty=1.1)),
        ("presence_penalty", dict(presence_penalty=0.5)),
        ("frequency_penalty", dict(frequency_penalty=-0.5)),
        ("min_tokens", dict(min_tokens=3)),
        ("allowed_token_ids", dict(allowed_token_ids=[1, 2])),
        ("bad_words", dict(bad_words=["no"])),
    ],
)
def test_validate_request_rejects_what_the_runner_would_ignore(parameter, kwargs):
    from vllm import SamplingParams
    from vllm.exceptions import VLLMClientError, VLLMValidationError
    from vllm_mlx_plugin.vllm_platform import MLXPlatform

    with pytest.raises(VLLMValidationError) as raised:
        MLXPlatform.validate_request(None, SamplingParams(**kwargs))

    assert raised.value.parameter == parameter
    assert isinstance(raised.value, VLLMClientError)  # answered as a 4xx
    assert parameter in str(raised.value)


@needs_vllm_and_torch
def test_validate_request_accepts_what_the_runner_honours():
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm_mlx_plugin.vllm_platform import MLXPlatform

    MLXPlatform.validate_request(
        None,
        SamplingParams(
            n=2,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            min_p=0.05,
            max_tokens=32,
            stop=["x"],
            ignore_eos=True,
            logit_bias={5: 1.0},
            structured_outputs=StructuredOutputsParams(json_object=True),
        ),
    )
    MLXPlatform.validate_request(None, SamplingParams())  # all defaults


@needs_vllm_and_torch
def test_validate_request_rejects_pooling():
    from vllm import PoolingParams
    from vllm.exceptions import VLLMValidationError
    from vllm_mlx_plugin.vllm_platform import MLXPlatform

    with pytest.raises(VLLMValidationError):
        MLXPlatform.validate_request(None, PoolingParams())


def _config(prefix_caching):
    from types import SimpleNamespace

    from vllm.config import CacheConfig

    cache = CacheConfig(enable_prefix_caching=prefix_caching)
    return SimpleNamespace(
        cache_config=cache,
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1]),
        parallel_config=SimpleNamespace(worker_cls="auto", enable_dbo=False),
    )


@needs_vllm_and_torch
@pytest.mark.parametrize("requested", [True, False])
def test_vllm_prefix_caching_is_off_whatever_was_requested(requested):
    """MLX keeps its own KV cache: a vLLM cache hit would tell the scheduler a
    prefix is computed that the runner still has to feed, in one oversized step."""
    from vllm_mlx_plugin.vllm_platform import MLXPlatform

    config = _config(requested)
    MLXPlatform.check_and_update_config(config)

    assert config.cache_config.enable_prefix_caching is False
    assert config.parallel_config.worker_cls == "vllm_mlx_plugin.worker.MLXWorker"
