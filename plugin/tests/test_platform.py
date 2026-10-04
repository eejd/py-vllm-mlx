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
