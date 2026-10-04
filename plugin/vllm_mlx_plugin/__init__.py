# SPDX-License-Identifier: Apache-2.0
"""
vLLM platform plugin for Apple Silicon via MLX.

This distribution owns the ``vllm.platform_plugins`` entry point that makes
vLLM run on the ``vllm-mlx`` engine. It is split from ``vllm-mlx`` itself so
the standalone server/CLI can be installed without vLLM (or PyTorch), and so
installing ``vllm-mlx`` never registers a platform plugin with vLLM
implicitly -- vLLM permits exactly one out-of-tree platform plugin to be
active at a time.

The heavy imports (torch, vllm) are deferred to the submodules; importing
this package is cheap.
"""

from vllm_mlx_plugin.plugin import (
    get_mlx_device_info,
    is_mlx_available,
    mlx_platform_plugin,
)

__all__ = [
    "MLXAttentionBackend",
    "MLXModelRunner",
    "MLXPlatform",
    "MLXWorker",
    "get_mlx_device_info",
    "is_mlx_available",
    "mlx_platform_plugin",
]


def __getattr__(name: str):
    if name == "MLXPlatform":
        from vllm_mlx_plugin.vllm_platform import MLXPlatform

        return MLXPlatform
    if name == "MLXWorker":
        from vllm_mlx_plugin.worker import MLXWorker

        return MLXWorker
    if name == "MLXModelRunner":
        from vllm_mlx_plugin.model_runner import MLXModelRunner

        return MLXModelRunner
    if name == "MLXAttentionBackend":
        from vllm_mlx_plugin.attention import MLXAttentionBackend

        return MLXAttentionBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
