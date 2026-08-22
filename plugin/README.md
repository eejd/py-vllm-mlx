# vllm-mlx-plugin

The [vLLM](https://github.com/vllm-project/vllm) platform plugin for Apple
Silicon, backed by the [vllm-mlx](https://github.com/vllm-mlx/vllm-mlx) engine.

```
pip install vllm-mlx-plugin   # pulls in vllm, torch and vllm-mlx
```

With this package installed, `import vllm` resolves `current_platform` to
`MLXPlatform` and `vllm serve` runs models on the GPU through MLX.

## Why a separate package?

`vllm-mlx` is a standalone server and engine that does not need vLLM or
PyTorch. vLLM discovers platform plugins through the `vllm.platform_plugins`
entry-point group and allows **exactly one** out-of-tree plugin to be active;
keeping the entry point here means installing the standalone server never
silently claims vLLM's platform slot, and never conflicts with another Apple
Silicon plugin (e.g. `vllm-metal`, `vllm-swift`) unless both are installed.

If several plugins are installed, pick one with vLLM's allow-list:

```
VLLM_PLUGINS=mlx vllm serve ...
```

## Contents

| Module | Role |
|---|---|
| `plugin` | entry point: `mlx_platform_plugin()` returns the platform class path or `None` |
| `vllm_platform` | `MLXPlatform(vllm.platforms.interface.Platform)` |
| `worker` | `MLXWorker` |
| `model_runner` | `MLXModelRunner` -- drives `vllm_mlx` |
| `attention` | `MLXAttentionBackend` |

## Tests

```
cd plugin && python -m pytest tests
```
