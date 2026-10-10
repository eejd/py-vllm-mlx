# SPDX-License-Identifier: Apache-2.0
"""Process-wide float32 matmul precision for MLX.

On GPUs with matrix units (Apple M5 neural accelerators, CUDA tensor cores)
MLX runs float32 matmul, quantized matmul, convolution and attention at
reduced TF32-class precision unless ``MLX_ENABLE_TF32=0``; see MLX's
``docs/src/usage/precision.rst``. The names follow
``torch.set_float32_matmul_precision`` and JAX's ``default_matmul_precision``:

* ``highest``: full float32 (``MLX_ENABLE_TF32=0``)
* ``high``: reduced precision allowed where the hardware has it
  (``MLX_ENABLE_TF32=1``, MLX's default)

Only float32 operands are affected; float16/bfloat16 and quantized weights
with half-precision activations behave the same under both settings.
"""

import logging
import os

logger = logging.getLogger(__name__)

FP32_MATMUL_PRECISIONS = ("highest", "high")

_MLX_ENABLE_TF32 = "MLX_ENABLE_TF32"


def set_fp32_matmul_precision(precision: str) -> None:
    """Select float32 matmul precision for this process.

    With an MLX that has the runtime ``mx.config`` module the setting takes
    effect immediately. Older MLX reads ``MLX_ENABLE_TF32`` once, at the first
    reduced-precision-eligible operation, so on those builds this must run
    before any model is loaded; the CLI calls it before loading.
    """
    if precision not in FP32_MATMUL_PRECISIONS:
        raise ValueError(
            f"fp32 matmul precision must be one of {FP32_MATMUL_PRECISIONS}, "
            f"got {precision!r}"
        )
    value = 0 if precision == "highest" else 1

    import mlx.core as mx

    config = getattr(mx, "config", None)
    if config is not None:
        config.update(_MLX_ENABLE_TF32, value)
        mechanism = "mx.config"
    else:
        os.environ[_MLX_ENABLE_TF32] = str(value)
        mechanism = "environment (applies from the first matmul)"
    logger.info(
        "float32 matmul precision: %s (%s=%d via %s)",
        precision,
        _MLX_ENABLE_TF32,
        value,
        mechanism,
    )
