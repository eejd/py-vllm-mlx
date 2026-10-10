# SPDX-License-Identifier: Apache-2.0
"""--fp32-matmul-precision selects full or reduced float32 matmul precision."""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.cli import create_parser  # noqa: E402
from vllm_mlx.precision import set_fp32_matmul_precision  # noqa: E402

requires_runtime_config = pytest.mark.skipif(
    not hasattr(mx, "config") or not mx.metal.is_available(),
    reason="needs MLX with the runtime mx.config module and a Metal GPU",
)


@pytest.fixture
def restore_precision():
    if hasattr(mx, "config"):
        before = mx.config.get("MLX_ENABLE_TF32", 1)
        yield
        mx.config.update("MLX_ENABLE_TF32", before)
    else:
        yield


def _gpu_matmul_error():
    """Max |GPU - exact| / (|A| @ |B|) for a float32 GEMM-sized product."""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((128, 256)).astype(np.float32)
    b = rng.standard_normal((256, 64)).astype(np.float32)
    exact = a.astype(np.float64) @ b.astype(np.float64)
    magnitude = np.abs(a).astype(np.float64) @ np.abs(b).astype(np.float64)
    out = np.array(mx.matmul(mx.array(a), mx.array(b), stream=mx.gpu))
    return float((np.abs(out - exact) / magnitude).max())


@requires_runtime_config
def test_highest_gives_float32_accuracy_and_high_stays_within_tf32(
    restore_precision,
):
    set_fp32_matmul_precision("highest")
    # float32 accumulation over K=256: well under 256 * 2^-24.
    assert _gpu_matmul_error() < 256 * 2.0**-24

    set_fp32_matmul_precision("high")
    # TF32 inputs: each product within 2 * 2^-10 of exact.
    assert _gpu_matmul_error() < 2 * 2.0**-10

    # Switching back in the same process takes effect.
    set_fp32_matmul_precision("highest")
    assert _gpu_matmul_error() < 256 * 2.0**-24


def test_rejects_unknown_precision():
    with pytest.raises(ValueError, match="highest"):
        set_fp32_matmul_precision("medium")


@pytest.mark.parametrize("command", ["serve", "bench"])
def test_parser_rejects_unknown_precision(command):
    with pytest.raises(SystemExit):
        create_parser().parse_args(
            [command, "local-test-model", "--fp32-matmul-precision", "medium"]
        )
