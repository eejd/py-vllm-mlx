# SPDX-License-Identifier: Apache-2.0
"""What a persisted prefix-cache entry is valid for.

The simulated KV here encodes the tokens it was computed for (the value at
position ``p`` is a checksum of ``tokens[:p + 1]``), so an entry reused behind
other tokens, another model or another configuration is visible in the tensors,
not just in a count.
"""

import json
import zlib

import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
from mlx_lm.models import llama  # noqa: E402
from mlx_lm.models.cache import KVCache  # noqa: E402

from vllm_mlx import cache_identity  # noqa: E402
from vllm_mlx.cache_identity import (  # noqa: E402
    kv_identity,
    weights_digest,
    window_identity,
)
from vllm_mlx.memory_cache import (  # noqa: E402
    MemoryAwarePrefixCache,
    MemoryCacheConfig,
    _compute_model_fingerprint,
)


def tiny_llama(seed=0, dtype=None, **overrides):
    """A random-init llama; the same seed and config give identical weights."""
    mx.random.seed(seed)
    cfg = dict(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=2000,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    cfg.update(overrides)
    model = llama.Model(llama.ModelArgs(**cfg))
    if dtype is not None:
        model.update(nn.utils.tree_map(lambda p: p.astype(dtype), model.parameters()))
    mx.eval(model.parameters())
    return model


def prefix_kv(tokens, layers=2, heads=2, head_dim=64):
    """KV whose value at position p is a checksum of tokens[:p + 1]."""
    marks = [
        float(zlib.crc32(repr(tuple(tokens[: p + 1])).encode()) % 4093)
        for p in range(len(tokens))
    ]
    keys = mx.array(marks, dtype=mx.float32).reshape(1, 1, len(tokens), 1)
    keys = mx.broadcast_to(keys, (1, heads, len(tokens), head_dim))
    out = []
    for _ in range(layers):
        layer = KVCache()
        layer.keys = mx.array(keys)
        layer.values = mx.array(keys) + 0.5
        layer.offset = len(tokens)
        out.append(layer)
    mx.eval([(layer.keys, layer.values) for layer in out])
    return out


def cache_for(model, **cfg):
    return MemoryAwarePrefixCache(
        model, MemoryCacheConfig(max_memory_mb=64, min_prefix_tokens=1, **cfg)
    )


TOKENS = list(range(10, 42))


def stored_keys(cache, tokens):
    layer = cache._entries[tuple(tokens)].cache[0]
    return layer.keys[..., : len(tokens), :]


# --- what the model-side identity covers ----------------------------------------------------


def test_the_same_checkpoint_has_the_same_identity():
    assert _compute_model_fingerprint(tiny_llama(1)) == _compute_model_fingerprint(
        tiny_llama(1)
    )


@pytest.mark.parametrize(
    "variant",
    [
        dict(seed=2),  # same architecture and dimensions, other weights
        dict(rope_theta=500000.0),
        dict(rope_scaling={"rope_type": "linear", "factor": 4.0}),
        dict(rms_norm_eps=1e-6),
        dict(tie_word_embeddings=False),
        dict(attention_bias=True),
        dict(dtype=mx.float16),
        dict(dtype=mx.bfloat16),
    ],
    ids=[
        "weights",
        "rope_theta",
        "rope_scaling",
        "rms_norm_eps",
        "untied_embeddings",
        "attention_bias",
        "float16",
        "bfloat16",
    ],
)
def test_anything_that_changes_the_kv_changes_the_identity(variant):
    base = {"seed": 1}
    assert _compute_model_fingerprint(tiny_llama(**base)) != _compute_model_fingerprint(
        tiny_llama(**{**base, **variant})
    )


def test_one_changed_weight_changes_the_identity():
    model = tiny_llama(1)
    before = weights_digest(model)
    weight = model.model.layers[1].mlp.down_proj.weight
    model.model.layers[1].mlp.down_proj.weight = weight.at[3, 5].add(0.01)
    mx.eval(model.parameters())
    assert weights_digest(model) != before


@pytest.mark.parametrize("dtype", [None, mx.bfloat16], ids=["float32", "bfloat16"])
def test_weight_digest_does_not_depend_on_how_the_device_pass_is_chunked(
    monkeypatch, dtype
):
    model = tiny_llama(1, dtype=dtype)
    whole = weights_digest(model)
    monkeypatch.setattr(cache_identity, "_CHECKSUM_CHUNK_BYTES", 1)
    assert weights_digest(model) == whole
    monkeypatch.setattr(cache_identity, "_SLICE_ELEMENTS", 4099)  # uneven last slice
    assert weights_digest(model) == whole


def test_unreadable_weights_are_reported_and_never_match_another_process(
    caplog, monkeypatch
):
    class Opaque:
        args = None

    with caplog.at_level("WARNING"):
        assert weights_digest(Opaque()) is None
    assert "cannot read the model's weights" in caplog.text

    def identity_in_process(nonce):
        monkeypatch.setattr(cache_identity, "_UNREADABLE_NONCE", nonce)
        return cache_identity.model_identity(Opaque(), "none")[0]

    assert identity_in_process("one") == identity_in_process("one")
    assert identity_in_process("one") != identity_in_process("other")


def test_the_identity_changes_with_the_mlx_and_mlx_lm_versions(monkeypatch):
    """KV numerics and the cache file layout come from these packages."""
    from importlib import metadata

    model = tiny_llama(1)
    real = metadata.version

    def with_version(package, version):
        monkeypatch.setattr(
            metadata, "version", lambda n: version if n == package else real(n)
        )
        return _compute_model_fingerprint(model)

    base = _compute_model_fingerprint(model)
    assert with_version("mlx-lm", "0.0.1") != base
    assert with_version("mlx", "0.0.1") != base


def test_where_the_checkpoint_lives_is_not_part_of_the_identity():
    from types import SimpleNamespace

    class Tiny(nn.Module):
        def __init__(self, path):
            super().__init__()
            mx.random.seed(0)
            self.proj = nn.Linear(8, 8, bias=False)
            self.args = SimpleNamespace(hidden_size=8, model_path=path)

    assert _compute_model_fingerprint(Tiny("/models/a")) == _compute_model_fingerprint(
        Tiny("/elsewhere/a")
    )


def test_kv_quantization_bits_only_matter_while_quantization_is_on():
    off = lambda bits: kv_identity(  # noqa: E731
        kv_quantize=False, kv_bits=bits, kv_group_size=64, kv_min_quantize_tokens=256
    )[0]
    on = lambda bits: kv_identity(  # noqa: E731
        kv_quantize=True, kv_bits=bits, kv_group_size=64, kv_min_quantize_tokens=256
    )[0]
    assert off(4) == off(8)
    assert on(4) != on(8)
    assert on(8) != off(8)
    assert window_identity(0)[0] != window_identity(512)[0]
    assert window_identity(0)[0] == window_identity(-3)[0]  # negative = unbounded


# --- persisted entries ------------------------------------------------------------------------


def test_persisted_kv_loads_into_the_model_that_computed_it_unchanged(tmp_path):
    src = cache_for(tiny_llama(1))
    assert src.store(TOKENS, prefix_kv(TOKENS))
    assert src.save_to_disk(str(tmp_path))

    dst = cache_for(tiny_llama(1))  # same checkpoint, loaded again
    assert dst.load_from_disk(str(tmp_path)) == 1
    assert mx.array_equal(stored_keys(dst, TOKENS), prefix_kv(TOKENS)[0].keys).item()


@pytest.mark.parametrize(
    "other",
    [
        dict(seed=2),
        dict(rope_theta=500000.0),
        dict(rope_scaling={"rope_type": "linear", "factor": 4.0}),
        dict(dtype=mx.float16),
    ],
    ids=["weights", "rope_theta", "rope_scaling", "dtype"],
)
def test_persisted_kv_is_not_loaded_into_a_different_model(tmp_path, other):
    src = cache_for(tiny_llama(1))
    assert src.store(TOKENS, prefix_kv(TOKENS))
    assert src.save_to_disk(str(tmp_path))
    assert (
        cache_for(tiny_llama(**{"seed": 1, **other})).load_from_disk(str(tmp_path)) == 0
    )


def test_persisted_kv_is_not_loaded_under_another_max_kv_size(tmp_path):
    src = cache_for(tiny_llama(1), max_kv_size=0)
    assert src.store(TOKENS, prefix_kv(TOKENS))
    assert src.save_to_disk(str(tmp_path))
    assert cache_for(tiny_llama(1), max_kv_size=64).load_from_disk(str(tmp_path)) == 0
    assert cache_for(tiny_llama(1), max_kv_size=0).load_from_disk(str(tmp_path)) == 1


QUANT = dict(kv_quantize=True, kv_bits=8, kv_group_size=64, kv_min_quantize_tokens=1)


def test_kv_saved_while_quantized_is_only_loaded_under_the_same_setting(tmp_path):
    src = cache_for(tiny_llama(1), **QUANT)
    assert src.store(TOKENS, prefix_kv(TOKENS))
    assert src.save_to_disk(str(tmp_path))
    model = tiny_llama(1)
    assert (
        cache_for(model, **{**QUANT, "kv_bits": 4}).load_from_disk(str(tmp_path)) == 0
    )
    assert cache_for(model).load_from_disk(str(tmp_path)) == 0  # quantization off
    assert cache_for(model, **QUANT).load_from_disk(str(tmp_path)) == 1


def test_kv_saved_exactly_is_quantized_like_a_fresh_entry(tmp_path):
    src = cache_for(tiny_llama(1))
    assert src.store(TOKENS, prefix_kv(TOKENS))
    assert src.save_to_disk(str(tmp_path))
    dst = cache_for(tiny_llama(1), **QUANT)
    assert dst.load_from_disk(str(tmp_path)) == 1
    assert type(dst._entries[tuple(TOKENS)].cache[0]).__name__ == (
        "_QuantizedCacheWrapper"
    )


# --- entries that do not describe their own tokens -------------------------------------------


def two_entries(tmp_path):
    a, b = TOKENS, [t + 1000 for t in TOKENS]
    src = cache_for(tiny_llama(1))
    assert src.store(a, prefix_kv(a)) and src.store(b, prefix_kv(b))
    assert src.save_to_disk(str(tmp_path))
    return a, b


def test_kv_is_not_loaded_behind_tokens_it_was_not_computed_for(tmp_path):
    """Two writers (or an interrupted save) can leave entry i's KV next to entry
    j's tokens; both are the same length, so only the content tells."""
    two_entries(tmp_path)
    t0, t1 = tmp_path / "entry_0_tokens.bin", tmp_path / "entry_1_tokens.bin"
    swapped = (t1.read_bytes(), t0.read_bytes())
    t0.write_bytes(swapped[0])
    t1.write_bytes(swapped[1])
    assert cache_for(tiny_llama(1)).load_from_disk(str(tmp_path)) == 0


def test_an_index_that_miscounts_the_tokens_loads_nothing(tmp_path):
    two_entries(tmp_path)
    index = json.loads((tmp_path / "index.json").read_text())
    for entry in index["entries"]:
        entry["num_tokens"] -= 8
    (tmp_path / "index.json").write_text(json.dumps(index))
    assert cache_for(tiny_llama(1)).load_from_disk(str(tmp_path)) == 0


def test_kv_longer_than_its_key_is_not_loaded(tmp_path):
    src = cache_for(tiny_llama(1))
    assert src.store(TOKENS, prefix_kv(TOKENS + list(range(100, 108))))
    assert src.save_to_disk(str(tmp_path))
    assert cache_for(tiny_llama(1)).load_from_disk(str(tmp_path)) == 0


def test_a_save_that_dies_midway_leaves_nothing_to_load(tmp_path, monkeypatch):
    """The old index must not survive to describe half-written new files."""
    import mlx_lm.models.cache as mlx_cache

    old = cache_for(tiny_llama(1))
    assert old.store(TOKENS, prefix_kv(TOKENS))
    assert old.save_to_disk(str(tmp_path))
    assert (tmp_path / "index.json").exists()

    new_tokens = [t + 5000 for t in TOKENS]
    new = cache_for(tiny_llama(1))
    assert new.store(new_tokens, prefix_kv(new_tokens))

    def killed(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(mlx_cache, "save_prompt_cache", killed)
    with pytest.raises(KeyboardInterrupt):
        new.save_to_disk(str(tmp_path))
    assert not (tmp_path / "index.json").exists()
    assert cache_for(tiny_llama(1)).load_from_disk(str(tmp_path)) == 0


def test_an_index_without_an_identity_is_refused(tmp_path):
    src = cache_for(tiny_llama(1))
    assert src.store(TOKENS, prefix_kv(TOKENS))
    assert src.save_to_disk(str(tmp_path))
    index = json.loads((tmp_path / "index.json").read_text())
    del index["identity"]
    (tmp_path / "index.json").write_text(json.dumps(index))
    assert cache_for(tiny_llama(1)).load_from_disk(str(tmp_path)) == 0


# --- the SSD tier is namespaced by the cache's identity ---------------------------------------


def test_ssd_tiers_of_different_models_sharing_a_directory_do_not_see_each_other(
    tmp_path,
):
    import time

    from vllm_mlx.ssd_cache import SSDCacheConfig, SSDCacheTier

    def tier_for(cache):
        tier = SSDCacheTier(
            SSDCacheConfig(cache_dir=str(tmp_path), identity=cache.identity)
        )
        tier.start_writer()
        cache.set_ssd_tier(tier)
        return tier

    a = cache_for(tiny_llama(1))
    tier_a = tier_for(a)
    assert tier_a.enqueue_spill(tuple(TOKENS), prefix_kv(TOKENS), 4096)
    deadline = time.monotonic() + 5.0
    while tier_a.get_stats()["spill_count"] < 1 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert a.check_ssd(TOKENS + [999]) is not None
    tier_a.close()

    b = cache_for(tiny_llama(2))  # same architecture, other weights
    tier_b = tier_for(b)
    try:
        assert b.check_ssd(TOKENS + [999]) is None
    finally:
        tier_b.close()
    a2 = cache_for(tiny_llama(1))  # the original model finds its entries again
    tier_a2 = tier_for(a2)
    try:
        assert a2.check_ssd(TOKENS + [999]) is not None
    finally:
        tier_a2.close()


# --- the weight digest -------------------------------------------------------------------------


def _holder(**arrays):
    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            for name, value in arrays.items():
                setattr(self, name, value)

    return Holder()


def test_permuting_a_tensor_changes_the_digest():
    """Plain and squared sums cannot tell rows apart; the positional term can."""
    w = mx.arange(24, dtype=mx.float32).reshape(6, 4)
    swapped = mx.concatenate([w[1:2], w[0:1], w[2:]])
    assert weights_digest(_holder(w=w)) != weights_digest(_holder(w=swapped))
    bf = w.astype(mx.bfloat16)
    flipped = bf.reshape(-1)[::-1].reshape(6, 4)
    assert weights_digest(_holder(w=bf)) != weights_digest(_holder(w=flipped))


def test_the_squared_term_separates_equal_sum_equal_position_sum_tensors():
    a = mx.array([1, 4, 1], dtype=mx.uint32)  # sum 6, sum(i * x) 12
    b = mx.array([2, 2, 2], dtype=mx.uint32)  # sum 6, sum(i * x) 12
    assert weights_digest(_holder(w=a)) != weights_digest(_holder(w=b))


def test_shape_dtype_and_name_are_part_of_the_digest():
    data = mx.arange(6, dtype=mx.uint32)
    base = weights_digest(_holder(w=data.reshape(2, 3)))
    assert weights_digest(_holder(w=data.reshape(3, 2))) != base  # shape
    assert (
        weights_digest(_holder(w=data.astype(mx.int32).reshape(2, 3))) != base
    )  # dtype
    assert weights_digest(_holder(v=data.reshape(2, 3))) != base  # name


def test_the_model_class_is_part_of_the_identity():
    class A(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = mx.arange(6, dtype=mx.float32)

    class B(A):
        pass

    assert _compute_model_fingerprint(A()) != _compute_model_fingerprint(B())


# --- the scheduler hands --max-kv-size to the cache that persists entries ------------------------


def test_the_scheduler_names_max_kv_size_in_the_cache_identity():
    from unittest.mock import MagicMock

    from vllm_mlx.scheduler import Scheduler, SchedulerConfig

    def window(max_kv_size):
        scheduler = Scheduler(
            tiny_llama(1),
            MagicMock(),
            SchedulerConfig(
                enable_prefix_cache=True,
                use_memory_aware_cache=True,
                cache_memory_mb=64,
                max_kv_size=max_kv_size,
            ),
        )
        return scheduler.memory_aware_cache.identity.window

    assert window(64) == window_identity(64)[0]
    assert window(64) != window(0)


# --- entry files of refused or larger earlier saves -------------------------------------------


def test_a_save_removes_entry_files_it_did_not_write(tmp_path):
    big = cache_for(tiny_llama(1))
    a, b = TOKENS, [t + 1000 for t in TOKENS]
    assert big.store(a, prefix_kv(a)) and big.store(b, prefix_kv(b))
    assert big.save_to_disk(str(tmp_path))
    assert (tmp_path / "entry_1.safetensors").exists()

    small = cache_for(tiny_llama(1))
    assert small.store(a, prefix_kv(a))
    assert small.save_to_disk(str(tmp_path))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["entry_0.safetensors", "entry_0_tokens.bin", "index.json"]
