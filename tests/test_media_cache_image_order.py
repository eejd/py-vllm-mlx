# SPDX-License-Identifier: Apache-2.0
"""Media cache keys keep the order of the images.

``compute_images_hash`` sorted the per-image hashes, so ``[A, B]`` and
``[B, A]`` (and a video's frames in another order) shared one entry: the second
order was served the first order's pixels / KV state and answered about the
wrong image in each placeholder.  The k-th image fills the k-th placeholder,
so the order is part of the request.
"""

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx import mllm_cache  # noqa: E402
from vllm_mlx import vision_embedding_cache as vision_cache  # noqa: E402
from vllm_mlx.mllm_batch_generator import MLLMBatchGenerator  # noqa: E402
from vllm_mlx.mllm_batch_generator import MLLMBatchRequest  # noqa: E402


@pytest.fixture
def images(tmp_path):
    paths = []
    for name in ("a", "b", "c"):
        path = tmp_path / f"{name}.png"
        path.write_bytes(f"image {name}".encode())
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("module", [vision_cache, mllm_cache])
def test_hash_depends_on_order(module, images):
    a, b, c = images
    assert module.compute_images_hash([a, b]) != module.compute_images_hash([b, a])
    assert module.compute_images_hash([a, b, c]) != module.compute_images_hash(
        [c, b, a]
    )


@pytest.mark.parametrize("module", [vision_cache, mllm_cache])
def test_hash_is_stable_and_distinguishes_repeats(module, images):
    a, b, _ = images
    assert module.compute_images_hash([a, b]) == module.compute_images_hash([a, b])
    assert module.compute_images_hash([a]) != module.compute_images_hash([a, a])
    assert module.compute_images_hash([a, a, b]) != module.compute_images_hash(
        [a, b, a]
    )


def test_pixel_cache_does_not_serve_swapped_images(images):
    a, b, _ = images
    cache = vision_cache.VisionEmbeddingCache()
    pixels_ab = mx.zeros((2, 1))
    cache.set_pixel_cache(
        images=[a, b],
        prompt="p",
        pixel_values=pixels_ab,
        input_ids=mx.arange(4)[None, :],
    )

    assert cache.get_pixel_cache([b, a], "p") is None
    assert cache.get_pixel_cache([a, b], "p").pixel_values is pixels_ab


def test_pixel_only_and_encoding_caches_do_not_serve_swapped_images(images):
    a, b, _ = images
    cache = vision_cache.VisionEmbeddingCache()
    cache.set_pixel_values([a, b], mx.zeros((2, 1)))
    cache.set_encoding_cache([a, b], "p", mx.zeros((1, 3)), 1, mx.zeros((3,)))

    assert cache.get_pixel_values([b, a]) is None
    assert cache.get_encoding_cache([b, a], "p") is None
    assert cache.get_pixel_values([a, b]) is not None
    assert cache.get_encoding_cache([a, b], "p") is not None


def test_video_frames_in_a_different_order_are_a_different_entry(images):
    frames = images
    cache = vision_cache.VisionEmbeddingCache()
    cache.set_pixel_cache(
        images=frames,
        prompt="p",
        pixel_values=mx.zeros((3, 1)),
        input_ids=mx.arange(4)[None, :],
    )
    assert cache.get_pixel_cache(list(reversed(frames)), "p") is None


def test_prefix_cache_manager_does_not_serve_swapped_images(images):
    a, b, _ = images
    manager = mllm_cache.MLLMPrefixCacheManager()
    manager.store_cache([a, b], "p", [object()], num_tokens=4)

    assert manager.fetch_cache([b, a], "p") == (None, False)
    kv, hit = manager.fetch_cache([a, b], "p")
    assert hit and kv is not None


def test_prefix_cache_manager_does_not_reuse_vision_embeddings_across_orders(images):
    a, b, _ = images
    manager = mllm_cache.MLLMPrefixCacheManager()
    manager.store(
        images=[a, b],
        prompt="p",
        vision_embeddings="features-ab",
        kv_cache=[object()],
        token_ids=[1, 2, 3, 4],
        num_image_tokens=2,
    )

    entry, _ = manager.fetch([b, a], "another prompt", [1, 2])
    assert entry is None


def test_batched_preprocessing_reprocesses_images_in_the_other_order(
    images, monkeypatch
):
    """Through ``_preprocess_request``: the swapped request gets its own
    pixels, the repeated one hits the pixel cache."""
    a, b, _ = images
    processed = []

    def prepare_inputs(processor, images, audio, prompts, image_token_index, **kw):
        processed.append(list(images))
        order = [0 if image == a else 1 for image in images]
        return {
            "input_ids": mx.array([[1, 2, 3]]),
            "pixel_values": mx.array(order, dtype=mx.float32),
            "attention_mask": mx.ones((1, 3)),
        }

    monkeypatch.setattr("mlx_vlm.utils.prepare_inputs", prepare_inputs)
    monkeypatch.setattr("vllm_mlx.models.mllm.process_image_input", lambda image: image)
    generator = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    generator.processor = object()
    generator.model = type("M", (), {"config": None})()
    generator.vision_cache = vision_cache.VisionEmbeddingCache()
    generator._stats = type(
        "S", (), {"num_images_processed": 0, "vision_encoding_time": 0.0}
    )()

    def run(order):
        request = MLLMBatchRequest(uid=1, request_id="r", prompt="p", images=order)
        generator._preprocess_request(request)
        return request.pixel_values.tolist()

    assert run([a, b]) == [0.0, 1.0]
    assert run([b, a]) == [1.0, 0.0]
    assert processed == [[a, b], [b, a]]
    assert run([a, b]) == [0.0, 1.0]
    assert len(processed) == 2  # the repeat of the first order was a cache hit
