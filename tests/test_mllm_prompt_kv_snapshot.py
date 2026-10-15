# SPDX-License-Identifier: Apache-2.0
"""The Simple-engine MLLM prompt-KV cache stores prompt snapshots, never live caches.

``mlx_vlm`` extends the cache it is given in place, with the prompt and then
every generated token. The Simple engine used to store that live list after
``generate()``, hand the stored list itself back on a hit, and call mlx-vlm with
the *whole* prompt again: the prompt was prefilled a second time on top of
``prompt + output`` positions, and every later hit grew the stored entry.

The KV in these tests encodes ``position * 1000 + token`` per row, so a cache
shows exactly which tokens it holds at which positions. The fake mlx-vlm
follows the real contract: it prefills the given ``input_ids`` onto whatever
cache it runs on, except where ``prompt_cache_state`` names a cached prefix of
the same ids, in which case only the rest is fed.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")
vlm_cache = pytest.importorskip("mlx_vlm.models.cache")
lm_cache = pytest.importorskip("mlx_lm.models.cache")
PromptCacheState = pytest.importorskip("mlx_vlm.generate").PromptCacheState

from vllm_mlx.mllm_cache import (  # noqa: E402
    MLLMPrefixCacheManager,
    snapshot_prompt_prefix,
)
from vllm_mlx.models.mllm import MLXMultimodalLM  # noqa: E402

WINDOW = 64
IMAGE = "data:image/png;base64,AAAA"


def feed(layer, token_ids, layer_no=0):
    """Append ``token_ids`` to ``layer`` as rows ``position * 1000 + token``."""
    first = layer.offset
    rows = [[(first + i) * 1000 + t, layer_no] for i, t in enumerate(token_ids)]
    keys = mx.array([[rows]], dtype=mx.float32)
    layer.update_and_fetch(keys, keys)


def rows_of(layer):
    """``[(position, token), ...]`` held by a plain or unwrapped rotating layer."""
    keys = layer.keys[0, 0, : layer.offset, 0]
    mx.eval(keys)
    return [(int(v) // 1000, int(v) % 1000) for v in keys.tolist()]


def positions(tokens):
    return [(i, t) for i, t in enumerate(tokens)]


def make_layers(plain=vlm_cache.KVCache, window=WINDOW):
    # Gemma-3 mixes sliding-window and global layers; so does this.
    return [plain(), vlm_cache.RotatingKVCache(max_size=window, keep=0)]


def live_cache(prompt, generated, plain=vlm_cache.KVCache, window=WINDOW):
    """The cache a request leaves behind: prompt, then all but the last output token."""
    layers = make_layers(plain, window)
    for n, layer in enumerate(layers):
        feed(layer, prompt, n)
        for token in generated[:-1]:
            feed(layer, [token], n)
    return layers


PROMPT_IDS = [2, 7, 7, 7, 7, 40, 41, 42, 43, 44]
OUTPUT_IDS = [901, 902, 903]


@pytest.fixture(params=[vlm_cache.KVCache, lm_cache.KVCache], ids=["vlm", "lm"])
def plain_kv(request):
    return request.param


class TestSnapshotPromptPrefix:
    def test_holds_exactly_the_prompt_positions_not_the_output(self, plain_kv):
        live = live_cache(PROMPT_IDS, OUTPUT_IDS, plain_kv)
        keep = len(PROMPT_IDS) - 1

        snapshot = snapshot_prompt_prefix(live, keep)

        assert [layer.offset for layer in snapshot] == [keep, keep]
        for layer in snapshot:
            assert rows_of(layer) == positions(PROMPT_IDS[:keep])

    def test_shares_nothing_with_the_live_cache(self, plain_kv):
        live = live_cache(PROMPT_IDS, OUTPUT_IDS, plain_kv)
        snapshot = snapshot_prompt_prefix(live, len(PROMPT_IDS) - 1)
        before = [rows_of(layer) for layer in snapshot]

        for n, layer in enumerate(live):  # generation keeps extending the live cache
            feed(layer, [555, 556], n)
        for n, layer in enumerate(snapshot):  # and a hit extends its own copy
            feed(layer, [777], n)

        for layer, kept in zip(snapshot, before):
            assert rows_of(layer)[: len(kept)] == kept
        assert all(
            s.keys is not live_layer.keys for s, live_layer in zip(snapshot, live)
        )

    def test_a_window_that_has_wrapped_cannot_be_cut_back(self):
        live = live_cache(PROMPT_IDS, OUTPUT_IDS, window=len(PROMPT_IDS))

        assert live[1].offset > live[1].max_size
        assert snapshot_prompt_prefix(live, len(PROMPT_IDS) - 1) is None

    def test_a_wrapped_window_is_refused_even_when_nothing_needs_trimming(self):
        live = live_cache(PROMPT_IDS, OUTPUT_IDS, window=len(PROMPT_IDS))

        assert snapshot_prompt_prefix(live, live[1].offset) is None

    def test_a_buffer_whose_start_has_advanced_cannot_be_cut_back(self):
        # A plain KV layer that has dropped its oldest rows (``start_position``
        # moved on): the first rows it holds are no longer positions 0..n, so
        # keeping the "first keep_tokens" would hand back later positions as
        # if they were the prompt's start. Only this guard refuses it (the
        # layer is plain KV and not a wrapped window).
        layer = vlm_cache.KVCache()
        feed(layer, list(range(1, 21)))
        layer.start_position = 12
        assert snapshot_prompt_prefix([layer], 10) is None

        layer.start_position = 0
        assert snapshot_prompt_prefix([layer], 10) is not None

    def test_a_chunked_buffer_that_dropped_its_front_is_refused(self):
        chunked = lm_cache.ChunkedKVCache(chunk_size=8)
        feed(chunked, list(range(1, 21)))
        chunked.maybe_trim_front()
        assert chunked.start_position > 0 and chunked.offset == 20

        assert snapshot_prompt_prefix([chunked], 10) is None

    def test_recurrent_state_cannot_be_cut_back(self):
        live = live_cache(PROMPT_IDS, OUTPUT_IDS)
        live[1] = lm_cache.ArraysCache(2)
        live[1].cache = [mx.ones((1, 4)), mx.ones((1, 4))]

        assert snapshot_prompt_prefix(live, len(PROMPT_IDS) - 1) is None

    def test_nothing_to_keep_stores_nothing(self):
        live = live_cache(PROMPT_IDS, OUTPUT_IDS)

        assert snapshot_prompt_prefix(live, 0) is None
        assert snapshot_prompt_prefix([], 3) is None
        assert snapshot_prompt_prefix(live, len(PROMPT_IDS) + len(OUTPUT_IDS)) is None


class TestManagerHandsOutCopies:
    @pytest.fixture
    def manager(self):
        return MLLMPrefixCacheManager(max_entries=8)

    def test_a_hit_is_a_copy_and_extending_it_leaves_the_entry_alone(self, manager):
        live = live_cache(PROMPT_IDS, OUTPUT_IDS)
        assert manager.store_prompt_cache([IMAGE], "p", live, PROMPT_IDS)
        keep = len(PROMPT_IDS) - 1

        layers, cached_ids = manager.fetch_prompt_cache([IMAGE], "p", PROMPT_IDS)
        for n, layer in enumerate(layers):  # what mlx-vlm does with it
            feed(layer, PROMPT_IDS[keep:] + OUTPUT_IDS, n)
        again, _ = manager.fetch_prompt_cache([IMAGE], "p", PROMPT_IDS)

        assert cached_ids == PROMPT_IDS[:keep]
        assert [layer.offset for layer in again] == [keep, keep]
        for layer in again:
            assert rows_of(layer) == positions(PROMPT_IDS[:keep])

    def test_the_stored_entry_is_a_snapshot_not_the_live_cache(self, manager):
        live = live_cache(PROMPT_IDS, OUTPUT_IDS)
        manager.store_prompt_cache([IMAGE], "p", live, PROMPT_IDS)
        for n, layer in enumerate(live):
            feed(layer, [555, 556], n)

        (entry,) = manager._cache.values()

        assert [layer.offset for layer in entry.kv_cache] == [len(PROMPT_IDS) - 1] * 2
        assert entry.token_ids == PROMPT_IDS[:-1]
        assert all(a is not b for a, b in zip(entry.kv_cache, live))

    def test_ids_that_do_not_extend_the_cached_ones_are_a_miss(self, manager):
        manager.store_prompt_cache(
            [IMAGE], "p", live_cache(PROMPT_IDS, OUTPUT_IDS), PROMPT_IDS
        )
        other = PROMPT_IDS[:5] + [99] + PROMPT_IDS[6:]

        assert manager.fetch_prompt_cache([IMAGE], "p", other) is None
        # No token left to prefill: the cached ids are not a *strict* prefix.
        assert manager.fetch_prompt_cache([IMAGE], "p", PROMPT_IDS[:-1]) is None
        assert manager.stats.hits == 0
        assert manager.stats.misses == 2

    def test_a_cache_that_cannot_be_cut_back_is_not_stored(self, manager):
        wrapped = live_cache(PROMPT_IDS, OUTPUT_IDS, window=len(PROMPT_IDS))

        assert manager.store_prompt_cache([IMAGE], "p", wrapped, PROMPT_IDS) is False
        assert len(manager) == 0

    def test_storing_a_key_again_replaces_it_without_double_counting(self, manager):
        for _ in range(3):
            manager.store_prompt_cache(
                [IMAGE], "p", live_cache(PROMPT_IDS, OUTPUT_IDS), PROMPT_IDS
            )

        (entry,) = manager._cache.values()
        assert len(manager) == 1
        assert manager._current_memory == entry.memory_size

    def test_legacy_fetch_cache_is_a_copy_of_the_stored_layers(self, manager):
        manager.store_cache([IMAGE], "p", [[1, 2, 3]])

        first, _ = manager.fetch_cache([IMAGE], "p")
        first[0].append(4)
        second, hit = manager.fetch_cache([IMAGE], "p")

        assert hit is True
        assert second == [[1, 2, 3]]
        assert second is not first

    def test_legacy_store_cache_copies_a_live_cache(self, manager):
        live = live_cache(PROMPT_IDS, OUTPUT_IDS)
        manager.store_cache([IMAGE], "p", live, num_tokens=len(PROMPT_IDS))
        for n, layer in enumerate(live):
            feed(layer, [555], n)

        (entry,) = manager._cache.values()
        assert entry.kv_cache[0].offset == len(PROMPT_IDS) + len(OUTPUT_IDS) - 1


# -- the Simple-engine model, on a fake mlx-vlm that follows its contract -----


class FakeMlxVlm:
    """``generate`` / ``stream_generate`` with mlx-vlm's prompt-cache contract.

    The model's output is a function of every (position, token) row in the
    cache after prefill, so a cache holding the wrong rows changes the answer.
    """

    def __init__(self, window=WINDOW, layers=None):
        self.window = window
        self.layers = layers
        self.calls = []

    def make_prompt_cache(self, language_model, max_kv_size=None):
        if self.layers is not None:
            return self.layers()
        return make_layers(window=self.window)

    def _run(self, prompt_cache, prompt_cache_state, input_ids, max_tokens):
        ids = [int(t) for t in input_ids.flatten().tolist()]
        cache, feed_ids, cached = prompt_cache, ids, 0
        call = {"prompt_ids": ids, "state_hit": False}
        if prompt_cache_state is not None and prompt_cache_state.cache is not None:
            prefix = prompt_cache_state.find_prefix_length(ids)
            if 0 < prefix < len(ids):
                cache, feed_ids, cached = prompt_cache_state.cache, ids[prefix:], prefix
                for layer in cache:  # mlx-vlm trims a longer cache to the prefix
                    layer.trim(max(0, layer.offset - prefix))
                call["state_hit"] = True
        call.update(
            fed=feed_ids,
            offsets_before=[getattr(layer, "offset", None) for layer in cache],
            cache=cache,
        )
        self.calls.append(call)
        for n, layer in enumerate(cache):
            if hasattr(layer, "update_and_fetch"):
                feed(layer, feed_ids, n)
            else:
                layer.cache = [mx.ones((1, 4)) * len(ids)] * len(layer.cache)
        seen = rows_of(cache[0])
        answer = 800 + sum(p * t for p, t in seen) % 97
        generated = [answer + j for j in range(max_tokens)]
        for token in generated[:-1]:
            for n, layer in enumerate(cache):
                if hasattr(layer, "update_and_fetch"):
                    feed(layer, [token], n)
        if prompt_cache_state is not None:
            prompt_cache_state.pending = (ids + generated, cache)
        return ids, generated, cached, seen

    def generate(
        self,
        model,
        processor,
        prompt,
        image=None,
        *,
        prompt_cache=None,
        prompt_cache_state=None,
        input_ids,
        max_tokens=3,
        **kwargs,
    ):
        ids, generated, cached, seen = self._run(
            prompt_cache, prompt_cache_state, input_ids, max_tokens
        )
        if prompt_cache_state is not None:
            prompt_cache_state.update(*prompt_cache_state.pending)
        return SimpleNamespace(
            text=" ".join(map(str, generated)),
            prompt_tokens=len(ids),
            generation_tokens=len(generated),
            cached_tokens=cached,
            finish_reason="length",
        )

    def stream_generate(
        self,
        model,
        processor,
        prompt,
        image=None,
        *,
        prompt_cache=None,
        prompt_cache_state=None,
        input_ids,
        max_tokens=3,
        **kwargs,
    ):
        ids, generated, cached, seen = self._run(
            prompt_cache, prompt_cache_state, input_ids, max_tokens
        )
        for n, token in enumerate(generated):
            yield SimpleNamespace(
                text=f"{token} ",
                prompt_tokens=len(ids),
                generation_tokens=n + 1,
                finish_reason="length" if n == len(generated) - 1 else None,
            )
        # mlx-vlm saves the cache after its last yield, i.e. only when the
        # consumer reads to the end.
        if prompt_cache_state is not None:
            prompt_cache_state.update(*prompt_cache_state.pending)


def prompt_ids_for(prompt, n_images):
    return [2] + [7] * (4 * n_images) + [10 + ord(c) % 90 for c in prompt]


@pytest.fixture
def engine(monkeypatch):
    vlm = FakeMlxVlm()

    def prepare_inputs(processor, images=None, prompts=None, **kwargs):
        ids = prompt_ids_for(prompts, len(images or []))
        return {
            "input_ids": mx.array([ids]),
            "pixel_values": None,
            "attention_mask": mx.ones((1, len(ids))),
        }

    package = ModuleType("mlx_vlm")
    package.generate = vlm.generate
    package.stream_generate = vlm.stream_generate
    generate_module = ModuleType("mlx_vlm.generate")
    generate_module.PromptCacheState = PromptCacheState
    prompt_utils = ModuleType("mlx_vlm.prompt_utils")
    prompt_utils.apply_chat_template = lambda *a, **k: "<bos>" + a[2]
    prompt_utils.get_chat_template = (
        lambda processor, messages, **k: "<bos>"
        + " ".join(
            part["text"]
            for m in messages
            for part in (m["content"] if isinstance(m["content"], list) else [])
            if part.get("type") == "text"
        )
    )
    cache = ModuleType("mlx_vlm.models.cache")
    cache.make_prompt_cache = vlm.make_prompt_cache
    models = ModuleType("mlx_vlm.models")
    models.cache = cache
    utils = ModuleType("mlx_vlm.utils")
    utils.prepare_inputs = prepare_inputs
    for module in (package, generate_module, prompt_utils, models, cache, utils):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    model = MLXMultimodalLM("fake", enable_cache=True)
    model._loaded = True
    model.model = SimpleNamespace(
        language_model=object(), config=SimpleNamespace(image_token_index=7)
    )
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(bos_token="<bos>"))
    model.config = {}
    monkeypatch.setattr(model, "_prepare_images", lambda images: list(images))
    return model, vlm


QUESTION = "what is shown"
CHAT = [
    {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": IMAGE}},
            {"type": "text", "text": QUESTION},
        ],
    }
]


def ask_generate(model):
    return model.generate(QUESTION, images=[IMAGE], max_tokens=3, temperature=0.0)


def ask_chat(model):
    return model.chat(CHAT, max_tokens=3, temperature=0.0)


def ask_stream(model):
    return "".join(
        o.text for o in model.stream_chat(CHAT, max_tokens=3, temperature=0.0)
    )


def stored_entry(model):
    (entry,) = model._cache_manager._cache.values()
    return entry


def text_of(result):
    return result if isinstance(result, str) else result.text


@pytest.mark.parametrize(
    "ask", [ask_generate, ask_chat, ask_stream], ids=["generate", "chat", "stream_chat"]
)
class TestSimpleEngineHit:
    def test_a_hit_prefills_only_the_last_prompt_token(self, engine, ask):
        model, vlm = engine
        cold = text_of(ask(model))

        hit = text_of(ask(model))

        first, second = vlm.calls
        prompt = first["prompt_ids"]
        assert first["fed"] == prompt and first["offsets_before"] == [0, 0]
        assert second["state_hit"] is True
        assert second["fed"] == prompt[-1:]
        assert second["offsets_before"] == [len(prompt) - 1] * 2
        assert hit == cold

    def test_the_hit_cache_holds_each_prompt_token_once_in_position(self, engine, ask):
        model, vlm = engine
        ask(model)
        ask(model)

        first, second = vlm.calls
        prompt = first["prompt_ids"]
        for n, layer in enumerate(second["cache"]):
            keep = len(prompt) + len(OUTPUT_IDS) - 1
            assert layer.offset == keep, n
        assert rows_of(second["cache"][0])[: len(prompt)] == positions(prompt)

    def test_hits_never_change_the_stored_entry(self, engine, ask):
        model, vlm = engine
        ask(model)
        prompt = vlm.calls[0]["prompt_ids"]
        keep = len(prompt) - 1
        stored = stored_entry(model)
        stored_layers = list(stored.kv_cache)

        for _ in range(3):
            ask(model)

        entry = stored_entry(model)
        assert entry is stored and list(entry.kv_cache) == stored_layers
        assert [layer.offset for layer in entry.kv_cache] == [keep, keep]
        assert rows_of(entry.kv_cache[0]) == positions(prompt[:keep])
        assert entry.token_ids == prompt[:keep]
        assert all(
            layer is not stored_layer
            for call in vlm.calls
            for layer in call["cache"]
            for stored_layer in entry.kv_cache
        )

    def test_the_entry_holds_the_prompt_not_the_output(self, engine, ask):
        model, vlm = engine
        ask(model)

        prompt = vlm.calls[0]["prompt_ids"]
        entry = stored_entry(model)
        assert [layer.offset for layer in entry.kv_cache] == [len(prompt) - 1] * 2
        assert vlm.calls[0]["cache"][0].offset == len(prompt) + len(OUTPUT_IDS) - 1


class TestSimpleEngineSharedKeySpace:
    def test_chat_stores_and_stream_chat_hits(self, engine):
        model, vlm = engine
        cold = ask_chat(model).text

        streamed = ask_stream(model)

        prompt = vlm.calls[0]["prompt_ids"]
        hit = vlm.calls[1]
        assert hit["state_hit"] is True
        assert hit["fed"] == prompt[-1:]
        assert rows_of(hit["cache"][0])[: len(prompt)] == positions(prompt)
        assert streamed.split() == cold.split()

    def test_stream_chat_stores_for_later_requests(self, engine):
        model, vlm = engine
        cold = ask_stream(model)

        hit = ask_chat(model)

        assert vlm.calls[1]["state_hit"] is True
        assert hit.text.split() == cold.split()

    def test_a_stream_that_stops_early_stores_nothing(self, engine):
        model, vlm = engine

        list(model.stream_chat(CHAT, max_tokens=3, temperature=0.0, stop=[" "]))

        assert len(model._cache_manager) == 0

    def test_a_request_without_media_does_not_touch_the_cache(self, engine):
        model, vlm = engine

        model.generate("no image here", max_tokens=3, temperature=0.0)
        model.chat([{"role": "user", "content": "hi"}], max_tokens=3)

        assert len(model._cache_manager) == 0
        assert all(call["fed"] == call["prompt_ids"] for call in vlm.calls)

    def test_use_cache_false_neither_reads_nor_writes(self, engine):
        model, vlm = engine
        ask_generate(model)

        model.generate(QUESTION, images=[IMAGE], max_tokens=3, use_cache=False)

        assert vlm.calls[1]["fed"] == vlm.calls[1]["prompt_ids"]
        assert stored_entry(model).prompt_tokens == len(vlm.calls[0]["prompt_ids"]) - 1

    def test_a_draft_model_request_bypasses_the_cache(self, engine):
        model, vlm = engine
        model._draft_model = SimpleNamespace(accept_lens=[], _vllm_mlx_draft_counts=[])
        model.default_draft_enabled = True

        ask_generate(model)
        ask_generate(model)

        assert len(model._cache_manager) == 0
        assert all(call["fed"] == call["prompt_ids"] for call in vlm.calls)


ASKERS = {"generate": ask_generate, "chat": ask_chat, "stream": ask_stream}


@pytest.mark.parametrize(
    "sequence",
    [
        ["generate", "generate", "generate"],
        ["chat", "stream", "stream", "chat"],
        ["stream", "chat", "stream", "stream"],
        ["chat", "chat", "chat"],
    ],
    ids="-".join,
)
def test_no_request_prefills_on_top_of_positions_a_cache_already_holds(
    engine, sequence
):
    model, vlm = engine

    answers = [text_of(ASKERS[name](model)).split() for name in sequence]

    for call in vlm.calls:
        prompt = call["prompt_ids"]
        # What is fed continues exactly where each layer's cache stands ...
        assert [o + len(call["fed"]) for o in call["offsets_before"]] == [
            len(prompt)
        ] * len(call["offsets_before"])
        # ... so every prompt token sits in the cache once, at its own position.
        assert rows_of(call["cache"][0])[: len(prompt)] == positions(prompt)
    assert all(answer == answers[0] for answer in answers)


class TestUncuttableCachesAreNotReused:
    def test_a_wrapped_sliding_window_means_cold_prefill_every_time(
        self, engine, monkeypatch
    ):
        model, vlm = engine
        vlm.window = 6  # shorter than the prompt: the window wraps while prefilling

        first = ask_generate(model)
        second = ask_generate(model)

        assert len(model._cache_manager) == 0
        assert all(call["fed"] == call["prompt_ids"] for call in vlm.calls)
        assert second.text == first.text

    def test_recurrent_layers_mean_cold_prefill_every_time(self, engine):
        model, vlm = engine
        vlm.layers = lambda: [vlm_cache.KVCache(), lm_cache.ArraysCache(2)]

        first = ask_generate(model)
        second = ask_generate(model)

        assert len(model._cache_manager) == 0
        assert all(call["fed"] == call["prompt_ids"] for call in vlm.calls)
        assert second.text == first.text
