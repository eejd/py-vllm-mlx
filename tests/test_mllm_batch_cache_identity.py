# SPDX-License-Identifier: Apache-2.0
"""MLLM batch generator: prompt KV that is stored or served must be the KV of
exactly the tokens it is keyed by.

The real ``MLLMBatchGenerator`` (prefix cache, chunked prefill, MTP) runs over a
simulated language model whose KV at position ``p`` is the chain hash of
``tokens[:p+1]`` (``tests/mllm_cache_sim.py``). A stored entry that holds one position
too many or too few, or KV computed behind another prefix, is visible in the tensors
and changes the tokens a later request is served.
"""

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from tests.mllm_cache_sim import (
    LAYERS,
    VOCAB,
    chain_hashes,
    greedy_continuation,
    layer_offsets,
    layer_values,
    make_generator,
    make_request,
    next_token,
    run_to_completion,
    stored_entries,
)

PROMPT = list(range(3, 15))


def kv_layers(entry_cache):
    return [layer for layer in entry_cache if hasattr(layer, "keys")]


def assert_entries_are_prompt_kv(gen):
    """Every stored entry holds, in every layer, exactly the KV of its own key."""
    entries = stored_entries(gen)
    assert entries, "nothing was stored; the check would be vacuous"
    for key, cache in entries.items():
        layers = kv_layers(cache)
        assert layer_offsets(layers) == [len(key)] * len(layers), key
        for layer in range(LAYERS):
            expected = [float(h + layer) for h in chain_hashes(key)]
            assert layer_values(layers, layer) == expected, key


def serve(gen, request_id, ids, max_tokens=6, **kwargs):
    out = run_to_completion(gen, [make_request(request_id, ids, max_tokens, **kwargs)])
    return out[request_id]


MTP_MODES = [
    pytest.param(True, id="draft-accepted"),
    pytest.param(False, id="draft-rejected"),
]


class TestMtpStoresPromptKv:
    """A speculative verify step advances the cache past the emitted tokens."""

    @pytest.mark.parametrize("draft_correct", MTP_MODES)
    @pytest.mark.parametrize("max_tokens", range(1, 9))
    def test_entry_holds_the_prompt_and_nothing_else(self, draft_correct, max_tokens):
        # max_tokens sweeps both ways a request can end under MTP: on the primary
        # token, or on the deferred draft emitted right after it.
        gen, _ = make_generator(mtp=True, draft_correct=draft_correct)
        tokens = serve(gen, "mtp", PROMPT, max_tokens)

        assert tokens == greedy_continuation(PROMPT, max_tokens)
        assert_entries_are_prompt_kv(gen)

    @pytest.mark.parametrize("draft_correct", MTP_MODES)
    def test_later_requests_are_served_cold_answers(self, draft_correct):
        gen, _ = make_generator(mtp=True, draft_correct=draft_correct)
        serve(gen, "cold", PROMPT)

        repeat = serve(gen, "repeat", PROMPT)
        extension_ids = PROMPT + [40, 41, 42]
        extension = serve(gen, "extension", extension_ids)

        assert repeat == greedy_continuation(PROMPT, 6)
        assert extension == greedy_continuation(extension_ids, 6)

    @pytest.mark.parametrize("draft_correct", MTP_MODES)
    def test_interleaved_prefill_and_mtp_together(self, draft_correct):
        gen, _ = make_generator(
            mtp=True, draft_correct=draft_correct, chunked_prefill=4
        )
        serve(gen, "cold", PROMPT)
        extension_ids = PROMPT + [40, 41, 42, 43, 44, 45]

        assert serve(gen, "extension", extension_ids) == greedy_continuation(
            extension_ids, 6
        )
        assert_entries_are_prompt_kv(gen)

    def test_stop_token_emitted_as_a_draft(self):
        gen, _ = make_generator(mtp=True, draft_correct=True)
        expected = greedy_continuation(PROMPT, 8)
        gen.stop_tokens = {expected[2]}  # the second deferred draft of the run

        tokens = serve(gen, "stop", PROMPT, 8)

        assert tokens[-1] == expected[2]
        assert_entries_are_prompt_kv(gen)

    def test_batched_requests_ending_at_different_steps(self):
        gen, _ = make_generator(mtp=True, draft_correct=True)
        short, long = PROMPT[:5], PROMPT
        out = run_to_completion(
            gen, [make_request("a", short, 3), make_request("b", long, 8)]
        )

        assert out["a"] == greedy_continuation(short, 3)
        assert out["b"] == greedy_continuation(long, 8)
        assert_entries_are_prompt_kv(gen)


class TestRequestsJoiningAnMtpBatch:
    """A request that joins while others decode must not disturb them."""

    @staticmethod
    def drive(gen, first, second, join_after):
        gen.insert([first])
        out = {first.request_id: [], second.request_id: []}
        done = set()
        for step in range(1, 300):
            if step == join_after:
                gen.insert([second])
            for resp in gen.next():
                if resp.request_id in done:
                    continue
                out[resp.request_id].append(resp.token)
                if resp.finish_reason is not None:
                    done.add(resp.request_id)
            if len(done) == 2:
                return out
        raise AssertionError(f"did not finish: {out}")

    @pytest.mark.parametrize("hybrid", [False, True], ids=["attention", "hybrid"])
    @pytest.mark.parametrize("draft_correct", MTP_MODES)
    @pytest.mark.parametrize("join_after", [2, 3, 4])
    @pytest.mark.parametrize(
        "processors", [None, "noop"], ids=["plain-joiner", "processor-joiner"]
    )
    def test_resident_request_keeps_its_cold_answer(
        self, hybrid, draft_correct, join_after, processors
    ):
        gen, _ = make_generator(
            mtp=True, draft_correct=draft_correct, prefix_cache=False, hybrid=hybrid
        )
        resident = make_request("resident", PROMPT, 14)
        joiner = make_request(
            "joiner",
            list(range(20, 27)),
            8,
            logits_processors=[lambda tokens, logits: logits] if processors else None,
        )

        out = self.drive(gen, resident, joiner, join_after)

        assert out["resident"] == greedy_continuation(PROMPT, 14)
        assert out["joiner"] == greedy_continuation(list(range(20, 27)), 8)

    @pytest.mark.parametrize("draft_correct", MTP_MODES)
    def test_interleaved_prefill_joiner(self, draft_correct):
        gen, _ = make_generator(
            mtp=True, draft_correct=draft_correct, chunked_prefill=4
        )
        resident = make_request("resident", PROMPT, 14)
        joiner = make_request("joiner", list(range(20, 36)), 8)

        out = self.drive(gen, resident, joiner, 3)

        assert out["resident"] == greedy_continuation(PROMPT, 14)
        assert out["joiner"] == greedy_continuation(list(range(20, 36)), 8)
        assert_entries_are_prompt_kv(gen)

    def test_a_partially_verified_batch_is_refused_not_decoded(self):
        gen, _ = make_generator(mtp=True, prefix_cache=False)
        gen.insert([make_request("resident", PROMPT, 14)])
        gen.next()
        gen.next()
        gen.insert([make_request("joiner", list(range(20, 27)), 8)])
        gen.active_batch.on_extend = None  # a join that skipped the admission

        with pytest.raises(RuntimeError, match="partially verified"):
            for _ in range(3):
                gen.next()


class TestPrefixCacheKeepsKeyAndKvAligned:
    """Batched, chunked and think-suffix stores/fetches (no speculative decoding)."""

    ROUNDS = [
        [(PROMPT[:10], 4), (PROMPT[:7], 9), (PROMPT, 2)],
        [(PROMPT[:10] + [50, 51], 5), (PROMPT[:7] + [60], 3), (PROMPT + [1, 2], 6)],
        [(PROMPT[:10], 3), (PROMPT[:12], 7)],
    ]

    @pytest.mark.parametrize("hybrid", [False, True], ids=["attention", "hybrid"])
    @pytest.mark.parametrize("chunked", [0, 6], ids=["blocking", "interleaved"])
    @pytest.mark.parametrize("suffix", [0, 2], ids=["no-suffix", "think-suffix"])
    def test_cold_and_warm_rounds(self, hybrid, chunked, suffix):
        gen, _ = make_generator(
            chunked_prefill=chunked, prefill_step_size=8, hybrid=hybrid
        )
        gen._think_suffix_len = suffix
        for round_no, specs in enumerate(self.ROUNDS):
            requests = [
                make_request(f"r{round_no}_{i}", ids, max_tokens)
                for i, (ids, max_tokens) in enumerate(specs)
            ]
            out = run_to_completion(gen, requests)
            for request, (ids, max_tokens) in zip(requests, specs):
                assert out[request.request_id] == greedy_continuation(ids, max_tokens)
            assert_entries_are_prompt_kv(gen)

    def test_a_request_joining_a_running_batch_stores_its_own_prompt(self):
        gen, _ = make_generator()
        resident = make_request("resident", PROMPT, 12)
        joiner = make_request("joiner", PROMPT[:6] + [70, 71], 5)

        out = TestRequestsJoiningAnMtpBatch.drive(gen, resident, joiner, 3)

        assert out["resident"] == greedy_continuation(PROMPT, 12)
        assert out["joiner"] == greedy_continuation(PROMPT[:6] + [70, 71], 5)
        assert_entries_are_prompt_kv(gen)


def kv_of(tokens, positions=None, layers=LAYERS):
    """A request cache holding the KV of ``tokens`` (``positions`` of them)."""
    cache = [KVCache() for _ in range(layers)]
    hashes = chain_hashes(tokens)[: len(tokens) if positions is None else positions]
    for layer, c in enumerate(cache):
        kv = mx.array([float(h + layer) for h in hashes])[None, None, :, None]
        c.update_and_fetch(kv, kv)
    return cache


class TestMisalignedEntries:
    """An entry whose positions do not match its key is never served or stored."""

    @pytest.mark.parametrize("extra", [1, 2, 3])
    def test_oversized_entry_is_not_served(self, extra):
        gen, _ = make_generator()
        # KV of ``extra`` more tokens than the key says (what an MTP verify
        # step left behind before the store was aligned).
        gen.prefix_cache.store(PROMPT, kv_of(PROMPT + [40, 41, 42][:extra]))

        repeat = serve(gen, "repeat", PROMPT)
        extension_ids = PROMPT + [50, 51]
        extension = serve(gen, "extension", extension_ids)

        assert repeat == greedy_continuation(PROMPT, 6)
        assert extension == greedy_continuation(extension_ids, 6)

    def test_short_entry_is_not_served(self):
        gen, _ = make_generator()
        gen.prefix_cache.store(PROMPT, kv_of(PROMPT, positions=len(PROMPT) - 2))

        assert serve(gen, "repeat", PROMPT) == greedy_continuation(PROMPT, 6)

    def test_oversized_entry_is_not_served_on_the_interleaved_path(self):
        gen, _ = make_generator(chunked_prefill=4)
        gen.prefix_cache.store(PROMPT, kv_of(PROMPT + [40, 41]))
        extension_ids = PROMPT + [50, 51, 52, 53, 54]

        assert serve(gen, "extension", extension_ids) == greedy_continuation(
            extension_ids, 6
        )

    def test_exact_entry_with_logits_is_not_served_for_a_longer_prompt(
        self, monkeypatch
    ):
        # The stored last-token logits answer the exact prompt only. If the
        # cache hands back a shorter entry for it (an eviction between the
        # exact lookup and the fetch), they must not be paired with that KV.
        gen, _ = make_generator()
        logits = mx.where(mx.arange(VOCAB)[None, :] == next_token(PROMPT), 10.0, 0.0)
        entry = gen.prefix_cache.prepare_store(
            PROMPT, kv_of(PROMPT), auxiliary={"last_logits": logits}
        )
        assert gen.prefix_cache.commit_prepared(entry)
        shorter = PROMPT[:8]
        monkeypatch.setattr(
            gen.prefix_cache,
            "fetch",
            lambda ids: (kv_of(shorter), list(ids[len(shorter) :])),
        )

        assert serve(gen, "repeat", PROMPT) == greedy_continuation(PROMPT, 6)

    def test_snapshot_with_the_wrong_number_of_positions_is_not_stored(self):
        gen, _ = make_generator()
        cache = kv_of(PROMPT + [40, 41])

        assert not gen._store_prefix_snapshot(PROMPT, cache, 1, "r", "completion")
        assert not gen._store_prefix_snapshot(PROMPT, cache, 3, "r", "completion")
        assert not stored_entries(gen)
        assert gen._store_prefix_snapshot(PROMPT, cache, 2, "r", "completion")
        assert_entries_are_prompt_kv(gen)

    def test_layers_that_disagree_on_their_position_are_not_stored(self):
        gen, _ = make_generator()
        cache = kv_of(PROMPT + [40, 41])
        cache[1] = kv_of(PROMPT + [40, 41, 42])[1]
        request = make_request("r", PROMPT, 3)

        class Batch:
            requests = [request]
            num_tokens = [3]

            @staticmethod
            def extract_cache(_):
                return cache

        gen._maybe_store_prefix_cache(Batch, [0])

        assert not stored_entries(gen)
