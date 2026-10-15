# SPDX-License-Identifier: Apache-2.0
"""A prefix-cache hit hands logits processors the same token history as a cold request.

A logits processor (repetition penalty, a thinking-budget state machine, ...)
is called with the tokens the BatchGenerator has seen for the sequence: the
prompt followed by what was generated.  A cache hit prefills only the prompt's
remainder, so unless the scheduler also says which tokens the cached prefix
holds, the processor sees the remainder alone -- for an exact repeat a single
token -- and a request answers differently from the same one served cold.

The model below is a small randomly initialised Llama (fixed seed, scaled up so
its greedy choices are distinct).  The prompt ends with the model's own
unpenalised continuation, so a penalty over the last 20 tokens of context
changes the answer exactly when the context includes the prompt's tail.
"""

import pytest

mx = pytest.importorskip("mlx.core")
llama = pytest.importorskip("mlx_lm.models.llama")
tree_map = pytest.importorskip("mlx.utils").tree_map

from vllm_mlx.request import Request, SamplingParams  # noqa: E402
from vllm_mlx.scheduler import Scheduler, SchedulerConfig  # noqa: E402

VOCAB = 128
CONTEXT = 20


class _Tokenizer:
    eos_token_id = 0
    eos_token_ids = {0}
    bos_token = None
    vocab_size = VOCAB

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))

    def encode(self, text, **kwargs):
        return [1]


@pytest.fixture(scope="module")
def model():
    mx.random.seed(0)
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=VOCAB,
    )
    net = llama.Model(args)
    net.update(tree_map(lambda p: p * 6.0 if p.ndim > 1 else p, net.parameters()))
    mx.eval(net.parameters())
    return net


class Recording:
    """A history-dependent processor: forbids the last ``CONTEXT`` tokens it was
    shown, and keeps every context it was called with."""

    def __init__(self):
        self.contexts = []

    def __call__(self, tokens, logits):
        context = tokens.tolist() if len(tokens) else []
        self.contexts.append(context)
        if context:
            logits[:, context[-CONTEXT:]] = -float("inf")
        return logits


def make_scheduler(model, memory_aware):
    return Scheduler(
        model=model,
        tokenizer=_Tokenizer(),
        config=SchedulerConfig(
            enable_prefix_cache=True,
            use_memory_aware_cache=memory_aware,
            cache_memory_mb=64,
        ),
    )


def generate(
    scheduler, prompt, request_id, processors, max_tokens=8, cached_insert_fails=False
):
    request = Request(
        request_id=request_id,
        prompt="p",
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            logits_processors=list(processors),
        ),
    )
    request.num_prompt_tokens = len(prompt)
    if cached_insert_fails:
        scheduler._ensure_batch_generator(request.sampling_params)
        real_insert = scheduler.batch_generator.insert

        def insert(prompts, **kwargs):
            if kwargs.get("caches"):
                raise RuntimeError("the cache does not fit")
            return real_insert(prompts, **kwargs)

        scheduler.batch_generator.insert = insert
    scheduler.add_request(request)
    for _ in range(200):
        scheduler.step()
        if not scheduler.has_requests():
            break
    return list(request.output_token_ids), request.cached_tokens


@pytest.fixture(scope="module")
def prompt(model):
    pool = [5, 9, 13, 21, 34, 55, 89, 7, 3, 11]
    head = [pool[(i * i + i // 3) % len(pool)] for i in range(150)]
    scheduler = make_scheduler(model, memory_aware=True)
    unpenalised, _ = generate(scheduler, head, "probe", [])
    return head + unpenalised


@pytest.mark.parametrize("memory_aware", [True, False], ids=["memory", "legacy"])
def test_a_hit_answers_like_the_cold_request_under_a_history_dependent_processor(
    model, prompt, memory_aware
):
    scheduler = make_scheduler(model, memory_aware)
    cold_processor, warm_processor = Recording(), Recording()
    cold, cold_cached = generate(scheduler, prompt, "cold", [cold_processor])
    unconstrained, _ = generate(make_scheduler(model, True), prompt, "free", [])
    assert cold_cached == 0
    assert cold != unconstrained, "the processor must change the answer"

    warm, warm_cached = generate(scheduler, prompt, "warm", [warm_processor])

    assert warm_cached > 0, "the second request must be served from the cache"
    assert warm == cold
    # Not just the same answer: the processor was shown the same history at
    # every step (the whole prompt, then the output so far).
    assert warm_processor.contexts == cold_processor.contexts
    assert cold_processor.contexts[0] == prompt


def test_a_cache_the_batch_generator_refuses_leaves_the_full_history(model, prompt):
    """When the insert with the cached prefix fails the scheduler prefills the
    whole prompt: the processor must see it once, not the cached prefix twice."""
    scheduler = make_scheduler(model, memory_aware=True)
    cold_processor, retried_processor = Recording(), Recording()
    cold, _ = generate(scheduler, prompt, "cold", [cold_processor])

    retried, cached = generate(
        scheduler, prompt, "retry", [retried_processor], cached_insert_fails=True
    )

    assert cached == 0
    assert retried == cold
    assert retried_processor.contexts == cold_processor.contexts
