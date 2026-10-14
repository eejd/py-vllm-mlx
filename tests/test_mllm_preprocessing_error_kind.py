# SPDX-License-Identifier: Apache-2.0
"""A request the multimodal scheduler cannot preprocess fails as the client's, not ours.

vLLM answers 400 for an input its processor rejects (an undecodable image).  The
scheduler here preprocesses after admission and reports the failure as an
aborted request (``finish_reason="error"``); ``error_kind`` is what tells it
apart from a failure while generating, which is a 500.  Every place that turns
a preprocessing exception into an error response must say so.
"""

import threading
from unittest.mock import MagicMock

import pytest

mx = pytest.importorskip("mlx.core")

from vllm_mlx.mllm_batch_generator import (  # noqa: E402
    MLLMBatchGenerator,
    MLLMBatchRequest,
    MLLMBatchResponse,
    MLLMBatchStats,
    install_chunked_prefill_mllm,
)
from vllm_mlx.mllm_scheduler import MLLMRequest, MLLMScheduler  # noqa: E402
from vllm_mlx.request import (  # noqa: E402
    ERROR_KIND_PREPROCESSING,
    SamplingParams,
)


def _undecodable(_request):
    raise ValueError("Failed to load image: cannot identify image file")


def _generator():
    gen = MLLMBatchGenerator.__new__(MLLMBatchGenerator)
    gen._stats = MLLMBatchStats()
    gen._pending_error_responses = []
    gen._aborted_request_ids = set()
    gen._prefill_progress = {}
    gen._prefix_checkpoint_lock = threading.Lock()
    gen._request_prefix_checkpoints = {}
    gen.active_batch = None
    gen.stop_tokens = set()
    gen.unprocessed_requests = []
    gen._think_suffix_len = 0
    gen.max_kv_size = 0
    gen.prefix_cache = None
    gen._compatible_pending_requests = lambda reqs, n, reference=None: list(reqs)[:n]
    gen._preprocess_request = _undecodable
    return gen


def _request(uid, request_id):
    req = MLLMBatchRequest(uid=uid, request_id=request_id, prompt="x")
    req.is_text_only = True
    return req


def _only(responses):
    (response,) = responses
    assert response.finish_reason == "error"
    return response


def test_process_prompts_marks_a_preprocessing_failure():
    gen = _generator()
    req = _request(1, "bad")

    assert MLLMBatchGenerator._process_prompts(gen, [req]) is None

    assert _only(gen._pending_error_responses).error_kind == ERROR_KIND_PREPROCESSING


def test_chunked_prefill_marks_a_preprocessing_failure_of_the_next_request():
    gen = _generator()
    gen._next = lambda: []
    install_chunked_prefill_mllm(gen, budget=1024)
    gen.unprocessed_requests.append(_request(2, "bad-next"))

    responses = gen._next()

    assert _only(responses).error_kind == ERROR_KIND_PREPROCESSING
    assert gen.unprocessed_requests == []


def _rope_failure(_request):
    raise RuntimeError("rope index could not be derived")


def _preprocessed_but_rope_fails():
    gen = _generator()
    gen._preprocess_request = lambda request: None
    gen._derive_request_rope_deltas = _rope_failure
    return gen


def test_process_prompts_does_not_blame_the_client_for_a_rope_failure():
    """The inputs were preprocessed; what failed afterwards is the server's."""
    gen = _preprocessed_but_rope_fails()

    assert MLLMBatchGenerator._process_prompts(gen, [_request(1, "rope")]) is None

    assert _only(gen._pending_error_responses).error_kind is None


def test_process_prompts_keeps_the_kinds_apart_within_one_batch():
    gen = _generator()
    gen._derive_request_rope_deltas = _rope_failure
    gen._preprocess_request = lambda request: (
        _undecodable(request) if request.request_id == "bad" else None
    )
    batch = [_request(1, "bad"), _request(2, "rope")]

    assert MLLMBatchGenerator._process_prompts(gen, batch) is None

    kinds = {r.request_id: r.error_kind for r in gen._pending_error_responses}
    assert kinds == {"bad": ERROR_KIND_PREPROCESSING, "rope": None}


def test_chunked_prefill_does_not_blame_the_client_for_a_rope_failure():
    gen = _preprocessed_but_rope_fails()
    gen._next = lambda: []
    install_chunked_prefill_mllm(gen, budget=1024)
    gen.unprocessed_requests.append(_request(2, "rope-next"))

    assert _only(gen._next()).error_kind is None


def test_chunked_prefill_marks_a_preprocessing_failure_of_an_inlined_request():
    """A short request picked up while a long prefill is in flight."""
    gen = _generator()
    gen._next = lambda: []
    install_chunked_prefill_mllm(gen, budget=2)
    gen.language_model = MagicMock()
    gen._language_model_kwargs = lambda *a, **k: {}
    long_request = _request(1, "long")
    gen._partial = {
        "request": long_request,
        "remaining_ids": mx.zeros((1, 10), dtype=mx.int32),
        "cached_count": 0,
        "processed": 0,
        "chunk_count": 0,
        "total": 10,
        "cache": [],
        "checkpoint_at": None,
    }
    gen.unprocessed_requests.append(_request(2, "bad-inline"))

    responses = gen._next()

    assert _only(responses).request_id == "bad-inline"
    assert responses[0].error_kind == ERROR_KIND_PREPROCESSING


def _scheduler():
    sched = MLLMScheduler.__new__(MLLMScheduler)
    sched.processor = MagicMock()
    sched.stop_tokens = set()
    sched.uid_to_request_id = {0: "r"}
    sched._detokenizer_pool = {}
    sched.batch_generator = None
    sched.total_completion_tokens = 0
    sched.num_requests_processed = 0
    sched.running = {
        "r": MLLMRequest(request_id="r", prompt="p", sampling_params=SamplingParams())
    }
    return sched


@pytest.mark.parametrize("error_kind", [ERROR_KIND_PREPROCESSING, None])
def test_scheduler_hands_the_error_kind_to_the_engine(error_kind):
    response = MLLMBatchResponse(
        uid=0,
        request_id="r",
        token=0,
        logprobs=mx.zeros(1),
        finish_reason="error",
        error_kind=error_kind,
    )

    outputs, finished = _scheduler()._process_batch_responses([response])

    assert finished == {"r"}
    assert outputs[0].finish_reason == "error"
    assert outputs[0].error_kind == error_kind
