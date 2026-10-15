# SPDX-License-Identifier: Apache-2.0
"""
Request management for vllm-mlx continuous batching.

This module provides Request and RequestStatus classes adapted from vLLM's
request management system, simplified for MLX backend.
"""

import enum
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from .mllm_specprefill import SpecPrefillOutcome
    from .paged_cache import BlockTable


class RequestStatus(enum.IntEnum):
    """Status of a request in the scheduling system."""

    # Request is waiting to be scheduled
    WAITING = enum.auto()
    # Request is currently being processed (generating tokens)
    RUNNING = enum.auto()
    # Request was preempted and needs to be resumed
    PREEMPTED = enum.auto()
    # Request finished successfully (hit stop token)
    FINISHED_STOPPED = enum.auto()
    # Request finished due to max_tokens limit
    FINISHED_LENGTH_CAPPED = enum.auto()
    # Request was aborted by user
    FINISHED_ABORTED = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        """Check if the status indicates a finished request."""
        return status > RequestStatus.PREEMPTED

    @staticmethod
    def get_finish_reason(status: "RequestStatus") -> Optional[str]:
        """Get the finish reason string for a finished status."""
        if status == RequestStatus.FINISHED_STOPPED:
            return "stop"
        elif status == RequestStatus.FINISHED_LENGTH_CAPPED:
            return "length"
        elif status == RequestStatus.FINISHED_ABORTED:
            return "abort"
        return None


# ``RequestOutput.error_kind`` of a request whose inputs could not be preprocessed.
ERROR_KIND_PREPROCESSING = "preprocessing"


class UnsupportedRequestOption(ValueError):
    """Raised when the route chosen for a request cannot honour one of its options.

    The server maps it to HTTP 400. Silently dropping an option such as
    ``ignore_eos`` would make a client believe it got semantics it did not.
    """


class ChatTemplateRejected(UnsupportedRequestOption):
    """The chat template refused the request's messages (HTTP 400).

    Templates validate their input by raising from inside the template
    (``raise_exception("Conversation roles must alternate ...")``: Gemma, Mistral,
    Llama-2 and many others), which surfaces as ``jinja2.TemplateError``. That is the
    template saying the *request* is malformed for this model, the case vLLM answers
    with a 400 carrying the template's message. Raised by the engines through
    ``chat_template_rejections``; a template that cannot be compiled, or any other
    exception, is a server problem and keeps its own type.
    """


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""

    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0  # 0 means disabled
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    # vLLM ``ignore_eos``: the model's EOS tokens do not end generation. Only
    # ``max_tokens`` (or an explicit ``stop_token_ids`` entry) does. Post-EOS
    # tokens are generated and counted like any other, but EOS tokens are
    # never rendered into the output text.
    ignore_eos: bool = False
    # vLLM ``skip_special_tokens`` (default true): BOS/PAD/UNK/EOS tokens the
    # model generates are left out of the output text (still generated and
    # counted). False renders them, as vLLM does.
    skip_special_tokens: bool = True
    # Extra per-request logits processors (e.g. JSON schema constrained
    # decoding via ``lm-format-enforcer``).  These are merged with any
    # built-in processors (repetition/presence penalty) at batch time.
    logits_processors: Optional[List[Callable]] = None

    def __post_init__(self):
        if self.stop is None:
            self.stop = []
        if self.stop_token_ids is None:
            self.stop_token_ids = []


@dataclass
class Request:
    """
    Represents a single inference request in the scheduling system.

    Adapted from vLLM's Request class with simplifications for MLX backend.

    Attributes:
        request_id: Unique identifier for this request
        prompt: The input prompt (string or token ids)
        prompt_token_ids: Tokenized prompt
        sampling_params: Parameters for generation
        arrival_time: When the request was received
        status: Current status of the request
        num_prompt_tokens: Number of tokens in the prompt
        num_computed_tokens: Number of tokens processed so far
        output_token_ids: Generated token ids
        output_text: Generated text (decoded)
    """

    request_id: str
    prompt: Union[str, List[int]]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.time)
    priority: int = 0  # Lower is higher priority

    # Set after tokenization
    prompt_token_ids: Optional[List[int]] = None
    num_prompt_tokens: int = 0

    # Generation state
    status: RequestStatus = RequestStatus.WAITING
    num_computed_tokens: int = 0
    output_token_ids: List[int] = field(default_factory=list)
    output_text: str = ""

    # For BatchGenerator integration
    batch_uid: Optional[int] = None  # UID assigned by BatchGenerator

    # Prefix cache fields
    prompt_cache: Optional[List[Any]] = None  # Cached KV state from prefix cache
    prompt_cache_key: Optional[List[int]] = None  # Actual shared-cache entry key
    cached_tokens: int = 0  # Number of tokens retrieved from cache
    cache_credit_tokens: int = 0  # Tokens the cache credited as a hit at fetch time
    remaining_tokens: Optional[List[int]] = None  # Tokens still needing processing
    # Request kind for tokenization (vLLM ``add_special_tokens``): False for a
    # prompt the chat template already rendered, None for a raw completion
    # prompt (tokenizer default, minus a duplicated leading BOS).
    add_special_tokens: Optional[bool] = None

    # Paged cache fields (for BlockAwarePrefixCache)
    block_table: Optional["BlockTable"] = None  # Block table for paged cache
    shared_prefix_blocks: int = 0  # Number of shared prefix blocks

    # Multimodal content (images, video) - raw inputs
    images: Optional[List[Any]] = None
    videos: Optional[List[Any]] = None

    # Processed multimodal inputs for VLM batching
    pixel_values: Optional[Any] = None  # Processed image tensors (mx.array)
    image_grid_thw: Optional[Any] = None  # Grid info for Qwen-VL models
    attention_mask: Optional[Any] = None  # Attention mask for multimodal input
    multimodal_kwargs: Optional[Dict[str, Any]] = None  # Model-specific kwargs
    is_multimodal: bool = False  # Flag indicating this is a multimodal request

    # Metadata
    finish_reason: Optional[str] = None
    first_token_time: Optional[float] = (
        None  # Time when first output token was generated
    )
    cache_hit_type: Optional[str] = (
        None  # Type of cache hit: exact/prefix/supersequence/lcp/miss
    )

    # MTP (multi-token prediction) per-request counters, mirroring
    # MLLMRequest.mtp_drafts/mtp_accepted (mllm_scheduler.py). Attributed by
    # UID in Scheduler._process_batch_responses from the deltas
    # _install_mtp's closures accumulate per step (see scheduler.py).
    mtp_drafts: int = 0
    mtp_accepted: int = 0

    @property
    def num_output_tokens(self) -> int:
        """Number of output tokens generated so far."""
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """Total number of tokens (prompt + output)."""
        return self.num_prompt_tokens + self.num_output_tokens

    @property
    def max_tokens(self) -> int:
        """Maximum output tokens for this request."""
        return self.sampling_params.max_tokens

    def is_finished(self) -> bool:
        """Check if request has finished."""
        return RequestStatus.is_finished(self.status)

    def get_finish_reason(self) -> Optional[str]:
        """Get the finish reason if finished."""
        if self.finish_reason:
            return self.finish_reason
        return RequestStatus.get_finish_reason(self.status)

    def append_output_token(self, token_id: int) -> None:
        """Append a generated token to the output."""
        self.output_token_ids.append(token_id)
        self.num_computed_tokens += 1

    def set_finished(self, status: RequestStatus, reason: Optional[str] = None) -> None:
        """Mark the request as finished."""
        self.status = status
        self.finish_reason = reason or RequestStatus.get_finish_reason(status)

    def __lt__(self, other: "Request") -> bool:
        """Compare requests for priority queue ordering."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.arrival_time < other.arrival_time

    def __hash__(self) -> int:
        return hash(self.request_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Request):
            return False
        return self.request_id == other.request_id


@dataclass
class RequestOutput:
    """
    Output for a single request after a generation step.

    This is returned by the engine to communicate results back to the API layer.
    """

    request_id: str
    # New tokens generated in this step
    new_token_ids: List[int] = field(default_factory=list)
    new_text: str = ""
    # Cumulative output
    output_token_ids: List[int] = field(default_factory=list)
    output_text: str = ""
    # Status
    finished: bool = False
    finish_reason: Optional[str] = None
    # Why a request that finished with ``finish_reason="error"`` failed, when the
    # scheduler can tell: ERROR_KIND_PREPROCESSING for inputs it could not
    # preprocess (the client's fault), None for a failure while generating.
    error_kind: Optional[str] = None
    # Timing
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # MTP speculative decoding counters. Zero means no MTP attempt occurred.
    mtp_drafts: int = 0
    mtp_accepted: int = 0
    # Prompt tokens served from the prefix cache. None means the engine did not report it
    # (as opposed to 0, which means it was consulted and nothing was reused).
    cached_tokens: Optional[int] = None
    # Request-level sparse-prefill decision and diagnostics.
    specprefill_outcome: Optional["SpecPrefillOutcome"] = None

    @property
    def usage(self) -> Dict[str, int]:
        """Return usage statistics compatible with OpenAI API."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
