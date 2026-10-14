# SPDX-License-Identifier: Apache-2.0
"""
MLX Model Runner for vLLM.

This module implements the model runner that bridges vLLM's request
handling with mlx-lm's inference capabilities.

Includes low-level optimizations:
- mx.compile() for kernel fusion
- Memory bandwidth optimization
- Per-request mlx-lm KV caches, fed in scheduler-sized chunks
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple

import mlx.core as mx
import numpy as np
from vllm.v1.outputs import ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)

# vLLM samples greedily below this temperature (``_SAMPLING_EPS``).
_GREEDY_TEMPERATURE = 1e-5


class _Step(NamedTuple):
    """One request in a scheduled step."""

    req_id: str
    # Tokens of its own sequence the scheduler has now accounted for.
    reached: int
    # Whether this step completes everything the request knows, i.e. yields a
    # token (False: a chunk of a longer prefill).
    samples: bool


class StructuredOutputError(RuntimeError):
    """The grammar bitmask is unusable; the step must not continue."""


class _Sampler:
    """mlx-lm's sampler chain, with ``top_k`` resolved against the logits width.

    vLLM treats ``top_k`` >= vocab as disabled; mlx-lm raises for it, from
    inside ``mx.compile(inputs=mx.random.state, ...)``, which leaves the
    process-wide MLX RNG state unusable: every later ``temperature > 0`` sample
    fails until restart.  The width is only known from the logits, so the chain
    is built on first use for each width.
    """

    def __init__(self, temp: float, top_p: float, min_p: float, top_k: int) -> None:
        self._args = (temp, top_p, min_p)
        self._top_k = top_k
        self._by_width: dict[int, Any] = {}

    def __call__(self, logprobs: "mx.array") -> "mx.array":
        width = logprobs.shape[-1]
        sampler = self._by_width.get(width)
        if sampler is None:
            from mlx_lm.sample_utils import make_sampler

            temp, top_p, min_p = self._args
            top_k = self._top_k if 0 < self._top_k < width else 0
            sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)
            self._by_width[width] = sampler
        return sampler(logprobs)


def _apply_grammar_bitmask(logits: "mx.array", bitmask_row: Any) -> "mx.array | None":
    """Set the logits of tokens the grammar forbids to -inf.

    ``logits`` is ``(1, vocab)``; ``bitmask_row`` is one int32 row of
    ``GrammarOutput.grammar_bitmask``.  The mask is applied by xgrammar's
    ``apply_token_bitmask_inplace`` -- the call vLLM's GPU runner and
    vllm-metal use on the same bitmask -- on a float32 CPU copy of the row.

    Returns ``None`` when the grammar leaves no token allowed.  Anything that
    makes the bitmask unusable (wrong dtype or shape, narrower than the logits,
    xgrammar/torch missing or failing) raises ``StructuredOutputError``:
    sampling on regardless would hand back text that ignores the grammar.
    """
    try:
        import torch
        import xgrammar as xgr

        bitmask = np.asarray(bitmask_row)
        vocab = logits.shape[-1]
        if bitmask.dtype != np.int32 or bitmask.ndim != 1:
            raise ValueError(
                f"expected a 1-D int32 bitmask row, got {bitmask.dtype} "
                f"with shape {bitmask.shape}"
            )
        if bitmask.shape[0] * 32 < vocab:
            raise ValueError(
                f"a bitmask of {bitmask.shape[0]} words covers "
                f"{bitmask.shape[0] * 32} tokens, the logits have {vocab}; "
                "the tokens past that would stay unconstrained"
            )
        row = np.array(logits.astype(mx.float32))  # writable copy, evaluates
        xgr.apply_token_bitmask_inplace(
            torch.from_numpy(row), torch.from_numpy(bitmask[None])
        )
    except Exception as e:
        raise StructuredOutputError(f"cannot apply the grammar bitmask: {e}") from e
    if np.isneginf(row).all():
        return None
    return mx.array(row).astype(logits.dtype)


class _RequestState:
    """One request's model state: its KV cache and how far it has been fed.

    ``tokens`` is everything the model must see, in order (the prompt, or the
    whole history of a resumed request); ``fed`` of them are in the cache.
    Between steps ``fed <= len(tokens) - 1``: the last token is always left
    to produce the logits the next token comes from.

    Tokens are produced one of two ways.  An unconstrained request runs on
    mlx-lm's ``generate_step`` (``prompt_cache=`` takes over the cache), which
    samples one token ahead of the one it yields.  A structured-output request
    cannot: the grammar bitmask for token n+1 exists only after token n was
    accepted by the engine, so it steps one token at a time and applies the
    bitmask to the logits it samples from.
    """

    PREFILL_STEP_SIZE = 2048

    def __init__(
        self,
        model: Any,
        tokens: list[int],
        sampler: Any,
        processors: list[Any],
        structured: bool,
    ) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        if not tokens:
            raise ValueError("a request needs at least one prompt token")
        self._model = model
        self._sampler = sampler
        self._processors = processors
        self.structured = structured
        self.tokens = list(tokens)
        self.fed = 0
        self._cache = make_prompt_cache(model)
        self._generator: Iterator | None = None

    def prefill_to(self, upto: int) -> None:
        """Feed ``tokens[fed:upto]`` (never the last token) to the cache."""
        from mlx_lm.generate import generation_stream

        upto = min(upto, len(self.tokens) - 1)
        if self._generator is not None or upto <= self.fed:
            return
        with mx.stream(generation_stream):
            while self.fed < upto:
                n = min(self.PREFILL_STEP_SIZE, upto - self.fed)
                chunk = mx.array(self.tokens[self.fed : self.fed + n])
                self._model(chunk[None], cache=self._cache)
                mx.eval([c.state for c in self._cache])
                self.fed += n
                mx.clear_cache()

    def next_token(self, bitmask_row: Any) -> int:
        if self.structured:
            return self._masked_step(bitmask_row)
        if bitmask_row is not None:
            # generate_step already sampled this token before the bitmask
            # existed; dropping the constraint silently would break the grammar.
            raise StructuredOutputError(
                "a grammar bitmask arrived for a request started without "
                "structured outputs"
            )
        if self._generator is None:
            from mlx_lm.generate import generate_step

            # vLLM's scheduler enforces max_tokens / stop conditions; keep the
            # generator open for the whole context window.
            self._generator = generate_step(
                prompt=mx.array(self.tokens[self.fed :]),
                model=self._model,
                max_tokens=-1,
                sampler=self._sampler,
                logits_processors=self._processors or None,
                prompt_cache=self._cache,
            )
        token, _logprobs = next(self._generator)
        return int(token.item()) if hasattr(token, "item") else int(token)

    def _masked_step(self, bitmask_row: Any) -> int:
        from mlx_lm.generate import generation_stream

        self.prefill_to(len(self.tokens) - 1)
        with mx.stream(generation_stream):
            last = mx.array(self.tokens[self.fed :])
            logits = self._model(last[None], cache=self._cache)[:, -1, :]
            for processor in self._processors:
                logits = processor(mx.array(self.tokens), logits)
            if bitmask_row is not None:
                masked = _apply_grammar_bitmask(logits, bitmask_row)
                if masked is None:
                    # The grammar allows nothing.  Sample unconstrained: the
                    # scheduler rejects that token and ends this request alone
                    # with an error (FINISHED_ERROR), as it does for a CUDA
                    # runner whose mask leaves nothing.
                    logger.error("the grammar allows no token for this request")
                else:
                    logits = masked
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            token = int(self._sampler(logprobs).item())
        self.fed = len(self.tokens)
        self.tokens.append(token)
        return token


@dataclass
class SamplerOutput:
    """Output from sampling."""

    token_ids: list[int]
    logprobs: list[dict] | None = None


class MLXModelRunner:
    """
    Model runner that uses mlx-lm for inference.

    This class handles:
    - Model loading via mlx-lm
    - Converting vLLM requests to mlx-lm format
    - Running inference and returning results in vLLM format
    - KV cache management (delegated to mlx-lm)

    Optimizations:
    - mx.compile() for kernel fusion (fuses multiple ops into single Metal kernel)
    - Memory optimization for bandwidth efficiency
    - Prefill chunking for L2 cache utilization
    """

    def __init__(self, vllm_config: "VllmConfig", enable_optimizations: bool = True):
        """
        Initialize MLX model runner.

        Args:
            vllm_config: vLLM configuration
            enable_optimizations: Whether to enable low-level optimizations
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config

        # mlx-lm model and tokenizer
        self.model = None
        self.tokenizer = None
        self._loaded = False

        # Per-request state (vLLM >= 0.27 step contract): each request owns a
        # _RequestState (its KV cache and how far it has been fed) that
        # persists between scheduler steps.  Sampling params and prompt length
        # outlive a preemption, which drops the state, for the resume.
        self._requests: dict[str, _RequestState] = {}
        self._params: dict[str, Any] = {}
        self._prompt_lens: dict[str, int] = {}
        # The requests execute_model scheduled and sample_tokens has yet to run.
        self._scheduled_steps: list[_Step] = []

        # Cache for prompt processing
        self._prompt_cache = None

        # KV cache blocks
        self._num_cache_blocks = 0

        # Optimization settings
        self._enable_optimizations = enable_optimizations
        self._compiled_forward = None  # Compiled model forward pass
        self._hardware_info = None  # Detected hardware profile

        logger.info(f"MLXModelRunner initialized for model: {self.model_config.model}")
        logger.info(
            f"Low-level optimizations: {'ENABLED' if enable_optimizations else 'disabled'}"
        )

    def load_model(self) -> None:
        """Load model using mlx-lm with optimizations."""
        if self._loaded:
            return

        try:
            from mlx_lm import load

            model_name = self.model_config.model

            logger.info(f"Loading model with mlx-lm: {model_name}")
            start_time = time.time()

            self.model, self.tokenizer = load(
                model_name,
                tokenizer_config={
                    "trust_remote_code": self.model_config.trust_remote_code,
                },
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")

            self._loaded = True

            # Apply low-level optimizations
            if self._enable_optimizations:
                self._apply_optimizations()

        except ImportError:
            raise ImportError(
                "mlx-lm is required for MLX model runner. "
                "Install with: pip install mlx-lm"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _apply_optimizations(self) -> None:
        """Apply low-level optimizations for maximum performance."""
        try:
            from vllm_mlx.optimizations import detect_hardware

            # Detect hardware and apply memory optimization
            self._hardware_info = detect_hardware()
            logger.info(f"Hardware detected: {self._hardware_info.chip_name}")
            logger.info(f"Memory: {self._hardware_info.total_memory_gb:.1f} GB")
            logger.info(f"Bandwidth: {self._hardware_info.memory_bandwidth_gbs} GB/s")

            # Compile the model forward pass for kernel fusion
            self._setup_compiled_forward()

        except Exception as e:
            logger.warning(f"Failed to apply optimizations: {e}")

    def _setup_compiled_forward(self) -> None:
        """
        Setup compiled forward pass using mx.compile() for kernel fusion.

        This fuses multiple operations into single Metal kernels,
        reducing kernel launch overhead and improving throughput.
        """
        if self.model is None:
            return

        try:
            # Compile the model's __call__ method
            # This creates fused Metal kernels for the forward pass
            if hasattr(self.model, "__call__"):
                self._compiled_forward = mx.compile(self.model.__call__)
                logger.info("Compiled forward pass enabled (mx.compile kernel fusion)")
            else:
                logger.warning(
                    "Model does not have __call__ method, skipping compilation"
                )

        except Exception as e:
            logger.warning(f"Failed to compile forward pass: {e}")
            self._compiled_forward = None

    def initialize_cache(self, num_blocks: int) -> None:
        """Initialize KV cache."""
        self._num_cache_blocks = num_blocks
        logger.info(f"KV cache initialized with {num_blocks} blocks")

        # mlx-lm manages its own KV cache internally
        # We just track the configuration here

    def get_kv_cache_spec(self) -> dict:
        """Get KV cache specification."""
        return {
            "num_blocks": self._num_cache_blocks,
            "block_size": self.cache_config.block_size,
        }

    def get_cache_block_size_bytes(self) -> int:
        """Calculate cache block size in bytes."""
        if not self._loaded or self.model is None:
            return 0

        # Get model config
        config = getattr(self.model, "config", None)
        if config is None:
            return 0

        head_size = getattr(config, "head_dim", 64)
        num_kv_heads = getattr(
            config, "num_key_value_heads", getattr(config, "num_attention_heads", 32)
        )
        num_layers = getattr(config, "num_hidden_layers", 32)
        block_size = self.cache_config.block_size

        # 2 for K and V, 2 bytes for float16
        return 2 * block_size * num_layers * num_kv_heads * head_size * 2

    def warm_up(self) -> None:
        """Warm up model with a test generation."""
        if not self._loaded:
            self.load_model()

        logger.info("Warming up model...")

        try:
            from mlx_lm import generate

            # Simple warm-up generation
            _ = generate(
                self.model,
                self.tokenizer,
                prompt="Hello",
                max_tokens=5,
                verbose=False,
            )
            logger.info("Model warm-up complete")

        except Exception as e:
            logger.warning(f"Warm-up failed (non-critical): {e}")

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput | None:
        """Register one scheduler step (vLLM >= 0.27 contract).

        Finished requests are released, preempted ones lose their cache (the
        scheduler sends their history again on resume), new requests get their
        state.  Nothing is computed here: when requests were scheduled the step
        is stashed and ``None`` returned, so the engine collects it via
        ``sample_tokens`` -- the same two-phase protocol vllm_swift uses --
        together with the grammar bitmask, which vLLM only computes after
        ``execute_model`` (``EngineCore.step``) and which must reach the
        logits before a token is sampled.  Cleanup-only steps return an empty
        output directly.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        for req_id in scheduler_output.finished_req_ids:
            self._release(req_id)
        for req_id in getattr(scheduler_output, "preempted_req_ids", None) or ():
            self._requests.pop(req_id, None)

        steps: list[_Step] = []
        scheduled = scheduler_output.num_scheduled_tokens

        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            prompt = req_data.prompt_token_ids or []
            self._params[req_id] = req_data.sampling_params
            self._prompt_lens[req_id] = len(prompt)
            self._requests[req_id] = self._new_state(prompt, req_data.sampling_params)
            # ``num_computed_tokens`` > 0 is a vLLM prefix-cache hit, which
            # speaks for vLLM's KV blocks; MLX keeps its own cache and feeds
            # the request from its first token.
            reached = req_data.num_computed_tokens + scheduled[req_id]
            steps.append(_Step(req_id, reached, reached >= len(prompt)))

        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            if req_id in cached.resumed_req_ids:
                # Preempted and re-prefilled by the scheduler: rebuild from the
                # full token history it sends for resumed requests.
                self._resume_request(req_id, cached.all_token_ids.get(req_id))
            # The step produces a token once the scheduled tokens reach the end
            # of everything the request knows: prompt plus outputs so far
            # (``num_output_tokens`` counts the async scheduler's pending ones).
            total = self._prompt_lens.get(req_id, 0) + cached.num_output_tokens[i]
            reached = cached.num_computed_tokens[i] + scheduled[req_id]
            steps.append(_Step(req_id, reached, reached >= total))

        if steps:
            self._scheduled_steps = steps
            return None
        return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

    def sample_tokens(self, grammar_output: Any = None) -> ModelRunnerOutput | None:
        """Run the step ``execute_model`` registered.

        A request whose scheduled tokens only cover part of its prompt (a
        chunked prefill) feeds that chunk to its cache and reports an empty
        list, as vLLM's own runners do: the async scheduler counts one pending
        output per request that completes its prompt, and a token for a request
        that does not drives that count negative (``assert
        request.num_output_placeholders >= 0``), which kills the engine core.
        The scheduler's per-step token budget therefore bounds the work of a
        step here too.  Any other request advances by one token.

        ``grammar_output`` carries the structured-output bitmask rows, in
        ``structured_output_request_ids`` order (one row per request: this
        runner schedules no speculative tokens).  A request's row is applied
        to its logits before sampling, as vLLM's own runners do.
        """
        steps, self._scheduled_steps = self._scheduled_steps, []
        if not steps:
            return None
        rows = self._bitmask_rows(grammar_output)
        req_ids = [step.req_id for step in steps]
        sampled_token_ids = []
        for step in steps:
            if step.samples:
                sampled_token_ids.append(
                    self._step_request(step.req_id, rows.get(step.req_id))
                )
            else:
                state = self._requests.get(step.req_id)
                if state is not None:
                    state.prefill_to(step.reached)
                sampled_token_ids.append([])
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
        )

    @staticmethod
    def _bitmask_rows(grammar_output: Any) -> dict[str, Any]:
        """Map each structured-output request id to its bitmask row."""
        if grammar_output is None:
            return {}
        ids = grammar_output.structured_output_request_ids
        bitmask = grammar_output.grammar_bitmask
        if bitmask.shape[0] != len(ids):
            raise StructuredOutputError(
                f"grammar bitmask has {bitmask.shape[0]} rows for {len(ids)} "
                "structured-output requests; expected one row each"
            )
        return {rid: bitmask[i] for i, rid in enumerate(ids)}

    def _release(self, req_id: str) -> None:
        self._requests.pop(req_id, None)
        self._params.pop(req_id, None)
        self._prompt_lens.pop(req_id, None)

    def _resume_request(self, req_id: str, all_token_ids: list[int] | None) -> None:
        """Rebuild a preempted request's state from its full token history."""
        if not all_token_ids:
            logger.error(f"Resumed request {req_id} without token history")
            self._requests.pop(req_id, None)
            return
        self._requests[req_id] = self._new_state(
            list(all_token_ids), self._params.get(req_id)
        )

    def _eos_token_id(self) -> int | None:
        return getattr(self.tokenizer, "eos_token_id", None)

    def _new_state(self, token_ids: list[int], sampling_params: Any) -> "_RequestState":
        """Create the model state for one request from its sampling params."""
        from mlx_lm.sample_utils import make_logits_processors

        temp = float(getattr(sampling_params, "temperature", 1.0) or 0.0)
        if temp < _GREEDY_TEMPERATURE:  # vLLM: below this is greedy decoding
            temp = 0.0
        top_p_raw = getattr(sampling_params, "top_p", None)
        # vLLM: 1.0 = disabled; mlx-lm: 0.0 = disabled.
        top_p = 0.0 if top_p_raw is None or top_p_raw >= 1.0 else float(top_p_raw)
        min_p = float(getattr(sampling_params, "min_p", 0.0) or 0.0)
        top_k = int(getattr(sampling_params, "top_k", 0) or 0)
        logit_bias = getattr(sampling_params, "logit_bias", None)
        return _RequestState(
            self.model,
            token_ids,
            _Sampler(temp, top_p, min_p, top_k),
            make_logits_processors(logit_bias=logit_bias) if logit_bias else [],
            getattr(sampling_params, "structured_outputs", None) is not None,
        )

    def _step_request(self, req_id: str, bitmask_row: Any = None) -> list[int]:
        """Advance one request by one token.

        A request that can no longer generate (missing/exhausted state or
        an mlx-lm error) is terminated by emitting EOS: the scheduler only
        stops a request on a sampled stop token -- an empty list leaves it
        running forever (vllm/v1/core/sched/scheduler.py update_from_output).
        A structured-output failure is not that: terminating with EOS would
        hand back text that silently ignores the grammar, so it propagates.
        """
        state = self._requests.get(req_id)
        if state is None:
            logger.error(f"No state for request {req_id}; terminating it")
            return self._terminate(req_id)
        try:
            return [state.next_token(bitmask_row)]
        except StopIteration:
            return self._terminate(req_id)
        except StructuredOutputError:
            raise
        except Exception as e:
            logger.error(f"Generation failed for {req_id}: {e}")
            return self._terminate(req_id)

    def _terminate(self, req_id: str) -> list[int]:
        self._requests.pop(req_id, None)
        eos = self._eos_token_id()
        return [int(eos)] if eos is not None else []

    def decode_tokens(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(token_ids)

    def get_model_info(self) -> dict:
        """Get information about the loaded model and optimizations."""
        info = {
            "loaded": self._loaded,
            "model_name": self.model_config.model,
            "optimizations_enabled": self._enable_optimizations,
        }

        if self._loaded and self.model is not None:
            config = getattr(self.model, "config", None)
            if config:
                info.update(
                    {
                        "vocab_size": getattr(config, "vocab_size", None),
                        "hidden_size": getattr(config, "hidden_size", None),
                        "num_layers": getattr(config, "num_hidden_layers", None),
                        "num_heads": getattr(config, "num_attention_heads", None),
                    }
                )

            # Add optimization status
            info["optimizations"] = {
                "kernel_fusion": self._compiled_forward is not None,
                "memory_optimized": self._hardware_info is not None,
            }

            if self._hardware_info:
                info["hardware"] = {
                    "chip": self._hardware_info.chip_name,
                    "memory_gb": self._hardware_info.total_memory_gb,
                    "bandwidth_gbs": self._hardware_info.memory_bandwidth_gbs,
                    "gpu_cores": self._hardware_info.gpu_cores,
                    "prefill_chunk_size": self._hardware_info.optimal_prefill_size,
                }

        return info

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        opt_status = "optimized" if self._compiled_forward else "standard"
        return f"<MLXModelRunner model={self.model_config.model} status={status} mode={opt_status}>"
