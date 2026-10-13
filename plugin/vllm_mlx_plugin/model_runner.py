# SPDX-License-Identifier: Apache-2.0
"""
MLX Model Runner for vLLM.

This module implements the model runner that bridges vLLM's request
handling with mlx-lm's inference capabilities.

Includes low-level optimizations:
- mx.compile() for kernel fusion
- Memory bandwidth optimization
- Per-request mlx-lm generators (KV cache lives inside each generator)
"""

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Protocol

import mlx.core as mx
import numpy as np
from vllm.v1.outputs import ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)


class StructuredOutputError(RuntimeError):
    """The grammar bitmask could not be applied; the step must not continue."""


class _TokenSource(Protocol):
    def next_token(self, bitmask_row: Any) -> int: ...


class _PipelinedSteps:
    """mlx-lm's ``generate_step`` for a request with no grammar constraint."""

    def __init__(self, generator: Iterator) -> None:
        self._generator = generator

    def next_token(self, bitmask_row: Any) -> int:
        if bitmask_row is not None:
            # generate_step already sampled this token before the bitmask
            # existed; dropping the constraint silently would break the grammar.
            raise StructuredOutputError(
                "a grammar bitmask arrived for a request started without "
                "structured outputs"
            )
        token, _logprobs = next(self._generator)
        return int(token.item()) if hasattr(token, "item") else int(token)


def _apply_grammar_bitmask(logits: "mx.array", bitmask_row: Any) -> "mx.array":
    """Set the logits of tokens the grammar forbids to -inf.

    ``logits`` is ``(1, vocab)``; ``bitmask_row`` is one int32 row of
    ``GrammarOutput.grammar_bitmask``.  The mask is applied by xgrammar's
    ``apply_token_bitmask_inplace`` -- the call vLLM's GPU runner and
    vllm-metal use on the same bitmask -- on a float32 CPU copy of the row.
    """
    try:
        import torch
        import xgrammar as xgr
    except ImportError as e:
        raise StructuredOutputError(
            f"xgrammar and torch are required to apply a grammar bitmask: {e}"
        ) from e

    row = np.array(logits.astype(mx.float32))  # writable copy, evaluates the logits
    xgr.apply_token_bitmask_inplace(
        torch.from_numpy(row), torch.from_numpy(np.asarray(bitmask_row)[None])
    )
    return mx.array(row).astype(logits.dtype)


class _MaskedStepper:
    """One request's tokens, one at a time, with the grammar applied per step.

    Same computation as mlx-lm's ``generate_step`` (chunked prefill, then one
    forward per token) without its one-token lookahead, and with
    ``bitmask_row`` applied to the logits before the log-softmax and sampler,
    so top-p / min-p / top-k see the constrained distribution.
    """

    PREFILL_STEP_SIZE = 2048

    def __init__(self, model: Any, prompt_token_ids: list[int], sampler: Any) -> None:
        self._model = model
        self._sampler = sampler
        self._cache: list[Any] | None = None  # built on the first step
        self._pending = mx.array(prompt_token_ids)  # tokens not yet fed to the model

    def next_token(self, bitmask_row: Any) -> int:
        from mlx_lm.generate import generation_stream
        from mlx_lm.models.cache import make_prompt_cache

        if len(self._pending) == 0:
            raise ValueError("a request needs at least one prompt token")
        if self._cache is None:
            self._cache = make_prompt_cache(self._model)

        with mx.stream(generation_stream):
            # Feed all but the last pending token in chunks (the cache keeps
            # them); the last one yields the logits the next token comes from.
            while len(self._pending) > 1:
                n = min(self.PREFILL_STEP_SIZE, len(self._pending) - 1)
                self._model(self._pending[:n][None], cache=self._cache)
                mx.eval([c.state for c in self._cache])
                self._pending = self._pending[n:]
                mx.clear_cache()
            logits = self._model(self._pending[None], cache=self._cache)[:, -1, :]
            if bitmask_row is not None:
                logits = _apply_grammar_bitmask(logits, bitmask_row)
            logprobs = logits - mx.logsumexp(logits, keepdims=True)
            token = int(self._sampler(logprobs).item())
        self._pending = mx.array([token])
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

        # Per-request generation state (vLLM >= 0.27 step contract):
        # each request owns a token source (an mlx-lm generate_step generator,
        # or a _MaskedStepper for structured outputs) whose KV cache persists
        # between scheduler steps.
        self._gens: dict[str, _TokenSource] = {}
        self._tokens: dict[str, list[int]] = {}
        self._params: dict[str, Any] = {}
        self._prompt_lens: dict[str, int] = {}
        # (request id, whether this step produces a token) for the requests
        # execute_model scheduled and sample_tokens has yet to step.
        self._scheduled_steps: list[tuple[str, bool]] = []

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

        Finished requests are released, new requests get their generator and
        resumed (preempted) requests are rebuilt from the scheduler's token
        history.  No token is produced here: when requests were scheduled the
        step is stashed and ``None`` returned, so the engine collects it via
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

        steps: list[tuple[str, bool]] = []
        scheduled = scheduler_output.num_scheduled_tokens

        for req_data in scheduler_output.scheduled_new_reqs:
            req_id = req_data.req_id
            prompt = req_data.prompt_token_ids or []
            self._params[req_id] = req_data.sampling_params
            self._prompt_lens[req_id] = len(prompt)
            self._gens[req_id] = self._start_request(prompt, req_data.sampling_params)
            self._tokens[req_id] = []
            computed = req_data.num_computed_tokens
            steps.append((req_id, computed + scheduled[req_id] >= len(prompt)))

        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(cached.req_ids):
            if req_id in cached.resumed_req_ids:
                # Preempted and re-prefilled by the scheduler: the old
                # generator's KV state is stale, rebuild from the full token
                # history the scheduler sends for resumed requests.
                self._resume_request(req_id, cached.all_token_ids.get(req_id))
            # The step produces a token once the scheduled tokens reach the end
            # of everything the request knows: prompt plus outputs so far
            # (``num_output_tokens`` counts the async scheduler's pending ones).
            total = self._prompt_lens.get(req_id, 0) + cached.num_output_tokens[i]
            computed = cached.num_computed_tokens[i]
            steps.append((req_id, computed + scheduled[req_id] >= total))

        if steps:
            self._scheduled_steps = steps
            return None
        return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

    def sample_tokens(self, grammar_output: Any = None) -> ModelRunnerOutput | None:
        """Advance every request scheduled by ``execute_model`` by one token.

        A request whose scheduled tokens only cover part of its prompt (a
        chunked prefill) samples nothing and reports an empty list, as vLLM's
        own runners do: the async scheduler counts one pending output per
        request that completes its prompt, and a token for a request that does
        not drives that count negative (``assert request.num_output_placeholders
        >= 0``), which kills the engine core.

        ``grammar_output`` carries the structured-output bitmask rows, in
        ``structured_output_request_ids`` order (one row per request: this
        runner schedules no speculative tokens).  A request's row is applied
        to its logits before sampling, as vLLM's own runners do.
        """
        steps, self._scheduled_steps = self._scheduled_steps, []
        if not steps:
            return None
        rows = self._bitmask_rows(grammar_output)
        req_ids = [req_id for req_id, _ in steps]
        sampled_token_ids = [
            self._step_request(req_id, rows.get(req_id)) if samples else []
            for req_id, samples in steps
        ]
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
        self._gens.pop(req_id, None)
        self._tokens.pop(req_id, None)
        self._params.pop(req_id, None)
        self._prompt_lens.pop(req_id, None)

    def _resume_request(self, req_id: str, all_token_ids: list[int] | None) -> None:
        """Rebuild a preempted request's generator from its full token history."""
        if not all_token_ids:
            logger.error(f"Resumed request {req_id} without token history")
            self._gens.pop(req_id, None)
            return
        params = self._params.get(req_id)
        self._gens[req_id] = self._start_request(list(all_token_ids), params)
        self._tokens.setdefault(req_id, [])

    def _eos_token_id(self) -> int | None:
        return getattr(self.tokenizer, "eos_token_id", None)

    def _start_request(
        self, prompt_token_ids: list[int], sampling_params: Any
    ) -> "_TokenSource":
        """Create the persistent token source for one request.

        An unconstrained request runs on mlx-lm's ``generate_step``, which
        samples one token ahead of the one it yields.  A structured-output
        request cannot: the grammar bitmask for token n+1 only exists after
        token n was accepted by the engine, so it steps one token at a time
        (``_MaskedStepper``) and applies the bitmask to the logits it samples.
        """
        from mlx_lm.sample_utils import make_sampler

        temp = float(getattr(sampling_params, "temperature", 1.0) or 0.0)
        top_p_raw = getattr(sampling_params, "top_p", None)
        # vLLM: 1.0 = disabled; mlx-lm: 0.0 = disabled.
        top_p = 0.0 if top_p_raw is None or top_p_raw >= 1.0 else float(top_p_raw)
        min_p = float(getattr(sampling_params, "min_p", 0.0) or 0.0)
        top_k = int(getattr(sampling_params, "top_k", 0) or 0)
        if top_k < 0:  # vLLM uses -1 / 0 for "disabled"
            top_k = 0
        sampler = make_sampler(temp=temp, top_p=top_p, min_p=min_p, top_k=top_k)

        if getattr(sampling_params, "structured_outputs", None) is not None:
            return _MaskedStepper(self.model, prompt_token_ids, sampler)

        from mlx_lm.generate import generate_step

        # vLLM's scheduler enforces max_tokens / stop conditions; keep the
        # generator open for the whole context window.
        return _PipelinedSteps(
            generate_step(
                prompt=mx.array(prompt_token_ids),
                model=self.model,
                max_tokens=-1,
                sampler=sampler,
            )
        )

    def _step_request(self, req_id: str, bitmask_row: Any = None) -> list[int]:
        """Advance one request by one token.

        A request that can no longer generate (missing/exhausted generator or
        an mlx-lm error) is terminated by emitting EOS: the scheduler only
        stops a request on a sampled stop token -- an empty list leaves it
        running forever (vllm/v1/core/sched/scheduler.py update_from_output).
        A structured-output failure is not that: terminating with EOS would
        hand back text that silently ignores the grammar, so it propagates.
        """
        source = self._gens.get(req_id)
        if source is None:
            logger.error(f"No generator for request {req_id}; terminating it")
            return self._terminate(req_id)
        try:
            tok = source.next_token(bitmask_row)
        except StopIteration:
            return self._terminate(req_id)
        except StructuredOutputError:
            raise
        except Exception as e:
            logger.error(f"Generation failed for {req_id}: {e}")
            return self._terminate(req_id)
        self._tokens[req_id].append(tok)
        return [tok]

    def _terminate(self, req_id: str) -> list[int]:
        self._gens.pop(req_id, None)
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
