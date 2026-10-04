# Local patches carried on `eejd/integration`

This line is upstream `waybarrios/vllm-mlx` main plus the changes below. One row per local
change: what it is, where it came from, its upstream status, and when we can drop it.
Keep this file current when a patch lands upstream or is retired.

Base: upstream `80e7fde` (2026-10-03, 57 commits past v0.5.0).

| # | Change | Commits | Origin | Upstream status | Retire when |
|---|--------|---------|--------|-----------------|-------------|
| 1 | Split the vLLM platform plugin into its own distribution `vllm-mlx-plugin` (`plugin/`) | 6025934 | local (packaging) | not proposed; structural | Upstream adopts the split, or we stop shipping the plugin |
| 2 | Plugin: vLLM >= 0.27 contract (`manual_seed_all`, `CompilationTimes`, torch `device_type`, `KVCacheSpec` + `initialize_from_config`, `reset_mm_cache`/`sample_tokens`, full worker RPC surface) | cd3c626 .. 57af44a | local; fixes eejd/macports-ports-local#561 | upstream candidate once ported back to `vllm_mlx/` layout | Upstream carries the plugin |
| 3 | Plugin: port `execute_model` to the step contract; EOS-terminate failed requests, re-prefill resumed requests, drop dead sampler | f76e008, 37e65c2, c84caf2 | local (#561) | as above | as above |
| 4 | MLLM: preserve tool-call/tool-result round trip on `--mllm`, Gemma-native `tool_code`/`tool_output` serialization (#82) | 0816c93 | local | candidate. Overlaps upstream #611 (argument normalization), a different concern: both coexist | Upstream gains template-aware Gemma tool serialization |
| 5 | Gemma 4 parser: emit each streamed tool call exactly once (parallel calls render as separate `<\|tool_call>` blocks; resent index 0, never emitted call 1; split END marker) + emit-once regression tests for qwen/hermes/gemma4 | cb288d4 | local; eejd/py-vllm-mlx#10 | candidate (not opened) | Upstream fixes gemma4 |

## Landed upstream (no longer carried)

| Change | Upstream |
|--------|----------|
| qwen `</tool_call>` branch re-emitted every call on each later delta | #774 (our qwen fix was dropped in favour of it; our tests pass against it) |
| Mistral/Ministral `[THINK]` reasoning parser | #599 |
| `chat_template_kwargs` honoured on the MLLM path | #600 |
| Gemma 4 paren / unfenced `tool_code` tool-call forms | #623 |

## Rejected or dropped

| Change | Disposition |
|--------|-------------|
| `--mllm` + continuous-batching guard (old main 547a4fa) | Upstream #601 closed unmerged: guard is incomplete because BatchedEngine auto-detects MLLM. Not carried |
| MLLM tool metadata (36625e5), replayed tool-call arg normalization (77a7ad2) from old main | Not carried. Believed superseded by upstream #611 / `_normalize_mllm_tool_calls` but **not yet verified**: needs a reproducing test on this line (open item) |

## Known issues on this base (tracked, not patches)

- mlx-lm 0.32 API drift breaks cache/prefill paths and 21 tests; upstream pins `<0.32`. See the
  mlx-lm 0.32 port series.
