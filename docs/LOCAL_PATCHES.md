# Local patches carried on `main`

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
| 6 | **mlx-lm 0.32 support**: `vllm_mlx/mlx_cache_compat.py` dispatches per cache-layer class between the legacy contract (mlx-vlm caches, mlx-lm <= 0.31: `meta_state`, `from_state(state, meta_state)`, offset-sliced `state`) and mlx-lm 0.32 (no `meta_state`, padded `state`). Wired into the paged/prefix cache, MLLM prefix copies, ArraysCache SSD spill/promotion/pricing, MTP rollback, persistence format; detokenizer public API; mlx-lm pin `>=0.31.3,<0.33` | d8ffa42, 9d19dd5, f356a12, e8d91c8, c1c1b2f, 10c6837, 6353b27, 598ed4a | local; upstream pins `<0.32` ("breaks cache/batch APIs") | candidate, to be split per concern | Upstream supports 0.32 |
| 7 | `BatchGenerator` gets an explicit `stream=` (the process-global `mlx_lm.generate.generation_stream` could name another thread's stream); tests restore the global per test | 39b0f02 | local | candidate, standalone | Upstream fixes it |
| 8 | Remove the pre-0.31.2 BatchGenerator monkey-patch layer (~600 lines that could not run on any supported mlx-lm) and their tests | 044134d | local | candidate | n/a |
| 9 | Test isolation: restore `sys.modules`/package attributes after fixtures that re-import `engine_core`/`engine.batched` (4 order-dependent failures present on upstream main); wrappable mock tokenizers for 0.32; plugin tests skip without vllm/torch | 4e50df2 | local | candidate | n/a |
| 10 | CI: triggers on `eejd/**` and `release/**`; Apple Silicon job runs mlx-lm 0.31.3 and latest; plugin test step; black-clean | 598ed4a and later | local | fork-only | n/a |
| 11 | Tool parsers `lfm2` (`lfm2.5`) and `minicpm` (`minicpm5`): the LFM2 Python-call list between `<\|tool_call_start\|>`/`<\|tool_call_end\|>` and the MiniCPM5 `<function name><param name>` XML, each with absolute-index emit-once streaming, `finalize_streaming`, and a round-trip test against the models' own chat templates (skipped without the local model cache) | this change | local; eejd/py-vllm-mlx#19 | candidate (not opened); no lfm/minicpm parser exists upstream at 80e7fde | Upstream ships LFM2/MiniCPM parsers |

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
| MLLM tool metadata (36625e5), replayed tool-call arg normalization (77a7ad2) from old main | Not carried; verified covered by upstream #608/#611 code. The 14 regression tests from 77a7ad2 pass unchanged on this line, and the real Gemma 4 chat template renders identically with and without the `name` field 36625e5 also forwarded (the template never reads it). **Residual gap, not carried:** 36625e5 also forwarded `reasoning` and legacy `tool_responses`; the Gemma 4 template reads both (`reasoning` or `reasoning_content`; `tool_responses` is its legacy non-OpenAI assistant-embedded form), upstream's builder forwards only `reasoning_content`. Matters only for clients that send those field names (eejd/py-vllm-mlx#14) |

## Known gaps on this line (measured; causes marked where not established)

- **Hybrid-model prefix cache never hits.** On Nemotron-3-Nano-4B (ArraysCache) a repeated prompt gets
  0 cache hits with `--continuous-batching` (outputs are correct, just uncached). Measured identically
  on clean upstream main + mlx-lm 0.32, on this line + mlx-lm 0.32, and on this line + mlx-lm 0.31.3, so
  it predates the 0.32 port. **Cause not established by experiment.** Working hypothesis: mlx-lm >= 0.31.2
  removed `_process_prompts`/`active_batch`, which the deleted layer used to capture prompt-only state, so a
  recurrent model only ever stores prompt-plus-output entries, and a recurrent state cannot be trimmed back to
  a shorter prompt. The server log (INFO level) does not show the reason. A native-API prefill observer
  (`next()` responses, `extract_cache`, `insert_segments`) would be the fix; not implemented here.
- **Mid-prefill saves, `prefix_boundary` two-phase prefill and LLM-path MTP are inactive** on every
  supported mlx-lm for the same reason (they log or degrade silently).
- An exact repeat of a prompt on plain-KV models is counted as a hit but is not noticeably faster;
  partial-prefix hits are (about 20x on a 2.2k-token prompt). Same on all three configurations measured.
