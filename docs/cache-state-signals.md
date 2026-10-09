# Cache state: which signals each engine mode really exposes

Written for py-vllm-mlx#37. The table below comes from `scripts/probe_cache_state.py`, which starts
a throwaway server per mode on Llama-3.2-1B-Instruct-4bit (persisted cache `none` + reset `both`, so
nothing earlier influences it) and sends four requests: **cold**, **exact repeat**, **shared prefix**
(same ~1530-token prefix, different last word) and a repeat after `DELETE /v1/cache`. After each one it
records the response `usage`, `GET /v1/cache/stats`, the cache part of `GET /v1/status` and the cache
lines of `GET /metrics`. Re-run it with

```
python scripts/probe_cache_state.py --model <small model dir> --out probe.json
```

A signal is **real** when it changed as the requests demand, **stub** when it is present but a constant
that does not follow behavior, **missing** when absent. "Unreported" below is the literal value the new
`cache_state` block uses when an engine cannot report something (see the schema).

## Signals by engine mode

| Signal | Simple | Batched (`--continuous-batching`) | Registry `--models-config` (simple / batched models) |
|---|---|---|---|
| `usage.prompt_tokens_details.cached_tokens` per request | **missing** (omitted, never a fake 0) | **real** (this PR): cold 0, exact repeat 0, shared prefix 1532 of 1539 | same as the model's engine mode |
| `engine_cache` counters (`hits`, `misses`, `evictions`, `tokens_saved`, `entry_count`, `current_memory_mb`) | **missing**: only `system_kv_cache` (capacity 4, all counters 0 in the probe) | **real**, but see the exact-repeat caveat below | `engine_cache` is null at top level (no default engine); per-model under `models` (this PR) |
| Effective memory limit and how it was derived | n/a (no memory-aware cache) | **real** (this PR): `memory_limit` = `{source: percent_of_available, percent, available_bytes, bytes}`; was invisible before | per model |
| `cache_state` block | **real** (this PR): engine class, launch options, inert options, versions; counters `unreported` with a reason | **real**; the batched MLLM engine nests its stats under `prefix_cache`, which `cache_state` unwraps | top level says "registry mode keeps no single default engine"; per model under `models`, where `engine.continuous_batching` comes from the model's engine class and `launch_options`/`inert_options` are `unreported` (registry entries choose their own engine, so the CLI flags do not describe them) |
| Persisted prefix cache (`persistence`) | not loaded or saved (no hooks): `persistence.applies: false` with the engine class named. In the single-model lazy-load mode `applies` is true until the engine is loaded and may turn false once a Simple engine is in place | loaded and saved per `--prefix-cache-*` | **not applicable**: `persistence.applies: false` (eejd/py-vllm-mlx#41) |
| `GET /v1/status` cache section | **missing** (`cache` key absent in the probe; per the code, `--prefix-trie-cache` stats come from `SimpleEngine.get_stats` and are not in `/v1/cache/stats`; the trie was off in the probe, so this is not verified live) | **real** (same numbers as `engine_cache`) | only `model_manager` |
| `GET /metrics` prefix-cache series | **missing**; only `vllm_mlx_metal_memory_bytes{kind="cache"}` | **missing**; same | **missing** (eejd/py-vllm-mlx#42) |
| `DELETE /v1/cache` | clears (nothing material to clear) | clears: entries and counters back to 0, next request cold again (probe: entry_count 2 -> 0, hits 2 -> 0) | per the engine |
| MLLM engines | text-only requests that consult the prefix cache report the reused prompt tokens (prefix hit: the matched prefix; exact hit: all but the last token; miss or fall-through: 0) and settle the cache counters like the text scheduler; media requests and cache-off engines report nothing (`None`, omitted from `usage`). The Responses API carries the same value as `usage.input_tokens_details.cached_tokens` and omits the object when the engine does not report it (eejd/py-vllm-mlx#43) | same | same |

Probe numbers (Batched, 1538-token prompt, one request at a time):

| Step | `cached_tokens` | cache `hits` | cache `tokens_saved` | wall time |
|---|---|---|---|---|
| cold | 0 | 0 | 0 | 0.193 s |
| exact repeat | **0** | **1** | **1538** | 0.163 s |
| shared prefix | 1532 | 2 | 3070 | 0.052 s |
| `DELETE /v1/cache`, then cold again | 0 | 0 | 0 | 0.169 s |

**Exact repeats (changed by eejd/py-vllm-mlx#39).** The probe above is from `0.5.0-local9`, where the
scheduler discarded an exact match and prefilled the whole prompt while the cache still credited a hit.
Now a plain-KV entry is reused for an exact repeat: `cached_tokens` is the prompt length minus one (the last
token is always fed) and the repeat is fast. A hit the scheduler cannot use (recurrent, rotating-window or
container layers, an entry rejected by the KV bound, a failed cache insert) is taken back from `hits` and
`tokens_saved`, counted as a miss and in `discarded_hits`. Engines older than this change still show the
discrepancy; there the per-request value is the truthful one.

Known limits: on an exact repeat only the last prompt token is fed to the batch generator, so a request with
`repetition_penalty` or custom logits processors sees just that token as context (as partial-prefix hits
already do); a request aborted before it is scheduled keeps its credit in `hits`; the SSD-tier and MLLM paths
are not settled. An exact SSD promotion, a paged hit that ends on a block boundary and a legacy prefix-cache
hit are rewound by one position like a memory-cache exact hit (or prefilled when the layers cannot be
rewound), so the last token is never in the KV cache twice (#47).

## Why the Simple engine shows no warm speedup

`serve` builds a `SchedulerConfig` only with `--continuous-batching` (`cli.py`); in Simple mode
`scheduler_config=None`, so `--enable-prefix-cache`, the memory-aware cache, `--cache-memory-*`,
KV-cache quantization, the paged cache and the SSD tier are not used at all. Its only reuse is the
system-KV LRU and the optional `--prefix-trie-cache`. In the probe all four 1537-token requests took
0.15-0.22 s with no pattern, and `system_kv_cache` counters stayed at 0. This is also why
`--kv-cache-quantization` is inert in Simple mode (ash#703).

This PR does not change that behavior. It makes it visible: the server prints a startup warning naming
the inert flags, and `cache_state.inert_options` lists them (eejd/py-vllm-mlx#44 tracks whether Simple
mode should reuse prefixes).

## The `cache_state` block

`GET /v1/cache/stats` gains `cache_state` (and, in registry mode, `models.<name>.cache_state`).
`schema_version` is 1. A value the engine cannot report is `"unreported"` (or `{"value": "unreported",
"reason": ...}`), never a default that looks like data.

| Field | Meaning |
|---|---|
| `schema_version` | 1 |
| `versions` | `vllm_mlx`, `mlx`, `mlx_lm` (each or `unreported`) |
| `engine` | `class`, `continuous_batching` (or `unreported` when not started through the CLI), `registry_mode` |
| `launch_options` | the cache-relevant options as requested: `continuous_batching`, `prefix_cache_requested`, `memory_aware_cache_requested`, `cache_memory_mb`, `cache_memory_percent`, `use_paged_cache`, `ssd_cache_dir`, `ssd_cache_max_gb`, `prefill_step_size`, `prefix_trie_cache`, and `kv_cache_quantization` = `{requested, bits, group_size, min_quantize_tokens, effective}` (`effective` is true only with continuous batching) |
| `inert_options` | flags the user set to a non-default value that this engine mode ignores |
| `memory_limit` | `{bytes, source}` with `source` one of `explicit` (`max_memory_mb`), `percent_of_available` (plus `percent`, `available_bytes`: **RAM free at cache creation, so it differs between starts**), `fallback_8gb`; or `unreported` |
| `counters` | `hits`, `misses`, `evictions`, `tokens_saved`, `entry_count`, `current_memory_mb`, `max_memory_mb`; or `unreported` with a reason. Monotonic until `DELETE /v1/cache` or a restart |
| `persistence` | the existing block (`policy`, `dirs`) plus `applies` (false in registry mode and for engines without persistence hooks) and `not_applied_reason`. In registry mode every model shows the same server-wide block; it is not per model |

## Fingerprint of a persisted cache

`load_from_disk` refuses a cache whose version, `cache_format` or model fingerprint differs. The
fingerprint is now also a function of the **weight quantization actually present in the loaded model**
(count of quantized layers by kind, bits, group size and mode; a layer counts as quantized when it has integer
`bits` and `group_size`, which includes MoE expert layers such as `QuantizedSwitchLinear`), so a 4-bit checkpoint and a 6-bit
checkpoint of one architecture no longer share a cache. Before this change the fingerprint hashed only
architecture fields (layers, hidden size, heads, vocab, ...) although its docstring claimed to reject a
different quantization. **Existing persisted caches are refused once and rebuilt**; the refusal is
logged as a fingerprint mismatch.

KV-cache quantization needs no fingerprint field: entries are dequantized when saved. The reverse is
not true: before eejd/py-vllm-mlx#40 entries read from disk were inserted unquantized even when KV
quantization is on; they are now quantized on load under the same rule as fresh entries.

## Open items (filed, not fixed here)

| Issue | Gap |
|---|---|
| #41 | registry mode never loads or saves persisted caches |
| #42 | no cache series in `/metrics` |
| #43 | Responses API / MLLM cached-token counts are constants |
| #44 | Simple engine reports no reuse |
