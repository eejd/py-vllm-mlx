# Persisted prefix cache: location, persistence mode and reset

The memory-aware prefix cache is written to disk when the server stops and read back when it
starts. Left alone, that is hidden state: a fixed directory under `$HOME`, always loaded, always
saved, so the starting cache of a run is whatever the previous run left behind. That makes timings
and cache-hit behavior unreproducible. Three options make it explicit.

| Option | Values | Meaning |
|---|---|---|
| `--prefix-cache-dir DIR` (or `VLLM_MLX_PREFIX_CACHE_DIR`) | absolute path | Base directory. Default `~/.cache/vllm-mlx/prefix_cache`. Each model keeps its own subdirectory (the model path with `/` replaced by `--`), so existing caches keep working and `--models-config` serves one directory per model. |
| `--prefix-cache-persist` | `auto` (default), `none`, `load-only`, `save-only` | `auto`: load at start, save at stop (the historical behavior). `none`: never read or write the persisted cache. `load-only`: start from the preserved cache and never modify it, so every run starts from the same warm state. `save-only`: start cold, write at stop (builds a snapshot). |
| `--prefix-cache-reset` | `never` (default), `start`, `stop`, `both` | Delete this model's persisted entries before the load (`start`), after the shutdown save (`stop`), or both. |

Valid combinations (anything else is refused at startup, before any model loads):

| persist \ reset | never | start | stop | both |
|---|---|---|---|---|
| `auto` | yes | yes (cold start, then save) | no (would delete what it just saved) | no |
| `none` | yes | yes | yes | yes (leaves nothing behind) |
| `load-only` | yes (a defined warm cache) | no | no | no |
| `save-only` | yes | yes (build a clean snapshot) | no | no |

Typical uses:

* **Cold, nothing persisted, nothing left behind**: `--prefix-cache-persist none --prefix-cache-reset both`.
* **Cold start that saves at stop**: `--prefix-cache-reset start`.
* **Build a snapshot, then reuse it identically**: run once with `--prefix-cache-persist save-only --prefix-cache-reset start`,
  then run with `--prefix-cache-persist load-only` as often as needed; the snapshot is never modified.
* Give each benchmark run its own directory with `--prefix-cache-dir` so runs cannot warm each other.

## What a reset deletes

Only the files the persistence code writes (`index.json`, `entry_<i>.safetensors`,
`entry_<i>_tokens.bin`) in that model's directory. It never follows a symlink, never acts on `/` or
the home directory or a parent of it, refuses a directory that is a symlink or a regular file, and
leaves a directory with none of those files untouched. Other files are reported and left in place.
A reset requested at start that cannot be done safely stops the server with an error instead of
continuing with an unknown cache; a failed reset at stop is logged.

## What the server reports

One startup line per model directory:

```
prefix_cache dir=... persist=auto reset=start entries_on_disk=0 bytes_on_disk=0 loaded=0
```

and the same information, under `persistence`, in `GET /v1/cache/stats` (policy, and per directory:
entries and bytes on disk after any reset, entries loaded, whether it was saved, what a reset
deleted).

## Not covered here

* The in-memory limit still defaults to 20% of *free* RAM when the server starts, so it varies between
  starts; pass `--cache-memory-mb` for a reproducible limit.
* Saving writes `entry_<i>` files for the entries it has and does not remove higher-numbered files
  from an earlier, larger save (the index decides what is loaded). Use `--prefix-cache-reset start`
  or `stop` (with `none`) to start from an empty directory.
* The runtime endpoints `DELETE /v1/cache` and `DELETE /v1/cache/prefix` clear the in-memory cache and
  are a separate mechanism from the persisted directory.
