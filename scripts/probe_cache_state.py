#!/usr/bin/env python3
"""Record which cache signals each engine mode really exposes (py-vllm-mlx#37).

For each mode it starts a throwaway server on a small model, sends the same four requests
(cold, exact repeat, a request sharing a long prefix, and a repeat after ``DELETE /v1/cache``)
and records, after each one: the response ``usage``, ``GET /v1/cache/stats``, the cache-related
part of ``GET /v1/status`` and the cache lines of ``GET /metrics``. The output JSON is the
evidence behind docs/cache-state-signals.md.

    probe_cache_state.py --model /path/to/small/model --out probe.json [--modes simple,batched,...]

Modes: simple (SimpleEngine), batched (--continuous-batching), registry-simple and
registry-batched (--models-config with one model). Each server gets its own persisted-cache
directory under --workdir, started with ``--prefix-cache-persist none --prefix-cache-reset both``
so no earlier run influences it. Stdlib only. The servers run from this checkout.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "The quick brown fox jumps over the lazy dog. " * 150


def http(method, url, body=None, timeout=300):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            ctype = r.headers.get("Content-Type", "")
            return r.status, (json.loads(raw) if "json" in ctype else raw)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:300]


def start(mode, model, port, workdir):
    cmd = [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(ROOT)!r}); from vllm_mlx.cli import main; main()", "serve"]
    pdir = workdir / f"{mode}-prefix"
    flags = [
        "--port", str(port), "--enable-metrics",
        "--prefix-cache-dir", str(pdir), "--prefix-cache-persist", "none", "--prefix-cache-reset", "both",
    ]
    name = "probe-model"
    if mode == "simple":
        cmd += [model, *flags]
    elif mode == "batched":
        cmd += [model, *flags, "--continuous-batching", "--enable-prefix-cache"]
    else:
        cb = "true" if mode == "registry-batched" else "false"
        cfg = workdir / f"{mode}.yaml"
        cfg.write_text(
            f"manager:\n  memory_budget_gb: 16\nmodels:\n  - name: {name}\n    path: {model}\n"
            f"    preload: true\n    continuous_batching: {cb}\n    estimated_memory_gb: 2\n"
        )
        cmd += ["--models-config", str(cfg), *flags]
    log = open(workdir / f"{mode}.log", "w")
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT), name


def wait_ready(proc, base, seconds=300):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early (rc={proc.returncode})")
        try:
            if http("GET", f"{base}/health", timeout=3)[0] == 200:
                return
        except Exception:  # noqa: BLE001
            time.sleep(1)
    raise SystemExit("server did not become ready")


def cache_lines(metrics_text):
    if not isinstance(metrics_text, str):
        return []
    return [ln for ln in metrics_text.splitlines() if "cache" in ln.lower() and not ln.startswith("#")][:20]


def snapshot(base):
    _, stats = http("GET", f"{base}/v1/cache/stats")
    _, status = http("GET", f"{base}/v1/status")
    _, metrics = http("GET", f"{base}/metrics")
    status_cache = {}
    if isinstance(status, dict):
        status_cache = {k: v for k, v in status.items() if "cache" in k.lower()}
        status_cache["_top_level_keys"] = sorted(status)
    return {"cache_stats": stats, "status_cache": status_cache, "metrics_cache_lines": cache_lines(metrics)}


def chat(base, model_id, suffix):
    t0 = time.time()
    code, body = http("POST", f"{base}/v1/chat/completions", {
        "model": model_id, "max_tokens": 8, "temperature": 0,
        "messages": [{"role": "user", "content": PROMPT + suffix}],
    })
    return {"http": code, "seconds": round(time.time() - t0, 3),
            "usage": body.get("usage") if isinstance(body, dict) else body}


def probe(mode, model, port, workdir):
    proc, name = start(mode, model, port, workdir)
    base = f"http://127.0.0.1:{port}"
    steps = []
    try:
        wait_ready(proc, base)
        _, models = http("GET", f"{base}/v1/models")
        model_id = models["data"][0]["id"]
        steps.append({"step": "ready", **snapshot(base)})
        for label, suffix in (("cold", " ALPHA"), ("exact_repeat", " ALPHA"), ("shared_prefix", " BRAVO")):
            r = chat(base, model_id, suffix)
            steps.append({"step": label, "request": r, **snapshot(base)})
        code, deleted = http("DELETE", f"{base}/v1/cache")
        steps.append({"step": "delete_cache", "http": code, "body": deleted, **snapshot(base)})
        r = chat(base, model_id, " ALPHA")
        steps.append({"step": "after_delete", "request": r, **snapshot(base)})
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
    log = (workdir / f"{mode}.log").read_text()
    warn = [ln for ln in log.splitlines() if "only take effect with --continuous-batching" in ln]
    return {"mode": mode, "returncode": proc.returncode, "startup_warnings": warn, "steps": steps}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--port", type=int, default=14168)
    ap.add_argument("--modes", default="simple,batched,registry-simple,registry-batched")
    a = ap.parse_args()
    workdir = Path(a.workdir or (Path(a.out).parent / "probe-work"))
    workdir.mkdir(parents=True, exist_ok=True)
    result = [probe(m, a.model, a.port, workdir) for m in a.modes.split(",")]
    Path(a.out).write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
