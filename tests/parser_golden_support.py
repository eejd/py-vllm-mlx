"""Vendored from eejd/py-quantbench (PR #48, src/quantbench/bfcl/parser_golden.py); keep in sync.

The corpus is tests/data/parser_golden.jsonl, sha256 cdeb531c941bd95caf2d64f995dc35c6a26e2a58041c17d601487ea79b099735.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

DATA = Path(__file__).parent / "data" / "parser_golden.jsonl"
POLICIES = ("exact", "salvage", "reject", "any_valid", "content")


def load_cases(path: Path = DATA) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def iter_parser_cases(
    policies: Sequence[str] | None = None,
) -> Iterator[dict[str, Any]]:
    for case in load_cases():
        if policies is None or case["policy"] in policies:
            yield case


def _arguments(call: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    raw = call.get("arguments")
    if isinstance(raw, dict):
        return raw, None
    if not isinstance(raw, str):
        return None, f"arguments is {type(raw).__name__}, not a JSON string or object"
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"arguments is not valid JSON ({exc.msg}): {raw[:80]!r}"
    if not isinstance(value, dict):
        return None, f"arguments is JSON but not an object: {raw[:80]!r}"
    return value, None


def _canonical(calls: Any) -> str:
    # json.dumps keeps 3 and 3.0 (and 1 and true) apart, which == does not.
    return json.dumps(calls, sort_keys=True)


def check_case(
    case: dict[str, Any], calls: Sequence[dict[str, Any]], content: str | None = None
) -> list[str]:
    """Problems with what a parser returned for ``case``; empty means the case passes.

    ``calls`` are ``{"name": str, "arguments": str | dict}``; ``content`` is the text returned
    alongside (or instead of) the calls.
    """
    problems: list[str] = []
    decoded: list[dict[str, Any]] = []
    for i, call in enumerate(calls):
        if not isinstance(call, dict):
            problems.append(f"call {i} is {type(call).__name__}, not an object")
            continue
        name = call.get("name")
        args, err = _arguments(call)
        if not isinstance(name, str) or not name:
            problems.append(f"call {i} has no name")
        if err:
            problems.append(f"call {i}: {err}")
        elif args is not None:
            decoded.append({"name": name, "arguments": args})
    policy = case["policy"]
    if policy in ("exact", "salvage"):
        if not problems and _canonical(decoded) != _canonical(case["expected_calls"]):
            problems.append(f"expected {case['expected_calls']!r}, got {decoded!r}")
    elif policy == "reject":
        if calls:
            problems.append(
                f"emitted {len(calls)} call(s) for output that must not become one"
            )
    elif policy == "content":
        if calls:
            problems.append(
                f"emitted {len(calls)} call(s) for output that is not a call"
            )
        if not (content or "").strip() or case["text"].strip() not in (content or ""):
            problems.append("text did not come back as content")
    return problems
