# SPDX-License-Identifier: Apache-2.0
"""A request the engine aborted (finish_reason "error") must reach the client as an error."""

import json

import pytest

from vllm_mlx.engine.base import GenerationError
from vllm_mlx.engine.batched import _raise_if_generation_error
from vllm_mlx.server import (
    _anthropic_stream_error_frame,
    _ensure_sse_terminal,
    _openai_stream_error_frame,
)


class _Out:
    def __init__(self, finish_reason):
        self.finish_reason = finish_reason


def test_error_finish_reason_raises_generation_error():
    with pytest.raises(GenerationError):
        _raise_if_generation_error(_Out("error"))


@pytest.mark.parametrize("reason", [None, "stop", "length", "abort"])
def test_other_finish_reasons_pass(reason):
    _raise_if_generation_error(_Out(reason))


async def _failing():
    yield "data: {}\n\n"
    raise GenerationError()


@pytest.mark.anyio
async def test_stream_failure_sends_error_frame_before_terminal():
    seen = []
    frames = [
        f
        async for f in _ensure_sse_terminal(
            _failing(),
            "data: [DONE]\n\n",
            error_frame=_openai_stream_error_frame,
            on_error=seen.append,
        )
    ]
    assert frames[0] == "data: {}\n\n"
    err = json.loads(frames[1][len("data: ") :])["error"]
    assert err["code"] == 500 and err["type"] == "InternalServerError"
    assert frames[2] == "data: [DONE]\n\n"
    assert len(seen) == 1 and isinstance(seen[0], GenerationError)


@pytest.mark.anyio
async def test_anthropic_error_event():
    frames = [
        f
        async for f in _ensure_sse_terminal(
            _failing(),
            "event: message_stop\ndata: {}\n\n",
            error_frame=_anthropic_stream_error_frame,
        )
    ]
    assert frames[1].startswith("event: error\n")
    assert json.loads(frames[1].split("data: ", 1)[1])["error"]["type"] == "api_error"


@pytest.mark.anyio
async def test_without_error_frame_stream_just_ends():
    frames = [f async for f in _ensure_sse_terminal(_failing(), "data: [DONE]\n\n")]
    assert frames == ["data: {}\n\n", "data: [DONE]\n\n"]


def test_route_maps_generation_error_to_500_json():
    from fastapi.testclient import TestClient

    from vllm_mlx import server

    @server.app.get("/__boom_openai")
    async def boom():
        raise GenerationError()

    r = TestClient(server.app, raise_server_exceptions=False).get("/__boom_openai")
    assert r.status_code == 500
    assert r.json()["error"]["type"] == "InternalServerError"
