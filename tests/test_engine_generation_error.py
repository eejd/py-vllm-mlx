# SPDX-License-Identifier: Apache-2.0
"""A request the engine aborted (finish_reason "error") must reach the client as an error.

The scheduler finishes the requests that were running when a step failed (a Metal
out-of-memory, a crashed command buffer) with ``finish_reason="error"`` and no text.
Served as ordinary output that is an empty HTTP 200.  These tests drive every
route that generates (chat, completions, messages, responses; stream and not)
through a real ``BatchedEngine`` whose scheduler hands back error-finished
outputs, on both engine paths (text scheduler, multimodal scheduler).
"""

import json
import platform
import re
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason="Requires Apple Silicon",
)

from fastapi.testclient import TestClient  # noqa: E402

import vllm_mlx.server as server  # noqa: E402
from vllm_mlx.engine.base import (  # noqa: E402
    EngineRequestError,
    GenerationError,
    RequestPreprocessingError,
)
from vllm_mlx.engine.batched import (  # noqa: E402
    BatchedEngine,
    _raise_if_request_failed,
)
from vllm_mlx.metrics import MetricsCollector  # noqa: E402
from vllm_mlx.request import ERROR_KIND_PREPROCESSING, RequestOutput  # noqa: E402

DONE = "[DONE]"


# --------------------------------------------------------------------------
# Engine under test: a real BatchedEngine over a scheduler that fails requests
# --------------------------------------------------------------------------


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(str(m["content"]) for m in messages) + "\nassistant:"

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


def _ok(text, finished=False):
    return RequestOutput(
        request_id="r",
        new_text=text,
        output_text=text,
        prompt_tokens=3,
        completion_tokens=1,
        finished=finished,
        finish_reason="stop" if finished else None,
    )


def _failed(error_kind=None, text=""):
    return RequestOutput(
        request_id="r",
        output_text=text,
        finished=True,
        finish_reason="error",
        error_kind=error_kind,
    )


class _Scheduler:
    """Stands in for AsyncEngineCore (text) and MLLMScheduler (multimodal)."""

    def __init__(self, outputs):
        self.outputs = outputs

    async def generate(self, **kwargs):
        return self.outputs[-1]

    async def add_request(self, **kwargs):
        return "r"

    add_request_async = add_request

    async def stream_outputs(self, request_id):
        for output in self.outputs:
            yield output


def _engine(kind, outputs):
    engine = BatchedEngine("test-model", force_mllm=kind == "mllm")
    engine._loaded = True
    engine._tokenizer = _Tokenizer()
    scheduler = _Scheduler(outputs)
    if kind == "mllm":
        engine._mllm_scheduler = scheduler
    else:
        engine._engine = scheduler
    return engine


@pytest.fixture()
def collector(monkeypatch):
    collector = MetricsCollector()
    collector.configure(enabled=True)
    monkeypatch.setattr(server, "_metrics", collector)
    monkeypatch.setattr(server, "_model_name", "test-model")
    monkeypatch.setattr(server, "_api_key", None)
    monkeypatch.setattr(server, "_mcp_manager", None)
    monkeypatch.setattr(server, "_reasoning_parser", None)
    monkeypatch.setattr(server, "_tool_parser_instance", None)
    monkeypatch.setattr(server, "_tool_call_parser", None)
    monkeypatch.setattr(server, "_enable_auto_tool_choice", False)
    monkeypatch.setattr(server, "_default_timeout", 30.0)
    monkeypatch.setattr(server, "_default_max_tokens", 128)
    monkeypatch.setattr(
        server,
        "_rate_limiter",
        server.RateLimiter(requests_per_minute=60, enabled=False),
    )
    return collector


@pytest.fixture()
def serve(collector, monkeypatch):
    """serve(kind, outputs) -> TestClient answering from a BatchedEngine."""
    monkeypatch.setattr(server, "_engine", None)

    def make(kind, outputs):
        monkeypatch.setattr(server, "_engine", _engine(kind, outputs))
        return TestClient(server.app, raise_server_exceptions=False)

    return make


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

ROUTES = {
    "chat": (
        "/v1/chat/completions",
        {"messages": [{"role": "user", "content": "hi there"}], "max_tokens": 8},
    ),
    "completions": ("/v1/completions", {"prompt": "hi there", "max_tokens": 8}),
    "messages": (
        "/v1/messages",
        {"messages": [{"role": "user", "content": "hi there"}], "max_tokens": 8},
    ),
    "responses": ("/v1/responses", {"input": "hi there", "max_output_tokens": 8}),
}
KINDS = ["text", "mllm"]


def _post(client, route, stream):
    path, body = ROUTES[route]
    return client.post(path, json={"model": "test-model", "stream": stream, **body})


def _stream(client, route):
    """The raw SSE body of a streaming request, as a list of (event, data) frames."""
    path, body = ROUTES[route]
    with client.stream(
        "POST", path, json={"model": "test-model", "stream": True, **body}
    ) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    frames = []
    for block in text.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        event, data = None, None
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                data = line[len("data: ") :]
        frames.append((event, data if data == DONE else json.loads(data)))
    return frames


def _openai_error(body, status=500):
    assert body["error"]["code"] == status
    return body["error"]


# --- non-stream -----------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ["chat", "completions", "responses"])
def test_nonstream_generation_failure_is_http_500_openai_json(serve, route, kind):
    response = _post(serve(kind, [_failed()]), route, stream=False)

    assert response.status_code == 500
    error = _openai_error(response.json())
    assert error["type"] == "InternalServerError"
    assert error["message"] == "Internal server error"


@pytest.mark.parametrize("kind", KINDS)
def test_nonstream_messages_failure_is_http_500_anthropic_envelope(serve, kind):
    response = _post(serve(kind, [_failed()]), "messages", stream=False)

    assert response.status_code == 500
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }


@pytest.mark.parametrize("route", ["chat", "completions", "responses"])
def test_nonstream_preprocessing_failure_is_http_400_bad_request(serve, route):
    outputs = [_failed(ERROR_KIND_PREPROCESSING)]
    response = _post(serve("mllm", outputs), route, stream=False)

    assert response.status_code == 400
    assert _openai_error(response.json(), 400)["type"] == "BadRequestError"


def test_nonstream_messages_preprocessing_failure_is_invalid_request(serve):
    response = _post(
        serve("mllm", [_failed(ERROR_KIND_PREPROCESSING)]), "messages", stream=False
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


# --- stream ---------------------------------------------------------------

AFTER_PARTIAL = [_ok("Hel"), _failed(text="Hel")]
FIRST = [_failed()]


def _openai_stream_frames(client, route):
    frames = [data for _, data in _stream(client, route)]
    errors = [i for i, f in enumerate(frames) if f != DONE and "error" in f]
    dones = [i for i, f in enumerate(frames) if f == DONE]
    return frames, errors, dones


@pytest.mark.parametrize("outputs", [FIRST, AFTER_PARTIAL], ids=["first", "partial"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ["chat", "completions"])
def test_openai_stream_error_frame_precedes_done(serve, route, kind, outputs):
    """A client stops reading at [DONE]: the error must already have been sent."""
    frames, errors, dones = _openai_stream_frames(serve(kind, outputs), route)

    assert len(errors) == 1 and len(dones) == 1
    assert errors[0] < dones[0] == len(frames) - 1
    error = _openai_error(frames[errors[0]])
    assert error["type"] == "InternalServerError"
    # nothing that reads as a normal end of the choice was sent
    for frame in frames[: errors[0]]:
        for choice in frame.get("choices", []):
            assert choice.get("finish_reason") is None


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("route", ["chat", "completions"])
def test_openai_stream_preprocessing_failure_is_a_400_error_frame(serve, route, kind):
    outputs = [_failed(ERROR_KIND_PREPROCESSING)]
    frames, errors, dones = _openai_stream_frames(serve(kind, outputs), route)

    assert len(errors) == 1 and errors[0] < dones[0]
    error = _openai_error(frames[errors[0]], 400)
    assert error["type"] == "BadRequestError"


@pytest.mark.parametrize("outputs", [FIRST, AFTER_PARTIAL], ids=["first", "partial"])
@pytest.mark.parametrize("kind", KINDS)
def test_messages_stream_error_event_precedes_message_stop(serve, kind, outputs):
    frames = _stream(serve(kind, outputs), "messages")
    names = [event for event, _ in frames]

    assert names.count("error") == 1 and names[-1] == "message_stop"
    assert names.index("error") < names.index("message_stop")
    assert "message_delta" not in names
    error = dict(frames)["error"]
    assert error == {
        "type": "error",
        "error": {"type": "api_error", "message": "Internal server error"},
    }


def test_messages_stream_preprocessing_failure_is_invalid_request(serve):
    frames = _stream(serve("mllm", [_failed(ERROR_KIND_PREPROCESSING)]), "messages")

    assert dict(frames)["error"]["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("outputs", [FIRST, AFTER_PARTIAL], ids=["first", "partial"])
@pytest.mark.parametrize("kind", KINDS)
def test_responses_stream_ends_in_error_and_response_failed(serve, kind, outputs):
    frames = _stream(serve(kind, outputs), "responses")
    names = [event for event, _ in frames]

    assert names[:2] == ["response.created", "response.in_progress"]
    assert names[-2:] == ["error", "response.failed"]
    assert "response.completed" not in names
    numbers = [data["sequence_number"] for _, data in frames]
    assert numbers == sorted(set(numbers))

    error, failed = frames[-2][1], frames[-1][1]
    assert error["type"] == "error"
    assert error["code"] == "server_error"
    assert error["message"] == "Internal server error"
    assert failed["type"] == "response.failed"
    assert failed["response"]["status"] == "failed"
    assert failed["response"]["error"] == {
        "code": "server_error",
        "message": "Internal server error",
    }
    assert failed["response"]["id"] == frames[0][1]["response"]["id"]


def test_responses_stream_preprocessing_failure_is_invalid_prompt(serve):
    frames = _stream(serve("mllm", [_failed(ERROR_KIND_PREPROCESSING)]), "responses")

    assert frames[-1][1]["response"]["error"]["code"] == "invalid_prompt"


def test_failed_responses_stream_stores_nothing(serve, monkeypatch):
    monkeypatch.setattr(server, "_responses_store", server.OrderedDict())
    _stream(serve("text", AFTER_PARTIAL), "responses")

    assert not server._responses_store


# --- a failure does not poison the next request ---------------------------


@pytest.mark.parametrize("route", ["chat", "completions", "responses"])
def test_next_request_after_a_failure_succeeds(serve, route):
    client = serve("text", [_failed()])
    assert _post(client, route, stream=False).status_code == 500

    client = serve("text", [_ok("Hello", finished=True)])
    assert _post(client, route, stream=False).status_code == 200


# --- metrics --------------------------------------------------------------


def _scrape(collector):
    body, _ = collector.render_metrics(engine=None, mcp_manager=None)
    return body.decode()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "route,endpoint",
    [
        ("chat", "chat_completions"),
        ("completions", "completions"),
        ("messages", "anthropic_messages"),
        ("responses", "responses"),
    ],
)
def test_failure_is_counted_as_an_error_result(
    serve, collector, route, endpoint, stream
):
    client = serve("text", [_failed()])
    if stream:
        _stream(client, route)
    else:
        _post(client, route, stream=False)

    assert re.search(
        rf'vllm_mlx_inference_requests_total\{{endpoint="{endpoint}",'
        rf'result="error",stream="(true|false)"\}} 1\.0',
        _scrape(collector),
    )


def test_preprocessing_failure_is_counted_as_a_client_error(serve, collector):
    client = serve("mllm", [_failed(ERROR_KIND_PREPROCESSING)])
    _post(client, "chat", stream=False)

    assert re.search(
        r'vllm_mlx_inference_requests_total\{endpoint="chat_completions",'
        r'result="client_error",stream="false"\} 1\.0',
        _scrape(collector),
    )


# --------------------------------------------------------------------------
# Engine classification (every engine path)
# --------------------------------------------------------------------------


def _drive(engine, method):
    import asyncio

    async def run():
        if method == "generate":
            return await engine.generate(prompt="hi", max_tokens=4)
        return [o async for o in engine.stream_generate(prompt="hi", max_tokens=4)]

    return asyncio.run(run())


@pytest.mark.parametrize("method", ["generate", "stream_generate"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize(
    "error_kind,expected",
    [(None, GenerationError), (ERROR_KIND_PREPROCESSING, RequestPreprocessingError)],
)
def test_every_engine_path_raises_the_matching_error(
    kind, method, error_kind, expected
):
    engine = _engine(kind, [_failed(error_kind)])

    with pytest.raises(expected):
        _drive(engine, method)


@pytest.mark.parametrize("method", ["generate", "stream_generate"])
@pytest.mark.parametrize("kind", KINDS)
def test_every_engine_path_passes_good_output_through(kind, method):
    result = _drive(_engine(kind, [_ok("Hello", finished=True)]), method)

    texts = [o.text for o in result] if method == "stream_generate" else [result.text]
    assert texts == ["Hello"]


@pytest.mark.parametrize("reason", [None, "stop", "length", "abort"])
def test_only_error_finish_reason_fails_a_request(reason):
    _raise_if_request_failed(RequestOutput(request_id="r", finish_reason=reason))


def test_engine_errors_say_how_they_are_reported():
    assert issubclass(GenerationError, EngineRequestError)
    assert (GenerationError.status_code, RequestPreprocessingError.status_code) == (
        500,
        400,
    )
