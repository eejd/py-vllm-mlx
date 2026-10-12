# SPDX-License-Identifier: Apache-2.0
"""``max_completion_tokens`` and ``ignore_eos`` at the API boundary.

load generators (guidellm, vllm bench) send ``max_completion_tokens`` plus
``ignore_eos: true`` to fix the output length. Both used to be dropped by the
request models, so a request for 32 tokens produced hundreds, or stopped early
at EOS. The server now resolves the cap like vLLM does
(``max_completion_tokens`` > ``max_tokens`` > server default) and hands
``ignore_eos`` to the engine.
"""

import pytest
from pydantic import ValidationError

import vllm_mlx.server as server
from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest, Message
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.engine.base import GenerationOutput
from vllm_mlx.request import UnsupportedRequestOption


@pytest.fixture
def limits(monkeypatch):
    monkeypatch.setattr(server, "_default_max_tokens", 1024)
    monkeypatch.setattr(server, "_max_request_tokens", 2048)


class TestRequestModels:
    def test_chat_accepts_max_completion_tokens_and_ignore_eos(self):
        request = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            max_completion_tokens=32,
            ignore_eos=True,
        )
        assert request.max_completion_tokens == 32
        assert request.max_tokens is None
        assert request.ignore_eos is True

    def test_chat_defaults_stop_on_eos(self):
        request = ChatCompletionRequest(
            model="m", messages=[Message(role="user", content="hi")]
        )
        assert request.max_completion_tokens is None
        assert request.ignore_eos is False

    @pytest.mark.parametrize("value", [0, -1])
    def test_chat_rejects_non_positive_max_completion_tokens(self, value):
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="m",
                messages=[Message(role="user", content="hi")],
                max_completion_tokens=value,
            )

    def test_completions_and_responses_accept_ignore_eos(self):
        assert CompletionRequest(model="m", prompt="p", ignore_eos=True).ignore_eos
        assert not CompletionRequest(model="m", prompt="p").ignore_eos
        assert ResponsesRequest(model="m", input="p", ignore_eos=True).ignore_eos
        assert not ResponsesRequest(model="m", input="p").ignore_eos


class TestResolveRequestMaxTokens:
    def test_precedence_completion_over_max_tokens_over_default(self, limits):
        resolve = server._resolve_request_max_tokens
        assert resolve(None) == 1024
        assert resolve(512) == 512
        assert resolve(None, max_completion_tokens=32) == 32
        # Both given: max_completion_tokens wins, whichever is larger.
        assert resolve(512, max_completion_tokens=32) == 32
        assert resolve(16, max_completion_tokens=32) == 32

    def test_server_ceiling_applies_to_max_completion_tokens(self, limits):
        with pytest.raises(server.HTTPException) as exc_info:
            server._resolve_request_max_tokens(None, max_completion_tokens=4096)
        assert exc_info.value.status_code == 400
        assert "max_completion_tokens" in exc_info.value.detail

    def test_ceiling_checks_the_effective_field_only(self, limits):
        # An over-limit max_tokens is superseded, so it must not be rejected.
        assert server._resolve_request_max_tokens(4096, max_completion_tokens=8) == 8
        with pytest.raises(server.HTTPException) as exc_info:
            server._resolve_request_max_tokens(4096)
        assert "max_tokens" in exc_info.value.detail


class _Engine:
    """Records the kwargs the server hands to each generation entry point."""

    model_name = "fake"
    is_mllm = False
    preserve_native_tool_format = False

    def __init__(self):
        self.calls = []
        self.validated = []

    async def validate_request_options(self, **kwargs):
        self.validated.append(kwargs)

    async def chat(self, messages, **kwargs):
        self.calls.append(("chat", kwargs))
        return GenerationOutput(
            text="ok", prompt_tokens=3, completion_tokens=2, finish_reason="length"
        )

    async def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return GenerationOutput(
            text="ok", prompt_tokens=3, completion_tokens=2, finish_reason="length"
        )

    async def stream_generate(self, **kwargs):
        self.calls.append(("stream_generate", kwargs))
        yield GenerationOutput(
            text="ok",
            new_text="ok",
            prompt_tokens=3,
            completion_tokens=2,
            finished=True,
            finish_reason="length",
        )


@pytest.fixture
def engine(monkeypatch, limits):
    fake = _Engine()

    async def acquire(raw_request, **kwargs):
        return fake

    async def release(**kwargs):
        return None

    monkeypatch.setattr(server, "_validate_model_name", lambda _m: None)
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", acquire)
    monkeypatch.setattr(server, "_release_default_engine", release)
    monkeypatch.setattr(server, "_model_name", "served")
    monkeypatch.setattr(server, "_default_timeout", 30.0)
    return fake


@pytest.mark.anyio
class TestEndpointsForwardTheCapAndIgnoreEos:
    async def test_chat_completions(self, engine):
        request = ChatCompletionRequest(
            model="served",
            messages=[Message(role="user", content="hi")],
            max_tokens=500,
            max_completion_tokens=32,
            ignore_eos=True,
        )
        await server.create_chat_completion(request, raw_request=None)

        _, kwargs = engine.calls[0]
        assert kwargs["max_tokens"] == 32
        assert kwargs["ignore_eos"] is True

    async def test_chat_completions_default_leaves_eos_alone(self, engine):
        request = ChatCompletionRequest(
            model="served",
            messages=[Message(role="user", content="hi")],
            max_tokens=500,
        )
        await server.create_chat_completion(request, raw_request=None)

        _, kwargs = engine.calls[0]
        assert kwargs["max_tokens"] == 500
        assert "ignore_eos" not in kwargs

    async def test_chat_completions_unsupported_option_is_a_400(self, engine):
        async def chat(messages, **kwargs):
            raise UnsupportedRequestOption("ignore_eos is not supported here")

        engine.chat = chat
        request = ChatCompletionRequest(
            model="served",
            messages=[Message(role="user", content="hi")],
            ignore_eos=True,
        )
        with pytest.raises(server.HTTPException) as exc_info:
            await server.create_chat_completion(request, raw_request=None)
        assert exc_info.value.status_code == 400
        assert "ignore_eos" in exc_info.value.detail

    async def test_unsupported_option_is_a_400_before_any_response_starts(self, engine):
        """Streaming cannot report an error after its first chunk."""

        async def refuse(**kwargs):
            raise UnsupportedRequestOption("ignore_eos is not supported here")

        engine.validate_request_options = refuse

        chat = ChatCompletionRequest(
            model="served",
            messages=[Message(role="user", content="hi")],
            ignore_eos=True,
            stream=True,
        )
        completion = CompletionRequest(
            model="served", prompt="p", ignore_eos=True, stream=True
        )
        for call in (
            server.create_chat_completion(chat, raw_request=None),
            server.create_completion(completion, raw_request=None),
        ):
            with pytest.raises(server.HTTPException) as exc_info:
                await call
            assert exc_info.value.status_code == 400
        assert engine.calls == []

    async def test_chat_preflight_sees_media_and_the_drafter_override(self, engine):
        request = ChatCompletionRequest(
            model="served",
            messages=[
                Message(
                    role="user",
                    content=[
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "file:///x.png"}},
                    ],
                )
            ],
            ignore_eos=True,
            mllm_draft=True,
        )
        await server.create_chat_completion(request, raw_request=None)

        assert engine.validated == [
            {"ignore_eos": True, "has_media": True, "mllm_draft": True}
        ]

    async def test_no_preflight_without_ignore_eos(self, engine):
        request = ChatCompletionRequest(
            model="served", messages=[Message(role="user", content="hi")]
        )
        await server.create_chat_completion(request, raw_request=None)

        assert engine.validated == []

    async def test_completions(self, engine):
        request = CompletionRequest(
            model="served", prompt="p", max_tokens=32, ignore_eos=True
        )
        await server.create_completion(request, raw_request=None)

        name, kwargs = engine.calls[0]
        assert name == "generate"
        assert kwargs["max_tokens"] == 32
        assert kwargs["ignore_eos"] is True

    async def test_streaming_completions(self, engine):
        request = CompletionRequest(
            model="served", prompt="p", max_tokens=32, ignore_eos=True, stream=True
        )
        chunks = [
            chunk async for chunk in server.stream_completion(engine, "p", request, 32)
        ]

        assert chunks
        name, kwargs = engine.calls[0]
        assert name == "stream_generate"
        assert kwargs["ignore_eos"] is True

    async def test_completions_default_leaves_eos_alone(self, engine):
        request = CompletionRequest(model="served", prompt="p", max_tokens=32)
        await server.create_completion(request, raw_request=None)

        assert "ignore_eos" not in engine.calls[0][1]


@pytest.mark.anyio
class TestResponsesPreflight:
    @pytest.fixture
    def refusing_engine(self, monkeypatch, limits):
        fake = _Engine()

        async def refuse(**kwargs):
            raise UnsupportedRequestOption("ignore_eos is not supported here")

        fake.validate_request_options = refuse
        monkeypatch.setattr(server, "get_engine", lambda: fake)
        monkeypatch.setattr(server, "_validate_model_name", lambda _m: None)
        monkeypatch.setattr(server, "_model_manager", None)

        async def no_preflight(raw_request):
            return None

        monkeypatch.setattr(server, "_preflight_response_format_backend", no_preflight)
        return fake

    @pytest.mark.parametrize("stream", [False, True])
    async def test_unsupported_option_is_a_400(self, refusing_engine, stream):
        request = ResponsesRequest(
            model="served", input="hi", ignore_eos=True, stream=stream
        )
        with pytest.raises(server.HTTPException) as exc_info:
            await server.create_response(request, raw_request=None)

        assert exc_info.value.status_code == 400
        assert refusing_engine.calls == []


class TestResponsesApi:
    def test_max_output_tokens_and_ignore_eos_reach_the_engine(self, limits):
        request = ResponsesRequest(
            model="served", input="hi", max_output_tokens=24, ignore_eos=True
        )
        _, _, _, chat_kwargs = server._prepare_responses_request(
            request, validate_remote_media=False, engine=_Engine()
        )

        assert chat_kwargs["max_tokens"] == 24
        assert chat_kwargs["ignore_eos"] is True

    def test_without_them_the_server_default_applies(self, limits):
        request = ResponsesRequest(model="served", input="hi")
        _, _, _, chat_kwargs = server._prepare_responses_request(
            request, validate_remote_media=False, engine=_Engine()
        )

        assert chat_kwargs["max_tokens"] == 1024
        assert "ignore_eos" not in chat_kwargs
