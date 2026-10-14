# SPDX-License-Identifier: Apache-2.0
"""``skip_special_tokens`` at the API boundary.

vLLM's chat, completion and responses requests carry ``skip_special_tokens``
(default true). The engines already leave BOS/PAD/UNK/EOS out of the text, so
only ``false`` has to travel: a client that wants the raw markers back asks for
them, as it can on vLLM.
"""

import pytest

import vllm_mlx.server as server
from tests.test_ignore_eos_api import _Engine, engine, limits  # noqa: F401
from vllm_mlx.api.models import ChatCompletionRequest, CompletionRequest, Message
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.api.utils import clean_output_text
from vllm_mlx.engine.base import GenerationOutput


class TestRequestModels:
    def test_default_is_true(self):
        chat = ChatCompletionRequest(
            model="m", messages=[Message(role="user", content="hi")]
        )
        assert chat.skip_special_tokens is True
        assert CompletionRequest(model="m", prompt="p").skip_special_tokens is True
        assert ResponsesRequest(model="m", input="p").skip_special_tokens is True

    def test_false_is_accepted(self):
        chat = ChatCompletionRequest(
            model="m",
            messages=[Message(role="user", content="hi")],
            skip_special_tokens=False,
        )
        assert chat.skip_special_tokens is False
        assert not CompletionRequest(
            model="m", prompt="p", skip_special_tokens=False
        ).skip_special_tokens
        assert not ResponsesRequest(
            model="m", input="p", skip_special_tokens=False
        ).skip_special_tokens


@pytest.mark.anyio
class TestEndpointsForwardTheOptOut:
    async def test_chat_completions(self, engine):  # noqa: F811
        request = ChatCompletionRequest(
            model="served",
            messages=[Message(role="user", content="hi")],
            skip_special_tokens=False,
        )
        await server.create_chat_completion(request, raw_request=None)

        assert engine.calls[0][1]["skip_special_tokens"] is False

    async def test_chat_completions_default_sends_nothing(self, engine):  # noqa: F811
        request = ChatCompletionRequest(
            model="served", messages=[Message(role="user", content="hi")]
        )
        await server.create_chat_completion(request, raw_request=None)

        assert "skip_special_tokens" not in engine.calls[0][1]

    async def test_completions(self, engine):  # noqa: F811
        request = CompletionRequest(
            model="served", prompt="p", max_tokens=32, skip_special_tokens=False
        )
        await server.create_completion(request, raw_request=None)

        name, kwargs = engine.calls[0]
        assert name == "generate"
        assert kwargs["skip_special_tokens"] is False

    async def test_streaming_completions(self, engine):  # noqa: F811
        request = CompletionRequest(
            model="served",
            prompt="p",
            max_tokens=32,
            skip_special_tokens=False,
            stream=True,
        )
        chunks = [
            chunk async for chunk in server.stream_completion(engine, "p", request, 32)
        ]

        assert chunks
        name, kwargs = engine.calls[0]
        assert name == "stream_generate"
        assert kwargs["skip_special_tokens"] is False

    async def test_completions_default_sends_nothing(self, engine):  # noqa: F811
        request = CompletionRequest(model="served", prompt="p", max_tokens=32)
        await server.create_completion(request, raw_request=None)

        assert "skip_special_tokens" not in engine.calls[0][1]

    def test_responses(self, limits, monkeypatch):  # noqa: F811
        monkeypatch.setattr(server, "_validate_model_name", lambda _m: None)
        for value, expected in ((False, False), (True, None)):
            request = ResponsesRequest(
                model="served", input="hi", skip_special_tokens=value
            )
            _, _, _, chat_kwargs = server._prepare_responses_request(
                request, validate_remote_media=False, engine=_Engine()
            )
            assert chat_kwargs.get("skip_special_tokens") is expected


class TestCleanOutputText:
    def test_default_removes_the_legacy_special_token_list(self):
        assert clean_output_text("hi<pad><|im_end|>") == "hi"

    def test_false_returns_the_text_as_generated(self):
        text = "<s>hi<pad><|im_end|> "
        assert clean_output_text(text, skip_special_tokens=False) == text


@pytest.mark.anyio
class TestNonStreamContentIsNotCleanedWhenAskedNotTo:
    """The server strips a list of special tokens from non-stream content."""

    @pytest.fixture
    def padded(self, engine):  # noqa: F811
        async def chat(messages, **kwargs):
            return GenerationOutput(
                text="hi<pad>",
                prompt_tokens=3,
                completion_tokens=2,
                finish_reason="stop",
            )

        engine.chat = chat
        return engine

    @pytest.mark.parametrize("skip, expected", [(True, "hi"), (False, "hi<pad>")])
    async def test_chat_completions(self, padded, skip, expected):
        request = ChatCompletionRequest(
            model="served",
            messages=[Message(role="user", content="hi")],
            skip_special_tokens=skip,
        )
        response = await server.create_chat_completion(request, raw_request=None)

        assert response.choices[0].message.content == expected
