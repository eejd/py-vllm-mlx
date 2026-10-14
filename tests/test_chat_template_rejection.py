# SPDX-License-Identifier: Apache-2.0
"""A chat template that refuses the request's messages is a 400, not a 500.

Gemma's template (like Mistral's and Llama-2's) validates its input with
``raise_exception("Conversation roles must alternate ...")``. That is the model's
template saying the *request* is malformed for it, the case vLLM answers with a 400
carrying the template's message (``safe_apply_chat_template``). Here it surfaced as an
HTTP 500, or, once a stream had started, as an empty 200. A template that cannot be
compiled, or any other error, is still a server error.
"""

from types import SimpleNamespace
from unittest.mock import patch

import jinja2
import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

import vllm_mlx.server as server
from vllm_mlx.api.models import ChatCompletionRequest, Message
from vllm_mlx.api.responses_models import ResponsesRequest
from vllm_mlx.engine.chat_template_safety import chat_template_rejections
from vllm_mlx.request import ChatTemplateRejected, UnsupportedRequestOption

ROLES_MESSAGE = "Conversation roles must alternate user/assistant/user/assistant/..."

# The role check of google/gemma-3-*-it's template (the part that raises).
GEMMA_TEMPLATE = (
    "{{ bos_token }}"
    "{%- for message in messages -%}"
    "{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}"
    f"{{{{ raise_exception({ROLES_MESSAGE!r}) }}}}"
    "{%- endif -%}"
    "<start_of_turn>{{ message['role'] }}\n{{ message['content'] | trim }}"
    "<end_of_turn>\n"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<start_of_turn>model\n{%- endif -%}"
)

VALID = [{"role": "user", "content": "hi"}]
# A conversation that opens with an assistant turn: the template's own rule rejects it.
REJECTED = [
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "hello?"},
]


def _tokenizer(template: str = GEMMA_TEMPLATE):
    tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
        bos_token="<unk>",
    )
    tok.chat_template = template
    return tok


class TestContextManager:
    def test_template_raise_exception_becomes_a_rejection(self):
        with pytest.raises(ChatTemplateRejected) as exc_info:
            with chat_template_rejections():
                _tokenizer().apply_chat_template(REJECTED, tokenize=False)
        assert str(exc_info.value) == ROLES_MESSAGE
        assert isinstance(exc_info.value, UnsupportedRequestOption)  # HTTP 400

    def test_undefined_error_while_rendering_the_messages_is_a_rejection(self):
        tok = _tokenizer("{{ messages[0].nope.deeper }}")
        with pytest.raises(ChatTemplateRejected):
            with chat_template_rejections():
                tok.apply_chat_template(VALID, tokenize=False)

    def test_a_template_that_does_not_compile_stays_a_server_error(self):
        tok = _tokenizer("{% for m in messages %}")
        with pytest.raises(jinja2.TemplateSyntaxError):
            with chat_template_rejections():
                tok.apply_chat_template(VALID, tokenize=False)

    @pytest.mark.parametrize("exc", [RuntimeError("boom"), KeyError("k"), OSError("x")])
    def test_other_errors_are_not_masked_as_client_errors(self, exc):
        with pytest.raises(type(exc)):
            with chat_template_rejections():
                raise exc


@pytest.fixture
def batched_engine():
    with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=False):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine("org/gemma-like")
    engine._tokenizer = _tokenizer()
    engine._loaded = True
    return engine


@pytest.mark.anyio
class TestBatchedEngine:
    def test_valid_conversation_renders(self, batched_engine):
        prompt = batched_engine._apply_chat_template(VALID)
        assert "<start_of_turn>user\nhi<end_of_turn>" in prompt
        assert prompt.endswith("<start_of_turn>model")

    def test_rejected_conversation_carries_the_templates_message(self, batched_engine):
        with pytest.raises(ChatTemplateRejected, match="must alternate"):
            batched_engine._apply_chat_template(REJECTED)

    async def test_pre_check_renders_like_the_request(self, batched_engine):
        await batched_engine.validate_chat_template(VALID)
        with pytest.raises(ChatTemplateRejected, match="must alternate"):
            await batched_engine.validate_chat_template(REJECTED)

    async def test_chat_and_stream_chat_reject_before_generating(self, batched_engine):
        # No scheduler exists: reaching generation would be an AttributeError, not a 400.
        with pytest.raises(ChatTemplateRejected):
            await batched_engine.chat(REJECTED)
        with pytest.raises(ChatTemplateRejected):
            async for _ in batched_engine.stream_chat(REJECTED):
                pass

    async def test_broken_template_is_not_a_rejection(self, batched_engine):
        batched_engine._tokenizer = _tokenizer("{% for m in messages %}")
        with pytest.raises(jinja2.TemplateSyntaxError):
            await batched_engine.validate_chat_template(VALID)


@pytest.fixture
def simple_engine():
    from vllm_mlx.engine.simple import SimpleEngine

    engine = SimpleEngine("org/gemma-like", force_mllm=False)
    engine._model = SimpleNamespace(tokenizer=_tokenizer())
    engine._loaded = True
    return engine


@pytest.mark.anyio
class TestSimpleEngine:
    async def test_pre_check_follows_the_llm_route(self, simple_engine):
        await simple_engine.validate_chat_template(VALID)
        with pytest.raises(ChatTemplateRejected, match="must alternate"):
            await simple_engine.validate_chat_template(REJECTED)

    async def test_chat_and_stream_chat_turn_a_template_error_into_a_rejection(
        self, simple_engine, monkeypatch
    ):
        async def template_refuses(*args, **kwargs):
            _tokenizer().apply_chat_template(REJECTED, tokenize=False)
            yield  # pragma: no cover - makes this an async generator

        monkeypatch.setattr(simple_engine, "_stream_chat_impl", template_refuses)
        with pytest.raises(ChatTemplateRejected, match="must alternate"):
            async for _ in simple_engine.stream_chat(REJECTED):
                pass
        # Non-streaming chat with tools aggregates the same stream.
        with pytest.raises(ChatTemplateRejected, match="must alternate"):
            await simple_engine.chat(REJECTED, tools=[{"type": "function"}])

    async def test_blocking_chat_route_turns_a_template_error_into_a_rejection(
        self, simple_engine
    ):
        # Without tools, a plain LLM request runs ``model.chat`` in a worker thread.
        def model_chat(**kwargs):
            _tokenizer().apply_chat_template(REJECTED, tokenize=False)

        simple_engine._model.chat = model_chat
        with pytest.raises(ChatTemplateRejected, match="must alternate"):
            await simple_engine.chat(REJECTED)

    async def test_runtime_errors_in_a_stream_are_not_rejections(
        self, simple_engine, monkeypatch
    ):
        async def broken(*args, **kwargs):
            raise RuntimeError("kernel failed")
            yield  # pragma: no cover

        monkeypatch.setattr(simple_engine, "_stream_chat_impl", broken)
        with pytest.raises(RuntimeError, match="kernel failed"):
            async for _ in simple_engine.stream_chat(VALID):
                pass


@pytest.mark.anyio
class TestSimpleEngineRoutes:
    """``validate_chat_template`` follows the route ``stream_chat`` takes."""

    @pytest.fixture
    def mllm_engine(self):
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine("org/gemma-like-vlm", force_mllm=True)
        engine._loaded = True
        engine._is_mllm = True
        return engine

    def _model(self):
        from vllm_mlx.models.mllm import MLXMultimodalLM

        tok = _tokenizer()
        lm = MLXMultimodalLM.__new__(MLXMultimodalLM)
        lm.processor = SimpleNamespace(
            tokenizer=tok,
            chat_template=GEMMA_TEMPLATE,
            apply_chat_template=tok.apply_chat_template,
        )
        lm.config = SimpleNamespace(model_type="gemma3")
        lm._loaded = True
        return lm

    async def test_text_model_route_uses_the_text_tokenizer(self, mllm_engine):
        mllm_engine._text_model = object()
        mllm_engine._text_tokenizer = _tokenizer()
        mllm_engine._model = SimpleNamespace()  # reaching it would be an AttributeError
        await mllm_engine.validate_chat_template(VALID)
        with pytest.raises(ChatTemplateRejected, match="must alternate"):
            await mllm_engine.validate_chat_template(REJECTED)

    async def test_mllm_route_renders_through_the_processor(self, mllm_engine):
        mllm_engine._text_model = None
        mllm_engine._model = self._model()
        await mllm_engine.validate_chat_template(VALID)
        with pytest.raises(ChatTemplateRejected, match="must alternate"):
            await mllm_engine.validate_chat_template(REJECTED)

    async def test_media_requests_are_pre_checked_too(self, mllm_engine):
        mllm_engine._text_model = None
        mllm_engine._model = self._model()
        video = {"type": "video_url", "video_url": {"url": "file:///x.mp4"}}
        image = {"type": "image_url", "image_url": {"url": "file:///x.png"}}
        for part in (video, image):
            messages = [
                {
                    "role": "assistant",
                    "content": [part, {"type": "text", "text": "hi"}],
                },
                {"role": "user", "content": "hello?"},
            ]
            with pytest.raises(ChatTemplateRejected, match="must alternate"):
                await mllm_engine.validate_chat_template(messages)

    async def test_a_video_only_turn_still_counts_as_a_turn(self, mllm_engine):
        # Dropping the turn (no frames, no text) would hide the bad order.
        mllm_engine._text_model = None
        mllm_engine._model = self._model()
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "video_url", "video_url": {"url": "file:///x.mp4"}}
                ],
            },
            {"role": "user", "content": "hello?"},
        ]
        with pytest.raises(ChatTemplateRejected, match="must alternate"):
            await mllm_engine.validate_chat_template(messages)


class TestMultimodalFormatter:
    def _model(self, processor):
        from vllm_mlx.models.mllm import MLXMultimodalLM

        lm = MLXMultimodalLM.__new__(MLXMultimodalLM)
        lm.processor = processor
        lm.config = SimpleNamespace(model_type="gemma3")
        lm._loaded = True
        return lm

    def test_template_rejection_is_not_answered_with_the_last_user_message(self):
        processor = SimpleNamespace(
            tokenizer=_tokenizer(),
            chat_template=GEMMA_TEMPLATE,
            apply_chat_template=_tokenizer().apply_chat_template,
        )
        lm = self._model(processor)
        with pytest.raises(jinja2.TemplateError, match="must alternate"):
            lm._format_chat_prompt(REJECTED, None, {}, True)

    def test_a_failure_that_is_not_a_template_rejection_is_not_hidden(self):
        # The conversation must never be answered by prompting with its last user
        # message alone: a processor that fails for any reason fails the request.
        class BrokenProcessor:
            chat_template = "x"
            tokenizer = None

            def apply_chat_template(self, *args, **kwargs):
                raise KeyError("processor bug")

        lm = self._model(BrokenProcessor())
        with pytest.raises(KeyError, match="processor bug"):
            lm._format_chat_prompt(
                [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}],
                None,
                {},
                True,
            )

    def test_pre_check_runs_the_requests_message_preparation(self):
        tok = _tokenizer()
        processor = SimpleNamespace(
            tokenizer=tok,
            chat_template=GEMMA_TEMPLATE,
            apply_chat_template=tok.apply_chat_template,
        )
        lm = self._model(processor)
        assert "<start_of_turn>user\n" in lm.render_chat_prompt(VALID)
        with pytest.raises(jinja2.TemplateError, match="must alternate"):
            lm.render_chat_prompt(REJECTED)


@pytest.fixture
def served(monkeypatch, batched_engine):
    async def acquire(raw_request, **kwargs):
        return batched_engine

    async def release(*args, **kwargs):
        return None

    async def no_preflight(raw_request):
        return None

    monkeypatch.setattr(server, "_validate_model_name", lambda _m: None)
    monkeypatch.setattr(server, "_acquire_default_engine_for_request", acquire)
    monkeypatch.setattr(server, "_release_default_engine", release)
    monkeypatch.setattr(server, "_release_engine_for_request", release)
    monkeypatch.setattr(server, "get_engine", lambda: batched_engine)
    monkeypatch.setattr(server, "_model_manager", None)
    monkeypatch.setattr(server, "_preflight_response_format_backend", no_preflight)
    monkeypatch.setattr(server, "_model_name", "served")
    monkeypatch.setattr(server, "_default_timeout", 30.0)
    monkeypatch.setattr(server, "_default_max_tokens", 64)
    monkeypatch.setattr(server, "_max_request_tokens", 128)
    return batched_engine


@pytest.mark.anyio
class TestEndpoints:
    """Every chat endpoint, streaming or not: 400 with the template's message."""

    @pytest.mark.parametrize("stream", [False, True])
    async def test_chat_completions(self, served, stream):
        request = ChatCompletionRequest(
            model="served",
            messages=[Message(role=m["role"], content=m["content"]) for m in REJECTED],
            stream=stream,
        )
        with pytest.raises(server.HTTPException) as exc_info:
            await server.create_chat_completion(request, raw_request=None)
        assert exc_info.value.status_code == 400
        assert ROLES_MESSAGE in exc_info.value.detail

    @pytest.mark.parametrize("stream", [False, True])
    async def test_responses(self, served, stream):
        request = ResponsesRequest(
            model="served",
            input=[
                {"type": "message", "role": m["role"], "content": m["content"]}
                for m in REJECTED
            ],
            stream=stream,
        )
        with pytest.raises(server.HTTPException) as exc_info:
            await server.create_response(request, raw_request=None)
        assert exc_info.value.status_code == 400
        assert ROLES_MESSAGE in exc_info.value.detail

    @pytest.mark.parametrize("stream", [False, True])
    async def test_anthropic_messages(self, served, stream):
        body = {
            "model": "served",
            "max_tokens": 16,
            "messages": REJECTED,
            "stream": stream,
        }
        with pytest.raises(server.HTTPException) as exc_info:
            await server.create_anthropic_message(_FakeRequest(body))
        assert exc_info.value.status_code == 400
        assert ROLES_MESSAGE in exc_info.value.detail

    async def test_a_server_error_is_still_a_500_not_a_400(self, served, monkeypatch):
        async def broken(*args, **kwargs):
            raise RuntimeError("kernel failed")

        monkeypatch.setattr(served, "validate_chat_template", broken)
        request = ChatCompletionRequest(
            model="served", messages=[Message(role="user", content="hi")], stream=True
        )
        with pytest.raises(RuntimeError, match="kernel failed"):
            await server.create_chat_completion(request, raw_request=None)


class _FakeRequest:
    """Enough of a Starlette request for the endpoints under test."""

    headers: dict = {}
    state = SimpleNamespace()

    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


# A template that refuses a request carrying tools, naming the first tool: the
# pre-check rejects it only if it hands the template the request's converted tools.
TOOLS_TEMPLATE = (
    "{%- if tools -%}"
    "{{ raise_exception('template refuses tool ' ~ tools[0]['function']['name']) }}"
    "{%- endif -%}"
    "{{ messages[0]['content'] }}"
)
WEATHER_TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object"}},
}


@pytest.mark.anyio
class TestPreCheckSeesTheRequestInputs:
    """The pre-check must render with the tools / thinking flags the request uses."""

    async def _expect_tools_reach_the_template(self, engine):
        await engine.validate_chat_template(VALID)
        with pytest.raises(ChatTemplateRejected, match="refuses tool get_weather"):
            await engine.validate_chat_template(VALID, tools=[WEATHER_TOOL])

    async def test_batched_engine(self, batched_engine):
        batched_engine._tokenizer = _tokenizer(TOOLS_TEMPLATE)
        await self._expect_tools_reach_the_template(batched_engine)

    async def test_simple_engine_llm_route(self, simple_engine):
        simple_engine._model = SimpleNamespace(tokenizer=_tokenizer(TOOLS_TEMPLATE))
        await self._expect_tools_reach_the_template(simple_engine)

    async def test_simple_engine_text_model_route(self):
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine("org/gemma-like-vlm", force_mllm=True)
        engine._loaded = True
        engine._is_mllm = True
        engine._text_model = object()
        engine._text_tokenizer = _tokenizer(TOOLS_TEMPLATE)
        engine._model = SimpleNamespace()  # reaching it would be an AttributeError
        await self._expect_tools_reach_the_template(engine)

    async def test_simple_engine_mllm_route(self):
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine("org/gemma-like-vlm", force_mllm=True)
        engine._loaded = True
        engine._is_mllm = True
        engine._text_model = None
        engine._model = _mllm_model(TOOLS_TEMPLATE, model_type="qwen2_vl")
        await self._expect_tools_reach_the_template(engine)


def _mllm_model(template: str, model_type: str = "gemma3"):
    from vllm_mlx.models.mllm import MLXMultimodalLM

    tok = _tokenizer(template)
    lm = MLXMultimodalLM.__new__(MLXMultimodalLM)
    lm.processor = SimpleNamespace(
        tokenizer=tok,
        chat_template=template,
        apply_chat_template=tok.apply_chat_template,
    )
    lm.config = SimpleNamespace(model_type=model_type)
    lm._loaded = True
    return lm


class TestMultimodalPreCheckPreparation:
    """``render_chat_prompt`` prepares messages and flags exactly like ``chat``."""

    def test_tool_turns_are_serialized_for_a_template_without_a_tool_role(self):
        # Gemma's template knows no ``tool`` role. ``chat`` rewrites tool turns to
        # user text first; a pre-check that skipped this would 400 a request that
        # is in fact served.
        lm = _mllm_model(GEMMA_TEMPLATE)
        conversation = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"},
        ]
        prompt = lm.render_chat_prompt(conversation)
        assert "get_weather()" in prompt
        assert "tool_output" in prompt and "sunny" in prompt

    @pytest.mark.parametrize("enable_thinking", [None, True, False])
    def test_enable_thinking_nested_in_chat_template_kwargs_wins(self, enable_thinking):
        lm = _mllm_model("{{ messages[0]['content'] }}|thinking={{ enable_thinking }}")
        render = lm.render_chat_prompt
        assert render(VALID).endswith("|thinking=True")  # chat's default
        assert render(VALID, enable_thinking=False).endswith("|thinking=False")
        nested = {"enable_thinking": False}
        assert render(
            VALID, enable_thinking=enable_thinking, chat_template_kwargs=nested
        ).endswith("|thinking=False")
        assert nested == {"enable_thinking": False}  # the caller's dict is not consumed


@pytest.mark.anyio
class TestPreCheckBusyEngine:
    """An engine that is busy initialising its text route answers 503, not 500."""

    @pytest.mark.parametrize("stream_endpoint", ["chat", "responses", "anthropic"])
    async def test_busy_engine_is_a_retryable_503(
        self, served, monkeypatch, stream_endpoint
    ):
        from vllm_mlx.engine.base import EngineBusy

        async def busy(*args, **kwargs):
            raise EngineBusy("text route is busy")

        monkeypatch.setattr(served, "validate_chat_template", busy)
        with pytest.raises(server.HTTPException) as exc_info:
            if stream_endpoint == "chat":
                request = ChatCompletionRequest(
                    model="served",
                    messages=[Message(role="user", content="hi")],
                    stream=True,
                )
                await server.create_chat_completion(request, raw_request=None)
            elif stream_endpoint == "responses":
                request = ResponsesRequest(model="served", input="hi", stream=True)
                await server.create_response(request, raw_request=None)
            else:
                body = {
                    "model": "served",
                    "max_tokens": 16,
                    "messages": VALID,
                    "stream": True,
                }
                await server.create_anthropic_message(_FakeRequest(body))
        assert exc_info.value.status_code == 503
        assert exc_info.value.detail["error"] == "text_generation_busy"
