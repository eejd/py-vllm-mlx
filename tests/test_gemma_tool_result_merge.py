# SPDX-License-Identifier: Apache-2.0
"""A tool result and the user turn right after it are one Gemma user turn.

Gemma's template knows only user/model turns and raises "Conversation roles must
alternate" on two user turns in a row. The tool result becomes a user turn
(``tool_output`` block), so the user's next message has to be merged into it, whatever
shape that message has: text, a list of parts (image, audio, video, text) or no content.
"""

import glob
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_mlx.models.mllm import (
    _build_mllm_chat_messages,
    _serialize_tool_messages_gemma_native,
)

IMAGE_URL = "data:image/png;base64,AAAA"
IMAGE = {"type": "image_url", "image_url": {"url": IMAGE_URL}}
BLOCK = "```tool_output\nSunny\n```"
TOOL_CONVERSATION = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "Sunny"},
]


def _serialize(next_turn):
    return _serialize_tool_messages_gemma_native([*TOOL_CONVERSATION, next_turn])


class TestSerializer:
    def test_text_content_gets_the_block_prepended(self):
        out = _serialize({"role": "user", "content": "And Rome?"})
        assert [m["role"] for m in out] == ["user", "assistant", "user"]
        assert out[-1]["content"] == f"{BLOCK}\n\nAnd Rome?"

    def test_empty_text_content_is_the_block_alone(self):
        out = _serialize({"role": "user", "content": ""})
        assert out[-1]["content"] == BLOCK

    def test_list_content_gets_the_block_as_a_leading_text_part(self):
        text = {"type": "text", "text": "what is this?"}
        out = _serialize({"role": "user", "content": [IMAGE, text]})
        assert [m["role"] for m in out] == ["user", "assistant", "user"]
        assert out[-1]["content"] == [{"type": "text", "text": BLOCK}, IMAGE, text]

    def test_image_only_list_content(self):
        out = _serialize({"role": "user", "content": [IMAGE]})
        assert [m["role"] for m in out] == ["user", "assistant", "user"]
        assert out[-1]["content"] == [{"type": "text", "text": BLOCK}, IMAGE]

    def test_empty_list_content(self):
        out = _serialize({"role": "user", "content": []})
        assert out[-1]["content"] == [{"type": "text", "text": BLOCK}]

    @pytest.mark.parametrize("missing", ["none", "absent"])
    def test_user_turn_without_content_is_the_block_alone(self, missing):
        turn = (
            {"role": "user", "content": None} if missing == "none" else {"role": "user"}
        )
        out = _serialize(turn)
        assert [m["role"] for m in out] == ["user", "assistant", "user"]
        assert out[-1]["content"] == BLOCK

    def test_parallel_results_share_the_one_merged_turn(self):
        out = _serialize_tool_messages_gemma_native(
            [
                *TOOL_CONVERSATION,
                {"role": "tool", "tool_call_id": "c2", "content": "Rainy"},
                {"role": "user", "content": [IMAGE]},
            ]
        )
        assert out[-1]["content"][0]["text"] == ("```tool_output\nSunny\nRainy\n```")
        assert [m["role"] for m in out] == ["user", "assistant", "user"]

    def test_the_callers_message_is_not_modified(self):
        parts = [IMAGE]
        turn = {"role": "user", "content": parts}
        _serialize(turn)
        assert turn == {"role": "user", "content": [IMAGE]} and parts == [IMAGE]

    def test_extra_fields_of_the_user_turn_survive(self):
        out = _serialize({"role": "user", "content": [IMAGE], "name": "bob"})
        assert out[-1]["name"] == "bob"

    def test_a_tool_result_followed_by_an_assistant_turn_stays_its_own_user_turn(self):
        out = _serialize_tool_messages_gemma_native(
            [*TOOL_CONVERSATION, {"role": "assistant", "content": "It is sunny."}]
        )
        assert [m["role"] for m in out] == ["user", "assistant", "user", "assistant"]
        assert out[2]["content"] == BLOCK

    def test_a_user_turn_without_a_pending_result_is_untouched(self):
        turn = {"role": "user", "content": [IMAGE]}
        assert _serialize_tool_messages_gemma_native([turn]) == [turn]


def _gemma3_snapshot():
    paths = glob.glob(
        "/hive-cache/*/models/hub/models--mlx-community--gemma-3-4b-it-qat-4bit/"
        "snapshots/*"
    ) + glob.glob(
        "~/.cache/huggingface/hub/models--mlx-community--gemma-3-4b-it-qat-4bit/"
        "snapshots/*"
    )
    return paths[0] if paths else None


NEXT_TURNS = {
    "text": {"role": "user", "content": "And Rome?"},
    "image+text": {
        "role": "user",
        "content": [IMAGE, {"type": "text", "text": "what is this?"}],
    },
    "image only": {"role": "user", "content": [IMAGE]},
    "none": {"role": "user", "content": None},
}


@pytest.mark.skipif(
    _gemma3_snapshot() is None, reason="Gemma-3 tokenizer not in the local cache"
)
class TestWithTheRealGemma3Template:
    """Gemma-3's own template, which refuses two user turns in a row."""

    @pytest.fixture(scope="class")
    def processor(self):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(_gemma3_snapshot())
        return SimpleNamespace(
            tokenizer=tok,
            chat_template=None,
            apply_chat_template=tok.apply_chat_template,
        )

    def _assert_one_merged_turn(self, prompt):
        assert "Conversation roles must alternate" not in prompt
        assert prompt.count("<start_of_turn>user") == 2
        assert prompt.count("<start_of_turn>model") == 2  # the call, then the reply
        assert "<start_of_turn>user\n```tool_output\nSunny\n```" in prompt
        assert prompt.endswith("<start_of_turn>model\n")

    @pytest.mark.parametrize("kind", NEXT_TURNS)
    def test_the_request_route_renders(self, processor, kind):
        from mlx_vlm.prompt_utils import get_chat_template

        next_turn = NEXT_TURNS[kind]
        messages = _serialize_tool_messages_gemma_native(
            [*TOOL_CONVERSATION, next_turn]
        )
        chat_messages = _build_mllm_chat_messages(
            messages,
            all_image_urls=[IMAGE_URL] if "image" in kind else [],
            video_frame_counts={},
        )
        prompt = get_chat_template(processor, chat_messages, add_generation_prompt=True)
        self._assert_one_merged_turn(prompt)

    @pytest.mark.parametrize("kind", NEXT_TURNS)
    def test_the_batched_engine_renders(self, processor, kind):
        with patch("vllm_mlx.engine.batched.is_mllm_model", return_value=True):
            from vllm_mlx.engine.batched import BatchedEngine

            engine = BatchedEngine("org/gemma-3-vlm")
        engine._processor = processor
        engine._is_mllm = True
        engine._mllm_instance = SimpleNamespace(
            config=SimpleNamespace(model_type="gemma3")
        )
        engine._loaded = True
        self._assert_one_merged_turn(
            engine._apply_chat_template([*TOOL_CONVERSATION, NEXT_TURNS[kind]])
        )

    def test_without_the_merge_the_template_refuses(self, processor):
        # The failure this guards against, on the same conversation.
        from mlx_vlm.prompt_utils import get_chat_template

        chat_messages = _build_mllm_chat_messages(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {"role": "assistant", "content": "```tool_code\nf()\n```"},
                {"role": "user", "content": BLOCK},
                NEXT_TURNS["image only"],
            ],
            all_image_urls=[IMAGE_URL],
            video_frame_counts={},
        )
        with pytest.raises(Exception, match="roles must alternate"):
            get_chat_template(processor, chat_messages, add_generation_prompt=True)
