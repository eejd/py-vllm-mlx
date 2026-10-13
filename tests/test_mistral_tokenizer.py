# SPDX-License-Identifier: Apache-2.0
"""Mistral ``tekken`` checkpoints must be prompted with the ids mistral-common builds.

``transformers`` loads ``MistralCommonBackend`` for a checkpoint with ``tekken.json``.
This server renders the chat to a string and encodes that string later, and that
backend encodes a string as plain text: ``<s>[INST]hi[/INST]`` becomes 16 ordinary
tokens instead of ``[1, 3, ..., 4]``. ``utils/mistral_tokenizer.py`` selects the
checkpoint's equivalent ``tokenizer.json`` tokenizer instead.

The synthetic checkpoint below is a real tekken tokenizer (byte-level vocabulary,
the Mistral control tokens, a Mistral-style chat template), small enough to build
per test run. ``TestRealDevstral`` repeats the id comparison on the Devstral Small 2
tokenizer files from the local Hugging Face cache and skips when they are absent.
"""

import base64
import glob
import json
import logging
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")
pytest.importorskip("mistral_common")

from transformers import AutoTokenizer  # noqa: E402

from vllm_mlx.utils import mistral_tokenizer as mt  # noqa: E402

N_SPECIAL = 100
_NAMED = {
    0: "<unk>",
    1: "<s>",
    2: "</s>",
    3: "[INST]",
    4: "[/INST]",
    5: "[AVAILABLE_TOOLS]",
    6: "[/AVAILABLE_TOOLS]",
    7: "[TOOL_RESULTS]",
    8: "[/TOOL_RESULTS]",
    9: "[TOOL_CALLS]",
    10: "[IMG]",
    11: "<pad>",
    12: "[IMG_BREAK]",
    13: "[IMG_END]",
    14: "[PREFIX]",
    15: "[MIDDLE]",
    16: "[SUFFIX]",
    17: "[SYSTEM_PROMPT]",
    18: "[/SYSTEM_PROMPT]",
    19: "[TOOL_CONTENT]",
    24: "[AUDIO]",
    25: "[BEGIN_AUDIO]",
    32: "[ARGS]",
    33: "[CALL_ID]",
}
_PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# Renders what mistral-common renders for system/user/assistant turns.
_TEMPLATE = (
    "{{ bos_token }}"
    "{% set ms = messages %}"
    "{% if ms[0]['role'] == 'system' %}"
    "[SYSTEM_PROMPT]{{ ms[0]['content'] }}[/SYSTEM_PROMPT]{% set ms = ms[1:] %}"
    "{% endif %}"
    "{% for m in ms %}"
    "{% if m['role'] == 'user' %}[INST]{{ m['content'] }}[/INST]"
    "{% else %}{{ m['content'] }}{{ eos_token }}{% endif %}"
    "{% endfor %}"
)


def _byte_to_unicode() -> dict[int, str]:
    keep = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    chars = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, map(chr, chars)))


def build_tekken_checkpoint(path: Path, template: str | None = _TEMPLATE) -> Path:
    """Write a tekken.json + tokenizer.json pair describing the same tokenizer."""
    path.mkdir(parents=True, exist_ok=True)
    names = [_NAMED.get(i, f"<SPECIAL_{i}>") for i in range(N_SPECIAL)]
    tekken = {
        "config": {
            "pattern": _PATTERN,
            "num_vocab_tokens": 256,
            "default_vocab_size": N_SPECIAL + 256,
            "default_num_special_tokens": N_SPECIAL,
            "version": "v13",
        },
        "vocab": [
            {
                "rank": i,
                "token_bytes": base64.b64encode(bytes([i])).decode(),
                "token_str": None,
            }
            for i in range(256)
        ],
        "special_tokens": [
            {"rank": i, "token_str": n, "is_control": True} for i, n in enumerate(names)
        ],
        "image": {
            "image_patch_size": 14,
            "max_image_size": 1540,
            "spatial_merge_size": 2,
        },
    }
    (path / "tekken.json").write_text(json.dumps(tekken))
    b2u = _byte_to_unicode()
    tokenizer = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": i,
                "content": n,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for i, n in enumerate(names)
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"Regex": _PATTERN},
                    "behavior": "Isolated",
                    "invert": False,
                },
                {
                    "type": "ByteLevel",
                    "add_prefix_space": False,
                    "trim_offsets": True,
                    "use_regex": False,
                },
            ],
        },
        "post_processor": None,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": True,
            "trim_offsets": True,
            "use_regex": True,
        },
        "model": {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": None,
            "end_of_word_suffix": None,
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": True,
            "vocab": {
                **{n: i for i, n in enumerate(names)},
                **{b2u[i]: N_SPECIAL + i for i in range(256)},
            },
            "merges": [],
        },
    }
    (path / "tokenizer.json").write_text(json.dumps(tokenizer))
    (path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "backend": "tokenizers",
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
                "tokenizer_class": "TokenizersBackend",
            }
        )
    )
    (path / "config.json").write_text(
        json.dumps({"model_type": "mistral", "eos_token_id": 2, "bos_token_id": 1})
    )
    if template is not None:
        (path / "chat_template.jinja").write_text(template)
    return path


CONVERSATIONS = {
    "plain": [{"role": "user", "content": "Reply with exactly: OK"}],
    "system": [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi there"},
    ],
    "multi_turn": [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c é 世界 😀\n\n  code:\n\tx=1\r\n"},
    ],
}


def _ids(encoded):
    return encoded["input_ids"] if not isinstance(encoded, list) else encoded


def reference_ids(reference, messages, **kwargs):
    """What mistral-common builds: render and encode in one step."""
    return _ids(
        reference.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, **kwargs
        )
    )


def string_pipeline_ids(tokenizer, messages, **kwargs):
    """What this server does: render to a string, encode that string afterwards."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
    return tokenizer.encode(text, add_special_tokens=False)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    return build_tekken_checkpoint(tmp_path_factory.mktemp("tekken"))


@pytest.fixture(scope="module")
def reference(checkpoint):
    tokenizer = mt.reference_tokenizer(checkpoint)
    assert tokenizer is not None, "transformers did not pick MistralCommonBackend"
    return tokenizer


class TestFastTokenizerSelection:
    @pytest.mark.parametrize("name", sorted(CONVERSATIONS))
    def test_rendered_prompt_encodes_to_the_reference_ids(
        self, checkpoint, reference, name
    ):
        config = mt.fast_tokenizer_config(checkpoint, {"trust_remote_code": False})
        assert config["mistral_format"] is False
        assert config["trust_remote_code"] is False
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, **config)

        messages = CONVERSATIONS[name]
        expected = reference_ids(reference, messages)
        assert expected[0] == 1  # <s> is a single id, not the text "<s>"
        assert string_pipeline_ids(tokenizer, messages) == expected

    def test_default_tokenizer_misreads_the_rendered_prompt(self, reference):
        # The defect: the backend transformers picks encodes the string as text.
        messages = CONVERSATIONS["plain"]
        assert string_pipeline_ids(reference, messages) != reference_ids(
            reference, messages
        )

    def test_unrelated_checkpoint_is_left_alone(self, tmp_path):
        build_tekken_checkpoint(tmp_path)
        (tmp_path / "tekken.json").unlink()
        config = {"trust_remote_code": True}
        assert mt.fast_tokenizer_config(tmp_path, config) is config
        assert mt.fast_tokenizer_config(tmp_path, None) is None

    def test_keeps_mistral_common_without_tokenizer_json(self, tmp_path, caplog):
        build_tekken_checkpoint(tmp_path)
        (tmp_path / "tokenizer.json").unlink()
        config = {}
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            assert mt.fast_tokenizer_config(tmp_path, config) is config
        assert "no tokenizer.json" in caplog.text

    def test_keeps_mistral_common_without_chat_template(self, tmp_path):
        build_tekken_checkpoint(tmp_path, template=None)
        assert mt.fast_tokenizer_config(tmp_path, None) is None

    def test_keeps_mistral_common_when_token_ids_disagree(self, tmp_path, caplog):
        build_tekken_checkpoint(tmp_path)
        path = tmp_path / "tokenizer.json"
        data = json.loads(path.read_text())
        vocab = data["model"]["vocab"]
        b2u = _byte_to_unicode()
        a, b = b2u[ord("a")], b2u[ord("b")]
        vocab[a], vocab[b] = vocab[b], vocab[a]  # 'a' and 'b' trade ids
        path.write_text(json.dumps(data))
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            assert mt.fast_tokenizer_config(tmp_path, None) is None
        assert "not equivalent" in caplog.text


class _StubTokenizer:
    """The slice of a tokenizer ``_mismatch`` looks at."""

    is_fast = True
    chat_template = "{{ messages }}"
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, texts=("<unk>", "<s>", "</s>", "a"), encode=None):
        self._texts = list(texts)
        self._encode = encode or (lambda text: [ord(c) for c in text])

    def __len__(self):
        return len(self._texts)

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self._texts[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return self._encode(text)


class TestEquivalenceCheck:
    """Each way a tokenizer.json can differ from tekken.json is caught."""

    def test_identical_tokenizers_match(self):
        assert mt._mismatch(_StubTokenizer(), _StubTokenizer()) is None

    @pytest.mark.parametrize(
        "change, reason",
        [
            ({"is_fast": False}, "not a fast tokenizer"),
            ({"chat_template": None}, "no chat template"),
            ({"bos_token_id": 0}, "bos_token_id differs"),
            ({"eos_token_id": 3}, "eos_token_id differs"),
        ],
    )
    def test_attribute_differences(self, change, reason):
        fast = _StubTokenizer()
        for name, value in change.items():
            setattr(fast, name, value)
        assert reason in mt._mismatch(fast, _StubTokenizer())

    def test_vocabulary_size_difference(self):
        fast = _StubTokenizer(texts=("<unk>", "<s>", "</s>", "a", "b"))
        assert "vocabulary size" in mt._mismatch(fast, _StubTokenizer())

    def test_token_text_difference(self):
        fast = _StubTokenizer(texts=("<unk>", "<s>", "</s>", "b"))
        assert mt._mismatch(fast, _StubTokenizer()) == "token texts differ"

    def test_encoding_difference_in_plain_text(self):
        # Same vocabulary, but this one never splits digits: only the probes tell.
        fast = _StubTokenizer(encode=lambda text: [len(text)])
        assert "encoding of" in mt._mismatch(fast, _StubTokenizer())



class TestAdoptFastTokenizer:
    def test_replaces_tokenizer_and_rewires_streaming_and_stopping(
        self, checkpoint, reference
    ):
        processor = SimpleNamespace(tokenizer=reference)
        assert mt.adopt_fast_tokenizer(processor, checkpoint, eos_token_ids=2)

        tokenizer = processor.tokenizer
        assert tokenizer is not reference and tokenizer.is_fast
        messages = CONVERSATIONS["system"]
        assert string_pipeline_ids(tokenizer, messages) == reference_ids(
            reference, messages
        )
        # mlx_vlm.generate reads both of these off the processor.
        assert tokenizer.stopping_criteria.eos_token_ids == [2]
        assert tokenizer.stopping_criteria.tokenizer is tokenizer
        detok = processor.detokenizer
        for token in tokenizer.encode("é 世界", add_special_tokens=False):
            detok.add_token(token)
        detok.finalize()
        assert detok.text == "é 世界"

    def test_other_tokenizers_are_untouched(self):
        tokenizer = SimpleNamespace(name="plain hf tokenizer")
        processor = SimpleNamespace(tokenizer=tokenizer)
        assert not mt.adopt_fast_tokenizer(processor, "/nonexistent")
        assert processor.tokenizer is tokenizer
        assert not hasattr(processor, "detokenizer")
        assert not mt.is_mistral_common_tokenizer(tokenizer)

    def test_keeps_tokenizer_when_no_equivalent_exists(self, tmp_path):
        build_tekken_checkpoint(tmp_path)
        (tmp_path / "tokenizer.json").unlink()
        reference = mt.reference_tokenizer(tmp_path)
        processor = SimpleNamespace(tokenizer=reference)
        assert not mt.adopt_fast_tokenizer(processor, tmp_path, eos_token_ids=2)
        assert processor.tokenizer is reference


class TestLoadSites:
    def test_mllm_load_serves_the_fast_tokenizer(
        self, checkpoint, reference, monkeypatch
    ):
        import mlx_vlm
        import mlx_vlm.utils

        from vllm_mlx.models.mllm import MLXMultimodalLM

        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))
        monkeypatch.setattr(
            mlx_vlm, "load", lambda name: (model, SimpleNamespace(tokenizer=reference))
        )
        monkeypatch.setattr(mlx_vlm.utils, "get_model_path", lambda name: checkpoint)
        monkeypatch.setattr(mlx_vlm.utils, "load_config", lambda name: {})

        engine = MLXMultimodalLM("tekken-test")
        engine.load()

        messages = CONVERSATIONS["multi_turn"]
        assert string_pipeline_ids(
            engine.processor.tokenizer, messages
        ) == reference_ids(reference, messages)

    def test_llm_load_serves_the_fast_tokenizer(
        self, checkpoint, reference, monkeypatch
    ):
        import mlx_lm
        from mlx_lm.tokenizer_utils import load as load_tokenizer

        from vllm_mlx.utils.tokenizer import load_model_with_fallback

        monkeypatch.setattr(
            mlx_lm,
            "load",
            lambda name, tokenizer_config=None: (
                None,
                load_tokenizer(Path(name), tokenizer_config),
            ),
        )
        _, tokenizer = load_model_with_fallback(str(checkpoint))

        messages = CONVERSATIONS["multi_turn"]
        assert string_pipeline_ids(tokenizer, messages) == reference_ids(
            reference, messages
        )


_HUB = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
)
_DEVSTRAL = "models--mlx-community--Devstral-Small-2-24B-Instruct-2512-8bit"


def _devstral_snapshot() -> str:
    candidates = [os.environ.get("VLLM_MLX_DEVSTRAL_DIR")]
    for hub in (_HUB, "/hive-cache/models/hub"):
        candidates += sorted(glob.glob(os.path.join(hub, _DEVSTRAL, "snapshots", "*")))
    for path in candidates:
        if path and all(
            os.path.exists(os.path.join(path, name))
            for name in ("tekken.json", "tokenizer.json", "chat_template.jinja")
        ):
            return path
    pytest.skip(
        f"{_DEVSTRAL} tokenizer files not found (set VLLM_MLX_DEVSTRAL_DIR or HF_HOME): "
        "the fixture tests above cover the code path, not this tokenizer"
    )


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "echo_args",
            "description": "Echo",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"},
                    "flag": {"type": "boolean"},
                },
                "required": ["text", "count", "flag"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
]


def _call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _server_shaped(messages):
    """Messages as ``api/utils.extract_multimodal_content`` hands them to a template:
    ``content`` never None, tool-call ``arguments`` parsed to a mapping."""
    shaped = json.loads(json.dumps(messages))
    for message in shaped:
        if message.get("content") is None:
            message["content"] = ""
        for call in message.get("tool_calls") or []:
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                call["function"]["arguments"] = json.loads(arguments)
    return shaped


_ARGS = json.dumps({"text": "hello world", "count": 42, "flag": True})
DEVSTRAL_CASES = {
    "plain": (CONVERSATIONS["plain"], None),
    "system": (CONVERSATIONS["system"], None),
    "multi_turn_unicode": (CONVERSATIONS["multi_turn"], None),
    "tools": (
        [
            {
                "role": "user",
                "content": "Call echo_args with text 'hello world', count 42 and flag true.",
            }
        ],
        _TOOLS[:1],
    ),
    "tools_and_system": (
        [{"role": "system", "content": "S"}, {"role": "user", "content": "go"}],
        _TOOLS,
    ),
    "tool_result": (
        [
            {"role": "user", "content": "Call echo"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_call("abc123XYZ", "echo_args", _ARGS)],
            },
            {"role": "tool", "tool_call_id": "abc123XYZ", "content": '{"ok": true}'},
        ],
        _TOOLS[:1],
    ),
    "parallel_calls_then_followup": (
        [
            {"role": "user", "content": "both"},
            {
                "role": "assistant",
                "content": "calling",
                "tool_calls": [
                    _call("aaaaaaaa1", "echo_args", _ARGS),
                    _call("bbbbbbbb2", "get_weather", json.dumps({"city": "Paris"})),
                ],
            },
            {"role": "tool", "tool_call_id": "aaaaaaaa1", "content": "r1"},
            {"role": "tool", "tool_call_id": "bbbbbbbb2", "content": "r2"},
            {"role": "assistant", "content": "All done."},
            {"role": "user", "content": "thanks"},
        ],
        _TOOLS,
    ),
}


class TestRealDevstral:
    @pytest.fixture(scope="class")
    def snapshot(self):
        return _devstral_snapshot()

    @pytest.fixture(scope="class")
    def real_reference(self, snapshot):
        tokenizer = mt.reference_tokenizer(snapshot)
        assert tokenizer is not None
        return tokenizer

    @pytest.fixture(scope="class")
    def real_fast(self, snapshot, real_reference):
        tokenizer = mt.equivalent_fast_tokenizer(snapshot, real_reference)
        assert tokenizer is not None, "Devstral tokenizer.json is not equivalent"
        return tokenizer

    @pytest.mark.parametrize("name", sorted(DEVSTRAL_CASES))
    def test_rendered_prompt_encodes_to_the_mistral_common_ids(
        self, real_reference, real_fast, name
    ):
        messages, tools = DEVSTRAL_CASES[name]
        kwargs = {"tools": tools} if tools else {}
        expected = reference_ids(real_reference, messages, **kwargs)
        # the server passes enable_thinking to every template
        got = string_pipeline_ids(
            real_fast, _server_shaped(messages), enable_thinking=True, **kwargs
        )
        assert got == expected
        assert got[0] == real_fast.bos_token_id

    def test_eos_is_the_generation_stop_token(self, real_fast, snapshot):
        from vllm_mlx.utils.tokenizer import collect_eos_token_ids

        assert real_fast.eos_token_id == 2
        assert collect_eos_token_ids(real_fast) == {2}
