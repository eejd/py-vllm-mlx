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
import functools
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
# Renders what mistral-common renders for system/user/assistant turns; an image part
# is the ``[IMG]`` placeholder the processor expands.
_TEMPLATE = (
    "{{ bos_token }}"
    "{% set ms = messages %}"
    "{% if ms[0]['role'] == 'system' %}"
    "[SYSTEM_PROMPT]{{ ms[0]['content'] }}[/SYSTEM_PROMPT]{% set ms = ms[1:] %}"
    "{% endif %}"
    "{% for m in ms %}"
    "{% if m['role'] == 'user' %}[INST]"
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for p in m['content'] %}"
    "{% if p['type'] == 'image' %}[IMG]{% else %}{{ p['text'] }}{% endif %}"
    "{% endfor %}{% endif %}"
    "[/INST]"
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


# The pre-fix pattern of Mistral-Small-3.1's tokenizer.json (Llama-3 style: groups
# up to three digits, where tekken encodes each digit on its own).
_OLD_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def build_tekken_checkpoint(
    path: Path,
    template: str | None = _TEMPLATE,
    *,
    merged_digits: bool = False,
    wrong_regex: bool = False,
    large_vocab: bool = False,
    model_type: str = "mistral",
) -> Path:
    """Write a tekken.json + tokenizer.json pair describing the same tokenizer.

    ``merged_digits`` adds the token ``12`` (so the digit-splitting rule of the
    pre-tokenizer shows in the ids); ``wrong_regex`` gives tokenizer.json the old
    Llama-3 style pattern that ``fix_mistral_regex`` repairs; ``large_vocab`` pads
    the vocabulary past 100000 tokens, which is when ``transformers`` applies that
    repair at all (3-byte runs of UTF-8 continuation bytes: never in valid text).
    """
    path.mkdir(parents=True, exist_ok=True)
    names = [_NAMED.get(i, f"<SPECIAL_{i}>") for i in range(N_SPECIAL)]
    b2u = _byte_to_unicode()
    pieces = [bytes([i]) for i in range(256)]
    if merged_digits:
        pieces.append(b"12")
    if large_vocab:
        pieces += [
            bytes([0x80 + (i >> 12), 0x80 + ((i >> 6) & 63), 0x80 + (i & 63)])
            for i in range(100_000)
        ]
    n_regular = len(pieces)
    vocab = [
        {
            "rank": rank,
            "token_bytes": base64.b64encode(piece).decode(),
            "token_str": None,
        }
        for rank, piece in enumerate(pieces)
    ]
    tekken = {
        "config": {
            "pattern": _PATTERN,
            "num_vocab_tokens": n_regular,
            "default_vocab_size": N_SPECIAL + n_regular,
            "default_num_special_tokens": N_SPECIAL,
            "version": "v13",
        },
        "vocab": vocab,
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
    token_ids = {
        **{n: i for i, n in enumerate(names)},
        **{
            "".join(b2u[byte] for byte in piece): N_SPECIAL + rank
            for rank, piece in enumerate(pieces)
        },
    }
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
                    "pattern": {"Regex": _OLD_PATTERN if wrong_regex else _PATTERN},
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
            "vocab": token_ids,
            "merges": ["1 2"] if merged_digits else [],
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
        json.dumps(
            {
                "model_type": model_type,
                "eos_token_id": 2,
                "bos_token_id": 1,
                "transformers_version": "5.0.0.dev0",
            }
        )
    )
    (path / "processor_config.json").write_text(
        json.dumps(
            {
                "image_token": "[IMG]",
                "image_break_token": "[IMG_BREAK]",
                "image_end_token": "[IMG_END]",
                "patch_size": 14,
                "spatial_merge_size": 2,
            }
        )
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


@pytest.fixture(scope="module")
def mistral3_checkpoint(tmp_path_factory) -> Path:
    """Same tokenizer, but ``config.json`` says ``model_type: mistral3`` (a VLM)."""
    return build_tekken_checkpoint(
        tmp_path_factory.mktemp("tekken_mistral3"), model_type="mistral3"
    )


def mistral3_processor(tokenizer):
    """The processor ``mlx_vlm.load`` returns for a Mistral3 checkpoint."""
    from mlx_vlm.models.mistral3.processing_mistral3 import Mistral3Processor
    from mlx_vlm.models.pixtral.image_processing_pixtral import PixtralImageProcessor

    return Mistral3Processor(
        image_processor=PixtralImageProcessor(),
        tokenizer=tokenizer,
        patch_size=14,
        spatial_merge_size=2,
    )


def _image_prompt_ids(processor, width=224, height=168):
    """``processor(text=..., images=...)`` on a rendered ``{"type": "image"}`` chat."""
    from PIL import Image

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": "What colour?"}],
        }
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    image = Image.new("RGB", (width, height), (200, 30, 30))
    out = processor(text=text, images=[image], add_special_tokens=False)
    ids = out["input_ids"]
    return list(ids[0] if hasattr(ids[0], "__len__") else ids)


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

    def _edit_tokenizer_json(self, tmp_path, edit):
        build_tekken_checkpoint(tmp_path)
        path = tmp_path / "tokenizer.json"
        data = json.loads(path.read_text())
        edit(data)
        path.write_text(json.dumps(data))

    def test_keeps_mistral_common_when_token_ids_disagree(self, tmp_path, caplog):
        def swap_a_b(data):
            vocab, b2u = data["model"]["vocab"], _byte_to_unicode()
            a, b = b2u[ord("a")], b2u[ord("b")]
            vocab[a], vocab[b] = vocab[b], vocab[a]

        self._edit_tokenizer_json(tmp_path, swap_a_b)
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            assert mt.fast_tokenizer_config(tmp_path, None) is None
        assert "not equivalent" in caplog.text

    def test_keeps_mistral_common_when_partial_utf8_ids_disagree(self, tmp_path):
        # Bytes 0xFE/0xFF never occur in UTF-8: both ids decode to U+FFFD and no
        # text encodes to them, so neither a comparison of decoded text nor plain-text
        # probes can see them trade places; the byte-for-byte check does.
        def swap_invalid_bytes(data):
            vocab, b2u = data["model"]["vocab"], _byte_to_unicode()
            a, b = b2u[0xFE], b2u[0xFF]
            vocab[a], vocab[b] = vocab[b], vocab[a]

        self._edit_tokenizer_json(tmp_path, swap_invalid_bytes)
        assert mt.fast_tokenizer_config(tmp_path, None) is None

    def test_keeps_mistral_common_when_a_control_token_is_renamed(self, tmp_path):
        def swap_inst(data):
            vocab = data["model"]["vocab"]
            vocab["[INST]"], vocab["[/INST]"] = vocab["[/INST]"], vocab["[INST]"]
            for token in data["added_tokens"]:
                if token["content"] in ("[INST]", "[/INST]"):
                    token["id"] = vocab[token["content"]]

        self._edit_tokenizer_json(tmp_path, swap_inst)
        assert mt.fast_tokenizer_config(tmp_path, None) is None

    def test_keeps_mistral_common_when_the_pretokenizer_regex_differs(
        self, tmp_path, caplog
    ):
        # The fixture's byte-only vocabulary encodes the same under any splitting
        # regex, so neither the probes nor the token check can tell: only comparing
        # the regex with tekken.json's pattern does.
        def split_on_digit_runs(data):
            split = data["pre_tokenizer"]["pretokenizers"][0]
            split["pattern"]["Regex"] = _PATTERN.replace(r"|\p{N}|", r"|\p{N}+|")
            assert split["pattern"]["Regex"] != _PATTERN

        self._edit_tokenizer_json(tmp_path, split_on_digit_runs)
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            assert mt.fast_tokenizer_config(tmp_path, None) is None
        assert "pre-tokenizer regex differs" in caplog.text

    def test_repairs_a_tokenizer_json_with_the_old_regex(self, tmp_path):
        build_tekken_checkpoint(
            tmp_path, merged_digits=True, wrong_regex=True, large_vocab=True
        )
        reference = mt.reference_tokenizer(tmp_path)
        expected = reference.encode("12345", add_special_tokens=False)
        assert expected == [N_SPECIAL + ord(c) for c in "12345"]
        # Without the repair the old pattern groups digits and merges "12".
        unrepaired = AutoTokenizer.from_pretrained(tmp_path, mistral_format=False)
        assert unrepaired.encode("12345", add_special_tokens=False) != expected

        config = mt.fast_tokenizer_config(tmp_path, None)
        assert config is not None
        repaired = AutoTokenizer.from_pretrained(tmp_path, **config)
        assert repaired.encode("12345", add_special_tokens=False) == expected


class _StubTokenizer:
    """The slice of a tokenizer ``_mismatch`` looks at."""

    is_fast = True
    chat_template = "{{ messages }}"
    bos_token_id = 1
    eos_token_id = 2

    def __init__(self, size=4, encode=None):
        self._size = size
        self._encode = encode or (lambda text: [ord(c) for c in text])

    def __len__(self):
        return self._size

    def encode(self, text, add_special_tokens=False):
        return self._encode(text)


class TestEquivalenceCheck:
    """The tokenizer-level reasons ``_mismatch`` rejects a tokenizer."""

    @pytest.fixture(autouse=True)
    def tokens_agree(self, monkeypatch):
        monkeypatch.setattr(mt, "_token_mismatch", lambda fast, path: None)

    def test_identical_tokenizers_match(self):
        assert mt._mismatch(_StubTokenizer(), _StubTokenizer(), Path("x")) is None

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
        assert reason in mt._mismatch(fast, _StubTokenizer(), Path("x"))

    def test_vocabulary_size_difference(self):
        reason = mt._mismatch(_StubTokenizer(5), _StubTokenizer(), Path("x"))
        assert "vocabulary size" in reason

    def test_encoding_difference_in_plain_text(self):
        # Same vocabulary, but never splits digits or words: only the probes tell.
        fast = _StubTokenizer(encode=lambda text: [len(text)])
        assert "encoding of" in mt._mismatch(fast, _StubTokenizer(), Path("x"))


class TestAdoptFastTokenizer:
    def test_swaps_the_tokenizer_of_a_mistral3_processor(
        self, mistral3_checkpoint, reference
    ):
        processor = mistral3_processor(reference)
        adopted = mt.adopt_fast_tokenizer(
            processor, mistral3_checkpoint, eos_token_ids=2
        )
        assert adopted is processor
        self._assert_serves_reference_ids(adopted, reference)

    def test_builds_a_processor_when_the_backend_is_the_processor(
        self, mistral3_checkpoint
    ):
        # mlx-vlm's AutoProcessor patch did not apply: the processor IS the backend,
        # with the image processor mlx_vlm.load assigns to it.
        from mlx_vlm.models.mistral3.processing_mistral3 import Mistral3Processor
        from mlx_vlm.models.pixtral.image_processing_pixtral import (
            PixtralImageProcessor,
        )

        backend = mt.reference_tokenizer(mistral3_checkpoint)
        backend.image_processor = PixtralImageProcessor()
        adopted = mt.adopt_fast_tokenizer(backend, mistral3_checkpoint, eos_token_ids=2)
        assert isinstance(adopted, Mistral3Processor)
        self._assert_serves_reference_ids(adopted, backend)

    def _assert_serves_reference_ids(self, processor, reference):
        tokenizer = processor.tokenizer
        assert tokenizer.is_fast and not mt.holds_mistral_common_tokenizer(processor)
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
        # An image: 8x6 patches of [IMG] (224x168 at 28 px), one [IMG_END], real ids.
        ids = _image_prompt_ids(processor)
        assert ids.count(10) == 48 and ids.count(13) == 1 and ids.count(12) == 5

    def test_warns_when_the_processor_cannot_be_rebuilt(self, checkpoint, caplog):
        # Backend as processor, but a model_type with no known mlx-vlm processor.
        backend = mt.reference_tokenizer(checkpoint)
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            assert mt.adopt_fast_tokenizer(backend, checkpoint) is backend
        assert "cannot build a multimodal processor" in caplog.text

    def test_warns_when_there_is_no_tokenizer_json(self, tmp_path, caplog):
        build_tekken_checkpoint(tmp_path, model_type="mistral3")
        (tmp_path / "tokenizer.json").unlink()
        backend = mt.reference_tokenizer(tmp_path)
        processor = mistral3_processor(backend)
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            assert mt.adopt_fast_tokenizer(processor, tmp_path) is processor
        assert processor.tokenizer is backend
        assert "no tokenizer.json" in caplog.text

    def test_other_processors_are_untouched(self):
        tokenizer = SimpleNamespace(name="plain hf tokenizer")
        processor = SimpleNamespace(tokenizer=tokenizer)
        assert mt.adopt_fast_tokenizer(processor, "/nonexistent") is processor
        assert processor.tokenizer is tokenizer
        assert not hasattr(processor, "detokenizer")
        assert not mt.holds_mistral_common_tokenizer(processor)


class TestLoadSites:
    @pytest.mark.parametrize(
        "shape", ["tokenizer_in_processor", "backend_is_processor"]
    )
    def test_mllm_load_serves_the_fast_tokenizer(
        self, mistral3_checkpoint, shape, monkeypatch
    ):
        import mlx_vlm
        import mlx_vlm.utils
        from mlx_vlm.models.pixtral.image_processing_pixtral import (
            PixtralImageProcessor,
        )

        from vllm_mlx.models.mllm import MLXMultimodalLM

        backend = mt.reference_tokenizer(mistral3_checkpoint)
        if shape == "tokenizer_in_processor":
            processor = mistral3_processor(backend)
        else:
            backend.image_processor = PixtralImageProcessor()
            processor = backend
        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))
        monkeypatch.setattr(mlx_vlm, "load", lambda name: (model, processor))
        monkeypatch.setattr(
            mlx_vlm.utils, "get_model_path", lambda name: mistral3_checkpoint
        )
        monkeypatch.setattr(mlx_vlm.utils, "load_config", lambda name: {})

        engine = MLXMultimodalLM("tekken-test")
        engine.load()

        messages = CONVERSATIONS["multi_turn"]
        assert not mt.holds_mistral_common_tokenizer(engine.processor)
        assert string_pipeline_ids(
            engine.processor.tokenizer, messages
        ) == reference_ids(backend, messages)
        assert _image_prompt_ids(engine.processor).count(10) == 48

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

    def test_encodes_a_corpus_like_tekken(self, real_reference, real_fast):
        # The docs: English plus Chinese, Spanish and French, prose, code and tables.
        # A chunk that spells out a control token ("[INST]") is left out: tekken
        # reads that as text, the fast tokenizer as the control token (documented).
        import re

        docs = Path(__file__).resolve().parent.parent / "docs"
        if not docs.is_dir():
            pytest.skip(f"{docs} not found: no corpus to encode")
        control = re.compile("|".join(map(re.escape, real_fast.get_added_vocab())))
        text = "\n".join(path.read_text() for path in sorted(docs.rglob("*.md")))
        chunks = [text[i : i + 4000] for i in range(0, len(text), 4000)]
        chunks = [chunk for chunk in chunks if not control.search(chunk)]
        if len(chunks) < 100:
            pytest.skip(f"only {len(chunks)} usable chunks under {docs}")
        for number, chunk in enumerate(chunks):
            assert real_fast.encode(chunk, add_special_tokens=False) == (
                real_reference.encode(chunk, add_special_tokens=False)
            ), f"chunk {number}"


class TestRealDevstralLoad:
    """``MLXMultimodalLM.load`` on the real Devstral files.

    Only the weights are stubbed out (``mlx_vlm.utils.load_model``); the processor is
    what ``mlx_vlm.load`` really builds: a ``Mistral3Processor`` whose ``.tokenizer``
    is the mistral-common backend. The second shape, the backend itself as the
    processor, is what ``AutoProcessor`` returns when mlx-vlm's processor patch does
    not apply.
    """

    @pytest.fixture(scope="class")
    def snapshot(self):
        return _devstral_snapshot()

    @pytest.fixture(params=["mlx_vlm_load", "backend_is_processor"])
    def engine(self, request, snapshot, monkeypatch):
        import mlx_vlm
        import mlx_vlm.utils

        from vllm_mlx.models.mllm import MLXMultimodalLM

        def no_weights(path, lazy=False, **kwargs):
            return SimpleNamespace(
                config=SimpleNamespace(eos_token_id=2), parameters=lambda: {}
            )

        monkeypatch.setattr(mlx_vlm.utils, "load_model", no_weights)
        if request.param == "backend_is_processor":
            real_load = mlx_vlm.utils.load

            def load_with_bare_backend(name):
                model, processor = real_load(name)
                backend = AutoTokenizer.from_pretrained(name)
                backend.image_processor = processor.image_processor
                return model, backend

            monkeypatch.setattr(mlx_vlm, "load", load_with_bare_backend)
        engine = MLXMultimodalLM(snapshot)
        engine.load()
        return engine

    def test_text_prompts_encode_to_the_mistral_common_ids(self, engine, snapshot):
        from mlx_vlm.utils import prepare_inputs

        reference = mt.reference_tokenizer(snapshot)
        processor = engine.processor
        assert not mt.holds_mistral_common_tokenizer(processor)
        for name in ("plain", "tools", "tool_result"):
            messages, tools = DEVSTRAL_CASES[name]
            kwargs = {"tools": tools} if tools else {}
            text = processor.tokenizer.apply_chat_template(
                _server_shaped(messages),
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
            # the mllm batch generator's text-only encoding of a rendered prompt
            ids = prepare_inputs(processor, prompts=text)["input_ids"].tolist()[0]
            assert ids == reference_ids(reference, messages, **kwargs), name

    @pytest.mark.parametrize("size", [(224, 168), (640, 480)])
    def test_image_prompt_encodes_to_the_mistral_common_ids(
        self, engine, snapshot, size
    ):
        # The image tiles with patch_size 14 / spatial_merge_size 2 (from
        # processor_config.json); the processor's own defaults (16 / 1) expand to a
        # different, much larger number of [IMG] tokens, so a processor rebuilt
        # without those settings fails here.
        import base64
        import io

        from PIL import Image

        reference = mt.reference_tokenizer(snapshot)
        image = Image.new("RGB", size, (200, 30, 30))
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
        expected = reference_ids(
            reference,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What colour?"},
                        {"type": "image_url", "image_url": {"url": url}},
                    ],
                }
            ],
        )
        processor = engine.processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What colour?"},
                    {"type": "image"},
                ],
            }
        ]
        text = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = processor(text=text, images=[image], add_special_tokens=False)
        assert list(out["input_ids"][0]) == expected

    def test_streaming_detokenizer_and_stop_criteria_are_wired(self, engine):
        tokenizer = engine.processor.tokenizer
        assert tokenizer.stopping_criteria.eos_token_ids == [2]
        detok = engine.processor.detokenizer
        for token in tokenizer.encode("café 世界", add_special_tokens=False):
            detok.add_token(token)
        detok.finalize()
        assert detok.text == "café 世界"


RAW_PROMPTS = (
    "Once upon a time",
    " Once upon a time",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n - 1)",
    "日本語のテキスト 世界",
    "12345 6789",
    "x",
)


class TestRawPromptBos:
    """A raw completion prompt gets the BOS, as with mistral-common (vLLM).

    vLLM tokenizes a ``/v1/completions`` prompt with mistral-common's
    ``encode(text, bos=add_special_tokens)``: ``[BOS, *plain_text_ids]``.  The
    ``tokenizer.json`` of a tekken checkpoint has no BOS post-processor, so the
    fast tokenizer that replaces mistral-common encoded the same prompt without
    one on every route (Simple, batched, MLLM): the model saw ``"Once upon a
    time"`` with nothing in front of it.
    """

    def test_fast_tokenizer_prepends_the_bos_like_mistral_common(
        self, checkpoint, reference
    ):
        fast = AutoTokenizer.from_pretrained(
            checkpoint, **mt.fast_tokenizer_config(checkpoint, None)
        )
        for prompt in RAW_PROMPTS:
            expected = reference.encode(prompt, add_special_tokens=True)
            assert expected[0] == 1 and expected[1:] != [1]
            assert fast.encode(prompt, add_special_tokens=True) == expected
            # a rendered chat prompt carries its own <s>: nothing is added
            assert fast.encode(prompt, add_special_tokens=False) == expected[1:]

    def test_gate_rejects_a_fast_tokenizer_that_drops_the_bos(
        self, checkpoint, reference, monkeypatch, caplog
    ):
        # The equivalence check compares add_special_tokens=True encodings too, so
        # a tokenizer.json without the BOS post-processor is never adopted silently.
        monkeypatch.setattr(
            mt,
            "FAST_TOKENIZER_KWARGS",
            {k: v for k, v in mt.FAST_TOKENIZER_KWARGS.items() if k != "add_bos_token"},
        )
        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            assert mt.equivalent_fast_tokenizer(checkpoint, reference) is None
        assert "add_special_tokens=True" in caplog.text

    def test_request_kinds_through_encode_prompt(self, checkpoint):
        from vllm_mlx.utils.tokenizer import encode_prompt

        fast = AutoTokenizer.from_pretrained(
            checkpoint, **mt.fast_tokenizer_config(checkpoint, None)
        )
        raw = "Once upon a time"
        plain = fast.encode(raw, add_special_tokens=False)
        assert encode_prompt(fast, raw, None) == [1, *plain]
        assert encode_prompt(fast, raw, True) == [1, *plain]
        assert encode_prompt(fast, raw, False) == plain
        # already opened with <s>: one BOS, never two
        assert encode_prompt(fast, "<s>" + raw, None) == [1, *plain]
        # a rendered chat prompt: the template's <s> only
        chat = fast.apply_chat_template(
            CONVERSATIONS["plain"], tokenize=False, add_generation_prompt=True
        )
        assert encode_prompt(fast, chat, False).count(1) == 1
        assert encode_prompt(fast, chat, False) == string_pipeline_ids(
            fast, CONVERSATIONS["plain"]
        )

    def test_mlx_lm_loaded_tokenizer_has_the_bos(self, checkpoint, monkeypatch):
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
        assert tokenizer.encode("hello", add_special_tokens=True)[0] == 1
        assert 1 not in tokenizer.encode("hello", add_special_tokens=False)


_MINISTRAL = "models--mlx-community--Ministral-3-3B-Instruct-2512-4bit"


def _tekken_snapshot(model_dir: str, env: str) -> str:
    candidates = [os.environ.get(env)]
    for hub in (
        _HUB,
        "/hive-cache/models/hub",
        "/hive-cache/codeagent/models/hub",
    ):
        candidates += sorted(glob.glob(os.path.join(hub, model_dir, "snapshots", "*")))
    for path in candidates:
        if path and all(
            os.path.exists(os.path.join(path, name))
            for name in ("tekken.json", "tokenizer.json", "chat_template.jinja")
        ):
            return path
    pytest.skip(f"{model_dir} tokenizer files not found (set {env} or HF_HOME)")


@functools.cache
def _real_tekken(snapshot: str):
    from mistral_common.tokens.tokenizers.tekken import Tekkenizer

    return Tekkenizer.from_file(os.path.join(snapshot, "tekken.json"))


@functools.cache
def _real_fast(snapshot: str):
    reference = mt.reference_tokenizer(snapshot)
    assert reference is not None
    fast = mt.equivalent_fast_tokenizer(snapshot, reference)
    assert fast is not None, "tokenizer.json is not equivalent to tekken.json"
    return fast


@pytest.mark.parametrize(
    "model_dir,env",
    [
        (_MINISTRAL, "VLLM_MLX_MINISTRAL_DIR"),
        (_DEVSTRAL, "VLLM_MLX_DEVSTRAL_DIR"),
    ],
    ids=["ministral-3-3b", "devstral-small-2"],
)
class TestRealTekkenRawPromptBos:
    """The real checkpoints against ``mistral_common``'s ``Tekkenizer`` itself."""

    @pytest.fixture
    def tekken(self, model_dir, env):
        return _real_tekken(_tekken_snapshot(model_dir, env))

    @pytest.fixture
    def fast(self, model_dir, env):
        return _real_fast(_tekken_snapshot(model_dir, env))

    @pytest.mark.parametrize("prompt", RAW_PROMPTS)
    def test_raw_prompt_ids_are_mistral_commons(self, tekken, fast, prompt):
        from vllm_mlx.utils.tokenizer import encode_prompt

        expected = tekken.encode(prompt, bos=True, eos=False)
        assert expected[0] == tekken.bos_id
        assert encode_prompt(fast, prompt, None) == expected
        assert encode_prompt(fast, prompt, False) == expected[1:]

    def test_rendered_chat_prompt_keeps_a_single_bos(self, tekken, fast):
        from vllm_mlx.utils.tokenizer import encode_prompt

        messages = [{"role": "user", "content": "hi"}]
        text = fast.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        expected = _ids(
            fast.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        )
        assert expected[0] == tekken.bos_id and expected.count(tekken.bos_id) == 1
        assert encode_prompt(fast, text, False) == expected
        # the rule already keeps the template's BOS when the kind is unknown
        assert encode_prompt(fast, text, None) == expected
