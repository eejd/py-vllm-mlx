# SPDX-License-Identifier: Apache-2.0
"""A tekken checkpoint's prompt is the one mistral-common builds, default system prompt included.

Several Mistral chat templates open with ``{% set default_system_message = '...' %}`` and
render it as a system prompt whenever the request has none. mistral-common renders no
system prompt then, and neither do ``transformers``' ``MistralCommonBackend``, mlx-lm and
vLLM's ``mistral`` tokenizer mode. Ministral 3's default is Mistral's ``SYSTEM_PROMPT.txt``
with ``{name}`` filled and ``{today}``/``{yesterday}`` left as literal text, so every
prompt carried ~500 tokens of it (Ministral-3-3B, temperature 0: ``count: 0, flag: false``
padded into tool arguments that mistral-common's prompt left out).
``mistral_tokenizer.drop_default_system_prompt`` empties the default at load.

The synthetic checkpoint (``build_tekken_checkpoint``) has a Mistral-style template with
such a default; ``TestRealCheckpoints`` compares the ids of the real Ministral-3-3B and
Devstral Small 2 snapshots with mistral-common's for plain, tool, tool-result and
system-message conversations, and skips when the snapshots are not in a local cache.
"""

import glob
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("transformers")
pytest.importorskip("mistral_common")

from transformers import AutoTokenizer  # noqa: E402

from tests.test_mistral_tokenizer import (  # noqa: E402
    _TEMPLATE,
    CONVERSATIONS,
    DEVSTRAL_CASES,
    _server_shaped,
    build_tekken_checkpoint,
    mistral3_processor,
    reference_ids,
    string_pipeline_ids,
)
from vllm_mlx.utils import mistral_tokenizer as mt  # noqa: E402

# The shape of Ministral 3's: a long single-quoted literal with \n and \' escapes and
# unfilled {placeholders}.
DEFAULT = (
    "You are Test-3B, a model.\\nThe current date is {today}.\\n"
    'It\\\'s "quoted" and {yesterday} too.'
)
_SYSTEM_BRANCH_END = "{% set ms = ms[1:] %}{% endif %}"
assert _SYSTEM_BRANCH_END in _TEMPLATE
TEMPLATE_WITH_DEFAULT = (
    "{%- set default_system_message = '"
    + DEFAULT
    + "' %}"
    + (
        _TEMPLATE.replace(
            _SYSTEM_BRANCH_END,
            "{% set ms = ms[1:] %}"
            "{% elif default_system_message != '' %}"
            "[SYSTEM_PROMPT]{{ default_system_message }}[/SYSTEM_PROMPT]"
            "{% endif %}",
        )
    )
)

PLAIN = CONVERSATIONS["plain"]
SYSTEM = CONVERSATIONS["system"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    return build_tekken_checkpoint(
        tmp_path_factory.mktemp("tekken_default"), TEMPLATE_WITH_DEFAULT
    )


@pytest.fixture(scope="module")
def reference(checkpoint):
    tokenizer = mt.reference_tokenizer(checkpoint)
    assert tokenizer is not None, "transformers did not pick MistralCommonBackend"
    return tokenizer


def _fast(checkpoint):
    return AutoTokenizer.from_pretrained(checkpoint, **mt.FAST_TOKENIZER_KWARGS)


def _text(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


class TestTemplateEdit:
    def test_the_synthetic_template_renders_its_default_as_shipped(self, checkpoint):
        """The control: without the fix the default (placeholders and all) is the
        system prompt of a request that has none."""
        text = _text(_fast(checkpoint), PLAIN)

        assert text.startswith("<s>[SYSTEM_PROMPT]You are Test-3B")
        assert "{today}" in text

    @pytest.mark.parametrize(
        "literal",
        [
            "'a {today} \\'b\\' \\n c'",
            '"a {today} \\"b\\" \\n c"',
            "'one\ntwo\n'",
        ],
    )
    def test_any_string_literal_is_emptied(self, literal):
        template = (
            "{%- set default_system_message = " + literal + " %}\n"
            "{%- if default_system_message != '' %}{{ default_system_message }}"
            "{%- endif %}"
        )

        emptied, dropped = mt.template_without_default_system_message(template)

        assert emptied == template.replace(literal, "''", 1)
        assert dropped == literal[1:-1]

    @pytest.mark.parametrize(
        "template",
        [
            "{{ x }}",
            "{%- set default_system_message = '' %}{{ x }}",
            "{% set default_system_message = '' %}"
            '{% if default_system_message != "" %}{{ x }}{% endif %}',
        ],
    )
    def test_nothing_to_drop_leaves_the_template_alone(self, template):
        assert mt.template_without_default_system_message(template) == (template, "")

    def test_a_default_that_is_not_a_literal_stays_and_is_reported(self, caplog):
        template = (
            "{% set default_system_message = prompts | join %}"
            "{% if default_system_message != '' %}{{ x }}{% endif %}"
        )

        with caplog.at_level(logging.WARNING, logger=mt.logger.name):
            assert mt.template_without_default_system_message(template) == (
                template,
                "",
            )

        assert "default system prompt stays" in caplog.text

    def test_only_the_default_is_edited(self):
        template = (
            "{% set default_system_message = 'x' %}"
            "{% if default_system_message != '' %}[SYSTEM_PROMPT]"
            "{{ default_system_message }}{% endif %}"
        )

        emptied, _ = mt.template_without_default_system_message(template)

        assert emptied == (
            "{% set default_system_message = '' %}"
            "{% if default_system_message != '' %}[SYSTEM_PROMPT]"
            "{{ default_system_message }}{% endif %}"
        )


_DATA = Path(__file__).parent / "data"


class TestUnguardedDefaults:
    """Mistral Small 3.x templates: ``"..." + today + "..."`` rendered whether or not it
    is empty. Emptying it would emit ``[SYSTEM_PROMPT][/SYSTEM_PROMPT]``, which
    mistral-common does not build either, so these keep their default (with the real
    date from ``strftime_now``) and nothing is logged above debug level.

    The files are the ``chat_template`` of mistralai/Mistral-Small-3.1-24B-Instruct-2503
    (``chat_template.json``) and mistralai/Mistral-Small-24B-Instruct-2501
    (``tokenizer_config.json``), unedited.
    """

    @pytest.fixture(params=["mistral_small_3_1_2503", "mistral_small_2501"])
    def small_template(self, request):
        return (_DATA / f"{request.param}_chat_template.jinja").read_text()

    def test_the_template_is_left_alone_without_a_warning(self, small_template, caplog):
        assert "default_system_message" in small_template

        with caplog.at_level(logging.INFO, logger=mt.logger.name):
            assert mt.template_without_default_system_message(small_template) == (
                small_template,
                "",
            )

        assert caplog.text == ""

    def test_a_load_keeps_the_default_and_renders_the_real_date(
        self, small_template, tmp_path, caplog
    ):
        import datetime

        checkpoint = build_tekken_checkpoint(tmp_path / "small", small_template)
        tokenizer = _fast(checkpoint)

        with caplog.at_level(logging.INFO, logger=mt.logger.name):
            assert not mt.drop_default_system_prompt(tokenizer, checkpoint)

        text = _text(tokenizer, PLAIN)
        assert tokenizer.chat_template == small_template
        assert text.startswith("<s>[SYSTEM_PROMPT]You are Mistral Small 3")
        assert datetime.date.today().isoformat() in text
        assert caplog.text == ""


class TestDropDefaultSystemPrompt:
    def test_a_request_without_a_system_message_gets_the_mistral_common_prompt(
        self, checkpoint, reference
    ):
        tokenizer = _fast(checkpoint)

        assert mt.drop_default_system_prompt(tokenizer, checkpoint)

        for messages in (PLAIN, CONVERSATIONS["multi_turn"][1:]):
            assert string_pipeline_ids(tokenizer, messages) == reference_ids(
                reference, messages
            )

    def test_a_system_message_is_rendered_as_before(self, checkpoint, reference):
        tokenizer = _fast(checkpoint)
        mt.drop_default_system_prompt(tokenizer, checkpoint)

        assert string_pipeline_ids(tokenizer, SYSTEM) == reference_ids(
            reference, SYSTEM
        )
        assert "{today}" not in _text(tokenizer, SYSTEM)

    def test_a_processor_and_its_tokenizer_both_lose_the_default(
        self, checkpoint, reference
    ):
        processor = mistral3_processor(_fast(checkpoint))
        assert mt.mentions_default_system_message(processor)

        assert mt.drop_default_system_prompt(processor, checkpoint)

        assert not processor.chat_template or "{today}" not in processor.chat_template
        assert "{today}" not in processor.tokenizer.chat_template
        assert not mt.mentions_default_system_message(
            SimpleNamespace(chat_template="{{ x }}")
        )

    def test_a_checkpoint_without_tekken_json_is_left_alone(self, tmp_path):
        tokenizer = SimpleNamespace(chat_template=TEMPLATE_WITH_DEFAULT)

        assert not mt.drop_default_system_prompt(tokenizer, tmp_path)
        assert tokenizer.chat_template == TEMPLATE_WITH_DEFAULT

    def test_dropping_twice_is_harmless(self, checkpoint):
        tokenizer = _fast(checkpoint)

        assert mt.drop_default_system_prompt(tokenizer, checkpoint)
        assert not mt.drop_default_system_prompt(tokenizer, checkpoint)

    def test_the_log_says_what_was_dropped_and_how_to_get_it_back(
        self, checkpoint, caplog
    ):
        """A Reasoning checkpoint's default is its thinking-format prompt: the operator
        must be able to see from the log that it is gone and what to do about it."""
        template = TEMPLATE_WITH_DEFAULT.replace(
            DEFAULT, "# HOW YOU SHOULD THINK AND ANSWER\\n\\nFirst draft your thinking."
        )
        tokenizer = SimpleNamespace(chat_template=template)

        with caplog.at_level(logging.INFO, logger=mt.logger.name):
            assert mt.drop_default_system_prompt(tokenizer, checkpoint)

        message = caplog.records[-1].getMessage()
        assert checkpoint.name in message
        assert "'# HOW YOU SHOULD THINK AND ANSWER" in message
        assert "SYSTEM_PROMPT.txt" in message
        assert "--chat-template" in message
        assert "send it as the system message" in message

    def test_a_default_dropped_from_one_holder_is_still_reported(
        self, checkpoint, caplog
    ):
        inner = SimpleNamespace(chat_template=TEMPLATE_WITH_DEFAULT)
        processor = SimpleNamespace(chat_template=None, tokenizer=inner)

        with caplog.at_level(logging.INFO, logger=mt.logger.name):
            assert mt.drop_default_system_prompt(processor, checkpoint)

        assert "{today}" not in inner.chat_template
        assert "dropped the chat template's default system prompt" in caplog.text


_OFFICIAL = {
    # name: (what the default starts with)
    "ministral_3_3b_instruct_2512": "You are Ministral-3-3B-Instruct-2512",
    "ministral_3_3b_reasoning_2512": "# HOW YOU SHOULD THINK AND ANSWER",
    "devstral_small_2_2512": "You are operating as and within Mistral Vibe",
}


class TestOfficialTemplates:
    """The ``chat_template.jinja`` of the mistralai repos, unedited (they guard the
    default as ``messages[0]['role'] != 'system' and default_system_message != ''``,
    unlike the mlx-community Ministral file). Rendered through a synthetic tekken
    checkpoint, the ids equal mistral-common's for every conversation once the default
    is dropped, and differ for those without a system message before."""

    @pytest.fixture(params=sorted(_OFFICIAL))
    def official(self, request, tmp_path):
        name = request.param
        template = (_DATA / f"mistral_{name}_official_chat_template.jinja").read_text()
        checkpoint = build_tekken_checkpoint(tmp_path / name, template)
        return name, checkpoint, mt.reference_tokenizer(checkpoint)

    def _equal(self, tokenizer, reference, case):
        messages, tools = DEVSTRAL_CASES[case]
        kwargs = {"tools": tools} if tools else {}
        return string_pipeline_ids(
            tokenizer, _server_shaped(messages), enable_thinking=True, **kwargs
        ) == reference_ids(reference, messages, **kwargs)

    def test_the_default_is_a_difference_before(self, official):
        _, checkpoint, reference = official
        tokenizer = _fast(checkpoint)

        assert not self._equal(tokenizer, reference, "plain")
        assert self._equal(tokenizer, reference, "system")

    @pytest.mark.parametrize("case", sorted(DEVSTRAL_CASES))
    def test_every_conversation_matches_mistral_common_after(self, official, case):
        _, checkpoint, reference = official
        tokenizer = _fast(checkpoint)

        assert mt.drop_default_system_prompt(tokenizer, checkpoint)

        assert self._equal(tokenizer, reference, case)

    def test_the_log_shows_the_start_of_the_dropped_default(self, official, caplog):
        name, checkpoint, _ = official

        with caplog.at_level(logging.INFO, logger=mt.logger.name):
            assert mt.drop_default_system_prompt(_fast(checkpoint), checkpoint)

        assert f"starting '{_OFFICIAL[name]}" in caplog.text


class TestLoadSites:
    def test_llm_load_renders_the_mistral_common_prompt(
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

        for messages in (PLAIN, SYSTEM):
            assert string_pipeline_ids(tokenizer, messages) == reference_ids(
                reference, messages
            )

    def test_llm_load_leaves_a_chat_template_override_in_charge(
        self, checkpoint, monkeypatch
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
        _, tokenizer = load_model_with_fallback(
            str(checkpoint), chat_template="{{ 'mine' }}"
        )

        assert _text(tokenizer, PLAIN) == "mine"

    @pytest.mark.parametrize(
        "shape", ["tokenizer_in_processor", "backend_is_processor"]
    )
    def test_mllm_load_renders_the_mistral_common_prompt(
        self, tmp_path, shape, monkeypatch
    ):
        import mlx_vlm
        import mlx_vlm.utils
        from mlx_vlm.models.pixtral.image_processing_pixtral import (
            PixtralImageProcessor,
        )

        from vllm_mlx.models.mllm import MLXMultimodalLM

        checkpoint = build_tekken_checkpoint(
            tmp_path / "ckpt", TEMPLATE_WITH_DEFAULT, model_type="mistral3"
        )
        reference = mt.reference_tokenizer(checkpoint)
        backend = mt.reference_tokenizer(checkpoint)
        if shape == "tokenizer_in_processor":
            processor = mistral3_processor(backend)
        else:
            backend.image_processor = PixtralImageProcessor()
            processor = backend
        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))
        monkeypatch.setattr(mlx_vlm, "load", lambda name: (model, processor))
        monkeypatch.setattr(mlx_vlm.utils, "get_model_path", lambda name: checkpoint)
        monkeypatch.setattr(mlx_vlm.utils, "load_config", lambda name: {})

        engine = MLXMultimodalLM("tekken-default-test")
        engine.load()

        for messages in (PLAIN, SYSTEM):
            assert string_pipeline_ids(
                engine.processor.tokenizer, messages
            ) == reference_ids(reference, messages)

    def test_a_model_without_a_default_never_looks_for_its_files(self, monkeypatch):
        """Only a template that has a default makes the load locate the checkpoint."""
        import mlx_vlm
        import mlx_vlm.utils

        from vllm_mlx.models.mllm import MLXMultimodalLM

        processor = SimpleNamespace(
            chat_template="{{ messages }}",
            tokenizer=SimpleNamespace(chat_template="{{ messages }}"),
        )
        model = SimpleNamespace(config=SimpleNamespace(eos_token_id=2))
        monkeypatch.setattr(mlx_vlm, "load", lambda name: (model, processor))
        monkeypatch.setattr(mlx_vlm.utils, "load_config", lambda name: {})

        def forbidden(name):
            raise AssertionError("get_model_path called")

        monkeypatch.setattr(mlx_vlm.utils, "get_model_path", forbidden)

        MLXMultimodalLM("plain-model").load()

    def test_llm_path_without_a_default_never_looks_for_its_files(self, monkeypatch):
        import mlx_lm.utils

        from vllm_mlx.utils.tokenizer import _drop_mistral_default_system_prompt

        calls = []
        monkeypatch.setattr(
            mlx_lm.utils, "_download", lambda name: calls.append(name) or name
        )

        _drop_mistral_default_system_prompt(
            "plain-model", SimpleNamespace(chat_template="{{ messages }}")
        )

        assert calls == []

    def test_llm_path_with_a_default_locates_the_files(self, checkpoint, monkeypatch):
        import mlx_lm.utils

        from vllm_mlx.utils.tokenizer import _drop_mistral_default_system_prompt

        monkeypatch.setattr(mlx_lm.utils, "_download", lambda name: checkpoint)
        tokenizer = SimpleNamespace(chat_template=TEMPLATE_WITH_DEFAULT)

        _drop_mistral_default_system_prompt("tekken-test", tokenizer)

        assert "{today}" not in tokenizer.chat_template


_HUBS = [
    os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
    ),
    "/hive-cache/models/hub",
    "/hive-cache/codeagent/models/hub",
]
_REAL = {
    "Ministral-3-3B": (
        "models--mlx-community--Ministral-3-3B-Instruct-2512-4bit",
        "VLLM_MLX_MINISTRAL_DIR",
    ),
    "Devstral-Small-2": (
        "models--mlx-community--Devstral-Small-2-24B-Instruct-2512-8bit",
        "VLLM_MLX_DEVSTRAL_DIR",
    ),
}


def _real_snapshot(name: str) -> str:
    repo, env = _REAL[name]
    candidates = [os.environ.get(env)]
    for hub in _HUBS:
        candidates += sorted(glob.glob(os.path.join(hub, repo, "snapshots", "*")))
    for path in candidates:
        if path and all(
            os.path.exists(os.path.join(path, f))
            for f in ("tekken.json", "tokenizer.json", "chat_template.jinja")
        ):
            return path
    pytest.skip(
        f"{repo} tokenizer files not found (set {env} or HF_HOME): the synthetic "
        "checkpoint tests cover the code path, not this template"
    )


class TestRealCheckpoints:
    """Ids the server prompts with, against mistral-common's, on the real files."""

    @pytest.fixture(scope="class", params=sorted(_REAL))
    def snapshot(self, request):
        return _real_snapshot(request.param)

    @pytest.fixture(scope="class")
    def reference(self, snapshot):
        tokenizer = mt.reference_tokenizer(snapshot)
        assert tokenizer is not None
        return tokenizer

    @pytest.fixture(scope="class")
    def llm_tokenizer(self, snapshot):
        """What ``load_model_with_fallback`` hands the engine (only mlx-lm's load is
        replaced by its tokenizer half)."""
        import mlx_lm
        from mlx_lm.tokenizer_utils import load as load_tokenizer

        from vllm_mlx.utils.tokenizer import load_model_with_fallback

        original = mlx_lm.load
        mlx_lm.load = lambda name, tokenizer_config=None: (
            None,
            load_tokenizer(Path(name), tokenizer_config),
        )
        try:
            return load_model_with_fallback(snapshot)[1]
        finally:
            mlx_lm.load = original

    @pytest.fixture(scope="class")
    def mllm_processor(self, snapshot):
        """What ``MLXMultimodalLM.load`` serves (only the weights are stubbed)."""
        import mlx_vlm.utils

        from vllm_mlx.models.mllm import MLXMultimodalLM

        original = mlx_vlm.utils.load_model
        mlx_vlm.utils.load_model = lambda path, lazy=False, **kw: SimpleNamespace(
            config=SimpleNamespace(eos_token_id=2), parameters=lambda: {}
        )
        try:
            engine = MLXMultimodalLM(snapshot)
            engine.load()
        finally:
            mlx_vlm.utils.load_model = original
        return engine.processor

    @pytest.mark.parametrize("name", sorted(DEVSTRAL_CASES))
    def test_llm_path_prompts_with_the_mistral_common_ids(
        self, llm_tokenizer, reference, name
    ):
        messages, tools = DEVSTRAL_CASES[name]
        kwargs = {"tools": tools} if tools else {}

        got = string_pipeline_ids(
            llm_tokenizer, _server_shaped(messages), enable_thinking=True, **kwargs
        )

        assert got == reference_ids(reference, messages, **kwargs)

    @pytest.mark.parametrize("name", sorted(DEVSTRAL_CASES))
    def test_mllm_path_prompts_with_the_mistral_common_ids(
        self, mllm_processor, reference, name
    ):
        messages, tools = DEVSTRAL_CASES[name]
        kwargs = {"tools": tools} if tools else {}

        got = string_pipeline_ids(
            mllm_processor.tokenizer,
            _server_shaped(messages),
            enable_thinking=True,
            **kwargs,
        )

        assert got == reference_ids(reference, messages, **kwargs)
