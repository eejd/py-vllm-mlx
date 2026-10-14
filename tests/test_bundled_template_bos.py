# SPDX-License-Identifier: Apache-2.0
"""Every bundled chat template follows its checkpoint family's BOS convention.

vllm-mlx tokenizes a rendered chat prompt without special tokens (as vLLM does), so
the template owns every marker: a template that leaves off a BOS the model expects
sends it a prompt with no BOS at all. A family whose tokenizer adds no BOS (Phi-4-mini)
must not get one from the template either.

The expectation is pinned per template, then cross-checked against the real tokenizer
of each family when it is in the local Hugging Face cache (skipped in CI, which has no
model cache): the template starts with the BOS exactly when the tokenizer adds one to
a plain string.
"""

import glob
import json
import os

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from vllm_mlx.templates import (
    GEMMA3_PYTHONIC_TOOL_TEMPLATE,
    HERMES_TOOL_TEMPLATE,
    PHI4_MINI_TOOL_TEMPLATE,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City"}},
                "required": ["city"],
            },
        },
    }
]
CONVERSATION = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi there"},
]

# name -> (template, checkpoint repo glob, bos_token, eos_token, starts with the BOS)
FAMILIES = {
    "hermes": (
        HERMES_TOOL_TEMPLATE,
        "models--mlx-community--Hermes-3-Llama-3.2-3B-4bit",
        "<|begin_of_text|>",
        "<|im_end|>",
        True,
    ),
    "gemma3": (
        GEMMA3_PYTHONIC_TOOL_TEMPLATE,
        "models--mlx-community--gemma-3-4b-it-qat-4bit",
        "<bos>",
        "<eos>",
        True,
    ),
    # Phi-4-mini's tokenizer has add_bos_token false and its own template emits none;
    # its bos_token is the same string as its eos_token.
    "phi4_mini": (
        PHI4_MINI_TOOL_TEMPLATE,
        "models--mlx-community--Phi-4-mini-instruct-4bit",
        "<|endoftext|>",
        "<|endoftext|>",
        False,
    ),
}


def _tojson(value, ensure_ascii=False, indent=None, **kwargs):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, **kwargs)


def _raise(message):
    raise ValueError(message)


def _env():
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise
    return env


def _render(name, tools, add_generation_prompt):
    template, _, bos, eos, _ = FAMILIES[name]
    return (
        _env()
        .from_string(template.read_text(encoding="utf-8"))
        .render(
            messages=CONVERSATION,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            bos_token=bos,
            eos_token=eos,
        )
    )


@pytest.mark.parametrize("name", sorted(FAMILIES))
@pytest.mark.parametrize("tools", [None, TOOLS], ids=["no_tools", "tools"])
@pytest.mark.parametrize("add_generation_prompt", [True, False])
def test_template_follows_its_checkpoints_bos_convention(
    name, tools, add_generation_prompt
):
    bos, expects_bos = FAMILIES[name][2], FAMILIES[name][4]
    prompt = _render(name, tools, add_generation_prompt)
    if expects_bos:
        assert prompt.startswith(bos)
        assert prompt.count(bos) == 1
    else:
        # Phi-4-mini renders its eos_token (the same string) after the last turn when
        # there is no generation prompt; it never leads the prompt.
        assert not prompt.startswith(bos)


def _snapshot(repo):
    paths = sorted(
        glob.glob(f"/hive-cache/*/models/hub/{repo}/snapshots/*")
        + glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{repo}/snapshots/*"))
    )
    return paths[0] if paths else None


@pytest.mark.parametrize("name", sorted(FAMILIES))
def test_template_agrees_with_the_real_tokenizer(name):
    template, repo, bos, _, expects_bos = FAMILIES[name]
    snapshot = _snapshot(repo)
    if snapshot is None:
        pytest.skip(f"{repo} is not in the local cache")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    assert tokenizer.bos_token == bos
    tokenizer.chat_template = template.read_text(encoding="utf-8")
    # The family's convention: does the tokenizer add a BOS to a plain string?
    adds_bos = tokenizer.encode("hi")[:1] == [tokenizer.bos_token_id]
    assert adds_bos is expects_bos
    for tools in (None, TOOLS):
        prompt = tokenizer.apply_chat_template(
            CONVERSATION, tools=tools, tokenize=False, add_generation_prompt=True
        )
        # What the batching engines feed the model for a chat prompt.
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        assert (ids[:1] == [tokenizer.bos_token_id]) is adds_bos
        if adds_bos:
            assert ids[1:2] != [tokenizer.bos_token_id]
