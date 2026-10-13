# SPDX-License-Identifier: Apache-2.0
"""``mlx_vlm.utils.prepare_inputs`` with a working ``add_special_tokens``."""

import inspect


class _SpecialTokensForwarder:
    """A processor whose ``__call__`` declares ``add_special_tokens``.

    ``mlx_vlm.utils.process_inputs`` forwards ``add_special_tokens`` only to
    processors that list it as a named parameter of ``__call__``.  The custom
    ``Gemma3Processor`` mlx-vlm registers takes ``(images, text, **kwargs)`` and
    hands ``**kwargs`` straight to its tokenizer, so the flag was silently
    dropped and every image request got the tokenizer's BOS on top of the one
    the chat template rendered (``[2, 2, 105, ...]``).  Everything else is
    delegated to the wrapped processor.
    """

    def __init__(self, processor):
        object.__setattr__(self, "_processor", processor)

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def __setattr__(self, name, value):
        setattr(self._processor, name, value)

    def __call__(self, *args, add_special_tokens=True, **kwargs):
        return self._processor(*args, add_special_tokens=add_special_tokens, **kwargs)


def _drops_add_special_tokens(processor) -> bool:
    """True for a processor that takes ``**kwargs`` but no named flag."""
    if (
        hasattr(processor, "process")
        or not hasattr(processor, "tokenizer")
        or not callable(processor)
    ):
        return False
    try:
        parameters = inspect.signature(processor.__call__).parameters.values()
    except (TypeError, ValueError):
        return False
    names = {p.name for p in parameters}
    return "add_special_tokens" not in names and any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters
    )


def prepare_inputs(processor, *, add_special_tokens: bool, **kwargs):
    """``mlx_vlm.utils.prepare_inputs`` honouring ``add_special_tokens`` for images."""
    from mlx_vlm.utils import prepare_inputs as mlx_vlm_prepare_inputs

    if _drops_add_special_tokens(processor):
        processor = _SpecialTokensForwarder(processor)
    return mlx_vlm_prepare_inputs(
        processor, add_special_tokens=add_special_tokens, **kwargs
    )
