# SPDX-License-Identifier: Apache-2.0
"""
Pythonic tool call parser for vllm-mlx (port of vLLM's ``pythonic`` parser).

Some models (Llama 3.2/4, ToolACE, Gemma 3 with the shipped template) answer a tool
request with a bare Python list of keyword-only calls and nothing else, no marker
tokens::

    [get_weather(city='San Francisco', metric='celsius'), get_time(tz='UTC')]

Semantics follow vLLM's ``vllm/tool_parsers/pythonic_tool_parser.py`` (v0.31):

* a response is a tool call only when it *starts* with ``[`` (a response that starts
  with anything else, whitespace included, is content, streaming and not);
* the list holds calls only, every call keyword-only; a positional argument,
  ``*args``/``**kwargs``, a non-call element, an empty list or a name that is not an
  identifier makes the whole response content;
* values are literals: strings, numbers, ``True``/``False``/``None`` and the JSON
  spellings ``true``/``false``/``null``, lists, dicts with literal keys, plus what
  vLLM's shared helper adds on top (tuples and sets become lists, an f-string without
  placeholders is a string, ``-1``/``+1`` are numbers; bytes, complex numbers,
  ``...``, names, operators and f-string placeholders are rejected);
* dotted names (``a.b.c(...)``) keep the dotted form;
* Python comments are not understood by the bracket scan: a comment between two calls
  or before the closing ``]`` leaves the response as content (vLLM's non-streaming regex
  rejects it too), and one inside a call's arguments is accepted unless it holds a
  quote or bracket.

Where this port differs from vLLM, on purpose:

* vLLM decides with a regex and then ``ast.parse``. The regex is a pre-filter that
  disagrees with the AST (non-streaming rejects ``f(a = 1)``, whitespace after the
  ``[``, ``_name`` and dotted names, which the streaming path, having no regex,
  accepts) and it backtracks exponentially on some inputs (seconds for a 100-byte
  response, hence vLLM's regex timeout). Here the AST, one call at a time (found by a
  string-aware bracket scan), decides in both modes: streaming and non-streaming
  agree on the same text, and there is nothing to time out;
* vLLM's streaming path accepts a positional argument and silently drops it
  (``[f(1)]`` streams ``f`` with ``{}``); here it is rejected as in non-streaming;
* only the leading ``[...]`` is the call list. Text after its ``]`` is returned as
  content next to the calls. vLLM parses the whole response: ``[f(a=1)]\n\nHi``
  returns the call and silently drops ``Hi`` (the later statement is ignored), and
  ``[f(a=1)] Hi.`` is content in non-streaming while streaming emits the call;
* vLLM streams argument fragments from the first character of a string value by
  closing the partial source and diffing the JSON, and a response that merely starts
  with ``[`` (``[1] is a list``) is swallowed by its streaming path (nothing is ever
  emitted, content included). This port, like the other Python-call parsers of this
  package, buffers the list and emits every call exactly once, under its absolute
  index, when the list closes (all or nothing, so streaming returns what
  non-streaming returns); a ``[...`` that is not a call list is released as content;
* ``NaN``/``Infinity`` (``1e999``) are rejected: vLLM puts ``Infinity`` into
  ``arguments``, which is not valid JSON.
"""

import ast
import json
import warnings
from collections.abc import Sequence
from typing import Any

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager,
)
from .hermes_tool_parser import generate_tool_id
from .lfm2_tool_parser import _call_name

# Python's own inter-token whitespace (``str.isspace`` also accepts NBSP and friends,
# which Python rejects).
_WS = " \t\r\n\f"


def _is_word(char: str) -> bool:
    """``\\w`` for one character: letters, digits and ``_`` of any script."""
    return char.isalnum() or char == "_"


_JSON_NAME_LITERALS = {"null": None, "true": True, "false": False}
_CLOSERS = {")": "(", "]": "[", "}": "{"}

_INCOMPLETE = "incomplete"
_INVALID = "invalid"
_COMPLETE = "complete"


class _Unsupported(Exception):
    """The source is Python but not a call this parser can stand behind."""


def _value(node: ast.expr) -> Any:
    """A keyword value as a JSON-serialisable Python value (vLLM ``get_parameter_value``)."""
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (str, bool, int, float)):
            return value  # (``1e999`` is inf: json.dumps(allow_nan=False) refuses it)
        raise _Unsupported("not a JSON value")
    if isinstance(node, ast.Dict):
        out: dict[Any, Any] = {}
        for key, val in zip(node.keys, node.values):
            if not isinstance(key, ast.Constant):  # also ``**spread`` (key is None)
                raise _Unsupported("dict keys must be literals")
            out[_value(key)] = _value(val)
        return out
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        # JSON has no tuple or set: both become a list, in source order.
        return [_value(v) for v in node.elts]
    if isinstance(node, ast.JoinedStr) and all(
        isinstance(part, ast.Constant) for part in node.values
    ):
        # ``f'text'`` without placeholders is a plain string.
        return "".join(str(part.value) for part in node.values)  # type: ignore[attr-defined]
    if isinstance(node, ast.Name) and node.id in _JSON_NAME_LITERALS:
        return _JSON_NAME_LITERALS[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _value(node.operand)
        if isinstance(operand, (int, float)) and not isinstance(operand, bool):
            return -operand if isinstance(node.op, ast.USub) else operand
    raise _Unsupported("not a literal")


def _convert_call(source: str) -> tuple[str, str]:
    """``name(k=v, ...)`` -> ``(name, arguments JSON)``; raises ``_Unsupported``."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # invalid escapes in model strings
            node = ast.parse(source, mode="eval").body
    except (SyntaxError, ValueError, MemoryError, RecursionError, OverflowError):
        raise _Unsupported("not Python") from None
    if not isinstance(node, ast.Call) or node.args:
        raise _Unsupported("only keyword-only calls")
    name = _call_name(node.func)
    if not name:
        raise _Unsupported("callee is not a name")
    arguments: dict[str, Any] = {}
    try:
        for keyword in node.keywords:
            if keyword.arg is None:  # ``**kwargs``
                raise _Unsupported("**kwargs")
            arguments[keyword.arg] = _value(keyword.value)
        return name, json.dumps(arguments, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError, RecursionError, OverflowError):
        raise _Unsupported("not JSON") from None


def _skip_ws(text: str, pos: int) -> int:
    n = len(text)
    while pos < n and text[pos] in _WS:
        pos += 1
    return pos


class _ListScanner:
    """Incremental scan of ``[call(k=v), call(k=v), ...]`` at the start of a text.

    ``advance(text)`` takes the whole text so far (each call extends the previous one)
    and returns ``_INCOMPLETE`` (could still become a call list), ``_INVALID`` (cannot)
    or ``_COMPLETE`` (the closing ``]`` is in: ``calls`` holds the parsed calls and
    ``end`` the offset just past it). Work is resumed, never repeated: a delta costs
    time proportional to itself, plus the re-read of an ambiguous quote at its end.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.end = 0
        self.status = _INCOMPLETE
        self._phase = "start"  # start | elem | sep | callee | args
        self._pos = 0
        self._callee = 0  # offset of the callee being read
        self._cstate = "start"  # callee: start | name | gap | dot
        self._stack: list[str] = []
        self._string: tuple[str, bool] | None = None  # (quote, triple)

    def _fail(self) -> str:
        self.status = _INVALID
        return _INVALID

    def advance(self, text: str) -> str:
        if self.status != _INCOMPLETE:
            return self.status
        n = len(text)
        while True:
            phase = self._phase
            if phase == "start":
                if not text:
                    return _INCOMPLETE
                if text[0] != "[":
                    return self._fail()
                self._pos, self._phase = 1, "elem"
            elif phase in ("elem", "sep"):
                i = _skip_ws(text, self._pos)
                self._pos = i
                if i >= n:
                    return _INCOMPLETE
                char = text[i]
                if phase == "sep":
                    if char == ",":
                        self._pos, self._phase = i + 1, "elem"
                    elif char == "]":
                        return self._finish(i + 1)
                    else:
                        return self._fail()
                elif char == "]":
                    # ``[]`` is not a call list; a trailing comma is legal Python.
                    return self._finish(i + 1) if self.calls else self._fail()
                else:
                    self._callee, self._cstate, self._phase = i, "start", "callee"
            elif phase == "callee":
                # ``name`` or ``a.b.c`` (blanks allowed around the dots) up to the
                # ``(``: one character at a time, resumed where the last delta ended,
                # so an endless identifier costs a delta only its own length.
                i, state = self._pos, self._cstate
                opened = False
                while i < n:
                    char = text[i]
                    if char == "(":
                        if state not in ("name", "gap"):
                            return self._fail()
                        opened = True
                        break
                    if char in _WS:
                        if state == "name":
                            state = "gap"
                        elif state == "start":
                            return self._fail()
                    elif char == ".":
                        if state not in ("name", "gap"):
                            return self._fail()
                        state = "dot"
                    elif _is_word(char):
                        if state == "gap" or (state != "name" and char.isdecimal()):
                            return self._fail()
                        state = "name"
                    else:
                        return self._fail()
                    i += 1
                self._pos, self._cstate = i, state
                if not opened:
                    return _INCOMPLETE
                self._phase = "args"
                self._stack, self._string = [], None
            else:  # args
                end = self._scan_args(text)
                if end is None:
                    return self.status
                try:
                    self.calls.append(_convert_call(text[self._callee : end]))
                except _Unsupported:
                    return self._fail()
                self._pos, self._phase = end, "sep"

    def _finish(self, end: int) -> str:
        self.end, self.status = end, _COMPLETE
        return _COMPLETE

    def _scan_args(self, text: str) -> int | None:
        """Offset past the ``)`` that closes the call, None while it is open.

        Brackets inside string literals do not count. Sets ``status`` to ``_INVALID``
        (and returns None) on a mismatched bracket or a newline inside a single-line
        string, which Python rejects whatever follows.
        """
        n = len(text)
        i = self._pos
        stack = self._stack
        string = self._string
        while i < n:
            char = text[i]
            if string is not None:
                quote, triple = string
                if char == "\\":
                    if i + 1 >= n:
                        break  # the escaped character has not arrived
                    i += 2
                elif triple:
                    if text.startswith(quote * 3, i):
                        string, i = None, i + 3
                    elif char == quote and n - i < 3 and text[i:] == quote * (n - i):
                        break  # one or two quotes: maybe the end of the string
                    else:
                        i += 1
                elif char == quote:
                    string, i = None, i + 1
                elif char == "\n":
                    self._fail()
                    return None
                else:
                    i += 1
            elif char in "'\"":
                if text.startswith(char * 3, i):
                    string, i = (char, True), i + 3
                elif n - i < 3 and text[i:] == char * (n - i):
                    break  # ``'`` or ``''`` at the end: a string, or the start of ``'''``
                else:
                    string, i = (char, False), i + 1
            elif char in "([{":
                stack.append(char)
                i += 1
            elif char in ")]}":
                if not stack or stack.pop() != _CLOSERS[char]:
                    self._fail()
                    return None
                i += 1
                if not stack:
                    self._pos, self._string = i, None
                    return i
            else:
                i += 1
        self._pos, self._string = i, string
        return None


def _make_call(name: str, arguments: str) -> dict[str, Any]:
    return {"id": generate_tool_id(), "name": name, "arguments": arguments}


@ToolParserManager.register_module(["pythonic", "gemma3_pythonic"])
class PythonicToolParser(ToolParser):
    """Tool call parser for models that answer with a Python list of calls.

    Used with ``--enable-auto-tool-choice --tool-call-parser pythonic``. ``pythonic``
    suits every model that writes ``[fn(k=v), ...]`` (Llama 3.2/4 with vLLM's pythonic
    templates, ToolACE). ``gemma3_pythonic`` is the same parser under the name that
    goes with the template shipped for Gemma 3
    (``vllm_mlx/templates/tool_chat_template_gemma3_pythonic.jinja``), whose own
    template never shows the model the tools.
    """

    # The shipped Gemma 3 template renders assistant ``tool_calls`` and ``tool`` turns
    # itself, in the format this parser reads.
    SUPPORTS_NATIVE_TOOL_FORMAT = True
    # There is no marker: the very first characters decide, so every delta must reach
    # the parser, and a list the model never closed is resolved when the response ends
    # (the server otherwise only finalizes after a delta that produced nothing).
    REQUIRES_EAGER_STREAMING = True

    def __init__(self, tokenizer=None):
        super().__init__(tokenizer)
        self._reset_stream_state()

    def _reset_stream_state(self) -> None:
        self._mode = "start"  # start | buffer | tail | prose
        self._scanner = _ListScanner()
        self._sent = 0  # offset into the text up to which it has been handed out
        self._tail_started = False

    def reset(self) -> None:
        super().reset()
        self._reset_stream_state()

    def extract_tool_calls(
        self, model_output: str, request: dict[str, Any] | None = None
    ) -> ExtractedToolCallInformation:
        scanner = _ListScanner()
        if scanner.advance(model_output) != _COMPLETE:
            return ExtractedToolCallInformation(
                tools_called=False, tool_calls=[], content=model_output
            )
        tail = model_output[scanner.end :].lstrip()
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[_make_call(n, a) for n, a in scanner.calls],
            content=tail or None,
        )

    @staticmethod
    def _format_streaming(calls: list[dict[str, Any]]) -> dict[str, Any]:
        """Render calls into the streaming delta shape under their absolute index."""
        return {
            "tool_calls": [
                {
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for i, tc in enumerate(calls)
            ]
        }

    def _release(self, current_text: str) -> dict[str, Any] | None:
        """Hand out everything not yet handed out as content; the rest is prose."""
        self._mode = "prose"
        part = current_text[self._sent :]
        self._sent = len(current_text)
        return {"content": part} if part else None

    def _advance(self, current_text: str, final: bool) -> dict[str, Any] | None:
        if self._mode == "prose":
            return self._release(current_text)
        if self._mode == "tail":
            part = current_text[self._sent :]
            self._sent = len(current_text)
            if not self._tail_started:
                part = part.lstrip()
            if not part:
                return None
            self._tail_started = True
            return {"content": part}
        # start / buffer: the text so far may still become a call list.
        status = self._scanner.advance(current_text)
        if status == _INVALID or (final and status == _INCOMPLETE):
            return self._release(current_text)
        if status == _INCOMPLETE:
            self._mode = "buffer"
            return None
        calls = [_make_call(n, a) for n, a in self._scanner.calls]
        self.current_tool_id = len(calls) - 1
        self._sent = self._scanner.end
        self._mode = "tail"
        result = self._format_streaming(calls)
        tail = self._advance(current_text, final)
        if tail:
            result["content"] = tail["content"]
        return result

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int] | None = None,
        current_token_ids: Sequence[int] | None = None,
        delta_token_ids: Sequence[int] | None = None,
        request: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Buffer a leading ``[...`` until it is known to be a call list or not.

        A response that does not start with ``[`` streams straight through as content
        (O(1) per delta). A call list is emitted once, when its ``]`` arrives; a
        ``[`` text that cannot be a call list is released as content at once.
        """
        return self._advance(current_text, final=False)

    def finalize_streaming(self, current_text: str) -> dict[str, Any] | None:
        """Release a ``[...`` that the model never finished as content.

        ``extract_tool_calls`` returns such a response as content, so a stream that
        withheld it while it might still have become a call must hand it out now.
        """
        if self._mode in ("start", "buffer"):
            return self._advance(current_text, final=True)
        return None
