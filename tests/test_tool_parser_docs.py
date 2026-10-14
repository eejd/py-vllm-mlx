# SPDX-License-Identifier: Apache-2.0
"""The parser table of the tool-calling guide matches the parser registry."""

import re
from pathlib import Path

from vllm_mlx.tool_parsers import ToolParserManager

GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guides" / "tool-calling.md"


def _table_rows() -> dict[str, list[str]]:
    """``{parser: [aliases]}`` from the "Supported Parsers" table."""
    text = GUIDE.read_text(encoding="utf-8")
    section = text.split("## Supported Parsers", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for match in re.finditer(r"^\| `([^`]+)` \| ([^|]*)\|", section, re.M):
        rows[match.group(1)] = re.findall(r"`([^`]+)`", match.group(2))
    return rows


def test_every_table_name_resolves_to_its_rows_parser():
    rows = _table_rows()
    assert rows, "parser table not found"
    for parser, aliases in rows.items():
        expected = ToolParserManager.get_tool_parser(parser)
        for alias in aliases:
            assert (
                ToolParserManager.get_tool_parser(alias) is expected
            ), f"{alias} is listed as an alias of {parser} but is another parser"


def test_every_registered_parser_has_a_row():
    rows = _table_rows()
    documented = {
        ToolParserManager.get_tool_parser(name)
        for parser, aliases in rows.items()
        for name in [parser, *aliases]
    }
    missing = sorted(
        name
        for name in ToolParserManager.list_registered()
        if ToolParserManager.get_tool_parser(name) not in documented
    )
    assert not missing, f"registered but not in the guide's parser table: {missing}"


def test_the_pythonic_row_lists_its_alias():
    assert _table_rows()["pythonic"] == ["gemma3_pythonic"]
