# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Tests for the capability snapshot renderer
"""Tests for ``tools/capability_snapshot.py``.

The snapshot is the diffable inventory of the public surface, so the tests pin
that it mirrors the live facade exactly (version, groups, and every symbol), that
its Markdown page lists every group and symbol, that the JSON view is stable, and
that the checked-in page matches the renderer output byte for byte.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import oscillatools as kuramoto

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAGE = _REPO_ROOT / "docs" / "capabilities.md"


def _load(name: str, relative: str) -> ModuleType:
    """Load a tool module from its repository file path."""

    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


snapshot_tool = _load("capability_snapshot", "tools/capability_snapshot.py")


def test_build_snapshot_mirrors_the_live_facade() -> None:
    """The snapshot must capture the exact facade version, groups, and symbols."""

    snapshot = snapshot_tool.build_snapshot()
    facade = kuramoto.capabilities()

    assert snapshot.version == kuramoto.__version__
    assert set(snapshot.groups) == set(facade)
    for group, symbols in facade.items():
        assert snapshot.groups[group] == tuple(symbols)
    assert snapshot.group_count == len(facade)
    assert snapshot.total_symbols == sum(len(symbols) for symbols in facade.values())


def test_to_json_dict_is_stable_and_complete() -> None:
    """The JSON view must round-trip and carry every group and count."""

    snapshot = snapshot_tool.build_snapshot()
    payload = snapshot.to_json_dict()

    assert payload["version"] == snapshot.version
    assert payload["group_count"] == snapshot.group_count
    assert payload["total_symbols"] == snapshot.total_symbols
    assert set(payload["groups"]) == set(snapshot.groups)
    for group, symbols in snapshot.groups.items():
        assert payload["groups"][group] == list(symbols)
    # Serialisable and order-stable.
    assert json.loads(json.dumps(payload, sort_keys=True))["version"] == snapshot.version


def test_render_markdown_lists_every_group_and_symbol() -> None:
    """The rendered page must enumerate every group heading and symbol."""

    snapshot = snapshot_tool.build_snapshot()
    document = snapshot_tool.render_markdown(snapshot)

    assert f"`oscillatools` {snapshot.version} exposes {snapshot.total_symbols} public" in document
    for group, symbols in snapshot.groups.items():
        assert f"| `{group}` | {len(symbols)} |" in document
        assert f"## `{group}` ({len(symbols)})" in document
        for symbol in symbols:
            assert f"`{symbol}`" in document


def test_checked_in_page_matches_renderer_output() -> None:
    """The tracked capability page must be generated from the current renderer."""

    expected = snapshot_tool.render_markdown(snapshot_tool.build_snapshot())
    assert _PAGE.read_text(encoding="utf-8") == expected


def test_main_writes_page(tmp_path: Path) -> None:
    """The CLI writes the capability page and reports success."""

    out = tmp_path / "capabilities.md"
    code = snapshot_tool.main(["--output", str(out)])
    assert code == 0
    assert "# Capability Snapshot" in out.read_text(encoding="utf-8")
