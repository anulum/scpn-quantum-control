# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the Kuramoto handbook renderer
"""Tests for ``tools/render_kuramoto_handbook.py``."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from scpn_quantum_control import kuramoto

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HANDBOOK = _REPO_ROOT / "docs" / "kuramoto_handbook.md"
_BENCHMARK = _REPO_ROOT / "docs" / "benchmarks" / "tiers" / "kuramoto_tiers.local.json"


def _load(name: str, relative: str) -> ModuleType:
    """Load a tool module from its repository file path."""

    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = _load("render_kuramoto_handbook", "tools/render_kuramoto_handbook.py")


def test_handbook_render_uses_live_facade_and_benchmark_artifact() -> None:
    """The rendered handbook must enumerate the live facade and benchmark operations."""

    artifact = renderer.load_benchmark_artifact(_BENCHMARK)
    document = renderer.render_handbook(artifact)

    for group, symbols in kuramoto.KURAMOTO_CAPABILITIES.items():
        assert f"| `{group}` | {len(symbols)} |" in document
        for symbol in symbols:
            assert f"`{symbol}`" in document

    operations = {result["operation"] for result in artifact["results"]}
    for operation in operations:
        assert f"| `{operation}` |" in document

    assert "Production performance claim allowed: `no`." in document
    assert "[Multi-language Kuramoto tier benchmark](tier_benchmarks.md)" in document
    assert (
        "[Kuramoto Standalone Package Decision](kuramoto_standalone_package_decision.md)"
        in document
    )


def test_checked_in_handbook_matches_renderer_output() -> None:
    """The tracked handbook must be generated from the current renderer."""

    artifact = renderer.load_benchmark_artifact(_BENCHMARK)
    assert _HANDBOOK.read_text(encoding="utf-8") == renderer.render_handbook(artifact)


def test_mkdocs_nav_exposes_the_handbook() -> None:
    """The public documentation site must link the generated handbook."""

    mkdocs = (_REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Kuramoto Handbook: kuramoto_handbook.md" in mkdocs
