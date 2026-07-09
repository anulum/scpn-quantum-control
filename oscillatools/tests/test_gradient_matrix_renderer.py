# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Tests for the gradient coverage matrix renderer
"""Tests for ``tools/render_gradient_coverage_matrix.py``.

The matrix is a generated public documentation page over the live facade
capability map. These tests pin the renderer to the same public surface users
import, verify representative derivative categories, and fail if the checked-in
page or MkDocs navigation drifts from the renderer.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import oscillatools as kuramoto

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAGE = _REPO_ROOT / "docs" / "gradient_coverage_matrix.md"


def _load(name: str, relative: str) -> ModuleType:
    """Load a tool module from its repository file path."""

    spec = importlib.util.spec_from_file_location(name, _REPO_ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = _load("render_gradient_coverage_matrix", "tools/render_gradient_coverage_matrix.py")


def _derivative_like_symbols() -> set[str]:
    """Return derivative-like symbols from the live facade using the renderer classifier."""

    return {
        symbol
        for group, symbols in kuramoto.capabilities().items()
        for symbol in symbols
        if renderer.is_derivative_matrix_symbol(group, symbol)
    }


def test_build_matrix_covers_every_derivative_like_facade_symbol() -> None:
    """The matrix rows must cover every derivative-bearing public facade symbol."""

    matrix = renderer.build_matrix()
    row_symbols = {row.symbol for row in matrix.rows}

    assert matrix.version == kuramoto.__version__
    assert matrix.total_symbols == sum(
        len(symbols) for symbols in kuramoto.capabilities().values()
    )
    assert row_symbols == _derivative_like_symbols()
    assert matrix.row_count == len(row_symbols)


def test_matrix_classifies_representative_derivative_surfaces() -> None:
    """Representative gradients, Hessians, Jacobians, adjoints, and containers are classified."""

    rows = {row.symbol: row for row in renderer.build_matrix().rows}

    assert rows["order_parameter_gradient"].category == "Gradient"
    assert rows["order_parameter_gradient"].base_symbol == "order_parameter"
    assert rows["order_parameter_gradient"].companion_present
    assert rows["order_parameter_hessian"].category == "Hessian"
    assert rows["networked_kuramoto_jacobian"].category == "Jacobian"
    assert rows["networked_kuramoto_jacobian"].base_symbol == "networked_kuramoto_force"
    assert rows["kuramoto_rk4_vjp"].category == "Adjoint"
    assert rows["delayed_delay_sensitivity"].category == "Sensitivity"
    assert rows["MpcOptimumSensitivity"].base_symbol == "MpcOptimum"
    assert rows["NetworkControlGradients"].category == "Gradient container"
    assert renderer.infer_base_symbol("plain_symbol") == "plain_symbol"


def test_render_markdown_lists_counts_boundary_and_every_row() -> None:
    """The generated page must publish counts, evidence boundary, and every row."""

    matrix = renderer.build_matrix()
    document = renderer.render_markdown(matrix)

    assert "# Gradient Coverage Matrix" in document
    assert f"`oscillatools` {matrix.version} exposes {matrix.row_count}" in document
    assert "unsupported derivative routes remain governed" in document
    for category, count in matrix.category_counts.items():
        assert f"| {category} | {count} |" in document
    for row in matrix.rows:
        assert f"| `{row.group}` | `{row.symbol}` | {row.category} |" in document


def test_checked_in_page_matches_renderer_output() -> None:
    """The tracked matrix page must be generated from the current renderer."""

    expected = renderer.render_markdown(renderer.build_matrix())
    assert _PAGE.read_text(encoding="utf-8") == expected


def test_main_writes_page(tmp_path: Path) -> None:
    """The CLI writes the generated matrix page and reports success."""

    out = tmp_path / "gradient_coverage_matrix.md"
    code = renderer.main(["--output", str(out)])

    assert code == 0
    assert "# Gradient Coverage Matrix" in out.read_text(encoding="utf-8")


def test_mkdocs_nav_exposes_the_gradient_matrix() -> None:
    """The documentation site must link the generated matrix."""

    mkdocs = (_REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "Gradient Coverage Matrix: gradient_coverage_matrix.md" in mkdocs
