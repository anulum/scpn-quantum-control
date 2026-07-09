#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Gradient coverage matrix renderer
"""Render the gradient coverage matrix from the live oscillatools facade."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path

import oscillatools as kuramoto

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "gradient_coverage_matrix.md"

_CATEGORY_ORDER: tuple[str, ...] = (
    "Gradient",
    "Hessian",
    "Jacobian",
    "Adjoint",
    "Sensitivity",
    "Gradient container",
)
_DERIVATIVE_GROUPS: frozenset[str] = frozenset(
    {
        "forces",
        "integrators",
        "observables",
        "diagnostics",
        "analysis",
        "control_and_design",
    }
)
_SUFFIX_BASES: tuple[tuple[str, str], ...] = (
    ("_trajectory_value_and_grad", "_trajectory_value"),
    ("_terminal_value_and_grad", "_terminal_value"),
    ("_control_value_and_grad", "_control_value"),
    ("_value_and_grad", "_value"),
    ("_parameter_sensitivity", ""),
    ("_state_sensitivity", ""),
    ("_delay_sensitivity", ""),
    ("_delay_gradient", "_delay"),
    ("_gradient", ""),
    ("_hessian", ""),
    ("_jacobian", ""),
    ("_sensitivity", ""),
    ("_vjp", "_trajectory"),
)


@dataclass(frozen=True)
class GradientCoverageRow:
    """One public facade symbol carrying derivative or sensitivity semantics.

    Parameters
    ----------
    group:
        Capability group that exposes the symbol.
    symbol:
        Public facade symbol detected from :func:`oscillatools.capabilities`.
    category:
        Derivative category inferred from the symbol name.
    base_symbol:
        Inferred primal or objective symbol. This is a documentation hint, not a
        proof that the exact symbol exists in the facade.
    companion_present:
        ``True`` when ``base_symbol`` is also exported by the public facade.
    """

    group: str
    symbol: str
    category: str
    base_symbol: str
    companion_present: bool


@dataclass(frozen=True)
class GradientCoverageMatrix:
    """Versioned derivative-surface inventory derived from the public facade."""

    version: str
    total_symbols: int
    rows: tuple[GradientCoverageRow, ...]

    @property
    def row_count(self) -> int:
        """Return the number of derivative-bearing public symbols."""

        return len(self.rows)

    @property
    def category_counts(self) -> Mapping[str, int]:
        """Return derivative rows grouped by category."""

        counts = Counter(row.category for row in self.rows)
        return {category: counts[category] for category in _CATEGORY_ORDER if counts[category]}


def build_matrix() -> GradientCoverageMatrix:
    """Build the derivative coverage matrix from ``oscillatools.capabilities``."""

    capabilities = kuramoto.capabilities()
    all_symbols = frozenset(_all_symbols(capabilities))
    rows: list[GradientCoverageRow] = []
    for group, symbols in capabilities.items():
        for symbol in symbols:
            category = derivative_category(symbol)
            if group not in _DERIVATIVE_GROUPS or category is None:
                continue
            base_symbol = infer_base_symbol(symbol, all_symbols)
            rows.append(
                GradientCoverageRow(
                    group=group,
                    symbol=symbol,
                    category=category,
                    base_symbol=base_symbol,
                    companion_present=base_symbol in all_symbols,
                )
            )
    return GradientCoverageMatrix(
        version=kuramoto.__version__,
        total_symbols=len(all_symbols),
        rows=tuple(rows),
    )


def is_derivative_matrix_symbol(group: str, symbol: str) -> bool:
    """Return whether ``symbol`` belongs in the public derivative matrix.

    The matrix covers capability groups that expose mathematical derivative
    APIs. Type aliases, dispatch helpers, visualisation helpers, and
    ``last_*_tier_used`` introspection helpers stay outside this page because
    they do not compute derivatives.
    """

    return group in _DERIVATIVE_GROUPS and derivative_category(symbol) is not None


def derivative_category(symbol: str) -> str | None:
    """Return the derivative category encoded in ``symbol``.

    Parameters
    ----------
    symbol:
        Public facade symbol name.

    Returns
    -------
    str | None
        Category label when the symbol names a derivative, adjoint,
        sensitivity, or gradient result container; otherwise ``None``.
    """

    if symbol.endswith("Gradients"):
        return "Gradient container"
    lowered = symbol.lower()
    if "hessian" in lowered:
        return "Hessian"
    if "jacobian" in lowered:
        return "Jacobian"
    if lowered.endswith("_vjp") or "_vjp_" in lowered:
        return "Adjoint"
    if "sensitivity" in lowered:
        return "Sensitivity"
    if "gradient" in lowered or lowered.endswith("_grad") or "value_and_grad" in lowered:
        return "Gradient"
    return None


def infer_base_symbol(symbol: str, exported_symbols: Set[str] | None = None) -> str:
    """Infer the primal or objective symbol associated with a derivative name."""

    exported = exported_symbols or frozenset()
    for suffix, replacement in _SUFFIX_BASES:
        if symbol.endswith(suffix):
            candidate = f"{symbol[: -len(suffix)]}{replacement}"
            if suffix == "_jacobian" and candidate not in exported:
                force_candidate = f"{candidate}_force"
                if force_candidate in exported:
                    return force_candidate
            return candidate
    if symbol.endswith("Sensitivity"):
        return symbol[: -len("Sensitivity")]
    if symbol.endswith("Gradients"):
        return symbol[: -len("Gradients")]
    return symbol


def render_markdown(matrix: GradientCoverageMatrix) -> str:
    """Render the matrix as a generated Markdown documentation page."""

    lines: list[str] = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "Generated by tools/render_gradient_coverage_matrix.py — do not edit by hand.",
        "-->",
        "",
        "# Gradient Coverage Matrix",
        "",
        f"`oscillatools` {matrix.version} exposes {matrix.row_count} derivative-bearing",
        f"public symbols across {matrix.total_symbols} facade symbols. This generated page",
        "is an inventory of the public gradient, Hessian, Jacobian, adjoint, sensitivity,",
        "and gradient-container surfaces declared by the live capability map.",
        "",
        "Evidence boundary: this page documents exported derivative surfaces. It is not a",
        "claim that every model family has automatic differentiation through every backend;",
        "unsupported derivative routes remain governed by their public API contracts and",
        "fail-closed tests.",
        "",
        "## Category Counts",
        "",
        "| Category | Public symbols |",
        "|---|--:|",
    ]
    for category, count in matrix.category_counts.items():
        lines.append(f"| {category} | {count} |")
    lines.extend(
        [
            "",
            "## Matrix",
            "",
            "| Group | Symbol | Category | Inferred primal/objective | Primal exported |",
            "|---|---|---|---|---|",
        ]
    )
    for row in matrix.rows:
        primal = "yes" if row.companion_present else "no"
        lines.append(
            f"| `{row.group}` | `{row.symbol}` | {row.category} | `{row.base_symbol}` | {primal} |"
        )
    lines.append("")
    return "\n".join(lines)


def _all_symbols(capabilities: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    return tuple(symbol for symbols in capabilities.values() for symbol in symbols)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the gradient coverage matrix page and write it to disk."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown output path")
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    document = render_markdown(build_matrix())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(f"[gradient-matrix] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
