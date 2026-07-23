# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — aggregate differentiable quality-gate registry
"""Aggregate focused differentiable quality gates in canonical order."""

from __future__ import annotations

from tools import differentiable_api_quality_gates as _api_gates
from tools import differentiable_rust_python_inventory_quality_gates as _inventory_gates
from tools import differentiable_scalar_kernels_quality_gates as _scalar_gates

Gate = tuple[str, list[str]]


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build every focused differentiable static gate.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Scalar-kernel, Rust/Python inventory, and unified-API static gates.

    """
    return [
        *_scalar_gates.build_static_quality_gates(python),
        *_inventory_gates.build_static_quality_gates(python),
        *_api_gates.build_static_quality_gates(python),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build every focused differentiable exact-coverage gate.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Scalar-kernel, Rust/Python inventory, and unified-API coverage gates.

    """
    return [
        *_scalar_gates.build_coverage_gates(python),
        *_inventory_gates.build_coverage_gates(python),
        *_api_gates.build_coverage_gates(python),
    ]


__all__ = ["build_coverage_gates", "build_static_quality_gates"]
