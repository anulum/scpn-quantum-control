# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Enzyme Audit Tests
"""Architecture tests for Enzyme/MLIR maturity and toolchain probing."""

from __future__ import annotations

import ast
import inspect

import scpn_quantum_control.compiler.mlir as facade
import scpn_quantum_control.compiler.mlir_enzyme_audit as leaf

PRIVATE_NAMES = (
    "_default_enzyme_mlir_audit_circuit",
    "_enzyme_mlir_toolchain_status",
    "_probe_toolchain_version",
    "_resolve_toolchain_executable",
)


def test_enzyme_audit_has_no_facade_back_edge() -> None:
    """Keep Enzyme audit imports one-way from the compiler facade."""
    tree = ast.parse(inspect.getsource(leaf))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "mlir" not in relative_imports


def test_enzyme_audit_facade_exports_are_exact_leaf_aliases() -> None:
    """Preserve the Enzyme audit and probe helper facade identities."""
    assert facade.run_enzyme_mlir_maturity_audit is leaf.run_enzyme_mlir_maturity_audit
    for name in PRIVATE_NAMES:
        assert getattr(facade, name) is getattr(leaf, name)


def test_enzyme_audit_public_export_remains_declared() -> None:
    """Retain the maturity audit in the facade export contract."""
    assert "run_enzyme_mlir_maturity_audit" in facade.__all__
