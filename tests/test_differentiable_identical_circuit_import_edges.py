# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — identical-circuit import hard-gap edges.
"""Import-boundary tests for identical-circuit external comparison rows."""

from __future__ import annotations

import builtins
from typing import Any

import pytest

import scpn_quantum_control.benchmarks.differentiable_external_comparison as comparison


def test_qiskit_identical_circuit_import_error_becomes_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Qiskit should produce a same-circuit dependency hard gap."""
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "qiskit" or name.startswith("qiskit."):
            raise ImportError(name)
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    _, values, operations, observable_label, fingerprint = comparison._identical_circuit_problem()

    row = comparison._qiskit_identical_circuit_row(
        values=values,
        operations=operations,
        observable_label=observable_label,
        fingerprint=fingerprint,
        scpn_value=1.0,
        scpn_gradient=(0.0,),
    )

    assert row.status == "hard_gap"
    assert row.failure_class == "dependency_missing"
    assert row.backend == "qiskit"


def test_pennylane_identical_circuit_import_error_becomes_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing PennyLane bridge should produce a dependency hard gap."""
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.endswith("pennylane_bridge") or name == "pennylane_bridge":
            raise ImportError(name)
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    circuit, values, operations, observable_label, fingerprint = (
        comparison._identical_circuit_problem()
    )

    row = comparison._pennylane_identical_circuit_row(
        circuit=circuit,
        values=values,
        operations=operations,
        observable_label=observable_label,
        fingerprint=fingerprint,
        scpn_value=1.0,
        scpn_gradient=(0.0,),
    )

    assert row.status == "hard_gap"
    assert row.failure_class == "dependency_missing"
    assert row.backend == "pennylane"


def test_pennylane_identical_circuit_runtime_import_error_becomes_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PennyLane runtime import errors should remain dependency gaps."""
    from scpn_quantum_control.phase import pennylane_bridge

    def missing_round_trip(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ImportError("pennylane runtime missing")

    monkeypatch.setattr(
        pennylane_bridge,
        "check_pennylane_phase_qnode_round_trip",
        missing_round_trip,
    )
    circuit, values, operations, observable_label, fingerprint = (
        comparison._identical_circuit_problem()
    )

    row = comparison._pennylane_identical_circuit_row(
        circuit=circuit,
        values=values,
        operations=operations,
        observable_label=observable_label,
        fingerprint=fingerprint,
        scpn_value=1.0,
        scpn_gradient=(0.0,),
    )

    assert row.status == "hard_gap"
    assert row.failure_class == "dependency_missing"
