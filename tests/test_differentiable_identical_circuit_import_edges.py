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
from dataclasses import dataclass
from typing import Any

import pytest

import scpn_quantum_control.benchmarks.differentiable_external_comparison as comparison


@dataclass(frozen=True)
class _FakeQiskitGradientResult:
    """Small statevector result for the Qiskit success row boundary."""

    value: float
    gradient: tuple[float, ...]
    evaluations: int


@dataclass(frozen=True)
class _FakePennyLaneRoundTripResult:
    """Small PennyLane round-trip result for the success row boundary."""

    pennylane_value: float
    pennylane_gradient: tuple[float, ...]
    evaluations: int


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


def test_qiskit_identical_circuit_success_row_uses_bridge_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qiskit success rows should record value and gradient from the bridge."""
    from scpn_quantum_control.phase import qiskit_bridge

    def statevector_parameter_shift(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return _FakeQiskitGradientResult(value=0.5, gradient=(0.25,), evaluations=2)

    monkeypatch.setattr(
        qiskit_bridge,
        "execute_qiskit_statevector_parameter_shift",
        statevector_parameter_shift,
    )
    _, values, operations, observable_label, fingerprint = comparison._identical_circuit_problem()

    row = comparison._qiskit_identical_circuit_row(
        values=values,
        operations=operations,
        observable_label=observable_label,
        fingerprint=fingerprint,
        scpn_value=0.5,
        scpn_gradient=(0.25,),
    )

    assert row.status == "success"
    assert row.backend == "qiskit"
    assert row.backend_value == 0.5
    assert row.backend_gradient == (0.25,)


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


def test_pennylane_identical_circuit_success_row_uses_bridge_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PennyLane success rows should record value and gradient from the bridge."""
    from scpn_quantum_control.phase import pennylane_bridge

    def round_trip(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return _FakePennyLaneRoundTripResult(
            pennylane_value=0.5,
            pennylane_gradient=(0.25,),
            evaluations=2,
        )

    monkeypatch.setattr(
        pennylane_bridge,
        "check_pennylane_phase_qnode_round_trip",
        round_trip,
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
        scpn_value=0.5,
        scpn_gradient=(0.25,),
    )

    assert row.status == "success"
    assert row.backend == "pennylane"
    assert row.backend_value == 0.5
    assert row.backend_gradient == (0.25,)


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
