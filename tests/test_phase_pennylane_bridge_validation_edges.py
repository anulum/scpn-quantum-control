# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PennyLane Bridge Validation Edge Tests
"""Exercise validation, compatibility and defensive PennyLane bridge edges."""

from __future__ import annotations

import builtins
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pytest
from _phase_pennylane_bridge_test_helpers import (
    _closed_form_gradient,
    _FakePennyLane,
    _objective,
)
from numpy.typing import NDArray

import scpn_quantum_control.phase.pennylane_bridge as pennylane_bridge
from scpn_quantum_control.phase import (
    PauliCovarianceObservable,
    PauliTerm,
    PennyLaneMaturityAuditResult,
    PhaseQNodeCircuit,
    PhaseQNodeSupportError,
    SparsePauliHamiltonian,
    build_pennylane_qnode_from_phase_qnode,
    check_pennylane_parameter_shift_agreement,
    check_pennylane_phase_qnode_round_trip,
    check_pennylane_qnode_round_trip,
    is_phase_pennylane_available,
    phase_qnode_support_report,
)

FloatArray = NDArray[np.float64]


def test_pennylane_bridge_translates_the_real_optional_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expose the install-extra remedy when PennyLane itself cannot import."""
    original_import = cast(Callable[..., Any], builtins.__import__)

    def blocked_import(name: str, *args: object, **kwargs: object) -> Any:
        if name == "pennylane":
            raise ImportError("blocked optional dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    assert not is_phase_pennylane_available()
    with pytest.raises(ImportError, match=r"scpn-quantum-control\[pennylane\]"):
        check_pennylane_parameter_shift_agreement(
            _objective,
            _closed_form_gradient,
            np.array([0.2, -0.4], dtype=float),
        )


@pytest.mark.parametrize(
    ("values", "tolerance", "gradient", "match"),
    [
        (np.array([[0.2]], dtype=float), 1e-6, _closed_form_gradient, "one-dimensional"),
        (np.array([0.2, -0.4]), 1e-6, lambda values: values[:1], "shape"),
        (np.array([np.nan]), 1e-6, lambda values: values, "finite"),
        (np.array([0.2, -0.4]), -1.0, _closed_form_gradient, "tolerance"),
        (np.array([0.2, -0.4]), float("inf"), _closed_form_gradient, "tolerance"),
    ],
)
def test_pennylane_bridge_rejects_invalid_gradient_contracts(
    monkeypatch: pytest.MonkeyPatch,
    values: FloatArray,
    tolerance: float,
    gradient: Any,
    match: str,
) -> None:
    """Reject malformed vectors and tolerances through the public agreement API."""
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    with pytest.raises(ValueError, match=match):
        check_pennylane_parameter_shift_agreement(
            _objective,
            gradient,
            values,
            tolerance=tolerance,
        )


@pytest.mark.parametrize("bad_value", [np.array([1.0]), True, 1.0 + 2.0j])
def test_pennylane_bridge_rejects_non_scalar_qnode_values(
    monkeypatch: pytest.MonkeyPatch,
    bad_value: object,
) -> None:
    """Reject non-real scalar results returned by an external QNode."""
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    with pytest.raises(ValueError, match="real numeric scalar"):
        check_pennylane_qnode_round_trip(
            _objective,
            lambda values: cast(float, bad_value),
            _closed_form_gradient,
            np.array([0.2, -0.4], dtype=float),
        )


def test_pennylane_bridge_accepts_empty_parameter_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve zero-width agreement semantics without reduction failures."""
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    def objective(_values: FloatArray) -> float:
        return 1.0

    def gradient(_values: FloatArray) -> FloatArray:
        return np.array([], dtype=float)

    agreement = check_pennylane_parameter_shift_agreement(
        objective,
        gradient,
        np.array([], dtype=float),
    )
    round_trip = check_pennylane_qnode_round_trip(
        objective,
        objective,
        gradient,
        np.array([], dtype=float),
    )

    assert agreement.passed
    assert agreement.max_abs_error == 0.0
    assert round_trip.passed
    assert round_trip.gradient_max_abs_error == 0.0


def test_pennylane_bridge_rejects_unsupported_phase_qnode_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface the Phase-QNode support report before constructing a device."""
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(1, (("u3", (0,), 0),), "pauli_z")

    with pytest.raises(PhaseQNodeSupportError, match="unsupported gates"):
        build_pennylane_qnode_from_phase_qnode(circuit)

    assert fake_qml.devices == []


def test_pennylane_bridge_uses_array_and_gradient_compatibility_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Support PennyLane variants without requires-grad or argnum keywords."""

    class _LegacyArrayNamespace:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def array(self, values: FloatArray, **kwargs: object) -> FloatArray:
            self.calls.append(kwargs)
            if kwargs:
                raise TypeError("requires_grad unavailable")
            return np.asarray(values, dtype=float)

    fake_qml = _FakePennyLane()
    array_namespace = _LegacyArrayNamespace()
    cast(Any, fake_qml).numpy = array_namespace
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(1, (("ry", (0,), 0),), "pauli_z")

    result = check_pennylane_phase_qnode_round_trip(circuit, np.array([0.3]))

    assert result.passed
    assert array_namespace.calls == [{"requires_grad": True}, {}]


def test_pennylane_bridge_accepts_plain_sequence_qnode_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Infer a missing shape attribute and reject the wrong inferred width."""
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(1, (("ry", (0,), 0),), "pauli_z")
    conversion = build_pennylane_qnode_from_phase_qnode(circuit)

    assert cast(float, conversion.qnode([0.3])) == pytest.approx(float(np.cos(0.3)))
    with pytest.raises(ValueError, match="shape"):
        conversion.qnode([])


def test_pennylane_bridge_converts_special_gates_and_sparse_paulis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Convert the registered controlled gates and multi-term Pauli observable."""
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        3,
        (
            ("cs", (0, 1)),
            ("ct", (0, 1)),
            ("ccz", (0, 1, 2)),
            ("h", (2,)),
        ),
        SparsePauliHamiltonian(
            (
                PauliTerm(0.5, ((0, "x"), (1, "y"))),
                PauliTerm(-0.25, ((2, "z"),)),
            )
        ),
    )
    conversion = build_pennylane_qnode_from_phase_qnode(circuit)

    conversion.qnode(np.array([], dtype=float))

    names = [name for name, _args, _kwargs in fake_qml.calls]
    assert names[:4] == [
        "ControlledPhaseShift",
        "ControlledPhaseShift",
        "ControlledQubitUnitary",
        "Hadamard",
    ]
    assert "Hamiltonian" in names
    assert conversion.observable_kind == "sparse_pauli_hamiltonian"


def test_pennylane_bridge_fails_closed_for_corrupted_generated_qnodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject impossible gate and observable corruption at generated-QNode call time."""
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    valid = PhaseQNodeCircuit(1, (("h", (0,)),), "pauli_z")
    valid_report = phase_qnode_support_report(valid, np.array([], dtype=float))
    monkeypatch.setattr(pennylane_bridge, "phase_qnode_support_report", lambda *_: valid_report)

    invalid_gate = PhaseQNodeCircuit(1, (("u3", (0,), 0),), "pauli_z")
    gate_conversion = build_pennylane_qnode_from_phase_qnode(invalid_gate)
    with pytest.raises(ValueError, match="gate 'u3'"):
        gate_conversion.qnode(np.array([0.2]))

    covariance = PauliCovarianceObservable(
        PauliTerm(1.0, ((0, "z"),)),
        PauliTerm(1.0, ((0, "x"),)),
    )
    covariance_circuit = PhaseQNodeCircuit(1, (("h", (0,)),), "pauli_z")
    covariance_conversion = build_pennylane_qnode_from_phase_qnode(covariance_circuit)
    object.__setattr__(covariance_circuit, "observable", covariance)
    with pytest.raises(ValueError, match="covariance observables"):
        covariance_conversion.qnode(np.array([], dtype=float))

    unknown_circuit = PhaseQNodeCircuit(1, (("h", (0,)),), "pauli_z")
    unknown_conversion = build_pennylane_qnode_from_phase_qnode(unknown_circuit)
    object.__setattr__(unknown_circuit, "observable", object())
    with pytest.raises(ValueError, match="object"):
        unknown_conversion.qnode(np.array([], dtype=float))

    term = PauliTerm(1.0, ((0, "z"),))
    label_circuit = PhaseQNodeCircuit(1, (("h", (0,)),), term)
    label_conversion = build_pennylane_qnode_from_phase_qnode(label_circuit)
    object.__setattr__(term, "factors", ((0, "q"),))
    with pytest.raises(ValueError, match="Pauli label 'q'"):
        label_conversion.qnode(np.array([], dtype=float))


def test_pennylane_maturity_payload_serialises_dataclass_and_raw_evidence() -> None:
    """Keep the public audit payload robust for non-bridge evidence objects."""

    @dataclass(frozen=True)
    class BareEvidence:
        value: int

    result = PennyLaneMaturityAuditResult(
        identical_circuit_ready=False,
        ready_for_provider_exceedance=False,
        evidence={"dataclass": BareEvidence(3), "raw": "pending"},
        required_capabilities={},
        promotion_metadata={},
        open_gaps=("raw",),
    )

    payload = cast(dict[str, Any], result.to_dict())
    assert payload["evidence"] == {"dataclass": {"value": 3}, "raw": "pending"}
