# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase PennyLane Bridge
"""Tests for optional PennyLane gradient-agreement checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.phase.pennylane_bridge as pennylane_bridge
from scpn_quantum_control.phase import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PennyLaneGradientAgreementResult,
    PennyLaneMaturityAuditResult,
    PennyLanePluginMatrixResult,
    PennyLaneQNodeConversionResult,
    PennyLaneRoundTripResult,
    PhaseQNodeCircuit,
    build_pennylane_qnode_from_phase_qnode,
    check_pennylane_parameter_shift_agreement,
    check_pennylane_phase_qnode_round_trip,
    check_pennylane_qnode_round_trip,
    execute_phase_qnode_circuit,
    is_phase_pennylane_available,
    multi_frequency_parameter_shift_rule,
    parameter_shift_phase_qnode_gradient,
    run_pennylane_maturity_audit,
    run_pennylane_plugin_matrix,
)


class _FakeObservable:
    def __init__(
        self,
        name: str,
        wires: int | tuple[int, ...] | list[int],
        *,
        coefficient: float = 1.0,
        terms: tuple[_FakeObservable, ...] = (),
    ) -> None:
        self.name = name
        self.wires = tuple(wires) if isinstance(wires, list | tuple) else (int(wires),)
        self.coefficient = float(coefficient)
        self.terms = terms

    def __matmul__(self, other: _FakeObservable) -> _FakeObservable:
        return _FakeObservable("Prod", self.wires + other.wires, terms=(self, other))

    def __rmul__(self, coefficient: float) -> _FakeObservable:
        return _FakeObservable(
            self.name,
            self.wires,
            coefficient=float(coefficient) * self.coefficient,
            terms=self.terms,
        )


class _FakePennyLane:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.devices: list[dict[str, object]] = []

    def device(self, name: str, *, wires: int, shots: int | None = None) -> dict[str, object]:
        payload = {"name": name, "wires": wires, "shots": shots}
        self.devices.append(payload)
        return payload

    def qnode(
        self,
        device: object,
        **metadata: object,
    ) -> Callable[[Callable[[np.ndarray], object]], Callable[[np.ndarray], float]]:
        def decorate(function: Callable[[np.ndarray], object]) -> Callable[[np.ndarray], float]:
            def wrapper(params: np.ndarray) -> float:
                function(params)
                circuit = cast(Any, wrapper)._scpn_phase_qnode_circuit
                return execute_phase_qnode_circuit(circuit, params).value

            cast(Any, wrapper).device = device
            cast(Any, wrapper).metadata = metadata
            return wrapper

        return decorate

    def grad(self, qnode: Callable[[np.ndarray], object]) -> Callable[[np.ndarray], np.ndarray]:
        def gradient(params: np.ndarray) -> np.ndarray:
            circuit = cast(Any, qnode)._scpn_phase_qnode_circuit
            return parameter_shift_phase_qnode_gradient(circuit, params).gradient

        return gradient

    def expval(self, observable: _FakeObservable) -> _FakeObservable:
        self.calls.append(("expval", (observable,), {}))
        return observable

    def Hamiltonian(
        self,
        coefficients: list[float],
        observables: list[_FakeObservable],
    ) -> _FakeObservable:
        self.calls.append(("Hamiltonian", (tuple(coefficients), tuple(observables)), {}))
        return _FakeObservable("Hamiltonian", (), terms=tuple(observables))

    def Hermitian(self, matrix: np.ndarray, *, wires: range) -> _FakeObservable:
        self.calls.append(("Hermitian", (np.asarray(matrix),), {"wires": tuple(wires)}))
        return _FakeObservable("Hermitian", tuple(wires))

    def __getattr__(self, name: str) -> Callable[..., _FakeObservable]:
        if name in {
            "Hadamard",
            "PauliX",
            "PauliY",
            "PauliZ",
            "S",
            "T",
            "SX",
            "CNOT",
            "CZ",
            "CY",
            "SWAP",
            "RX",
            "RY",
            "RZ",
            "PhaseShift",
            "CRX",
            "CRY",
            "CRZ",
            "IsingXX",
            "IsingYY",
            "IsingZZ",
        }:
            return lambda *args, **kwargs: self._operation(name, *args, **kwargs)
        raise AttributeError(name)

    def _operation(self, name: str, *args: object, **kwargs: object) -> _FakeObservable:
        self.calls.append((name, args, kwargs))
        wires = kwargs.get("wires", ())
        wire_value = cast(int | tuple[int, ...] | list[int], wires)
        if name in {"PauliX", "PauliY", "PauliZ"}:
            return _FakeObservable(name, wire_value)
        return _FakeObservable(name, wire_value)


def _objective(values: np.ndarray) -> float:
    return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))


def _closed_form_gradient(values: np.ndarray) -> np.ndarray:
    return np.array([-np.sin(values[0]), 0.25 * np.cos(values[1])], dtype=float)


def test_pennylane_bridge_reports_gradient_agreement(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    result = check_pennylane_parameter_shift_agreement(
        _objective,
        _closed_form_gradient,
        np.array([0.2, -0.4], dtype=float),
        tolerance=1e-12,
    )

    assert isinstance(result, PennyLaneGradientAgreementResult)
    assert is_phase_pennylane_available()
    assert result.passed
    assert result.max_abs_error <= 1e-12
    assert result.evaluations == 5
    np.testing.assert_allclose(result.scpn_gradient, result.pennylane_gradient, atol=1e-12)


def test_pennylane_bridge_reports_gradient_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    def shifted_gradient(values: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, _closed_form_gradient(values) + np.array([0.01, 0.0], dtype=float))

    result = check_pennylane_parameter_shift_agreement(
        _objective,
        shifted_gradient,
        np.array([0.2, -0.4], dtype=float),
        tolerance=1e-4,
    )

    assert not result.passed
    assert result.max_abs_error > result.tolerance
    assert result.l2_error > 0.0


def test_pennylane_bridge_reports_multi_frequency_gradient_agreement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    def external_gradient(values: np.ndarray) -> np.ndarray:
        return np.array([np.cos(values[0]) - 0.2 * np.sin(2.0 * values[0])], dtype=float)

    result = check_pennylane_parameter_shift_agreement(
        objective,
        external_gradient,
        np.array([0.4], dtype=float),
        tolerance=1e-12,
        rule=rule,
    )
    payload = result.to_dict()

    assert result.passed
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    assert payload["shift_terms"] == len(rule.terms)
    np.testing.assert_allclose(result.scpn_gradient, result.pennylane_gradient, atol=1e-12)


def test_pennylane_bridge_reports_qnode_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    result = check_pennylane_qnode_round_trip(
        _objective,
        _objective,
        _closed_form_gradient,
        np.array([0.2, -0.4], dtype=float),
        value_tolerance=1e-12,
        gradient_tolerance=1e-12,
    )
    payload = result.to_dict()

    assert isinstance(result, PennyLaneRoundTripResult)
    assert result.passed
    assert result.value_abs_error <= 1e-12
    assert result.gradient_max_abs_error <= 1e-12
    assert result.evaluations == 5
    assert payload["passed"] is True
    assert payload["evaluations"] == 5
    np.testing.assert_allclose(result.scpn_gradient, result.pennylane_gradient, atol=1e-12)


def test_pennylane_bridge_reports_multi_frequency_qnode_round_trip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])

    def objective(values: np.ndarray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    def external_gradient(values: np.ndarray) -> np.ndarray:
        return np.array([np.cos(values[0]) - 0.2 * np.sin(2.0 * values[0])], dtype=float)

    result = check_pennylane_qnode_round_trip(
        objective,
        objective,
        external_gradient,
        np.array([0.4], dtype=float),
        value_tolerance=1e-12,
        gradient_tolerance=1e-12,
        rule=rule,
    )

    assert result.passed
    assert result.method == "multi_frequency_parameter_shift"
    assert result.shift_terms == len(rule.terms)
    assert result.evaluations == 1 + 2 * len(rule.terms)
    np.testing.assert_allclose(result.scpn_gradient, result.pennylane_gradient, atol=1e-12)


def test_pennylane_bridge_builds_phase_qnode_conversion_and_round_trips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0), ("rx", (0,), 1)),
        PauliTerm(1.0, ((0, "z"),)),
    )
    params = np.array([0.37, -0.29], dtype=float)

    conversion = build_pennylane_qnode_from_phase_qnode(circuit, shots=None)
    value = cast(float, conversion.qnode(params))
    gradient = cast(np.ndarray, conversion.gradient(params))
    round_trip = check_pennylane_phase_qnode_round_trip(circuit, params)
    payload = cast(dict[str, Any], conversion.to_dict())

    assert isinstance(conversion, PennyLaneQNodeConversionResult)
    assert conversion.device_name == "default.qubit"
    assert conversion.shots is None
    assert conversion.diff_method == "parameter-shift"
    assert conversion.observable_kind == "pauli_z"
    assert conversion.differentiable_parameters == (0, 1)
    assert payload["hardware_execution"] is False
    assert cast(str, payload["claim_boundary"]).startswith("bounded PennyLane QNode conversion")
    np.testing.assert_allclose(value, execute_phase_qnode_circuit(circuit, params).value)
    np.testing.assert_allclose(
        gradient,
        parameter_shift_phase_qnode_gradient(circuit, params).gradient,
        atol=1e-12,
    )
    assert round_trip.passed
    assert round_trip.value_abs_error <= 1e-12
    assert round_trip.gradient_max_abs_error <= 1e-12
    assert fake_qml.devices
    assert all(
        device == {"name": "default.qubit", "wires": 1, "shots": None}
        for device in fake_qml.devices
    )
    assert ("RY", (params[0],), {"wires": 0}) in fake_qml.calls
    assert ("RX", (params[1],), {"wires": 0}) in fake_qml.calls
    assert any(call[0] == "expval" for call in fake_qml.calls)


def test_pennylane_maturity_audit_records_export_metadata_and_provider_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0), ("rx", (0,), 1)),
        PauliTerm(1.0, ((0, "z"),)),
    )
    values = np.array([0.37, -0.29], dtype=float)

    result = run_pennylane_maturity_audit(
        objective=_objective,
        pennylane_objective=_objective,
        pennylane_gradient=_closed_form_gradient,
        values=np.array([0.2, -0.4], dtype=float),
        circuit=circuit,
        phase_qnode_values=values,
        value_tolerance=1e-12,
        gradient_tolerance=1e-12,
    )

    assert isinstance(result, PennyLaneMaturityAuditResult)
    assert not result.identical_circuit_ready
    assert not result.ready_for_provider_exceedance
    gradient_agreement = cast(
        PennyLaneGradientAgreementResult,
        result.evidence["gradient_agreement"],
    )
    caller_qnode_round_trip = cast(
        PennyLaneRoundTripResult,
        result.evidence["caller_qnode_round_trip"],
    )
    phase_qnode_export_round_trip = cast(
        PennyLaneRoundTripResult,
        result.evidence["phase_qnode_export_round_trip"],
    )
    assert gradient_agreement.passed
    assert caller_qnode_round_trip.passed
    assert phase_qnode_export_round_trip.passed
    assert result.evidence["phase_qnode_import_round_trip"] is None
    plugin_matrix = result.evidence["pennylane_plugin_matrix"]
    assert isinstance(plugin_matrix, PennyLanePluginMatrixResult)
    assert plugin_matrix.local_plugin_parity_ready
    assert not plugin_matrix.provider_plugin_execution_ready
    assert not plugin_matrix.ready_for_provider_exceedance
    assert plugin_matrix.route_status("default_qubit_exact_state") == "passed"
    assert plugin_matrix.route_status("provider_plugin_execution") == "blocked"
    assert result.required_capabilities["phase_qnode_import_round_trip"] == "blocked"
    assert result.required_capabilities["pennylane_plugin_matrix"] == "passed"
    assert result.promotion_metadata["device_name"] == "default.qubit"
    assert result.promotion_metadata["shots"] is None
    assert result.promotion_metadata["diff_method"] == "parameter-shift"
    assert result.promotion_metadata["phase_qnode_parameter_shift_evaluations"] == 4
    evaluation_groups = cast(
        list[dict[str, object]],
        result.promotion_metadata["phase_qnode_evaluation_groups"],
    )
    assert len(evaluation_groups) == 2
    assert "provider_plugin_execution" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    required_capabilities = cast(dict[str, str], payload["required_capabilities"])
    assert payload["claim_boundary"] == "bounded_pennylane_provider_maturity_audit"
    assert required_capabilities["device_metadata"] == "passed"
    assert required_capabilities["pennylane_plugin_matrix"] == "passed"


def test_pennylane_plugin_matrix_fails_closed_for_provider_plugins() -> None:
    result = run_pennylane_plugin_matrix()

    assert isinstance(result, PennyLanePluginMatrixResult)
    assert result.local_plugin_parity_ready
    assert not result.provider_plugin_execution_ready
    assert not result.hardware_plugin_execution_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("default_qubit_exact_state") == "passed"
    assert result.route_status("phase_qnode_export_default_qubit") == "passed"
    assert result.route_status("provider_plugin_execution") == "blocked"
    assert result.route_status("hardware_plugin_execution") == "blocked"
    assert "provider_plugin_execution" in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    assert result.claim_boundary == "bounded_pennylane_plugin_matrix"


def test_pennylane_bridge_converts_dense_hermitian_observable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0),),
        DenseHermitianObservable(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)),
    )

    conversion = build_pennylane_qnode_from_phase_qnode(circuit, shots=128)
    value = cast(float, conversion.qnode(np.array([0.4], dtype=float)))

    np.testing.assert_allclose(
        value,
        execute_phase_qnode_circuit(circuit, np.array([0.4], dtype=float)).value,
    )
    assert conversion.shots == 128
    assert conversion.observable_kind == "dense_hermitian"
    assert any(call[0] == "Hermitian" for call in fake_qml.calls)


def test_pennylane_bridge_fails_closed_for_covariance_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: _FakePennyLane())
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0),),
        PauliCovarianceObservable(
            PauliTerm(1.0, ((0, "z"),)),
            PauliTerm(1.0, ((0, "x"),)),
        ),
    )

    with pytest.raises(ValueError, match="PennyLane QNode conversion does not support"):
        build_pennylane_qnode_from_phase_qnode(circuit)


def test_pennylane_bridge_reports_qnode_round_trip_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    def shifted_objective(values: np.ndarray) -> float:
        return _objective(values) + 0.01

    result = check_pennylane_qnode_round_trip(
        _objective,
        shifted_objective,
        _closed_form_gradient,
        np.array([0.2, -0.4], dtype=float),
        value_tolerance=1e-4,
        gradient_tolerance=1e-12,
    )

    assert not result.passed
    assert result.value_abs_error > result.value_tolerance
    assert result.gradient_max_abs_error <= result.gradient_tolerance


def test_pennylane_bridge_round_trip_rejects_bad_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: object())

    def non_finite_objective(values: np.ndarray) -> float:
        return float("nan")

    with pytest.raises(ValueError, match="PennyLane objective"):
        check_pennylane_qnode_round_trip(
            _objective,
            non_finite_objective,
            _closed_form_gradient,
            np.array([0.2, -0.4], dtype=float),
        )


def test_pennylane_bridge_fails_closed_when_pennylane_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable():
        raise ImportError("blocked")

    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", unavailable)

    assert not is_phase_pennylane_available()
    with pytest.raises(ImportError, match="blocked"):
        check_pennylane_parameter_shift_agreement(
            _objective,
            _closed_form_gradient,
            np.array([0.2, -0.4], dtype=float),
        )
    with pytest.raises(ImportError, match="blocked"):
        check_pennylane_qnode_round_trip(
            _objective,
            _objective,
            _closed_form_gradient,
            np.array([0.2, -0.4], dtype=float),
        )
