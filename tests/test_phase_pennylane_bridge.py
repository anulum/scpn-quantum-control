# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase PennyLane Bridge Tests
"""Tests for local PennyLane gradient, conversion, and round-trip behavior."""

from __future__ import annotations

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
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PennyLaneGradientAgreementResult,
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
)

FloatArray = NDArray[np.float64]


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

    def shifted_gradient(values: FloatArray) -> FloatArray:
        return cast(FloatArray, _closed_form_gradient(values) + np.array([0.01, 0.0], dtype=float))

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

    def objective(values: FloatArray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    def external_gradient(values: FloatArray) -> FloatArray:
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

    def objective(values: FloatArray) -> float:
        return float(np.sin(values[0]) + 0.1 * np.cos(2.0 * values[0]))

    def external_gradient(values: FloatArray) -> FloatArray:
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
    gradient = cast(FloatArray, conversion.gradient(params))
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


def test_pennylane_bridge_canonicalises_conversion_metadata_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0),),
        PauliTerm(1.0, ((0, "z"),)),
    )

    conversion = build_pennylane_qnode_from_phase_qnode(
        circuit,
        device_name="  plugin.simulator  ",
        interface="  autograd  ",
        diff_method="  parameter-shift  ",
        shots=64,
    )

    assert conversion.device_name == "plugin.simulator"
    assert conversion.interface == "autograd"
    assert conversion.diff_method == "parameter-shift"
    assert fake_qml.devices == [{"name": "plugin.simulator", "wires": 1, "shots": 64}]
    assert cast(Any, conversion.qnode).metadata == {
        "interface": "autograd",
        "diff_method": "parameter-shift",
    }


@pytest.mark.parametrize(
    ("field_name", "device_name", "interface", "diff_method"),
    [
        ("device_name", "", "autograd", "parameter-shift"),
        ("device_name", "default.qubit\nbad", "autograd", "parameter-shift"),
        ("interface", "default.qubit", "autograd\x7fbad", "parameter-shift"),
        ("diff_method", "default.qubit", "autograd", "   "),
    ],
)
def test_pennylane_bridge_rejects_unsafe_conversion_metadata(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    device_name: str,
    interface: str,
    diff_method: str,
) -> None:
    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0),),
        PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(ValueError, match=field_name):
        build_pennylane_qnode_from_phase_qnode(
            circuit,
            device_name=device_name,
            interface=interface,
            diff_method=diff_method,
        )

    assert fake_qml.devices == []


@pytest.mark.parametrize("interface", ["numpy", "tensorflow", "jax-jit"])
def test_pennylane_bridge_rejects_undocumented_conversion_interfaces(
    monkeypatch: pytest.MonkeyPatch,
    interface: str,
) -> None:
    """Generated-QNode conversion accepts only documented PennyLane interfaces."""

    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0),),
        PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(ValueError, match="interface"):
        build_pennylane_qnode_from_phase_qnode(circuit, interface=interface)

    assert fake_qml.devices == []


@pytest.mark.parametrize("diff_method", ["parameter_shift", "finite_diff", "stoch-pulse"])
def test_pennylane_bridge_rejects_undocumented_conversion_diff_methods(
    monkeypatch: pytest.MonkeyPatch,
    diff_method: str,
) -> None:
    """Generated-QNode conversion accepts only documented PennyLane diff methods."""

    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0),),
        PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(ValueError, match="diff_method"):
        build_pennylane_qnode_from_phase_qnode(circuit, diff_method=diff_method)

    assert fake_qml.devices == []


@pytest.mark.parametrize(
    "shots", [cast(Any, True), cast(Any, 0), cast(Any, -1), cast(Any, 1.5), cast(Any, "64")]
)
def test_pennylane_bridge_rejects_non_integer_shots_before_device_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    shots: Any,
) -> None:
    """Finite-shot conversion metadata must be an explicit positive integer."""

    fake_qml = _FakePennyLane()
    monkeypatch.setattr(pennylane_bridge, "_load_pennylane", lambda: fake_qml)
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0),),
        PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(ValueError, match="shots"):
        build_pennylane_qnode_from_phase_qnode(circuit, shots=shots)

    assert fake_qml.devices == []


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

    def shifted_objective(values: FloatArray) -> float:
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

    def non_finite_objective(values: FloatArray) -> float:
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
    def unavailable() -> None:
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
