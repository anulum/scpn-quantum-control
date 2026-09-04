# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Framework Parity
"""Tests for phase/qnode_framework_parity.py framework parity evidence."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.phase.qnode_framework_parity as framework_parity
from scpn_quantum_control.phase.qnode_circuit import execute_phase_qnode_circuit
from scpn_quantum_control.phase.qnode_framework_parity import (
    ParityScenario,
    run_phase_qnode_framework_parity_suite,
)


class _ArrayValue:
    """Small TensorFlow-like eager value backed by a NumPy array."""

    def __init__(self, value: object) -> None:
        self._value = np.asarray(value, dtype=np.float64)

    def numpy(self) -> np.ndarray:
        """Expose the eager value through TensorFlow's NumPy boundary."""
        return self._value


class _VariableValue(_ArrayValue):
    """TensorFlow-like variable exposing dtype and indexed scalar values."""

    dtype = np.dtype(np.float64)

    def __getitem__(self, index: int) -> float:
        """Return one scalar parameter."""
        return float(self._value[index])


class _ScalarValue:
    """TensorFlow-like scalar supporting eager multiplication."""

    def __init__(self, value: float) -> None:
        self._value = float(value)

    def __mul__(self, other: _ScalarValue) -> _ScalarValue:
        """Multiply two eager scalar values."""
        return _ScalarValue(self._value * other._value)

    def numpy(self) -> np.ndarray:
        """Expose the eager scalar through TensorFlow's NumPy boundary."""
        return np.asarray(self._value)


class _GradientTape:
    """TensorFlow-like tape returning a prescribed analytic gradient."""

    def __init__(self, gradient: np.ndarray) -> None:
        self._gradient = gradient

    def __enter__(self) -> _GradientTape:
        """Enter the tape context."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Leave the tape context."""

    def gradient(self, _value: object, _tensor: object) -> _ArrayValue:
        """Return the prescribed gradient through an eager value."""
        return _ArrayValue(self._gradient)


def test_phase_qnode_framework_parity_executes_or_classifies_every_local_framework() -> None:
    """Execute or honestly classify every supported local framework."""
    suite = run_phase_qnode_framework_parity_suite()

    assert suite.scenario == "single_qubit_ry_rx_pauli_z"
    assert suite.frameworks == ("scpn", "jax", "torch", "tensorflow", "pennylane")
    assert suite.record_count == 5
    assert suite.record_by_framework("scpn").status == "passed"
    assert suite.record_by_framework("scpn").value is not None
    assert suite.record_by_framework("scpn").gradient is not None
    assert suite.dependency_sparse in {True, False}
    assert not suite.hardware_execution
    assert "provider" in suite.claim_boundary
    with pytest.raises(KeyError, match="unknown framework parity row"):
        suite.record_by_framework("unknown")

    for record in suite.records:
        assert record.status in {"passed", "dependency_missing", "failed"}
        assert record.failure_class in {
            "none",
            "dependency_missing",
            "value_mismatch",
            "gradient_mismatch",
            "runtime_error",
        }
        if record.status == "passed":
            assert record.value_abs_error is not None
            assert record.gradient_max_abs_error is not None
            assert record.value_abs_error <= suite.tolerance
            assert record.gradient_max_abs_error <= suite.tolerance
            assert record.dtype
            assert record.device
            gradient = record.gradient
            assert gradient is not None
            np.testing.assert_allclose(gradient, suite.reference_gradient, atol=suite.tolerance)


def test_phase_qnode_framework_parity_supports_registered_two_qubit_scenario() -> None:
    """Cross-check the registered entangling statevector scenario."""
    suite = run_phase_qnode_framework_parity_suite(
        scenario="registered_two_qubit_entangling_statevector"
    )

    assert suite.scenario == "registered_two_qubit_entangling_statevector"
    assert suite.frameworks == ("scpn", "jax", "torch", "tensorflow", "pennylane")
    assert suite.record_by_framework("scpn").status == "passed"
    assert suite.record_by_framework("scpn").gradient is not None
    assert suite.reference_gradient.shape == (3,)
    assert suite.passed
    assert "registered two-qubit" in suite.claim_boundary

    payload = suite.to_dict()
    assert payload["scenario"] == "registered_two_qubit_entangling_statevector"


def test_phase_qnode_framework_parity_validates_scenarios_and_parameters() -> None:
    """The public suite accepts finite parameters and rejects invalid requests."""
    valid = np.array([0.2, -0.4], dtype=np.float64)

    suite = run_phase_qnode_framework_parity_suite(params=valid)

    assert suite.reference_value == pytest.approx(float(np.cos(0.2) * np.cos(-0.4)))
    assert suite.record_by_framework("scpn").status == "passed"
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        run_phase_qnode_framework_parity_suite(
            params=np.array([0.2], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="finite vector"):
        run_phase_qnode_framework_parity_suite(
            params=np.array([0.2, -0.4, np.nan], dtype=np.float64),
            scenario="registered_two_qubit_entangling_statevector",
        )
    with pytest.raises(ValueError, match="unsupported Phase-QNode"):
        run_phase_qnode_framework_parity_suite(scenario=cast(ParityScenario, "unsupported"))


def test_framework_record_classifies_dependency_runtime_and_parity_failures() -> None:
    """Every adapter outcome should retain a precise fail-closed classification."""
    reference = np.array([0.1, -0.2], dtype=np.float64)

    dependency = framework_parity._run_framework_record(
        "missing",
        lambda: (_ for _ in ()).throw(ImportError("optional runtime unavailable")),
        reference_value=1.0,
        reference_gradient=reference,
        tolerance=1.0e-8,
    )
    runtime = framework_parity._run_framework_record(
        "broken",
        lambda: (_ for _ in ()).throw(RuntimeError("adapter failed")),
        reference_value=1.0,
        reference_gradient=reference,
        tolerance=1.0e-8,
    )
    shape = framework_parity._run_framework_record(
        "shape",
        lambda: (1.0, np.array([0.1]), "float64", "cpu"),
        reference_value=1.0,
        reference_gradient=reference,
        tolerance=1.0e-8,
    )
    value = framework_parity._run_framework_record(
        "value",
        lambda: (1.5, reference, "float64", "cpu"),
        reference_value=1.0,
        reference_gradient=reference,
        tolerance=1.0e-8,
    )
    gradient = framework_parity._run_framework_record(
        "gradient",
        lambda: (1.0, reference + 0.5, "float64", "cpu"),
        reference_value=1.0,
        reference_gradient=reference,
        tolerance=1.0e-8,
    )

    assert dependency.failure_class == "dependency_missing"
    assert dependency.to_dict()["gradient"] is None
    assert runtime.failure_class == "runtime_error"
    assert runtime.failure_reason == "adapter failed"
    assert shape.failure_class == "gradient_mismatch"
    assert "does not match" in shape.failure_reason
    assert value.failure_class == "value_mismatch"
    assert gradient.failure_class == "gradient_mismatch"
    assert gradient.to_dict()["gradient"] == pytest.approx([0.6, 0.3])


def test_tensorflow_adapter_contracts_without_claiming_tensorflow_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise eager-adapter boundaries while TensorFlow remains optional."""
    gradient = np.array([0.25, -0.5], dtype=np.float64)
    fake_tf = SimpleNamespace(
        float64=np.float64,
        Variable=lambda values, dtype: _VariableValue(values),
        GradientTape=lambda: _GradientTape(gradient),
        cos=lambda value: _ScalarValue(float(np.cos(value))),
    )
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake_tf if name == "tensorflow" else __import__(name),
    )

    value, observed_gradient, dtype, device = framework_parity._run_tensorflow(
        np.array([0.2, -0.4], dtype=np.float64),
        "single_qubit_ry_rx_pauli_z",
    )
    assert value == pytest.approx(float(np.cos(0.2) * np.cos(-0.4)))
    np.testing.assert_array_equal(observed_gradient, gradient)
    assert (dtype, device) == ("float64", "cpu")

    monkeypatch.setattr(
        framework_parity,
        "_registered_two_qubit_tensorflow_objective",
        lambda _tf, _theta: _ScalarValue(1.25),
    )
    two_value, _, _, _ = framework_parity._run_tensorflow(
        np.array([0.2, -0.4, 0.3], dtype=np.float64),
        "registered_two_qubit_entangling_statevector",
    )
    assert two_value == pytest.approx(1.25)


def test_tensorflow_matrix_facade_matches_registered_reference() -> None:
    """The TensorFlow matrix formulation should match the registered circuit."""
    tf = SimpleNamespace(
        complex128=np.complex128,
        float64=np.float64,
        eye=lambda size, dtype: np.eye(size, dtype=dtype),
        constant=lambda values, dtype: np.asarray(values, dtype=dtype),
        shape=np.shape,
        reshape=np.reshape,
        zeros=lambda shape, dtype: np.zeros(shape, dtype=dtype),
        zeros_like=np.zeros_like,
        cos=np.cos,
        sin=np.sin,
        complex=lambda real, imag: np.asarray(real) + 1.0j * np.asarray(imag),
        stack=np.stack,
        exp=np.exp,
        cast=lambda values, dtype: np.asarray(values, dtype=dtype),
        linalg=SimpleNamespace(matvec=np.matmul, diag=np.diag),
        math=SimpleNamespace(real=np.real, conj=np.conj),
        tensordot=np.tensordot,
    )
    params = np.array([0.37, -0.29, 0.23], dtype=np.float64)
    circuit = framework_parity._scenario_circuit("registered_two_qubit_entangling_statevector")
    reference = execute_phase_qnode_circuit(circuit, params).value

    observed = framework_parity._registered_two_qubit_tensorflow_objective(tf, params)

    assert float(observed) == pytest.approx(reference, abs=1.0e-12)
