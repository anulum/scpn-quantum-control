# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase QNode Framework Parity
"""Tests for phase/qnode_framework_parity.py framework parity evidence."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import execute_phase_qnode_circuit
from scpn_quantum_control.phase import qnode_framework_parity as parity_module
from scpn_quantum_control.phase.qnode_framework_parity import (
    ParityScenario,
    PhaseQNodeFrameworkParitySuiteResult,
    run_phase_qnode_framework_parity_suite,
)

FloatArray = NDArray[np.float64]


class _FakeDType:
    """Minimal dtype descriptor returned by the TensorFlow test double."""

    name = "float64"


class _FakeVariable:
    """Vector variable implementing the TensorFlow boundary used by the runner."""

    dtype = _FakeDType()

    def __init__(self, values: object) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def __getitem__(self, index: int) -> float:
        return float(self.values[index])


class _FakeScalar:
    """Eager scalar carrying TensorFlow's ``numpy`` conversion method."""

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __mul__(self, other: object) -> _FakeScalar:
        rhs = other.value if isinstance(other, _FakeScalar) else float(cast(float, other))
        return _FakeScalar(self.value * rhs)

    __rmul__ = __mul__

    def numpy(self) -> float:
        return self.value


class _FakeArray:
    """Eager vector carrying TensorFlow's ``numpy`` conversion method."""

    def __init__(self, values: object) -> None:
        self.values = np.asarray(values, dtype=np.float64)

    def numpy(self) -> FloatArray:
        return self.values.copy()


class _FakeGradientTape:
    """Deterministic gradient-tape boundary for optional TensorFlow coverage."""

    def __enter__(self) -> _FakeGradientTape:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback

    def gradient(self, value: object, tensor: _FakeVariable) -> _FakeArray:
        del value
        return _FakeArray(np.arange(1, tensor.values.size + 1, dtype=np.float64))


class _FakeTensorFlowRunner:
    """Bounded TensorFlow facade for the runner's eager/tape control flow."""

    float64 = np.float64

    def Variable(self, values: object, *, dtype: object) -> _FakeVariable:
        del dtype
        return _FakeVariable(values)

    def GradientTape(self) -> _FakeGradientTape:
        return _FakeGradientTape()

    def cos(self, value: float) -> _FakeScalar:
        return _FakeScalar(float(np.cos(value)))


def _numpy_tensorflow_ops() -> SimpleNamespace:
    """Return NumPy-backed TensorFlow array operations for helper parity."""

    def constant(values: object, *, dtype: Any) -> NDArray[Any]:
        return np.asarray(values, dtype=dtype)

    def cast_array(values: object, dtype: Any) -> NDArray[Any]:
        return np.asarray(values, dtype=dtype)

    def complex_array(real: object, imaginary: object) -> NDArray[np.complex128]:
        return np.asarray(real, dtype=np.float64) + 1.0j * np.asarray(
            imaginary,
            dtype=np.float64,
        )

    return SimpleNamespace(
        complex128=np.complex128,
        float64=np.float64,
        eye=np.eye,
        constant=constant,
        cast=cast_array,
        shape=np.shape,
        reshape=np.reshape,
        zeros=np.zeros,
        zeros_like=np.zeros_like,
        cos=np.cos,
        sin=np.sin,
        complex=complex_array,
        stack=np.stack,
        exp=np.exp,
        tensordot=np.tensordot,
        math=SimpleNamespace(real=np.real, conj=np.conj),
        linalg=SimpleNamespace(matvec=np.matmul, diag=np.diag),
    )


def test_phase_qnode_framework_parity_executes_or_classifies_every_local_framework() -> None:
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
    """Scenario and parameter guards cover valid, malformed, and non-finite vectors."""
    valid = np.array([0.2, -0.4], dtype=np.float64)

    np.testing.assert_array_equal(
        parity_module._as_params(valid, "single_qubit_ry_rx_pauli_z"),
        valid,
    )
    with pytest.raises(ValueError, match=r"shape \(2,\)"):
        parity_module._as_params(
            np.array([0.2], dtype=np.float64),
            "single_qubit_ry_rx_pauli_z",
        )
    with pytest.raises(ValueError, match="finite vector"):
        parity_module._as_params(
            np.array([0.2, -0.4, np.nan], dtype=np.float64),
            "registered_two_qubit_entangling_statevector",
        )
    with pytest.raises(ValueError, match="unsupported Phase-QNode"):
        run_phase_qnode_framework_parity_suite(scenario=cast(ParityScenario, "unsupported"))


def test_framework_record_classifies_every_failure_boundary() -> None:
    """Framework rows distinguish dependency, runtime, shape, value, and gradient failures."""
    reference_gradient = np.array([0.5, -0.25], dtype=np.float64)

    def dependency_missing() -> tuple[float, FloatArray, str, str]:
        raise ImportError("optional runtime absent")

    def runtime_failure() -> tuple[float, FloatArray, str, str]:
        raise RuntimeError("framework execution failed")

    def row(
        runner: object,
    ) -> parity_module.PhaseQNodeFrameworkParityRecord:
        return parity_module._run_framework_record(
            "framework",
            runner,
            reference_value=1.0,
            reference_gradient=reference_gradient,
            tolerance=1.0e-7,
        )

    dependency = row(dependency_missing)
    runtime = row(runtime_failure)
    shape = row(lambda: (1.0, np.array([0.5]), "float64", "cpu"))
    value = row(lambda: (1.1, reference_gradient, "float64", "cpu"))
    gradient = row(lambda: (1.0, reference_gradient + np.array([0.0, 0.1]), "float64", "cpu"))

    assert dependency.status == "dependency_missing"
    assert dependency.failure_class == "dependency_missing"
    assert runtime.status == "failed"
    assert runtime.failure_class == "runtime_error"
    assert shape.failure_class == "gradient_mismatch"
    assert shape.gradient_max_abs_error is None
    assert value.failure_class == "value_mismatch"
    assert gradient.failure_class == "gradient_mismatch"

    failed_suite = PhaseQNodeFrameworkParitySuiteResult(
        records=(value,),
        reference_value=1.0,
        reference_gradient=reference_gradient,
        tolerance=1.0e-7,
    )
    assert not failed_suite.passed
    assert not failed_suite.dependency_sparse
    assert failed_suite.to_dict()["passed"] is False


def test_tensorflow_runner_control_flow_without_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TensorFlow runner covers both scenarios through its bounded eager contract."""
    fake_tf = _FakeTensorFlowRunner()
    monkeypatch.setattr(
        "scpn_quantum_control.phase.qnode_framework_parity.importlib.import_module",
        lambda name: fake_tf,
    )

    single_value, single_gradient, dtype, device = parity_module._run_tensorflow(
        np.array([0.2, -0.4], dtype=np.float64),
        "single_qubit_ry_rx_pauli_z",
    )
    monkeypatch.setattr(
        parity_module,
        "_registered_two_qubit_tensorflow_objective",
        lambda tf, theta: _FakeScalar(1.25),
    )
    two_value, two_gradient, _, _ = parity_module._run_tensorflow(
        np.array([0.2, -0.4, 0.1], dtype=np.float64),
        "registered_two_qubit_entangling_statevector",
    )

    assert single_value == pytest.approx(float(np.cos(0.2) * np.cos(-0.4)))
    np.testing.assert_array_equal(single_gradient, np.array([1.0, 2.0]))
    assert dtype == "float64"
    assert device == "cpu"
    assert two_value == 1.25
    np.testing.assert_array_equal(two_gradient, np.array([1.0, 2.0, 3.0]))


def test_tensorflow_two_qubit_array_helpers_match_scpn_reference() -> None:
    """NumPy-backed TensorFlow operations reproduce the registered circuit value."""
    values = np.array([0.37, -0.29, 0.23], dtype=np.float64)
    tf_value = parity_module._registered_two_qubit_tensorflow_objective(
        _numpy_tensorflow_ops(),
        values,
    )
    circuit = parity_module._scenario_circuit("registered_two_qubit_entangling_statevector")
    reference = execute_phase_qnode_circuit(circuit, values).value

    assert float(tf_value) == pytest.approx(reference, abs=1.0e-12)
