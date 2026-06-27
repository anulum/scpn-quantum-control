# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — canonical diff namespace tests.
"""Tests for the canonical differentiable user namespace."""

from __future__ import annotations

import json
import runpy

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn.diff as short_diff
import scpn_quantum_control.diff as diff


def _scalar_objective(values: NDArray[np.float64]) -> float:
    return float(np.sin(values[0]) + values[1] ** 2)


def _vector_objective(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([values[0] ** 2 + values[1], values[0] - values[1]], dtype=np.float64)


def test_short_namespace_reexports_canonical_surface() -> None:
    """The packaged short namespace re-exports the canonical production surface."""
    assert short_diff.grad is diff.grad
    assert short_diff.differentiable_circuit is diff.differentiable_circuit
    assert short_diff.namespace_metadata()["compatibility_namespace"] == "scpn.diff"
    assert set(diff.supported_transforms()) == {
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "jacobian",
        "hessian",
        "jvp",
        "vjp",
        "vmap",
        "gradient_tape",
    }


def test_canonical_transforms_execute_real_numeric_routes() -> None:
    """The namespace dispatches to real numeric transform implementations."""
    values = np.array([0.2, -0.4], dtype=np.float64)

    value_grad = diff.value_and_grad(_scalar_objective, values, method="finite_difference")
    gradient = diff.grad(_scalar_objective, values, method="finite_difference")
    jacobian = diff.jacobian(_vector_objective, values)
    jacfwd = diff.jacfwd(_vector_objective, values)
    jacrev = diff.jacrev(_vector_objective, values)
    hessian = diff.hessian(_scalar_objective, values)
    jvp = diff.jvp(_vector_objective, values, np.array([1.0, 0.5], dtype=np.float64))
    vjp = diff.vjp(_vector_objective, values, np.array([2.0, -1.0], dtype=np.float64))
    vectorized = diff.vmap(lambda row: float(row[0] + row[1]))(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    )

    assert value_grad.value == pytest.approx(_scalar_objective(values))
    assert gradient == pytest.approx(np.array([np.cos(values[0]), 2.0 * values[1]]))
    assert jacobian == pytest.approx(np.array([[2.0 * values[0], 1.0], [1.0, -1.0]]))
    assert jacfwd == pytest.approx(jacobian)
    assert jacrev == pytest.approx(jacobian)
    assert hessian == pytest.approx(
        np.array([[-np.sin(values[0]), 0.0], [0.0, 2.0]]),
        abs=1.0e-8,
    )
    assert jvp == pytest.approx(jacobian @ np.array([1.0, 0.5], dtype=np.float64))
    assert vjp == pytest.approx(jacobian.T @ np.array([2.0, -1.0], dtype=np.float64))
    assert vectorized == pytest.approx(np.array([3.0, 7.0], dtype=np.float64))


def test_differentiable_circuit_evaluates_and_serializes_supported_route() -> None:
    """A supported circuit evaluates, differentiates, and serializes metadata."""
    circuit = diff.differentiable_circuit(
        _scalar_objective,
        name="two_parameter_phase_objective",
        parameter_names=("theta", "bias"),
    )
    values = np.array([0.3, 0.5], dtype=np.float64)

    assert circuit(values) == pytest.approx(_scalar_objective(values))
    result = circuit.value_and_grad(values, method="finite_difference")
    assert result.gradient == pytest.approx(np.array([np.cos(values[0]), 2.0 * values[1]]))
    assert circuit.grad(values, method="finite_difference") == pytest.approx(result.gradient)
    assert circuit.diagnostics.supported is True
    assert circuit.capability.fail_closed is False

    payload = circuit.to_dict()
    assert payload["name"] == "two_parameter_phase_objective"
    assert payload["parameter_names"] == ["theta", "bias"]
    assert json.loads(circuit.to_json())["diagnostics"]["supported"] is True


def test_differentiable_circuit_fails_closed_for_unsupported_hardware_route() -> None:
    """Unsupported hardware routes fail closed before objective execution."""
    circuit = diff.differentiable_circuit(
        _scalar_objective,
        name="blocked_hardware_route",
        backend="hardware",
        shot_policy=diff.ShotPolicy(shots=128, allow_hardware=False),
    )

    assert circuit.fail_closed is True
    assert circuit.diagnostics.capability.requires_hardware_policy is True
    with pytest.raises(ValueError, match="unsupported"):
        circuit(np.array([0.1, 0.2], dtype=np.float64))


def test_jit_or_explain_returns_fail_closed_diagnostics() -> None:
    """The JIT entry point returns explicit diagnostics instead of eager fallback."""
    explanation = diff.jit_or_explain(_scalar_objective, backend="statevector")

    assert explanation.compiled is False
    assert explanation.fail_closed is True
    assert "grad" in explanation.suggested_alternatives
    with pytest.raises(RuntimeError, match="unsupported"):
        explanation.require_compiled()


def test_gradient_tape_entry_point_records_real_phase_route() -> None:
    """The namespace exposes the real phase gradient-tape context manager."""
    values = np.array([0.25], dtype=np.float64)

    with diff.gradient_tape(backend="statevector") as tape:
        record = tape.record_parameter_shift("phase", lambda x: float(np.sin(x[0])), values)

    assert record.value == pytest.approx(np.sin(values[0]))
    assert record.gradient == pytest.approx(np.array([np.cos(values[0])]))
    assert tape.records == (record,)


def test_shot_policy_rejects_unsafe_hardware_configuration() -> None:
    """Shot policy validation rejects unsafe hardware and confidence settings."""
    with pytest.raises(ValueError, match="shots"):
        diff.ShotPolicy(allow_hardware=True)

    with pytest.raises(ValueError, match="confidence_level"):
        diff.ShotPolicy(confidence_level=1.5)


def test_first_path_example_runs_real_namespace(capsys: pytest.CaptureFixture[str]) -> None:
    """The documented first-path example executes against the real namespace."""
    runpy.run_path("examples/30_diff_first_path.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "canonical diff namespace" in output
    assert "jit fail_closed: True" in output
