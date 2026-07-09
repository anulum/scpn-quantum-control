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
from importlib.metadata import PackageNotFoundError
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn.diff as short_diff
import scpn_quantum_control as scpn_qc
import scpn_quantum_control.diff as diff
from scpn_quantum_control.differentiable_parameter_contracts import Parameter


def _scalar_objective(values: NDArray[np.float64]) -> float:
    return float(np.sin(values[0]) + values[1] ** 2)


def _vector_objective(values: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.array([values[0] ** 2 + values[1], values[0] - values[1]], dtype=np.float64)


def _nonfinite_objective(values: NDArray[np.float64]) -> float:
    return float(values[0] * np.inf)


def test_short_namespace_reexports_canonical_surface() -> None:
    """The packaged short namespace re-exports the canonical production surface."""
    assert short_diff.grad is diff.grad
    assert short_diff.differentiable_circuit is diff.differentiable_circuit
    assert scpn_qc.run_differentiable_circuit_contract_audit is (
        diff.run_differentiable_circuit_contract_audit
    )
    assert short_diff.run_differentiable_circuit_contract_audit is (
        diff.run_differentiable_circuit_contract_audit
    )
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
        gradient_method="finite_difference",
    )
    values = np.array([0.3, 0.5], dtype=np.float64)

    assert circuit(values) == pytest.approx(_scalar_objective(values))
    result = circuit.value_and_grad(values)
    assert result.gradient == pytest.approx(np.array([np.cos(values[0]), 2.0 * values[1]]))
    assert circuit.grad(values, method="finite_difference") == pytest.approx(result.gradient)
    assert circuit.diagnostics.supported is True
    assert circuit.capability.fail_closed is False

    payload = circuit.to_dict()
    assert payload["name"] == "two_parameter_phase_objective"
    assert payload["schema"] == diff.DIFFERENTIABLE_CIRCUIT_SCHEMA
    assert payload["gradient_method"] == "finite_difference"
    assert payload["parameter_names"] == ["theta", "bias"]
    serialized = json.loads(circuit.to_json())
    provenance = serialized["serialization_provenance"]
    assert serialized["diagnostics"]["supported"] is True
    assert provenance["serializes_executable_code"] is False
    assert len(provenance["metadata_digest"]) == 64
    assert serialized["diagnostics"]["estimator_provenance"]["route"].endswith(
        ":finite_difference"
    )


def test_differentiable_circuit_rejects_invalid_gradient_method() -> None:
    """Circuit construction validates the bound canonical gradient method."""
    with pytest.raises(ValueError, match="gradient_method"):
        diff.differentiable_circuit(
            _scalar_objective,
            name="invalid_gradient_method",
            gradient_method="spsa",
        )


def test_record_contracts_reject_malformed_metadata() -> None:
    """Public record constructors reject malformed shot and provenance metadata."""
    with pytest.raises(ValueError, match="shots"):
        diff.ShotPolicy(shots=0)
    with pytest.raises(ValueError, match="seed"):
        diff.ShotPolicy(seed=-1)

    with pytest.raises(ValueError, match="estimator"):
        diff.EstimatorProvenance("", "route", "1.0")
    with pytest.raises(ValueError, match="route"):
        diff.EstimatorProvenance("estimator", "", "1.0")
    with pytest.raises(ValueError, match="package_version"):
        diff.EstimatorProvenance("estimator", "route", "")
    with pytest.raises(ValueError, match="artifact_ids"):
        diff.EstimatorProvenance("estimator", "route", "1.0", artifact_ids=("",))
    with pytest.raises(ValueError, match="claim_boundary"):
        diff.EstimatorProvenance("estimator", "route", "1.0", claim_boundary="")


def test_differentiable_circuit_rejects_malformed_construction_inputs() -> None:
    """Circuit construction validates public metadata before execution."""
    with pytest.raises(ValueError, match="name"):
        diff.DifferentiableCircuit(name=" ", objective=_scalar_objective)
    with pytest.raises(ValueError, match="objective"):
        diff.DifferentiableCircuit(
            name="not_callable",
            objective=cast(diff.ScalarObjective, cast(Any, 3.0)),
        )
    with pytest.raises(ValueError, match="parameter_names"):
        diff.differentiable_circuit(
            _scalar_objective,
            name="empty_parameter_name",
            parameter_names=("theta", ""),
        )
    with pytest.raises(ValueError, match="claim_boundary"):
        diff.DifferentiableCircuit(
            name="empty_claim_boundary",
            objective=_scalar_objective,
            claim_boundary="",
        )


def test_differentiable_circuit_rejects_invalid_values_and_objectives() -> None:
    """Circuit call and gradient paths reject invalid parameter/objective contracts."""
    circuit = diff.differentiable_circuit(
        _scalar_objective,
        name="value_contracts",
        parameter_names=("theta", "bias"),
        gradient_method="finite_difference",
    )
    unnamed = diff.differentiable_circuit(
        _scalar_objective,
        name="unnamed_parameters",
        gradient_method="finite_difference",
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        circuit(np.array([[0.1, 0.2]], dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        circuit(np.array([np.inf, 0.2], dtype=np.float64))
    with pytest.raises(ValueError, match="length"):
        circuit.value_and_grad(np.array([0.1], dtype=np.float64))

    explicit = unnamed.value_and_grad(
        np.array([0.1, 0.2], dtype=np.float64),
        parameters=(Parameter("theta"), Parameter("bias")),
    )
    assert explicit.gradient.shape == (2,)

    nonfinite = diff.differentiable_circuit(
        _nonfinite_objective,
        name="nonfinite_objective",
        gradient_method="finite_difference",
    )
    with pytest.raises(ValueError, match="non-finite"):
        nonfinite(np.array([0.1], dtype=np.float64))


def test_differentiable_circuit_runtime_provenance_guard() -> None:
    """Diagnostics fail closed if provenance is removed after construction."""
    circuit = diff.differentiable_circuit(_scalar_objective, name="missing_provenance")
    object.__setattr__(circuit, "estimator_provenance", None)

    with pytest.raises(RuntimeError, match="provenance"):
        _ = circuit.diagnostics


def test_differentiable_circuit_version_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default provenance records editable-local when package metadata is absent."""

    def _missing_version(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr(diff, "version", _missing_version)
    circuit = diff.differentiable_circuit(_scalar_objective, name="editable_local")

    assert circuit.diagnostics.estimator_provenance.package_version == "editable-local"


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
    assert explanation.to_dict()["fail_closed"] is True
    with pytest.raises(RuntimeError, match="unsupported"):
        explanation.require_compiled()


def test_differentiable_circuit_contract_audit_covers_dp004_boundaries() -> None:
    """The DP-004 audit records supported and fail-closed public contracts."""
    result = diff.run_differentiable_circuit_contract_audit()

    assert result.passed is True
    assert {check.name for check in result.supported_checks} == {
        "call_semantics_flat_vector",
        "transform_composition",
        "backend_capability_contract",
        "serialization_provenance",
    }
    assert {check.name for check in result.fail_closed_checks} == {
        "dataclass_parameter_container_boundary",
        "scalar_objective_boundary",
    }
    payload = result.to_dict()
    assert payload["passed"] is True
    assert payload["supported_checks"] == 4
    assert payload["fail_closed_checks"] == 2
    assert payload["claim_boundary"] == diff.DIFFERENTIABLE_CIRCUIT_CONTRACT_CLAIM_BOUNDARY
    assert json.loads(json.dumps(payload))["passed"] is True


def test_contract_check_flags_missing_evidence() -> None:
    """Audit-result helpers expose malformed synthetic checks as failures."""
    malformed = diff.DifferentiableCircuitContractCheck(
        name="missing_evidence",
        status="fail_closed",
        evidence=(),
    )
    result = diff.DifferentiableCircuitContractAuditResult(checks=(malformed,))

    assert result.passed is False
    assert result.failing_checks == (malformed,)


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
