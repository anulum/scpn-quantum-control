# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase torch bridge compile boundaries tests
# scpn-quantum-control -- PyTorch compile-boundary audit tests
"""Compile-boundary conformance tests for registered Phase-QNode PyTorch routes."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.phase.torch_qnode_transforms as qnode_transforms
from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseTorchCompileBoundaryAuditResult,
    PhaseTorchPhaseQNodeCompileResult,
    run_torch_phase_qnode_lowering_matrix,
    torch_phase_qnode_compile_boundary_audit,
)

pytest.importorskip("torch")  # the audit requires the optional PyTorch runtime


def _phase_circuit() -> PhaseQNodeCircuit:
    """Return a deterministic registered circuit with two trainable parameters."""
    return PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )


@pytest.fixture(scope="module")
def compile_boundary_result() -> PhaseTorchCompileBoundaryAuditResult:
    """Execute the expensive real compiler-boundary audit once per test module."""
    return torch_phase_qnode_compile_boundary_audit(
        _phase_circuit(),
        np.array([0.37, -0.21], dtype=np.float64),
        tolerance=1.0e-8,
    )


def test_torch_phase_qnode_compile_boundary_audit_is_fail_closed(
    compile_boundary_result: PhaseTorchCompileBoundaryAuditResult,
) -> None:
    """The public audit should classify compile routes without promoting gaps."""
    result = compile_boundary_result

    assert isinstance(result, PhaseTorchCompileBoundaryAuditResult)
    assert result.passed
    assert result.route_status("non_fullgraph_compile") == "passed"
    assert result.route_status("dynamic_non_fullgraph_compile") == "blocked"
    assert result.route_status("fullgraph_compile") == "blocked"
    assert result.route_status("aot_autograd_export_boundary") == "blocked"
    assert result.non_fullgraph_passed is True
    assert result.persistent_export_claim is False
    assert result.provider_claim is False
    assert result.performance_claim is False
    assert result.max_abs_reference_error <= result.tolerance
    assert result.non_fullgraph_gradient.shape == (2,)
    assert result.parameter_shift_gradient.shape == (2,)
    assert "dynamic_non_fullgraph_compile" in result.open_gaps
    assert "fullgraph_compile" in result.open_gaps
    assert "aot_autograd_export_boundary" in result.open_gaps

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert routes["non_fullgraph_compile"]["execution_passed"] is True
    assert routes["dynamic_non_fullgraph_compile"]["execution_passed"] in {True, False}
    assert routes["fullgraph_compile"]["execution_passed"] in {True, False}
    assert "variable_shape_compile_artifact" in routes["dynamic_non_fullgraph_compile"]["requires"]
    assert "graph_break_free_fullgraph_artifact" in routes["fullgraph_compile"]["requires"]
    assert "AOTAutograd" in str(routes["aot_autograd_export_boundary"]["reason"])
    assert "no persistent export" in str(payload["claim_boundary"])


def test_torch_phase_qnode_compile_boundary_result_rejects_unknown_route(
    compile_boundary_result: PhaseTorchCompileBoundaryAuditResult,
) -> None:
    """Route lookups should fail closed for unknown compile-boundary rows."""
    result = compile_boundary_result

    with pytest.raises(KeyError, match="unknown PyTorch compile-boundary route"):
        result.route_status("missing")


def test_torch_phase_qnode_lowering_matrix_exposes_boundary_diagnostic() -> None:
    """The lowering matrix should advertise the diagnostic without promotion."""
    matrix = run_torch_phase_qnode_lowering_matrix()
    payload = matrix.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])

    assert matrix.route_status("registered_phase_qnode_torch_compile_boundary_diagnostic") == (
        "passed"
    )
    assert routes["registered_phase_qnode_torch_compile_boundary_diagnostic"]["requires"] == []
    assert (
        "fullgraph" in routes["registered_phase_qnode_torch_compile_boundary_diagnostic"]["reason"]
    )
    assert "registered_phase_qnode_torch_compile_fullgraph_lowering" in matrix.open_gaps


def test_torch_phase_qnode_compile_boundary_audit_records_missing_runtime() -> None:
    """The public boundary audit should preserve execution failures as blockers."""
    result = qnode_transforms.torch_phase_qnode_compile_boundary_audit(
        _phase_circuit(),
        np.array([0.37, -0.21], dtype=np.float64),
        _torch_loader=object,
    )

    assert not result.passed
    assert result.non_fullgraph_value == result.parameter_shift_value
    np.testing.assert_array_equal(result.non_fullgraph_gradient, result.parameter_shift_gradient)
    assert result.max_abs_reference_error > result.tolerance
    assert all(route.status == "blocked" for route in result.routes)
    assert result.routes[0].exception_type == "RuntimeError"


def test_compile_boundary_exception_reason_preserves_empty_exception_type() -> None:
    """An empty exception should retain a useful representation in diagnostics."""
    assert qnode_transforms._compile_boundary_exception_reason(RuntimeError()) == (
        "RuntimeError: RuntimeError()"
    )


def test_torch_phase_qnode_compile_boundary_audit_records_reference_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public boundary audit should block successful but incorrect execution."""
    bad_result = PhaseTorchPhaseQNodeCompileResult(
        value=10.0,
        gradient=np.array([10.0, 10.0], dtype=np.float64),
        parameter_shift_value=0.0,
        parameter_shift_gradient=np.zeros(2, dtype=np.float64),
        torch_value=10.0,
        torch_gradient=np.array([10.0, 10.0], dtype=np.float64),
        max_abs_error=10.0,
        l2_error=10.0,
        tolerance=1.0e-8,
        passed=False,
        torch_compile_supported=True,
        compiled_value_supported=True,
        compiled_gradient_supported=True,
        fullgraph=False,
        dynamic=False,
    )
    monkeypatch.setattr(
        qnode_transforms,
        "torch_phase_qnode_compile_audit",
        lambda *_args, **_kwargs: bad_result,
    )

    result = qnode_transforms.torch_phase_qnode_compile_boundary_audit(
        _phase_circuit(),
        np.array([0.37, -0.21], dtype=np.float64),
        _torch_loader=object,
    )

    assert not result.passed
    assert result.routes[0].status == "blocked"
    assert "disagrees with the SCPN parameter-shift reference" in result.routes[0].reason
    assert result.non_fullgraph_value == bad_result.value
    np.testing.assert_array_equal(result.non_fullgraph_gradient, bad_result.gradient)
