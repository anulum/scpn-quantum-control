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

from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseTorchCompileBoundaryAuditResult,
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


def test_torch_phase_qnode_compile_boundary_audit_is_fail_closed() -> None:
    """The public audit should classify compile routes without promoting gaps."""

    result = torch_phase_qnode_compile_boundary_audit(
        _phase_circuit(),
        np.array([0.37, -0.21], dtype=np.float64),
        tolerance=1.0e-8,
    )

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


def test_torch_phase_qnode_compile_boundary_result_rejects_unknown_route() -> None:
    """Route lookups should fail closed for unknown compile-boundary rows."""

    result = torch_phase_qnode_compile_boundary_audit(
        _phase_circuit(),
        np.array([0.37, -0.21], dtype=np.float64),
    )

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
