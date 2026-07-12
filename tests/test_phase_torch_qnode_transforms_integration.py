# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch QNode Transform Integration Tests
"""Integration tests for registered Phase-QNode Torch transforms."""

from __future__ import annotations

import json

import numpy as np
import pytest
from _phase_torch_bridge_test_helpers import _FakeTorchWithoutFunc

import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    PhaseTorchPhaseQNodeCompileResult,
    PhaseTorchPhaseQNodeStatevectorResult,
    PhaseTorchPhaseQNodeTransformResult,
    parameter_shift_phase_qnode_gradient,
    torch_phase_qnode_compile_audit,
    torch_phase_qnode_transform_audit,
    torch_phase_qnode_value_and_grad,
)


def test_torch_phase_qnode_compile_audit_lowers_registered_statevector() -> None:
    """Verify that PyTorch phase QNode compile audit lowers registered statevector."""
    pytest.importorskip("torch", reason="native Torch Phase-QNode compile requires PyTorch")

    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.37, -0.21], dtype=float)

    result = torch_phase_qnode_compile_audit(
        circuit,
        params,
        tolerance=1e-8,
        fullgraph=False,
    )
    reference = parameter_shift_phase_qnode_gradient(circuit, params)

    assert isinstance(result, PhaseTorchPhaseQNodeCompileResult)
    assert result.passed
    assert result.torch_compile_supported
    assert result.compiled_value_supported
    assert result.compiled_gradient_supported
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert not result.fullgraph
    assert result.claim_boundary == "registered_phase_qnode_torch_compile_lowering"
    np.testing.assert_allclose(result.value, reference.value, atol=1e-8)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=1e-8)
    payload = result.to_dict()
    assert payload["compiled_gradient_supported"] is True
    json.dumps(payload)


def test_torch_phase_qnode_value_and_grad_lowers_registered_statevector() -> None:
    """Verify that PyTorch phase QNode value and grad lowers registered statevector."""
    pytest.importorskip("torch", reason="native Torch phase-QNode lowering requires PyTorch")

    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.37, -0.21], dtype=float)

    result = torch_phase_qnode_value_and_grad(circuit, params, tolerance=1e-8)
    reference = parameter_shift_phase_qnode_gradient(circuit, params)

    assert isinstance(result, PhaseTorchPhaseQNodeStatevectorResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.claim_boundary == "registered_phase_qnode_torch_statevector_lowering"
    np.testing.assert_allclose(result.value, reference.value, atol=1e-8)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=1e-8)
    assert result.max_abs_error <= 1e-8
    assert result.state.shape == (4,)
    payload = result.to_dict()
    assert payload["method"] == "torch_native_registered_phase_qnode_statevector_value_and_grad"
    assert payload["host_boundary"] is False
    json.dumps(payload)


def test_torch_phase_qnode_transform_audit_checks_grad_jacrev_and_vmap() -> None:
    """Verify that PyTorch phase QNode transform audit checks grad jacrev and vmap."""
    pytest.importorskip("torch", reason="native Torch Phase-QNode transforms require PyTorch")

    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.37, -0.21], dtype=float)
    params_batch = np.array([[0.37, -0.21], [0.4, -0.25]], dtype=float)

    result = torch_phase_qnode_transform_audit(
        circuit,
        params,
        params_batch=params_batch,
        tolerance=1e-8,
    )
    reference = parameter_shift_phase_qnode_gradient(circuit, params)
    reference_batch = np.vstack(
        [parameter_shift_phase_qnode_gradient(circuit, row).gradient for row in params_batch]
    )

    assert isinstance(result, PhaseTorchPhaseQNodeTransformResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_boundary
    assert result.func_grad_supported
    assert result.func_vmap_supported
    assert result.func_jacrev_supported
    assert result.claim_boundary == "registered_phase_qnode_torch_func_transform_lowering"
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=1e-8)
    np.testing.assert_allclose(result.jacrev_gradient, reference.gradient, atol=1e-8)
    np.testing.assert_allclose(result.vmap_gradients, reference_batch, atol=1e-8)
    payload = result.to_dict()
    assert payload["host_boundary"] is False
    assert payload["func_vmap_supported"] is True
    json.dumps(payload)


def test_torch_phase_qnode_transform_audit_fails_closed_without_torch_func(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch phase QNode transform audit fails closed without PyTorch
    func.
    """
    fake_torch = _FakeTorchWithoutFunc()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(RuntimeError, match="torch.func"):
        torch_phase_qnode_transform_audit(
            circuit,
            np.array([0.2], dtype=float),
            params_batch=np.array([[0.2]], dtype=float),
        )
