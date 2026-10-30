# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native Phase-QNode gate operator tests
"""Compare the lower Torch gate producer with the executor and the historical route."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import registered_phase_qnode_gates
from scpn_quantum_control.phase.qnode_circuit_contracts import _PARAMETRIC_GATES
from scpn_quantum_control.phase.qnode_circuit_execution import _gate_matrix
from scpn_quantum_control.phase_qnode_native_gates import jax_gate_operator, torch_gate_operator

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("gate", registered_phase_qnode_gates())
def test_torch_operator_matches_the_executor_and_the_historical_route(gate: str) -> None:
    """Build every registered gate natively and compare both references exactly.

    Parameters
    ----------
    gate
        Registered Phase-QNode gate name.

    """
    historical = importlib.import_module("scpn_quantum_control.phase.torch_qnode_transforms")
    theta = torch.tensor(0.37, dtype=torch.float64)
    actual = torch_gate_operator(torch, gate, theta)
    assert actual.dtype == torch.complex128
    np.testing.assert_allclose(actual.numpy(), _gate_matrix(gate, 0.37), rtol=0, atol=1e-15)
    assert torch.equal(actual, historical._torch_gate_matrix(torch, gate, theta))


@pytest.mark.parametrize("gate", sorted(_PARAMETRIC_GATES))
def test_parametric_operator_keeps_the_angle_gradient(gate: str) -> None:
    """Differentiate a real scalar of each parametric operator through its angle.

    Parameters
    ----------
    gate
        Registered parametric gate name.

    """
    theta = torch.tensor(0.37, dtype=torch.float64, requires_grad=True)
    operator = torch_gate_operator(torch, gate, theta)
    (operator.real.sum() + operator.imag.sum()).backward()
    assert theta.grad is not None and torch.isfinite(theta.grad)
    shifted = _gate_matrix(gate, 0.37 + 1e-6) - _gate_matrix(gate, 0.37 - 1e-6)
    expected = float((shifted.real.sum() + shifted.imag.sum()) / 2e-6)
    assert abs(float(theta.grad) - expected) < 1e-6


def test_unregistered_gate_is_refused() -> None:
    """Refuse a gate name outside the registered family with the route's message."""
    with pytest.raises(ValueError, match="unsupported PyTorch Phase-QNode gate: toffoli"):
        torch_gate_operator(torch, "toffoli", torch.tensor(0.0, dtype=torch.float64))


@pytest.mark.parametrize("gate", registered_phase_qnode_gates())
def test_jax_operator_matches_the_executor_and_the_historical_route(gate: str) -> None:
    """Build every registered gate on JAX and compare both references exactly.

    Parameters
    ----------
    gate
        Registered Phase-QNode gate name.

    """
    jnp = pytest.importorskip("jax.numpy")
    if jnp.asarray(0.0).dtype != np.float64:
        pytest.skip("JAX operator parity requires JAX_ENABLE_X64=true")
    historical = importlib.import_module("scpn_quantum_control.phase.jax_qnode_transforms")
    theta = jnp.asarray(0.37)
    actual = jax_gate_operator(jnp, gate, theta)
    assert actual.dtype == np.complex128
    np.testing.assert_allclose(np.asarray(actual), _gate_matrix(gate, 0.37), rtol=0, atol=1e-15)
    np.testing.assert_array_equal(
        np.asarray(actual), np.asarray(historical._jax_gate_matrix(jnp, gate, theta))
    )


@pytest.mark.parametrize("gate", sorted(_PARAMETRIC_GATES))
def test_jax_parametric_operator_keeps_the_angle_gradient(gate: str) -> None:
    """Differentiate a real scalar of each parametric JAX operator through its angle.

    Parameters
    ----------
    gate
        Registered parametric gate name.

    """
    jax = pytest.importorskip("jax")
    if jax.numpy.asarray(0.0).dtype != np.float64:
        pytest.skip("JAX gradient parity requires JAX_ENABLE_X64=true")

    def scalar(angle: object) -> object:
        operator = jax_gate_operator(jax.numpy, gate, angle)
        return operator.real.sum() + operator.imag.sum()

    gradient = float(jax.grad(scalar)(0.37))
    shifted = _gate_matrix(gate, 0.37 + 1e-6) - _gate_matrix(gate, 0.37 - 1e-6)
    expected = float((shifted.real.sum() + shifted.imag.sum()) / 2e-6)
    assert abs(gradient - expected) < 1e-6


def test_unregistered_jax_gate_is_refused() -> None:
    """Refuse a JAX gate name outside the registered family."""
    jnp = pytest.importorskip("jax.numpy")
    with pytest.raises(ValueError, match="unsupported JAX Phase-QNode gate: toffoli"):
        jax_gate_operator(jnp, "toffoli", jnp.asarray(0.0))


def _recording(events: list[str], label: str, function: Any) -> Any:
    """Return ``function`` wrapped to append ``label`` to ``events`` per call."""

    def record(*args: Any) -> Any:
        events.append(label)
        return function(*args)

    return record


@pytest.mark.parametrize(
    ("gate", "constructor"), [("ch", "_hadamard_operator"), ("crx", "_pauli_rotation")]
)
def test_historical_torch_route_captures_the_control_supplier_before_its_target(
    gate: str, constructor: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the control supplier bound when the gate starts, not one rebound by its target.

    Parameters
    ----------
    gate
        Controlled gate whose target construction rebinds the supplier.
    constructor
        Original module dependency called while that target is built.
    monkeypatch
        Restore the original module globals after the case.

    """
    historical = importlib.import_module("scpn_quantum_control.phase.torch_qnode_transforms")
    events: list[str] = []
    real_controlled = historical._torch_controlled
    real_constructor = getattr(historical, constructor)

    def rebinding_constructor(*args: Any, **kwargs: Any) -> Any:
        monkeypatch.setattr(
            historical, "_torch_controlled", _recording(events, "late", real_controlled)
        )
        return real_constructor(*args, **kwargs)

    monkeypatch.setattr(
        historical, "_torch_controlled", _recording(events, "early", real_controlled)
    )
    monkeypatch.setattr(historical, constructor, rebinding_constructor)
    theta = torch.tensor(0.37, dtype=torch.float64)
    actual = historical._torch_gate_matrix(torch, gate, theta)
    assert events == ["early"]
    np.testing.assert_allclose(actual.numpy(), _gate_matrix(gate, 0.37), rtol=0, atol=1e-15)


@pytest.mark.parametrize(("gate", "label"), [("rxx", "x"), ("ryy", "y"), ("rzz", "z")])
def test_historical_torch_route_captures_kron_before_building_its_factors(
    gate: str, label: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use the native Kronecker product bound before the Pauli factors are built.

    Parameters
    ----------
    gate
        Two-qubit rotation whose generator is a Kronecker product.
    label
        Pauli label of both factors.
    monkeypatch
        Restore the PyTorch attribute and module globals after the case.

    """
    historical = importlib.import_module("scpn_quantum_control.phase.torch_qnode_transforms")
    events: list[str] = []
    real_kron = torch.kron
    real_entries = historical._pauli_operator_entries

    def rebinding_entries(name: str) -> Any:
        monkeypatch.setattr(torch, "kron", _recording(events, "late", real_kron))
        return real_entries(name)

    monkeypatch.setattr(torch, "kron", _recording(events, "early", real_kron))
    monkeypatch.setattr(historical, "_pauli_operator_entries", rebinding_entries)
    actual = historical._torch_gate_matrix(torch, gate, torch.tensor(0.37, dtype=torch.float64))
    monkeypatch.setattr(torch, "kron", real_kron)
    assert events == ["early"], label
    np.testing.assert_allclose(actual.numpy(), _gate_matrix(gate, 0.37), rtol=0, atol=1e-15)


def test_historical_jax_route_captures_the_control_supplier_before_its_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use the JAX control supplier bound before the controlled rotation is built."""
    jnp = pytest.importorskip("jax.numpy")
    historical = importlib.import_module("scpn_quantum_control.phase.jax_qnode_transforms")
    events: list[str] = []
    real_controlled = historical._jax_controlled
    real_rotation = historical._pauli_rotation

    def rebinding_rotation(*args: Any, **kwargs: Any) -> Any:
        monkeypatch.setattr(
            historical, "_jax_controlled", _recording(events, "late", real_controlled)
        )
        return real_rotation(*args, **kwargs)

    monkeypatch.setattr(
        historical, "_jax_controlled", _recording(events, "early", real_controlled)
    )
    monkeypatch.setattr(historical, "_pauli_rotation", rebinding_rotation)
    actual = historical._jax_gate_matrix(jnp, "crx", jnp.asarray(0.37))
    assert events == ["early"]
    np.testing.assert_allclose(np.asarray(actual), _gate_matrix("crx", 0.37), rtol=0, atol=1e-7)
