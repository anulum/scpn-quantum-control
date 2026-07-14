# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Phase-QNode Statevector Edge Tests
"""Public fail-closed and device-description edges for JAX Phase-QNodes."""

from __future__ import annotations

import numpy as np
import pytest
from _phase_jax_bridge_test_helpers import _FakeJAX
from _phase_jax_qnode_test_helpers import (
    _NUMPY_JNP,
    _FakeAOTJAX,
    _FakePyTreeJAX,
    _single_parameter_circuit,
)

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import PhaseQNodeSupportError


def test_statevector_transform_rejects_unsupported_registered_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Statevector autodiff should preserve the structured support boundary."""
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_FakeJAX(), np))

    with pytest.raises(PhaseQNodeSupportError, match="unsupported gates: u3"):
        jax_bridge.jax_phase_qnode_value_and_grad(
            _single_parameter_circuit(gate="u3"),
            np.array([0.2], dtype=float),
        )


def test_native_transform_rejects_unsupported_registered_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native transforms should preserve the structured support boundary."""
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_FakeJAX(), np))

    with pytest.raises(PhaseQNodeSupportError, match="unsupported gates: u3"):
        jax_bridge.jax_phase_qnode_native_transform_audit(
            _single_parameter_circuit(gate="u3"),
            np.array([0.2], dtype=float),
        )


def test_pytree_transform_rejects_unsupported_registered_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PyTree transforms should preserve the structured support boundary."""
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_FakePyTreeJAX(), np))

    with pytest.raises(PhaseQNodeSupportError, match="unsupported gates: u3"):
        jax_bridge.jax_phase_qnode_pytree_transform_audit(
            _single_parameter_circuit(gate="u3"),
            {"theta": np.array([0.2], dtype=float)},
        )


def test_sharding_transform_rejects_unsupported_registered_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PMAP transforms should preserve the structured support boundary."""
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = 1
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(PhaseQNodeSupportError, match="unsupported gates: u3"):
        jax_bridge.jax_phase_qnode_sharding_transform_audit(
            _single_parameter_circuit(gate="u3"),
            np.array([[0.2]], dtype=float),
        )


def test_aot_transform_rejects_unsupported_registered_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AOT diagnostics should preserve the structured support boundary."""
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_FakeAOTJAX(), np))

    with pytest.raises(PhaseQNodeSupportError, match="unsupported gates: u3"):
        jax_bridge.jax_phase_qnode_aot_export_audit(
            _single_parameter_circuit(gate="u3"),
            np.array([0.2], dtype=float),
        )


@pytest.mark.parametrize("device_mode", ("missing", "mismatched"))
def test_sharding_transform_uses_deterministic_device_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
    device_mode: str,
) -> None:
    """Missing or inconsistent JAX device descriptions should stay deterministic."""
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = 1
    if device_mode == "missing":
        monkeypatch.setattr(fake_jax, "local_devices", None)
    else:
        monkeypatch.setattr(fake_jax, "local_devices", lambda: [])
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, _NUMPY_JNP))

    result = jax_bridge.jax_phase_qnode_sharding_transform_audit(
        _single_parameter_circuit(),
        np.array([[0.2]], dtype=float),
        tolerance=2e-4,
    )

    assert result.passed
    assert result.device_descriptions == ("local-device-0",)
    assert result.sharding_mode == "single_device_pmap_smoke"
