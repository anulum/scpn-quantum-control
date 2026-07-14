# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Phase-QNode Input Validation Tests
"""Public-path input and PMAP validation tests for JAX Phase-QNodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _phase_jax_bridge_test_helpers import FakeCallable, _FakeJAX
from _phase_jax_qnode_test_helpers import _NUMPY_JNP, _single_parameter_circuit
from numpy.typing import NDArray

import scpn_quantum_control.phase.jax_bridge as jax_bridge


@pytest.mark.parametrize(
    ("batch_offsets", "message"),
    (
        (np.array([0.0], dtype=float), "two-dimensional"),
        (np.empty((0, 1), dtype=float), "at least one parameter row"),
        (np.zeros((1, 2), dtype=float), r"shape \(batch, 1\)"),
        (np.array([[np.nan]], dtype=float), "only finite values"),
    ),
)
def test_native_transform_rejects_invalid_batch_offsets(
    monkeypatch: pytest.MonkeyPatch,
    batch_offsets: NDArray[np.float64],
    message: str,
) -> None:
    """The public native-transform route should validate every batch invariant."""
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_FakeJAX(), np))

    with pytest.raises(ValueError, match=message):
        jax_bridge.jax_phase_qnode_native_transform_audit(
            _single_parameter_circuit(),
            np.array([0.2], dtype=float),
            tangent=np.array([0.1], dtype=float),
            batch_offsets=batch_offsets,
        )


@pytest.mark.parametrize(
    ("attribute", "message"),
    (
        ("pmap", "JAX pmap"),
        ("value_and_grad", "JAX value_and_grad"),
        ("local_device_count", "JAX local_device_count"),
    ),
)
def test_sharding_transform_requires_complete_pmap_runtime(
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    message: str,
) -> None:
    """The public PMAP route should reject each missing runtime capability."""
    fake_jax = _FakeJAX()
    monkeypatch.setattr(fake_jax, attribute, None)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match=message):
        jax_bridge.jax_phase_qnode_sharding_transform_audit(
            _single_parameter_circuit(),
            np.array([[0.2]], dtype=float),
        )


def test_sharding_transform_rejects_non_positive_device_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PMAP lowering should fail closed when JAX reports no local devices."""
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = 0
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match="must be positive"):
        jax_bridge.jax_phase_qnode_sharding_transform_audit(
            _single_parameter_circuit(),
            np.array([[0.2]], dtype=float),
        )


@pytest.mark.parametrize(
    ("params_batch", "device_count", "message"),
    (
        (np.array([0.2], dtype=float), 1, "two-dimensional"),
        (np.array([[0.2]], dtype=float), 2, "exactly one row per local JAX device"),
    ),
)
def test_sharding_transform_rejects_invalid_device_batches(
    monkeypatch: pytest.MonkeyPatch,
    params_batch: NDArray[np.float64],
    device_count: int,
    message: str,
) -> None:
    """PMAP lowering should require a two-dimensional one-row-per-device batch."""
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = device_count
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(ValueError, match=message):
        jax_bridge.jax_phase_qnode_sharding_transform_audit(
            _single_parameter_circuit(),
            params_batch,
        )


def test_native_transform_builds_deterministic_default_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitted tangent and batch offsets should produce deterministic defaults."""
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, _NUMPY_JNP))

    result = jax_bridge.jax_phase_qnode_native_transform_audit(
        _single_parameter_circuit(),
        np.array([0.2], dtype=float),
        tolerance=2e-4,
    )

    np.testing.assert_allclose(result.tangent, np.array([0.25]), atol=0.0)
    assert result.batch_params.shape == (3, 1)
    assert result.passed


def test_native_transform_rejects_malformed_hessian_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed JAX Hessian should fail before result promotion."""

    class _WrongHessianJAX(_FakeJAX):
        """Fake runtime returning a non-square-width Hessian."""

        def hessian(self, _fn: Any) -> FakeCallable:
            """Return a malformed Hessian callable."""

            def malformed(_values: Any) -> NDArray[np.float64]:
                """Return a two-element vector instead of a matrix."""
                return np.zeros(2, dtype=np.float64)

            return malformed

    monkeypatch.setattr(
        jax_bridge,
        "_load_jax",
        lambda: (_WrongHessianJAX(), _NUMPY_JNP),
    )

    with pytest.raises(RuntimeError, match="hessian has an unexpected shape"):
        jax_bridge.jax_phase_qnode_native_transform_audit(
            _single_parameter_circuit(),
            np.array([0.2], dtype=float),
        )
