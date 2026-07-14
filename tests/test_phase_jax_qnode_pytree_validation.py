# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Phase-QNode PyTree Validation Tests
"""Public-path PyTree capability and result validation for JAX Phase-QNodes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from _phase_jax_bridge_test_helpers import FakeCallable, _FakeJAX
from _phase_jax_qnode_test_helpers import (
    _NUMPY_JNP,
    _FakePyTreeJAX,
    _single_parameter_circuit,
)

import scpn_quantum_control.phase.jax_bridge as jax_bridge


@pytest.mark.parametrize(
    ("capability", "message"),
    (
        ("tree_util", "tree_util support"),
        ("tree_flatten", "tree_flatten"),
        ("tree_unflatten", "tree_unflatten"),
        ("value_and_grad", "value_and_grad"),
        ("grad", "PyTree transforms.*grad"),
    ),
)
def test_pytree_transform_requires_complete_runtime_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    capability: str,
    message: str,
) -> None:
    """The PyTree route should fail closed for every required JAX capability."""
    fake_jax = _FakePyTreeJAX()
    if capability == "tree_util":
        monkeypatch.setattr(fake_jax, "tree_util", None)
    elif capability == "tree_flatten":
        monkeypatch.setattr(
            fake_jax,
            "tree_util",
            SimpleNamespace(
                tree_flatten=None,
                tree_unflatten=_FakeJAX._TreeUtil.tree_unflatten,
            ),
        )
    elif capability == "tree_unflatten":
        monkeypatch.setattr(
            fake_jax,
            "tree_util",
            SimpleNamespace(
                tree_flatten=_FakeJAX._TreeUtil.tree_flatten,
                tree_unflatten=None,
            ),
        )
    else:
        monkeypatch.setattr(fake_jax, capability, None)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(RuntimeError, match=message):
        jax_bridge.jax_phase_qnode_pytree_transform_audit(
            _single_parameter_circuit(),
            {"theta": np.array([0.2], dtype=float)},
        )


@pytest.mark.parametrize(
    ("params_pytree", "message"),
    (
        ({}, "at least one numeric leaf"),
        ({"theta": np.array([np.nan], dtype=float)}, "leaf 0.*finite"),
    ),
)
def test_pytree_transform_rejects_invalid_parameter_leaves(
    monkeypatch: pytest.MonkeyPatch,
    params_pytree: object,
    message: str,
) -> None:
    """The PyTree route should reject empty and non-finite parameter leaves."""
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_FakePyTreeJAX(), np))

    with pytest.raises(ValueError, match=message):
        jax_bridge.jax_phase_qnode_pytree_transform_audit(
            _single_parameter_circuit(),
            params_pytree,
        )


def test_pytree_transform_supports_single_leaf_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One-leaf PyTrees should preserve structure under all default transforms."""
    fake_jax = _FakePyTreeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, _NUMPY_JNP))

    result = jax_bridge.jax_phase_qnode_pytree_transform_audit(
        _single_parameter_circuit(),
        {"theta": np.array([0.2], dtype=float)},
        tolerance=2e-4,
    )

    assert result.passed
    assert result.leaf_shapes == ((1,),)
    np.testing.assert_allclose(result.tangent, np.array([0.25]), atol=0.0)
    assert result.batch_params.shape == (3, 1)


@pytest.mark.parametrize(
    ("gradient", "message"),
    (
        ({}, "at least one gradient leaf"),
        (np.zeros(2, dtype=float), r"flatten to shape \(1,\)"),
        (np.array([np.nan], dtype=float), "only finite values"),
    ),
)
def test_pytree_transform_rejects_malformed_gradients(
    monkeypatch: pytest.MonkeyPatch,
    gradient: object,
    message: str,
) -> None:
    """Malformed scalar-output gradient PyTrees should fail before promotion."""

    class _MalformedGradientJAX(_FakePyTreeJAX):
        """Fake runtime returning the configured malformed primary gradient."""

        def grad(self, _fn: Any) -> FakeCallable:
            """Return a callable that emits the malformed gradient."""

            def malformed(_values: Any) -> object:
                """Return the test's malformed gradient object."""
                return gradient

            return malformed

    monkeypatch.setattr(
        jax_bridge,
        "_load_jax",
        lambda: (_MalformedGradientJAX(), _NUMPY_JNP),
    )

    with pytest.raises(ValueError, match=message):
        jax_bridge.jax_phase_qnode_pytree_transform_audit(
            _single_parameter_circuit(),
            {"theta": np.array([0.2], dtype=float)},
        )


@pytest.mark.parametrize(
    ("batched_gradient", "message"),
    (
        ({}, "at least one gradient leaf"),
        (np.zeros((2, 1), dtype=float), "leading batch size 3"),
        (np.zeros((3, 2), dtype=float), r"flatten to shape \(3, 1\)"),
        (np.full((3, 1), np.nan, dtype=float), "only finite values"),
    ),
)
def test_pytree_transform_rejects_malformed_batched_gradients(
    monkeypatch: pytest.MonkeyPatch,
    batched_gradient: object,
    message: str,
) -> None:
    """Malformed VMAP gradient PyTrees should fail before result promotion."""

    class _MalformedVmapJAX(_FakePyTreeJAX):
        """Fake runtime replacing only the VMAP gradient output."""

        def vmap(self, fn: FakeCallable) -> FakeCallable:
            """Return a VMAP callable with a malformed gradient tree."""
            valid_vmap = super().vmap(fn)

            def malformed(values: Any) -> tuple[object, object]:
                """Preserve values while replacing the batched gradient."""
                value_output, _gradient_output = valid_vmap(values)
                return value_output, batched_gradient

            return malformed

    monkeypatch.setattr(
        jax_bridge,
        "_load_jax",
        lambda: (_MalformedVmapJAX(), _NUMPY_JNP),
    )

    with pytest.raises(ValueError, match=message):
        jax_bridge.jax_phase_qnode_pytree_transform_audit(
            _single_parameter_circuit(),
            {"theta": np.array([0.2], dtype=float)},
        )


@pytest.mark.parametrize(
    ("hessian", "message"),
    (
        ((), "must contain 1 Hessian blocks"),
        (((np.zeros(2, dtype=float),),), "block 0 must have shape"),
        (((np.array([[np.nan]], dtype=float),),), "only finite values"),
    ),
)
def test_pytree_transform_rejects_malformed_hessian_blocks(
    monkeypatch: pytest.MonkeyPatch,
    hessian: object,
    message: str,
) -> None:
    """Malformed Hessian PyTrees should fail before result promotion."""

    class _MalformedHessianJAX(_FakePyTreeJAX):
        """Fake runtime returning the configured malformed Hessian PyTree."""

        def hessian(self, _fn: Any) -> FakeCallable:
            """Return a callable that emits the malformed Hessian."""

            def malformed(_values: Any) -> object:
                """Return the test's malformed Hessian object."""
                return hessian

            return malformed

    monkeypatch.setattr(
        jax_bridge,
        "_load_jax",
        lambda: (_MalformedHessianJAX(), _NUMPY_JNP),
    )

    with pytest.raises(ValueError, match=message):
        jax_bridge.jax_phase_qnode_pytree_transform_audit(
            _single_parameter_circuit(),
            {"theta": np.array([0.2], dtype=float)},
        )
