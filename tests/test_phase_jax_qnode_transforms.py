# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Registered-QNode Transform Leaf Tests
"""Compatibility tests for the registered-QNode JAX transform leaf."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.jax_bridge as jax_bridge
import scpn_quantum_control.phase.jax_qnode_transforms as qnode_transforms
from scpn_quantum_control.phase import PhaseQNodeCircuit

QNODE_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("jax_phase_qnode_value_and_grad", "_jax_phase_qnode_value_and_grad"),
    (
        "jax_phase_qnode_native_transform_audit",
        "_jax_phase_qnode_native_transform_audit",
    ),
    (
        "jax_phase_qnode_pytree_transform_audit",
        "_jax_phase_qnode_pytree_transform_audit",
    ),
    (
        "jax_phase_qnode_sharding_transform_audit",
        "_jax_phase_qnode_sharding_transform_audit",
    ),
    ("jax_phase_qnode_aot_export_audit", "_jax_phase_qnode_aot_export_audit"),
)


def test_qnode_transform_leaf_has_no_facade_back_edge() -> None:
    """Keep registered-QNode execution independent from the compatibility facade."""
    tree = ast.parse(inspect.getsource(qnode_transforms))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "jax_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_phase_exports_keep_qnode_facade_wrappers() -> None:
    """Expose signature-compatible facade wrappers without public loader parameters."""
    for facade_name, _ in QNODE_FUNCTIONS:
        facade_function = getattr(jax_bridge, facade_name)
        leaf_function = getattr(qnode_transforms, facade_name)
        assert getattr(phase, facade_name) is facade_function
        assert facade_function is not leaf_function
        assert "_jax_loader" not in inspect.signature(facade_function).parameters
        assert "_jax_loader" in inspect.signature(leaf_function).parameters


def test_all_qnode_facades_inject_their_active_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the facade loader binding into every registered-QNode implementation."""
    sentinel = object()
    circuit = cast(PhaseQNodeCircuit, object())
    params = np.array([0.2], dtype=np.float64)
    cases: dict[str, tuple[object, ...]] = {
        "jax_phase_qnode_value_and_grad": (circuit, params),
        "jax_phase_qnode_native_transform_audit": (circuit, params),
        "jax_phase_qnode_pytree_transform_audit": (circuit, {"theta": params}),
        "jax_phase_qnode_sharding_transform_audit": (circuit, params[None, :]),
        "jax_phase_qnode_aot_export_audit": (circuit, params),
    }

    def active_loader() -> tuple[object, object]:
        """Return the sentinel framework pair bound at the facade."""
        return sentinel, sentinel

    def implementation_for(target: dict[str, object]) -> Callable[..., object]:
        """Build one delegation stub bound to its own capture mapping."""

        def implementation(*args: object, **kwargs: object) -> object:
            """Capture wrapper delegation arguments at the dynamic test boundary."""
            target.update(kwargs)
            return sentinel

        return implementation

    monkeypatch.setattr(jax_bridge, "_load_jax", active_loader)
    for facade_name, implementation_name in QNODE_FUNCTIONS:
        captured: dict[str, object] = {}
        monkeypatch.setattr(jax_bridge, implementation_name, implementation_for(captured))
        facade_function = cast(Callable[..., object], getattr(jax_bridge, facade_name))

        result = facade_function(*cases[facade_name])

        assert result is sentinel
        assert captured["_jax_loader"] is active_loader
