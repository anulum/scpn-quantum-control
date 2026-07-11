# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Compatibility Leaf Tests
"""Compatibility tests for the bounded phase-QNN JAX audit leaf."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.jax_bridge as jax_bridge
import scpn_quantum_control.phase.jax_compatibility as jax_compatibility

COMPATIBILITY_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("run_jax_jit_compatibility_audit", "_run_jax_jit_compatibility_audit"),
    ("run_jax_vmap_compatibility_audit", "_run_jax_vmap_compatibility_audit"),
    (
        "run_jax_sharding_compatibility_audit",
        "_run_jax_sharding_compatibility_audit",
    ),
    ("run_jax_pytree_compatibility_audit", "_run_jax_pytree_compatibility_audit"),
    (
        "run_jax_nested_transform_algebra_audit",
        "_run_jax_nested_transform_algebra_audit",
    ),
)


def test_jax_compatibility_leaf_has_no_facade_back_edge() -> None:
    """Keep compatibility execution independent from the public facade."""
    tree = ast.parse(inspect.getsource(jax_compatibility))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "jax_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_phase_exports_keep_jax_compatibility_facade_wrappers() -> None:
    """Expose signature-compatible facade wrappers without loader parameters."""
    for facade_name, _ in COMPATIBILITY_FUNCTIONS:
        facade_function = getattr(jax_bridge, facade_name)
        leaf_function = getattr(jax_compatibility, facade_name)
        assert getattr(phase, facade_name) is facade_function
        assert facade_function is not leaf_function
        assert "_jax_loader" not in inspect.signature(facade_function).parameters
        assert "_jax_loader" in inspect.signature(leaf_function).parameters


def test_all_jax_compatibility_facades_inject_their_active_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the facade loader binding into every compatibility implementation."""
    sentinel = object()
    features = np.array([[0.0], [np.pi]], dtype=np.float64)
    labels = np.array([0.0, 1.0], dtype=np.float64)
    params = np.array([0.45], dtype=np.float64)
    params_batch = np.array([[0.25], [0.45]], dtype=np.float64)
    params_pytree = {"phase": params}
    cases: dict[str, dict[str, object]] = {
        "run_jax_jit_compatibility_audit": {
            "features": features,
            "labels": labels,
            "params": params,
        },
        "run_jax_vmap_compatibility_audit": {
            "features": features,
            "labels": labels,
            "params_batch": params_batch,
        },
        "run_jax_sharding_compatibility_audit": {
            "features": features,
            "labels": labels,
            "params_batch": params_batch,
        },
        "run_jax_pytree_compatibility_audit": {
            "features": features,
            "labels": labels,
            "params_pytree": params_pytree,
        },
        "run_jax_nested_transform_algebra_audit": {
            "features": features,
            "labels": labels,
            "params_batch": params_batch,
            "params_pytree": params_pytree,
        },
    }

    def active_loader() -> tuple[object, object]:
        """Return the sentinel framework pair bound at the facade."""
        return sentinel, sentinel

    def implementation_for(target: dict[str, object]) -> Callable[..., object]:
        """Build one delegation stub bound to its own capture mapping."""

        def implementation(*args: object, **kwargs: object) -> object:
            """Capture wrapper delegation arguments at the dynamic boundary."""
            target.update(kwargs)
            return sentinel

        return implementation

    monkeypatch.setattr(jax_bridge, "_load_jax", active_loader)
    for facade_name, implementation_name in COMPATIBILITY_FUNCTIONS:
        captured: dict[str, object] = {}
        monkeypatch.setattr(jax_bridge, implementation_name, implementation_for(captured))
        facade_function = cast(Callable[..., object], getattr(jax_bridge, facade_name))

        result = facade_function(**cases[facade_name])

        assert result is sentinel
        assert captured["_jax_loader"] is active_loader
