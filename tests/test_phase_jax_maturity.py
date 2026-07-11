# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Maturity Leaf Tests
"""Compatibility tests for JAX lowering, cloud, and maturity orchestration."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.jax_bridge as jax_bridge
import scpn_quantum_control.phase.jax_maturity as jax_maturity

MATURITY_FUNCTIONS: tuple[tuple[str, str], ...] = (
    (
        "run_jax_phase_qnode_lowering_matrix",
        "_run_jax_phase_qnode_lowering_matrix",
    ),
    ("plan_jax_cloud_validation_batch", "_plan_jax_cloud_validation_batch"),
    ("run_jax_maturity_audit", "_run_jax_maturity_audit"),
)
LOADER_FUNCTIONS = frozenset({"plan_jax_cloud_validation_batch", "run_jax_maturity_audit"})


def test_jax_maturity_leaf_has_no_facade_back_edge() -> None:
    """Keep maturity orchestration independent from the public facade."""
    tree = ast.parse(inspect.getsource(jax_maturity))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "jax_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_phase_exports_keep_jax_maturity_facade_wrappers() -> None:
    """Expose signature-compatible facade wrappers for all maturity routes."""
    for facade_name, _ in MATURITY_FUNCTIONS:
        facade_function = getattr(jax_bridge, facade_name)
        leaf_function = getattr(jax_maturity, facade_name)
        assert getattr(phase, facade_name) is facade_function
        assert facade_function is not leaf_function
        assert "_jax_loader" not in inspect.signature(facade_function).parameters
        if facade_name in LOADER_FUNCTIONS:
            assert "_jax_loader" in inspect.signature(leaf_function).parameters
        else:
            assert inspect.signature(facade_function) == inspect.signature(leaf_function)


def test_all_jax_maturity_facades_delegate_with_expected_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject the active loader only into cloud and executable maturity routes."""
    sentinel = object()
    features = np.array([[0.0], [np.pi]], dtype=np.float64)
    labels = np.array([0.0, 1.0], dtype=np.float64)
    params = np.array([0.45], dtype=np.float64)
    params_batch = np.array([[0.25], [0.45]], dtype=np.float64)
    cases: dict[str, dict[str, object]] = {
        "run_jax_phase_qnode_lowering_matrix": {},
        "plan_jax_cloud_validation_batch": {},
        "run_jax_maturity_audit": {
            "features": features,
            "labels": labels,
            "params": params,
            "params_batch": params_batch,
            "params_pytree": {"phase": params},
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
    for facade_name, implementation_name in MATURITY_FUNCTIONS:
        captured: dict[str, object] = {}
        monkeypatch.setattr(jax_bridge, implementation_name, implementation_for(captured))
        facade_function = cast(Callable[..., object], getattr(jax_bridge, facade_name))

        result = facade_function(**cases[facade_name])

        assert result is sentinel
        if facade_name in LOADER_FUNCTIONS:
            assert captured["_jax_loader"] is active_loader
        else:
            assert "_jax_loader" not in captured
