# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Compatibility Leaf Tests
"""Compatibility tests for the bounded Torch transform and module leaf."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.torch_bridge as torch_bridge
import scpn_quantum_control.phase.torch_compatibility as torch_compatibility

COMPATIBILITY_FUNCTIONS: tuple[tuple[str, str], ...] = (
    (
        "run_torch_func_compatibility_audit",
        "_run_torch_func_compatibility_audit",
    ),
    (
        "run_torch_compile_compatibility_audit",
        "_run_torch_compile_compatibility_audit",
    ),
    ("torch_bounded_qnn_module", "_torch_bounded_qnn_module"),
    ("torch_bounded_qnn_layer", "_torch_bounded_qnn_layer"),
    ("run_torch_module_wrapper_audit", "_run_torch_module_wrapper_audit"),
    ("run_torch_training_loop_audit", "_run_torch_training_loop_audit"),
)
PRIVATE_HELPERS = (
    "_as_positive_learning_rate",
    "_as_positive_step_count",
    "_torch_nn_module_and_parameter",
    "_torch_parameter_count",
    "_torch_bounded_qnn_loss_tensor",
)


def test_torch_compatibility_leaf_has_no_facade_back_edge() -> None:
    """Keep bounded transforms and module execution independent from the facade."""
    tree = ast.parse(inspect.getsource(torch_compatibility))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "torch_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_torch_compatibility_private_helpers_remain_facade_aliases() -> None:
    """Preserve helper objects used by existing satellites and boundary tests."""
    for name in PRIVATE_HELPERS:
        assert getattr(torch_bridge, name) is getattr(torch_compatibility, name)


def test_phase_exports_keep_torch_compatibility_facade_wrappers() -> None:
    """Expose signature-compatible facade wrappers without loader parameters."""
    for facade_name, _ in COMPATIBILITY_FUNCTIONS:
        facade_function = getattr(torch_bridge, facade_name)
        leaf_function = getattr(torch_compatibility, facade_name)
        assert getattr(phase, facade_name) is facade_function
        assert facade_function is not leaf_function
        assert "_torch_loader" not in inspect.signature(facade_function).parameters
        assert "_torch_loader" in inspect.signature(leaf_function).parameters


def test_all_torch_compatibility_facades_inject_their_active_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the facade loader binding into every compatibility implementation."""
    sentinel = object()
    features = np.array([[0.0]], dtype=np.float64)
    labels = np.array([0.0], dtype=np.float64)
    params = np.array([0.2], dtype=np.float64)
    cases: dict[str, dict[str, object]] = {
        "run_torch_func_compatibility_audit": {
            "features": features,
            "labels": labels,
            "params": params,
            "params_batch": params[None, :],
        },
        "run_torch_compile_compatibility_audit": {
            "features": features,
            "labels": labels,
            "params": params,
        },
        "torch_bounded_qnn_module": {
            "features": features,
            "labels": labels,
            "initial_params": params,
        },
        "torch_bounded_qnn_layer": {
            "features": features,
            "labels": labels,
            "initial_params": params,
        },
        "run_torch_module_wrapper_audit": {
            "features": features,
            "labels": labels,
            "initial_params": params,
        },
        "run_torch_training_loop_audit": {
            "features": features,
            "labels": labels,
            "initial_params": params,
        },
    }

    def active_loader() -> object:
        """Return the sentinel framework module bound at the facade."""
        return sentinel

    def implementation_for(target: dict[str, object]) -> Callable[..., object]:
        """Build one delegation stub bound to its own capture mapping."""

        def implementation(**kwargs: object) -> object:
            """Capture wrapper delegation arguments at the dynamic boundary."""
            target.update(kwargs)
            return sentinel

        return implementation

    monkeypatch.setattr(torch_bridge, "_load_torch", active_loader)
    for facade_name, implementation_name in COMPATIBILITY_FUNCTIONS:
        captured: dict[str, object] = {}
        monkeypatch.setattr(torch_bridge, implementation_name, implementation_for(captured))
        facade_function = cast(Callable[..., object], getattr(torch_bridge, facade_name))

        result = facade_function(**cases[facade_name])

        assert result is sentinel
        assert captured["_torch_loader"] is active_loader
