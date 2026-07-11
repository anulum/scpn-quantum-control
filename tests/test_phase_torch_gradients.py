# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Gradient Leaf Tests
"""Compatibility tests for the bounded Torch gradient leaf."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.torch_bridge as torch_bridge
import scpn_quantum_control.phase.torch_gradients as torch_gradients

GRADIENT_FUNCTIONS: tuple[tuple[str, str], ...] = (
    (
        "torch_parameter_shift_value_and_grad",
        "_torch_parameter_shift_value_and_grad",
    ),
    ("torch_bounded_qnn_value_and_grad", "_torch_bounded_qnn_value_and_grad"),
    ("torch_autograd_qnn_value_and_grad", "_torch_autograd_qnn_value_and_grad"),
)
PRIVATE_HELPERS = (
    "_load_torch",
    "_as_parameter_vector",
    "_as_feature_matrix",
    "_as_label_vector",
    "_as_non_negative_tolerance",
    "_torch_values_to_numpy",
    "_torch_tensor",
    "_torch_autograd_function",
    "_torch_autograd_grad",
    "_torch_trainable_tensor",
    "_bounded_qnn_loss_gradient_reference",
)


def test_torch_gradient_leaf_has_no_facade_back_edge() -> None:
    """Keep bounded gradient execution independent from the public facade."""
    tree = ast.parse(inspect.getsource(torch_gradients))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "torch_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_torch_gradient_private_helpers_remain_facade_aliases() -> None:
    """Preserve the helper surface consumed by later bridge and satellite code."""
    for name in PRIVATE_HELPERS:
        assert getattr(torch_bridge, name) is getattr(torch_gradients, name)


def test_phase_exports_keep_torch_gradient_facade_wrappers() -> None:
    """Expose signature-compatible facade wrappers without loader parameters."""
    for facade_name, _ in GRADIENT_FUNCTIONS:
        facade_function = getattr(torch_bridge, facade_name)
        leaf_function = getattr(torch_gradients, facade_name)
        assert getattr(phase, facade_name) is facade_function
        assert facade_function is not leaf_function
        assert "_torch_loader" not in inspect.signature(facade_function).parameters
        assert "_torch_loader" in inspect.signature(leaf_function).parameters


def test_all_torch_gradient_facades_inject_their_active_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the facade loader binding into every bounded gradient implementation."""
    sentinel = object()
    features = np.array([[0.0], [np.pi]], dtype=np.float64)
    labels = np.array([0.0, 1.0], dtype=np.float64)
    params = np.array([0.45], dtype=np.float64)
    cases: dict[str, tuple[object, ...]] = {
        "torch_parameter_shift_value_and_grad": (lambda values: float(values[0]), params),
        "torch_bounded_qnn_value_and_grad": (features, labels, params),
        "torch_autograd_qnn_value_and_grad": (features, labels, params),
    }

    def active_loader() -> object:
        """Return the sentinel framework module bound at the facade."""
        return sentinel

    def implementation_for(target: dict[str, object]) -> Callable[..., object]:
        """Build one delegation stub bound to its own capture mapping."""

        def implementation(*args: object, **kwargs: object) -> object:
            """Capture wrapper delegation arguments at the dynamic boundary."""
            target.update(kwargs)
            return sentinel

        return implementation

    monkeypatch.setattr(torch_bridge, "_load_torch", active_loader)
    for facade_name, implementation_name in GRADIENT_FUNCTIONS:
        captured: dict[str, object] = {}
        monkeypatch.setattr(torch_bridge, implementation_name, implementation_for(captured))
        facade_function = cast(Callable[..., object], getattr(torch_bridge, facade_name))

        result = facade_function(*cases[facade_name])

        assert result is sentinel
        assert captured["_torch_loader"] is active_loader
