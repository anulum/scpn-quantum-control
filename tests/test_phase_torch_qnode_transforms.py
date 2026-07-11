# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Registered-QNode Leaf Tests
"""Compatibility tests for the registered-QNode Torch execution leaf."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.torch_bridge as torch_bridge
import scpn_quantum_control.phase.torch_qnode_transforms as qnode_transforms
from scpn_quantum_control.phase import PhaseQNodeCircuit

QNODE_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("torch_phase_qnode_value_and_grad", "_torch_phase_qnode_value_and_grad"),
    ("torch_phase_qnode_transform_audit", "_torch_phase_qnode_transform_audit"),
    ("torch_phase_qnode_compile_audit", "_torch_phase_qnode_compile_audit"),
    (
        "torch_phase_qnode_compile_boundary_audit",
        "_torch_phase_qnode_compile_boundary_audit",
    ),
)
PRIVATE_HELPERS = (
    "_as_parameter_matrix",
    "_torch_batch_to_numpy",
    "_torch_matrix_to_numpy",
    "_torch_scalar_to_float",
    "_torch_func_transforms",
    "_torch_compile",
    "_torch_complex_tensor",
    "_torch_real_tensor",
    "_torch_phase_qnode_value_and_state",
    "_torch_operation_theta",
    "_torch_gate_matrix",
    "_torch_controlled",
    "_torch_ccnot_matrix",
    "_torch_cswap_matrix",
    "_torch_apply_gate_matrix",
    "_torch_expectation_value",
    "_torch_term_expectation",
    "_torch_symmetrized_product_expectation",
    "_torch_term_product_expectation",
    "_torch_apply_term_operator",
    "_torch_pauli_matrix",
    "_compile_boundary_exception_reason",
    "_torch_compile_boundary_execution_route",
    "_torch_aot_autograd_boundary_route",
)


def test_torch_qnode_leaf_has_no_facade_back_edge() -> None:
    """Keep registered-QNode execution independent from the public facade."""
    tree = ast.parse(inspect.getsource(qnode_transforms))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "torch_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_torch_qnode_private_helpers_remain_facade_aliases() -> None:
    """Preserve shared transform helpers for later compatibility and edge tests."""
    for name in PRIVATE_HELPERS:
        assert getattr(torch_bridge, name) is getattr(qnode_transforms, name)


def test_phase_exports_keep_torch_qnode_facade_wrappers() -> None:
    """Expose signature-compatible facade wrappers without loader parameters."""
    for facade_name, _ in QNODE_FUNCTIONS:
        facade_function = getattr(torch_bridge, facade_name)
        leaf_function = getattr(qnode_transforms, facade_name)
        assert getattr(phase, facade_name) is facade_function
        assert facade_function is not leaf_function
        assert "_torch_loader" not in inspect.signature(facade_function).parameters
        assert "_torch_loader" in inspect.signature(leaf_function).parameters


def test_all_torch_qnode_facades_inject_their_active_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass the facade loader binding into every registered-QNode implementation."""
    sentinel = object()
    circuit = cast(PhaseQNodeCircuit, object())
    params = np.array([0.2], dtype=np.float64)
    cases: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {
        "torch_phase_qnode_value_and_grad": ((circuit, params), {}),
        "torch_phase_qnode_transform_audit": (
            (circuit, params),
            {"params_batch": params[None, :]},
        ),
        "torch_phase_qnode_compile_audit": ((circuit, params), {}),
        "torch_phase_qnode_compile_boundary_audit": ((circuit, params), {}),
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
    for facade_name, implementation_name in QNODE_FUNCTIONS:
        captured: dict[str, object] = {}
        monkeypatch.setattr(torch_bridge, implementation_name, implementation_for(captured))
        facade_function = cast(Callable[..., object], getattr(torch_bridge, facade_name))
        args, kwargs = cases[facade_name]

        result = facade_function(*args, **kwargs)

        assert result is sentinel
        assert captured["_torch_loader"] is active_loader
