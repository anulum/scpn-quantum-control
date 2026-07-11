# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Maturity Leaf Tests
"""Compatibility tests for Torch lowering and maturity orchestration."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.torch_bridge as torch_bridge
import scpn_quantum_control.phase.torch_maturity as torch_maturity

MATURITY_FUNCTIONS: tuple[tuple[str, str], ...] = (
    (
        "run_torch_phase_qnode_lowering_matrix",
        "_run_torch_phase_qnode_lowering_matrix",
    ),
    (
        "run_torch_ecosystem_maturity_audit",
        "_run_torch_ecosystem_maturity_audit",
    ),
    ("plan_torch_cloud_validation_batch", "_plan_torch_cloud_validation_batch"),
    ("run_torch_maturity_audit", "_run_torch_maturity_audit"),
)
PRIVATE_HELPERS = (
    "_torch_cuda_metadata",
    "_load_torch_live_overlay_evidence",
    "_required_str",
    "_required_float",
    "_required_int",
)


def test_torch_maturity_leaf_has_no_facade_back_edge() -> None:
    """Keep evidence orchestration independent from the public facade."""
    tree = ast.parse(inspect.getsource(torch_maturity))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "torch_bridge" not in relative_imports
    assert "__init__" not in relative_imports


def test_torch_maturity_private_helpers_remain_facade_aliases() -> None:
    """Preserve CUDA and overlay helpers used by satellites and boundary tests."""
    for name in PRIVATE_HELPERS:
        assert getattr(torch_bridge, name) is getattr(torch_maturity, name)


def test_phase_exports_keep_torch_maturity_facade_wrappers() -> None:
    """Expose facade-owned public orchestration functions over the new leaf."""
    for facade_name, _ in MATURITY_FUNCTIONS:
        facade_function = getattr(torch_bridge, facade_name)
        leaf_function = getattr(torch_maturity, facade_name)
        assert getattr(phase, facade_name) is facade_function
        assert facade_function is not leaf_function
        assert not any(
            name.startswith("_") for name in inspect.signature(facade_function).parameters
        )


def test_torch_maturity_facades_inject_live_orchestration_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pass loader and facade callables into every orchestration implementation."""
    sentinel = object()
    params = np.array([0.2], dtype=np.float64)
    cases: dict[str, dict[str, object]] = {
        "run_torch_phase_qnode_lowering_matrix": {},
        "run_torch_ecosystem_maturity_audit": {},
        "plan_torch_cloud_validation_batch": {},
        "run_torch_maturity_audit": {
            "features": np.array([[0.0]], dtype=np.float64),
            "labels": np.array([0.0], dtype=np.float64),
            "params": params,
            "params_batch": params[None, :],
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
    captures: dict[str, dict[str, object]] = {}
    for facade_name, implementation_name in MATURITY_FUNCTIONS:
        captured: dict[str, object] = {}
        captures[facade_name] = captured
        monkeypatch.setattr(torch_bridge, implementation_name, implementation_for(captured))
        facade_function = cast(Callable[..., object], getattr(torch_bridge, facade_name))

        result = facade_function(**cases[facade_name])

        assert result is sentinel

    assert captures["run_torch_phase_qnode_lowering_matrix"] == {}
    assert captures["run_torch_ecosystem_maturity_audit"]["_torch_loader"] is active_loader

    cloud_capture = captures["plan_torch_cloud_validation_batch"]
    assert cloud_capture["_ecosystem_runner"] is torch_bridge.run_torch_ecosystem_maturity_audit
    assert cloud_capture["_lowering_runner"] is torch_bridge.run_torch_phase_qnode_lowering_matrix

    maturity_capture = captures["run_torch_maturity_audit"]
    expected_callbacks = {
        "_analytic_tensor_runner": torch_bridge.torch_bounded_qnn_value_and_grad,
        "_custom_autograd_runner": torch_bridge.torch_autograd_qnn_value_and_grad,
        "_func_runner": torch_bridge.run_torch_func_compatibility_audit,
        "_compile_runner": torch_bridge.run_torch_compile_compatibility_audit,
        "_module_wrapper_runner": torch_bridge.run_torch_module_wrapper_audit,
        "_training_loop_runner": torch_bridge.run_torch_training_loop_audit,
        "_ecosystem_runner": torch_bridge.run_torch_ecosystem_maturity_audit,
        "_cloud_runner": torch_bridge.plan_torch_cloud_validation_batch,
        "_lowering_runner": torch_bridge.run_torch_phase_qnode_lowering_matrix,
        "_overlay_loader": torch_bridge._load_torch_live_overlay_evidence,
    }
    for name, expected in expected_callbacks.items():
        assert maturity_capture[name] is expected
