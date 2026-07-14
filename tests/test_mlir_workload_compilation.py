# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Workload Compilation Tests
"""Architecture tests for Kuramoto and custom executable compilation."""

from __future__ import annotations

import ast
import inspect
from typing import Any, cast

import numpy as np
import pytest

import scpn_quantum_control.compiler.mlir as facade
import scpn_quantum_control.compiler.mlir_workload_compilation as leaf
from scpn_quantum_control.compiler.mlir_records import (
    CompilerADExecutableConfig,
    MLIRCompileConfig,
)
from scpn_quantum_control.differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
    PrimitiveTransformRule,
)

PUBLIC_NAMES = (
    "compile_kuramoto_to_mlir",
    "compile_custom_derivative_rule_to_executable",
    "compile_registered_primitive_to_executable",
)
LOWERING_FACTORY_NAMES = (
    "make_program_ad_linalg_matrix_power_executable_lowering_rule",
    "make_program_ad_linalg_multi_dot_executable_lowering_rule",
)


def test_workload_compilation_has_no_facade_back_edge() -> None:
    """Keep workload compilation imports one-way from the facade."""
    tree = ast.parse(inspect.getsource(leaf))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "mlir" not in relative_imports


def test_workload_compilation_facade_exports_are_exact_leaf_aliases() -> None:
    """Preserve workload functions and compatibility helpers as exact aliases."""
    for name in (*PUBLIC_NAMES, *LOWERING_FACTORY_NAMES, "_coupling_terms"):
        assert getattr(facade, name) is getattr(leaf, name)


def test_workload_compilation_public_exports_remain_declared() -> None:
    """Retain workload compiler names in the facade export contract."""
    assert set(PUBLIC_NAMES) <= set(facade.__all__)


def _quadratic_rule() -> CustomDerivativeRule:
    """Return a deterministic scalar quadratic rule for workload edge tests."""
    return CustomDerivativeRule(
        name="workload_quadratic",
        value_fn=lambda values: np.array([values[0] ** 2], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [2.0 * values[0] * tangent[0]], dtype=np.float64
        ),
        vjp_rule=lambda values, cotangent: np.array(
            [2.0 * values[0] * cotangent[0]], dtype=np.float64
        ),
        parameter_names=("x",),
        trainable=(True,),
    )


def test_workload_compilers_fail_closed_on_invalid_inputs() -> None:
    """Reject incomplete Kuramoto and executable-rule inputs at their boundaries."""
    with pytest.raises(ValueError, match="omega is required"):
        leaf.compile_kuramoto_to_mlir(
            np.eye(2, dtype=np.float64),
            MLIRCompileConfig(time=0.1),
        )
    with pytest.raises(ValueError, match="CustomDerivativeRule"):
        leaf.compile_custom_derivative_rule_to_executable(
            cast(Any, object()),
            np.array([0.2], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="backend='mlir_runtime'"):
        leaf.compile_custom_derivative_rule_to_executable(
            _quadratic_rule(),
            np.array([0.2], dtype=np.float64),
            CompilerADExecutableConfig(backend="native_llvm_jit"),
        )
    with pytest.raises(ValueError, match="CustomDerivativeRegistry"):
        leaf.compile_registered_primitive_to_executable(
            cast(Any, object()),
            "test:missing@1",
            np.array([0.2], dtype=np.float64),
        )


def test_workload_runtime_closures_snapshot_rules_and_reject_shape_drift() -> None:
    """Preserve compiled directional availability and reject tangent shape drift."""
    rule = _quadratic_rule()
    kernel = leaf.compile_custom_derivative_rule_to_executable(
        rule,
        np.array([0.2], dtype=np.float64),
        sample_tangent=np.array([0.5], dtype=np.float64),
        sample_cotangent=np.array([1.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="tangent shape"):
        kernel.jvp(
            np.array([0.2], dtype=np.float64),
            np.array([0.5, 0.7], dtype=np.float64),
        )

    jvp_only = CustomDerivativeRule(
        name="jvp_only",
        value_fn=lambda values: np.array([values[0] ** 2], dtype=np.float64),
        jvp_rule=lambda values, tangent: np.array(
            [2.0 * values[0] * tangent[0]], dtype=np.float64
        ),
    )
    vjp_only = CustomDerivativeRule(
        name="vjp_only",
        value_fn=lambda values: np.array([values[0] ** 2], dtype=np.float64),
        vjp_rule=lambda values, cotangent: np.array(
            [2.0 * values[0] * cotangent[0]], dtype=np.float64
        ),
    )
    jvp_kernel = leaf.compile_custom_derivative_rule_to_executable(
        jvp_only,
        np.array([0.2], dtype=np.float64),
        sample_tangent=np.array([0.5], dtype=np.float64),
    )
    vjp_kernel = leaf.compile_custom_derivative_rule_to_executable(
        vjp_only,
        np.array([0.2], dtype=np.float64),
        sample_cotangent=np.array([1.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="has no VJP"):
        jvp_kernel.vjp(np.array([0.2]), np.array([1.0]))
    with pytest.raises(ValueError, match="has no JVP"):
        vjp_kernel.jvp(np.array([0.2]), np.array([0.5]))


def test_kuramoto_workload_compiles_raw_public_inputs_without_metadata() -> None:
    """Compile raw Kuramoto arrays and honour metadata suppression."""
    coupling = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    omega = np.array([0.1, -0.2], dtype=np.float64)

    module = leaf.compile_kuramoto_to_mlir(
        coupling,
        MLIRCompileConfig(time=0.25, include_metadata=False),
        omega,
    )

    assert module.resource_counts == {
        "n_oscillators": 2,
        "omega_terms": 2,
        "coupling_terms": 1,
        "trotter_steps": 1,
        "trotter_order": 1,
    }
    assert "scpn.coupling" in module.text
    assert "scpn.metadata" not in module.text


def test_registered_and_linalg_lowering_rules_reject_invalid_results() -> None:
    """Reject foreign registered lowerings and non-rule linalg dispatch inputs."""
    identity = PrimitiveIdentity("test", "bad_lowering", "1")
    rule = _quadratic_rule()
    registry = CustomDerivativeRegistry()

    def bad_lowering(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return object()

    registry.register_transform(
        PrimitiveTransformRule(
            identity=identity,
            derivative_rule=rule,
            lowering_rule=bad_lowering,
        )
    )
    with pytest.raises(ValueError, match="ExecutableCompilerADKernel"):
        leaf.compile_registered_primitive_to_executable(
            registry,
            identity,
            np.array([0.2], dtype=np.float64),
        )

    power = leaf.make_program_ad_linalg_matrix_power_executable_lowering_rule(
        2,
        np.eye(2, dtype=np.float64).reshape(-1),
    )
    multi_dot = leaf.make_program_ad_linalg_multi_dot_executable_lowering_rule(
        ((2, 2), (2, 1)),
        np.arange(6, dtype=np.float64),
    )
    with pytest.raises(ValueError, match="CustomDerivativeRule"):
        power(cast(Any, object()))
    with pytest.raises(ValueError, match="CustomDerivativeRule"):
        multi_dot(cast(Any, object()))
