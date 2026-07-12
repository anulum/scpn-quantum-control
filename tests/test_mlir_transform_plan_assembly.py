# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR Transform-Plan Assembly Tests
"""Architecture tests for the compiler-AD transform-plan assembly leaf."""

from __future__ import annotations

import ast
import inspect

import pytest

import scpn_quantum_control.compiler.mlir as facade
import scpn_quantum_control.compiler.mlir_transform_plan_assembly as leaf
from scpn_quantum_control.compiler.mlir_records import (
    CompilerADTransformPlan,
    PrimitiveLoweringStatus,
)
from scpn_quantum_control.differentiable import PrimitiveIdentity

PUBLIC_NAMES = (
    "build_compiler_ad_transform_plan",
    "compile_compiler_ad_transform_plan_to_mlir",
)
PRIVATE_NAMES = (
    "_status_has_native_llvm_jit_backend",
    "_status_has_verified_rust_backend",
)


def test_transform_plan_assembly_has_no_facade_back_edge() -> None:
    """Keep transform-plan implementation imports one-way from the facade."""
    tree = ast.parse(inspect.getsource(leaf))
    relative_imports = {
        node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    }
    assert "mlir" not in relative_imports


def test_transform_plan_facade_exports_are_exact_leaf_aliases() -> None:
    """Preserve public and private transform-plan facade identities."""
    for name in (*PUBLIC_NAMES, *PRIVATE_NAMES):
        assert getattr(facade, name) is getattr(leaf, name)


def test_transform_plan_public_exports_remain_declared() -> None:
    """Retain the public transform-plan names in the facade export contract."""
    assert set(PUBLIC_NAMES) <= set(facade.__all__)


def test_transform_plan_lowering_rejects_non_plan_inputs() -> None:
    """Fail closed before lowering a value outside the plan contract."""
    with pytest.raises(ValueError, match="CompilerADTransformPlan"):
        leaf.compile_compiler_ad_transform_plan_to_mlir(object())  # type: ignore[arg-type]


def test_transform_plan_reports_unverified_runtime_and_registry_only_status() -> None:
    """Classify an unverified VJP-only registry row without overstating readiness."""
    status = PrimitiveLoweringStatus(
        identity=PrimitiveIdentity("test", "vjp_only", "1"),
        rule_name="vjp_only_rule",
        has_jvp=False,
        has_vjp=True,
        mlir_op="test.vjp_only",
        has_batching_rule=True,
        has_shape_rule=True,
        has_dtype_rule=True,
        has_lowering_rule=True,
        static_derivative_factory="test.factory",
        static_signature="vector[n]",
        nondifferentiable_policy="zero_cotangent",
        nondifferentiable_boundary="declared",
        nondifferentiable_boundary_policy="fail_closed",
        mlir_lowering="available: MLIR-runtime lowering",
        mlir_runtime_verification="verified: test runtime",
    )
    object.__setattr__(
        status,
        "mlir_runtime_verification",
        "available: no verified provenance",
    )

    module = leaf.compile_compiler_ad_transform_plan_to_mlir(CompilerADTransformPlan((status,)))
    readiness = module.metadata["primitive_readiness"]["test:vjp_only@1"]

    assert readiness["verdict"] == "registry_contract_only"
    assert module.metadata["mlir_runtime_blockers"]["test:vjp_only@1"] == (
        "blocked: no verified MLIR-runtime provenance"
    )
