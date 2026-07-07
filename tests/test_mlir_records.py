# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- MLIR record validation tests
"""Contract tests for MLIR compiler record value objects."""

from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import cast

import pytest

from scpn_quantum_control.compiler import (
    CompilerADExecutableConfig,
    CompilerADKernelVerification,
    CompilerADTransformPlan,
    DifferentiableMLIRCompileConfig,
    MLIRCompileConfig,
    MLIRModule,
    PrimitiveLoweringStatus,
)
from scpn_quantum_control.program_ad_registry import PrimitiveIdentity


def _primitive_identity(name: str = "primitive") -> PrimitiveIdentity:
    """Return a deterministic primitive identity for record tests."""
    return PrimitiveIdentity("scpn.test.mlir_records", name, "1")


def _lowering_status(
    *,
    name: str = "primitive",
    **overrides: object,
) -> PrimitiveLoweringStatus:
    """Return a valid primitive lowering status with optional field overrides."""
    fields: dict[str, object] = {
        "identity": _primitive_identity(name),
        "rule_name": f"{name}_rule",
        "has_jvp": True,
        "has_vjp": False,
        "mlir_op": "scpn_diff.test_op",
    }
    fields.update(overrides)
    return PrimitiveLoweringStatus(**fields)  # type: ignore[arg-type]  # dynamic negative cases


def test_mlir_compile_config_rejects_invalid_runtime_contracts() -> None:
    """Compile config validation should fail closed on invalid public fields."""
    MLIRCompileConfig(time=0.25, trotter_steps=2, trotter_order=2, dialect="scpn_test")
    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"time": 0.0}, "time"),
        ({"time": float("nan")}, "time"),
        ({"time": 0.25, "trotter_steps": 0}, "trotter_steps"),
        ({"time": 0.25, "trotter_steps": cast(int, 1.5)}, "trotter_steps"),
        ({"time": 0.25, "trotter_order": 3}, "trotter_order"),
        ({"time": 0.25, "dialect": "bad-dialect"}, "dialect"),
        ({"time": 0.25, "include_metadata": cast(bool, 1)}, "include_metadata"),
    )

    for kwargs, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            MLIRCompileConfig(**kwargs)  # type: ignore[arg-type]  # negative records


def test_mlir_module_hash_and_mapping_contracts_are_immutable() -> None:
    """MLIR modules should verify text provenance and freeze mapping fields."""
    text = "module { func.func @main() }"
    module = MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect="scpn_test",
        resource_counts={"ops": 1},
        metadata={"origin": "unit"},
    )

    assert isinstance(module.resource_counts, MappingProxyType)
    assert isinstance(module.metadata, MappingProxyType)
    assert module.resource_counts["ops"] == 1
    assert module.metadata["origin"] == "unit"

    with pytest.raises(ValueError, match="text"):
        MLIRModule(
            text=" ", sha256=hashlib.sha256(b" ").hexdigest(), dialect="x", resource_counts={}
        )
    with pytest.raises(ValueError, match="sha256"):
        MLIRModule(text=text, sha256="bad", dialect="x", resource_counts={})

    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"dialect": "bad-dialect", "resource_counts": {}}, "dialect"),
        ({"resource_counts": {"": 1}}, "resource_counts"),
        ({"resource_counts": {cast(str, 1): 1}}, "resource_counts"),
        ({"resource_counts": {"ops": -1}}, "resource_counts"),
        ({"resource_counts": {"ops": cast(int, True)}}, "resource_counts"),
        ({"metadata": {"": "value"}}, "metadata"),
        ({"metadata": {cast(str, 1): "value"}}, "metadata"),
    )
    for overrides, match in invalid_cases:
        kwargs: dict[str, object] = {
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "dialect": "scpn_test",
            "resource_counts": {"ops": 1},
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=match):
            MLIRModule(**kwargs)  # type: ignore[arg-type]  # negative records


def test_primitive_lowering_status_validates_static_and_boundary_metadata() -> None:
    """Primitive lowering status should cross-check derivative and boundary metadata."""
    status = _lowering_status(
        static_derivative_factory="make_rule",
        static_signature="vector:2",
        lowering_metadata={
            "static_derivative_factory": "make_rule",
            "static_signature": "vector:2",
        },
    )
    assert status.lowering_metadata["static_signature"] == "vector:2"
    assert isinstance(status.lowering_metadata, MappingProxyType)

    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"identity": cast(PrimitiveIdentity, object())}, "identity"),
        ({"rule_name": ""}, "rule_name"),
        ({"has_jvp": False, "has_vjp": False}, "JVP or VJP"),
        ({"mlir_op": "bad-op"}, "mlir_op"),
        ({"has_batching_rule": cast(bool, 1)}, "has_batching_rule"),
        ({"has_shape_rule": cast(bool, 1)}, "has_shape_rule"),
        ({"has_static_argument_rule": cast(bool, 1)}, "has_static_argument_rule"),
        ({"has_lowering_rule": cast(bool, 1)}, "has_lowering_rule"),
        ({"lowering_metadata": {"": "value"}}, "metadata keys"),
        ({"lowering_metadata": {"key": ""}}, "metadata values"),
        ({"static_derivative_factory": ""}, "static_derivative_factory"),
        ({"static_signature": ""}, "static_signature"),
        (
            {
                "static_derivative_factory": "make_rule",
                "lowering_metadata": {"static_derivative_factory": "other"},
            },
            "static_derivative_factory",
        ),
        (
            {
                "static_derivative_factory": "make_rule",
                "static_signature": "vector:2",
                "lowering_metadata": {"static_signature": "vector:3"},
            },
            "static_signature",
        ),
        ({"static_derivative_factory": "make_rule"}, "static_signature"),
        ({"static_signature": "vector:2"}, "static_derivative_factory"),
        ({"nondifferentiable_policy": ""}, "nondifferentiable_policy"),
        ({"nondifferentiable_boundary": ""}, "nondifferentiable_boundary"),
        ({"nondifferentiable_boundary_policy": ""}, "nondifferentiable_boundary_policy"),
        (
            {
                "nondifferentiable_boundary": "cusp",
                "lowering_metadata": {"nondifferentiable_boundary": "other"},
            },
            "nondifferentiable_boundary",
        ),
        (
            {
                "nondifferentiable_boundary": "cusp",
                "nondifferentiable_boundary_policy": "warn",
            },
            "fail_closed",
        ),
        (
            {"nondifferentiable_boundary_policy": "fail_closed"},
            "nondifferentiable_boundary",
        ),
        (
            {
                "nondifferentiable_boundary": "cusp",
                "nondifferentiable_boundary_policy": "fail_closed",
            },
            "nondifferentiable_policy",
        ),
        ({"effect": ""}, "effect"),
        ({"mlir_lowering": ""}, "mlir_lowering"),
    )

    for overrides, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            _lowering_status(**overrides)  # type: ignore[arg-type]  # negative records


def test_primitive_lowering_status_validates_backend_provenance() -> None:
    """Backend availability claims should require matching verification metadata."""
    verified_mlir = _lowering_status(
        has_lowering_rule=True,
        mlir_lowering="available: MLIR-runtime executable lowering",
        mlir_runtime_verification="verified: local parity",
    )
    assert verified_mlir.mlir_runtime_verification.startswith("verified:")

    verified_rust = _lowering_status(
        static_derivative_factory="make_rule",
        static_signature="vector:2",
        rust_lowering="available: Rust PyO3 backend",
        lowering_metadata={
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": "verified: parity",
            "rust_backend_signature": "vector:2",
            "rust_backend_functions": "value,jvp,vjp",
        },
    )
    assert verified_rust.rust_lowering.startswith("available:")

    verified_native = _lowering_status(
        llvm_lowering="available: native LLVM lowering",
        jit_lowering="available: native JIT lowering",
        lowering_metadata={
            "native_backend": "native_llvm_jit",
            "native_backend_verification": "verified: parity",
        },
    )
    assert verified_native.llvm_lowering.startswith("available:")

    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        (
            {
                "mlir_lowering": "available: MLIR-runtime executable lowering",
                "has_lowering_rule": False,
            },
            "has_lowering_rule",
        ),
        ({"has_lowering_rule": True}, "MLIR-runtime"),
        (
            {
                "has_lowering_rule": True,
                "mlir_lowering": "available: MLIR-runtime executable lowering",
            },
            "verified",
        ),
        ({"mlir_runtime_verification": "claimed"}, "mlir_runtime_verification"),
        (
            {
                "mlir_runtime_verification": "verified: local parity",
                "has_lowering_rule": False,
            },
            "has_lowering_rule",
        ),
        ({"rust_lowering": "available: Rust backend"}, "rust_backend"),
        (
            {
                "static_signature": "vector:2",
                "static_derivative_factory": "make_rule",
                "rust_lowering": "available: Rust backend",
                "lowering_metadata": {"rust_backend": "rust_pyo3"},
            },
            "verified Rust",
        ),
        (
            {
                "static_signature": "vector:2",
                "static_derivative_factory": "make_rule",
                "rust_lowering": "available: Rust backend",
                "lowering_metadata": {
                    "rust_backend": "rust_pyo3",
                    "rust_backend_verification": "verified: parity",
                    "rust_backend_signature": "vector:3",
                    "rust_backend_functions": "value",
                },
            },
            "rust_backend_signature",
        ),
        ({"llvm_lowering": "available: native LLVM lowering"}, "native_llvm_jit"),
        (
            {
                "jit_lowering": "available: native JIT lowering",
                "lowering_metadata": {"native_backend": "native_llvm_jit"},
            },
            "native_llvm_jit",
        ),
        (
            {
                "llvm_lowering": "available: native LLVM lowering",
                "lowering_metadata": {
                    "native_backend": "native_llvm_jit",
                    "native_backend_verification": "verified: parity",
                },
            },
            "LLVM and JIT",
        ),
    )

    for overrides, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            _lowering_status(**overrides)  # type: ignore[arg-type]  # negative records


def test_compiler_ad_transform_plan_rejects_invalid_rows_and_duplicates() -> None:
    """Transform plans should reject malformed rows and duplicate identities."""
    first = _lowering_status(name="first")
    second = _lowering_status(name="second", has_jvp=False, has_vjp=True)
    plan = CompilerADTransformPlan(
        statuses=(first, second),
        dialect="scpn_diff",
        transform="adjoint",
        executable_backend="native_llvm_jit",
        claim_boundary="verified local contract only",
    )

    assert [status.identity.key for status in plan.statuses] == [
        "scpn.test.mlir_records:first@1",
        "scpn.test.mlir_records:second@1",
    ]

    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"statuses": ()}, "at least one primitive"),
        ({"statuses": (cast(PrimitiveLoweringStatus, object()),)}, "PrimitiveLoweringStatus"),
        ({"statuses": (first,), "dialect": "bad-dialect"}, "dialect"),
        ({"statuses": (first,), "transform": "hessian"}, "transform"),
        ({"statuses": (first,), "executable_backend": "gpu"}, "executable_backend"),
        ({"statuses": (first, first)}, "duplicate"),
        ({"statuses": (first,), "claim_boundary": ""}, "claim_boundary"),
    )

    for kwargs, match in invalid_cases:
        with pytest.raises(ValueError, match=match):
            CompilerADTransformPlan(**kwargs)  # type: ignore[arg-type]  # negative records


def test_differentiable_mlir_compile_config_and_executable_config_validation() -> None:
    """Differentiable MLIR and executable configs should validate public fields."""
    config = CompilerADExecutableConfig(
        backend="native_llvm_jit",
        atol=0.0,
        rtol=1.0e-9,
        verify=False,
        mlir_config=DifferentiableMLIRCompileConfig(dialect="scpn_diff"),
    )
    assert config.backend == "native_llvm_jit"
    assert config.mlir_config.include_metadata is True

    invalid_mlir_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"dialect": "bad-dialect"}, "dialect"),
        ({"target": "llvm"}, "target"),
        ({"include_numeric_payload": cast(bool, 1)}, "include_numeric_payload"),
        ({"include_metadata": cast(bool, 1)}, "include_metadata"),
    )
    for kwargs, match in invalid_mlir_cases:
        with pytest.raises(ValueError, match=match):
            DifferentiableMLIRCompileConfig(**kwargs)  # type: ignore[arg-type]  # bad fields

    invalid_executable_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"backend": "gpu"}, "backend"),
        ({"atol": -1.0}, "atol"),
        ({"atol": float("nan")}, "atol"),
        ({"rtol": -1.0}, "rtol"),
        ({"verify": cast(bool, 1)}, "verify"),
        ({"mlir_config": cast(DifferentiableMLIRCompileConfig, object())}, "mlir_config"),
    )
    for kwargs, match in invalid_executable_cases:
        with pytest.raises(ValueError, match=match):
            CompilerADExecutableConfig(**kwargs)  # type: ignore[arg-type]  # bad fields


def test_compiler_ad_kernel_verification_passed_contract_and_failures() -> None:
    """Verification evidence should expose all-false-aware pass/fail semantics."""
    assert CompilerADKernelVerification(
        value_close=True,
        jvp_close=None,
        vjp_close=True,
        gradient_close=True,
        max_abs_error=0.0,
        samples=2,
    ).passed
    assert not CompilerADKernelVerification(
        value_close=True,
        jvp_close=False,
        vjp_close=True,
        gradient_close=None,
        max_abs_error=0.0,
        samples=1,
    ).passed

    invalid_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"value_close": cast(bool, 1)}, "value_close"),
        ({"jvp_close": cast(bool | None, 1)}, "jvp_close"),
        ({"vjp_close": cast(bool | None, 1)}, "vjp_close"),
        ({"gradient_close": cast(bool | None, 1)}, "gradient_close"),
        ({"max_abs_error": -1.0}, "max_abs_error"),
        ({"max_abs_error": float("nan")}, "max_abs_error"),
        ({"samples": 0}, "samples"),
        ({"samples": cast(int, 1.5)}, "samples"),
    )

    for overrides, match in invalid_cases:
        kwargs: dict[str, object] = {
            "value_close": True,
            "jvp_close": True,
            "vjp_close": True,
            "max_abs_error": 0.0,
            "samples": 1,
        }
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=match):
            CompilerADKernelVerification(**kwargs)  # type: ignore[arg-type]  # bad fields
