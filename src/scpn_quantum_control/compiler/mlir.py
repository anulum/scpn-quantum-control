# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- MLIR textual compiler surface
"""Deterministic MLIR-style export for Kuramoto-XY workloads.

The module emits a conservative textual interchange layer for the SCPN
Kuramoto-XY compiler. It does not require an MLIR Python runtime. Compiler AD
native LLVM/JIT execution is available only for primitives with verified native
lowering metadata; unrelated QIR, provider-pulse, and hardware execution claims
remain outside this boundary. The value is a stable, auditable IR boundary for
compiler passes and external tooling.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    Parameter,
    PrimitiveBatchingRule,
    PrimitiveIdentity,
    PrimitiveTransformRule,
    WholeProgramADResult,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
    program_adjoint_gradient,
    value_and_custom_jacobian,
    whole_program_value_and_grad,
)
from ..kuramoto_core import KuramotoProblem, build_kuramoto_problem


@dataclass(frozen=True)
class MLIRCompileConfig:
    """Configuration for Kuramoto-XY MLIR-style export."""

    time: float
    trotter_steps: int = 1
    trotter_order: int = 1
    dialect: str = "scpn_kuramoto"
    include_metadata: bool = True

    def __post_init__(self) -> None:
        if not np.isfinite(self.time) or self.time <= 0.0:
            raise ValueError("time must be finite and positive")
        if not isinstance(self.trotter_steps, int) or self.trotter_steps < 1:
            raise ValueError("trotter_steps must be a positive integer")
        if self.trotter_order not in {1, 2}:
            raise ValueError("trotter_order must be 1 or 2")
        if not self.dialect or not self.dialect.replace("_", "").isalnum():
            raise ValueError("dialect must be a non-empty MLIR-safe identifier")


@dataclass(frozen=True)
class MLIRModule:
    """Textual MLIR module plus deterministic provenance."""

    text: str
    sha256: str
    dialect: str
    resource_counts: Mapping[str, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        expected = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise ValueError("sha256 must match text")
        object.__setattr__(self, "resource_counts", MappingProxyType(dict(self.resource_counts)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True)
class PrimitiveLoweringStatus:
    """Compiler-backed AD lowering status for one primitive identity."""

    identity: PrimitiveIdentity
    rule_name: str
    has_jvp: bool
    has_vjp: bool
    mlir_op: str
    has_batching_rule: bool = False
    has_shape_rule: bool = False
    has_dtype_rule: bool = False
    has_static_argument_rule: bool = False
    has_lowering_rule: bool = False
    lowering_metadata: Mapping[str, str] = field(default_factory=dict)
    static_derivative_factory: str = "not_declared"
    static_signature: str = "none"
    nondifferentiable_policy: str = "not_declared"
    nondifferentiable_boundary: str = "not_declared"
    nondifferentiable_boundary_policy: str = "not_declared"
    effect: str = "pure"
    mlir_lowering: str = "available: scpn_diff dialect interchange"
    mlir_runtime_verification: str = "not_declared"
    rust_lowering: str = "blocked: no Rust differentiable primitive backend"
    llvm_lowering: str = "blocked: no LLVM/JIT differentiable primitive backend"
    jit_lowering: str = "blocked: no JIT differentiable primitive backend"

    def __post_init__(self) -> None:
        if not isinstance(self.identity, PrimitiveIdentity):
            raise ValueError("identity must be a PrimitiveIdentity")
        if not self.rule_name:
            raise ValueError("rule_name must be non-empty")
        if not isinstance(self.has_jvp, bool) or not isinstance(self.has_vjp, bool):
            raise ValueError("has_jvp and has_vjp must be booleans")
        if not self.has_jvp and not self.has_vjp:
            raise ValueError("primitive lowering requires a JVP or VJP rule")
        if not self.mlir_op or not self.mlir_op.replace(".", "").replace("_", "").isalnum():
            raise ValueError("mlir_op must be a non-empty MLIR-safe operation name")
        if not isinstance(self.has_batching_rule, bool):
            raise ValueError("has_batching_rule must be a boolean")
        if not isinstance(self.has_shape_rule, bool) or not isinstance(self.has_dtype_rule, bool):
            raise ValueError("has_shape_rule and has_dtype_rule must be booleans")
        if not isinstance(self.has_static_argument_rule, bool):
            raise ValueError("has_static_argument_rule must be a boolean")
        if not isinstance(self.has_lowering_rule, bool):
            raise ValueError("has_lowering_rule must be a boolean")
        metadata = dict(self.lowering_metadata)
        if any(not isinstance(key, str) or not key for key in metadata):
            raise ValueError("lowering metadata keys must be non-empty strings")
        if any(not isinstance(value, str) or not value for value in metadata.values()):
            raise ValueError("lowering metadata values must be non-empty strings")
        object.__setattr__(self, "lowering_metadata", MappingProxyType(metadata))
        if (
            not isinstance(self.static_derivative_factory, str)
            or not self.static_derivative_factory
        ):
            raise ValueError("static_derivative_factory must be non-empty")
        if not isinstance(self.static_signature, str) or not self.static_signature:
            raise ValueError("static_signature must be non-empty")
        metadata_factory = metadata.get("static_derivative_factory")
        if metadata_factory is not None and metadata_factory != self.static_derivative_factory:
            raise ValueError("static_derivative_factory must match lowering metadata")
        metadata_signature = metadata.get("static_signature")
        if metadata_signature is not None and metadata_signature != self.static_signature:
            raise ValueError("static_signature must match lowering metadata")
        factory_declared = self.static_derivative_factory not in {
            "not_declared",
            "not_required",
        }
        signature_declared = self.static_signature != "none"
        if factory_declared and not signature_declared:
            raise ValueError("static_signature is required for static derivative factories")
        if signature_declared and not factory_declared:
            raise ValueError(
                "static_derivative_factory is required for static derivative signatures"
            )
        if not isinstance(self.nondifferentiable_policy, str) or not self.nondifferentiable_policy:
            raise ValueError("nondifferentiable_policy must be non-empty")
        if (
            not isinstance(self.nondifferentiable_boundary, str)
            or not self.nondifferentiable_boundary
        ):
            raise ValueError("nondifferentiable_boundary must be non-empty")
        if (
            not isinstance(self.nondifferentiable_boundary_policy, str)
            or not self.nondifferentiable_boundary_policy
        ):
            raise ValueError("nondifferentiable_boundary_policy must be non-empty")
        metadata_boundary = metadata.get("nondifferentiable_boundary")
        if metadata_boundary is not None and metadata_boundary != self.nondifferentiable_boundary:
            raise ValueError("nondifferentiable_boundary must match lowering metadata")
        metadata_boundary_policy = metadata.get("nondifferentiable_boundary_policy")
        if (
            metadata_boundary_policy is not None
            and metadata_boundary_policy != self.nondifferentiable_boundary_policy
        ):
            raise ValueError("nondifferentiable_boundary_policy must match lowering metadata")
        if (
            self.nondifferentiable_boundary != "not_declared"
            and self.nondifferentiable_boundary_policy != "fail_closed"
        ):
            raise ValueError("declared nondifferentiable boundaries must be fail_closed")
        if (
            self.nondifferentiable_boundary == "not_declared"
            and self.nondifferentiable_boundary_policy != "not_declared"
        ):
            raise ValueError("nondifferentiable_boundary is required when policy is declared")
        if (
            self.nondifferentiable_boundary != "not_declared"
            and self.nondifferentiable_policy == "not_declared"
        ):
            raise ValueError(
                "nondifferentiable_policy is required when boundary metadata is declared"
            )
        if not isinstance(self.effect, str) or not self.effect:
            raise ValueError("effect must be non-empty")
        for label, status in (
            ("mlir_lowering", self.mlir_lowering),
            ("mlir_runtime_verification", self.mlir_runtime_verification),
            ("rust_lowering", self.rust_lowering),
            ("llvm_lowering", self.llvm_lowering),
            ("jit_lowering", self.jit_lowering),
        ):
            if not isinstance(status, str) or not status:
                raise ValueError(f"{label} must be non-empty")
        mlir_runtime_claimed = "MLIR-runtime" in self.mlir_lowering
        mlir_runtime_verified = self.mlir_runtime_verification.startswith("verified:")
        if mlir_runtime_claimed and not self.has_lowering_rule:
            raise ValueError(
                "has_lowering_rule must be true when mlir_lowering claims MLIR-runtime"
            )
        if self.has_lowering_rule and not mlir_runtime_claimed:
            raise ValueError("mlir_lowering must declare MLIR-runtime lowering")
        if mlir_runtime_claimed and not mlir_runtime_verified:
            raise ValueError(
                "mlir_runtime_verification must start with 'verified:' when "
                "mlir_lowering claims MLIR-runtime"
            )
        if self.mlir_runtime_verification != "not_declared" and not mlir_runtime_verified:
            raise ValueError("mlir_runtime_verification must be 'not_declared' or verified")
        if mlir_runtime_verified and not self.has_lowering_rule:
            raise ValueError(
                "has_lowering_rule must be true when mlir_runtime_verification is verified"
            )
        rust_available = "blocked" not in self.rust_lowering.lower()
        if rust_available:
            rust_backend = metadata.get("rust_backend", "")
            rust_backend_verification = metadata.get("rust_backend_verification", "")
            rust_backend_signature = metadata.get("rust_backend_signature", "")
            rust_backend_functions = metadata.get("rust_backend_functions", "")
            if rust_backend not in {"rust_pyo3"}:
                raise ValueError("rust_lowering requires rust_backend='rust_pyo3' metadata")
            if not rust_backend_verification.startswith("verified:"):
                raise ValueError("rust_lowering requires verified Rust backend metadata")
            if not rust_backend_signature:
                raise ValueError("rust_lowering requires rust_backend_signature metadata")
            if rust_backend_signature != self.static_signature:
                raise ValueError("rust_backend_signature must match static_signature")
            if not rust_backend_functions:
                raise ValueError("rust_lowering requires rust_backend_functions metadata")
        native_backend = metadata.get("native_backend")
        native_backend_verification = metadata.get("native_backend_verification", "")
        native_llvm_jit_verified = (
            native_backend == "native_llvm_jit"
            and native_backend_verification.startswith("verified:")
        )
        llvm_available = "blocked" not in self.llvm_lowering.lower()
        jit_available = "blocked" not in self.jit_lowering.lower()
        if llvm_available and not native_llvm_jit_verified:
            raise ValueError("llvm_lowering requires verified native_llvm_jit lowering metadata")
        if jit_available and not native_llvm_jit_verified:
            raise ValueError("jit_lowering requires verified native_llvm_jit lowering metadata")
        if native_llvm_jit_verified and (not llvm_available or not jit_available):
            raise ValueError("verified native_llvm_jit lowering requires LLVM and JIT status")


def _status_has_native_llvm_jit_backend(status: PrimitiveLoweringStatus) -> bool:
    metadata = status.lowering_metadata
    return (
        metadata.get("native_backend") == "native_llvm_jit"
        and metadata.get("native_backend_verification", "").startswith("verified:")
        and "blocked" not in status.llvm_lowering.lower()
        and "blocked" not in status.jit_lowering.lower()
    )


def _status_has_verified_rust_backend(status: PrimitiveLoweringStatus) -> bool:
    metadata = status.lowering_metadata
    return (
        metadata.get("rust_backend") == "rust_pyo3"
        and metadata.get("rust_backend_verification", "").startswith("verified:")
        and metadata.get("rust_backend_signature") == status.static_signature
        and bool(metadata.get("rust_backend_functions"))
        and "blocked" not in status.rust_lowering.lower()
    )


@dataclass(frozen=True)
class CompilerADTransformPlan:
    """Deterministic compiler AD plan over registered differentiable primitives."""

    statuses: tuple[PrimitiveLoweringStatus, ...]
    dialect: str = "scpn_diff"
    transform: str = "jvp_vjp_adjoint"
    executable_backend: str = "none"
    claim_boundary: str = (
        "compiler-backed AD planning and MLIR dialect interchange only; "
        "no executable Rust, LLVM, or JIT differentiated runtime"
    )

    def __post_init__(self) -> None:
        if not self.statuses:
            raise ValueError("compiler AD transform plan requires at least one primitive")
        if not self.dialect or not self.dialect.replace("_", "").isalnum():
            raise ValueError("dialect must be a non-empty MLIR-safe identifier")
        if self.transform not in {"jvp", "vjp", "adjoint", "jvp_vjp_adjoint"}:
            raise ValueError("transform must be one of jvp, vjp, adjoint, jvp_vjp_adjoint")
        if self.executable_backend not in {"none", "native_llvm_jit"}:
            raise ValueError("executable_backend must be 'none' or 'native_llvm_jit'")
        keys = [status.identity.key for status in self.statuses]
        if len(set(keys)) != len(keys):
            raise ValueError("compiler AD transform plan contains duplicate primitive identities")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")


def build_compiler_ad_transform_plan(
    registry: CustomDerivativeRegistry,
    *,
    dialect: str = "scpn_diff",
    transform: str = "jvp_vjp_adjoint",
) -> CompilerADTransformPlan:
    """Build a deterministic compiler AD plan from registered primitive rules."""

    if not isinstance(registry, CustomDerivativeRegistry):
        raise ValueError("registry must be a CustomDerivativeRegistry")
    statuses = []
    transform_snapshot = registry.transform_snapshot()
    for identity, rule in sorted(registry.snapshot().items(), key=lambda item: item[0].key):
        transform_rule = transform_snapshot.get(identity)
        metadata = (
            {}
            if transform_rule is None or transform_rule.lowering_metadata is None
            else dict(transform_rule.lowering_metadata)
        )
        default_mlir_status = (
            "available: executable scpn_diff MLIR-runtime primitive kernel"
            if transform_rule is not None and transform_rule.lowering_rule is not None
            else "available: scpn_diff dialect interchange"
        )
        statuses.append(
            PrimitiveLoweringStatus(
                identity=identity,
                rule_name=rule.name,
                has_jvp=rule.jvp_rule is not None,
                has_vjp=rule.vjp_rule is not None,
                mlir_op=metadata.get("mlir_op", f"{dialect}.{identity.namespace}_{identity.name}"),
                has_batching_rule=transform_rule is not None
                and transform_rule.batching_rule is not None,
                has_shape_rule=transform_rule is not None
                and transform_rule.shape_rule is not None,
                has_dtype_rule=transform_rule is not None
                and transform_rule.dtype_rule is not None,
                has_static_argument_rule=transform_rule is not None
                and transform_rule.static_argument_rule is not None,
                has_lowering_rule=transform_rule is not None
                and transform_rule.lowering_rule is not None,
                lowering_metadata=metadata,
                static_derivative_factory=metadata.get(
                    "static_derivative_factory", "not_declared"
                ),
                static_signature=metadata.get("static_signature", "none"),
                nondifferentiable_policy="not_declared"
                if transform_rule is None
                else transform_rule.nondifferentiable_policy,
                nondifferentiable_boundary=metadata.get(
                    "nondifferentiable_boundary", "not_declared"
                ),
                nondifferentiable_boundary_policy=metadata.get(
                    "nondifferentiable_boundary_policy", "not_declared"
                ),
                effect="pure" if transform_rule is None else transform_rule.effect,
                mlir_lowering=metadata.get("mlir", default_mlir_status),
                mlir_runtime_verification=metadata.get(
                    "mlir_runtime_verification", "not_declared"
                ),
                rust_lowering=metadata.get(
                    "rust", "blocked: no Rust differentiable primitive backend"
                ),
                llvm_lowering=metadata.get(
                    "llvm", "blocked: no LLVM/JIT differentiable primitive backend"
                ),
                jit_lowering=metadata.get(
                    "jit", "blocked: no JIT differentiable primitive backend"
                ),
            )
        )
    executable_backend = (
        "native_llvm_jit"
        if statuses and all(_status_has_native_llvm_jit_backend(status) for status in statuses)
        else "none"
    )
    claim_boundary = (
        "verified executable native LLVM/JIT primitive AD kernels for all planned primitives; "
        "Rust differentiated runtime remains fail-closed"
        if executable_backend == "native_llvm_jit"
        else (
            "compiler-backed AD planning and MLIR dialect interchange only; "
            "no executable Rust, LLVM, or JIT differentiated runtime"
        )
    )
    return CompilerADTransformPlan(
        tuple(statuses),
        dialect=dialect,
        transform=transform,
        executable_backend=executable_backend,
        claim_boundary=claim_boundary,
    )


def compile_compiler_ad_transform_plan_to_mlir(plan: CompilerADTransformPlan) -> MLIRModule:
    """Emit deterministic MLIR-style dialect metadata for compiler-backed AD planning."""

    if not isinstance(plan, CompilerADTransformPlan):
        raise ValueError("compiler AD MLIR lowering requires a CompilerADTransformPlan")
    lines = [
        f'module attributes {{scpn.module = "compiler_ad_transform_plan", '
        f'scpn.dialect = "{plan.dialect}", '
        f'scpn.transform = "{plan.transform}", '
        f"scpn.n_primitives = {len(plan.statuses)}}} {{",
        "  func.func @main() {",
    ]
    execution = (
        plan.executable_backend if plan.executable_backend != "none" else "interchange_only"
    )
    for index, status in enumerate(plan.statuses):
        lines.append(
            "    scpn_diff.primitive "
            f'%p{index} {{identity = "{_escape_mlir_string(status.identity.key)}", '
            f'rule = "{_escape_mlir_string(status.rule_name)}", '
            f'op = "{_escape_mlir_string(status.mlir_op)}", '
            f"jvp = {_fmt_bool(status.has_jvp)}, vjp = {_fmt_bool(status.has_vjp)}, "
            f"batching_rule = {_fmt_bool(status.has_batching_rule)}, "
            f"shape_rule = {_fmt_bool(status.has_shape_rule)}, "
            f"dtype_rule = {_fmt_bool(status.has_dtype_rule)}, "
            f"static_argument_rule = {_fmt_bool(status.has_static_argument_rule)}, "
            f"lowering_rule = {_fmt_bool(status.has_lowering_rule)}, "
            f'mlir_runtime_verification = "{_escape_mlir_string(status.mlir_runtime_verification)}", '
            f'static_derivative_factory = "{_escape_mlir_string(status.static_derivative_factory)}", '
            f'static_signature = "{_escape_mlir_string(status.static_signature)}", '
            f'policy = "{_escape_mlir_string(status.nondifferentiable_policy)}", '
            f'boundary = "{_escape_mlir_string(status.nondifferentiable_boundary)}", '
            f'boundary_policy = "{_escape_mlir_string(status.nondifferentiable_boundary_policy)}", '
            f'effect = "{_escape_mlir_string(status.effect)}"}}'
        )
        lines.append(
            "    scpn_diff.lowering_status "
            f'{{identity = "{_escape_mlir_string(status.identity.key)}", '
            f'mlir = "{_escape_mlir_string(status.mlir_lowering)}", '
            f'verification = "{_escape_mlir_string(status.mlir_runtime_verification)}", '
            f'rust = "{_escape_mlir_string(status.rust_lowering)}", '
            f'llvm = "{_escape_mlir_string(status.llvm_lowering)}", '
            f'jit = "{_escape_mlir_string(status.jit_lowering)}"}}'
        )
        for key, value in sorted(status.lowering_metadata.items()):
            lines.append(
                "    scpn_diff.lowering_metadata "
                f'{{identity = "{_escape_mlir_string(status.identity.key)}", '
                f'key = "{_escape_mlir_string(key)}", '
                f'value = "{_escape_mlir_string(value)}"}}'
            )
    lines.append(
        "    scpn_diff.ad_transform "
        f'{{kind = "{_escape_mlir_string(plan.transform)}", '
        f'execution = "{_escape_mlir_string(execution)}"}}'
    )
    lines.append("    return")
    lines.append("  }")

    def has_registry_contract(status: PrimitiveLoweringStatus) -> bool:
        return (
            (status.has_jvp or status.has_vjp)
            and status.has_batching_rule
            and status.has_shape_rule
            and status.has_dtype_rule
            and status.static_derivative_factory not in {"not_declared", "not_required"}
            and status.static_signature != "none"
            and status.nondifferentiable_policy != "not_declared"
            and status.nondifferentiable_boundary != "not_declared"
            and status.nondifferentiable_boundary_policy == "fail_closed"
        )

    def has_reverse_contract(status: PrimitiveLoweringStatus) -> bool:
        return status.has_vjp and has_registry_contract(status)

    def has_forward_contract(status: PrimitiveLoweringStatus) -> bool:
        return status.has_jvp and has_registry_contract(status)

    def has_adjoint_contract(status: PrimitiveLoweringStatus) -> bool:
        return status.effect == "pure" and has_reverse_contract(status)

    def has_transform_contract(status: PrimitiveLoweringStatus) -> bool:
        return (
            has_forward_contract(status)
            and has_reverse_contract(status)
            and has_adjoint_contract(status)
        )

    def exposes_nondifferentiable_policy(status: PrimitiveLoweringStatus) -> bool:
        has_static_contract = (
            status.static_derivative_factory not in {"not_declared", "not_required"}
            and status.static_signature != "none"
        )
        return status.nondifferentiable_policy != "not_declared" and (
            status.nondifferentiable_boundary != "not_declared" or has_static_contract
        )

    def has_rust_backend_contract(status: PrimitiveLoweringStatus) -> bool:
        return _status_has_verified_rust_backend(status)

    def has_native_llvm_jit_proof(status: PrimitiveLoweringStatus) -> bool:
        return _status_has_native_llvm_jit_backend(status)

    def has_llvm_backend_contract(status: PrimitiveLoweringStatus) -> bool:
        return has_native_llvm_jit_proof(status) and "blocked" not in status.llvm_lowering.lower()

    def has_jit_backend_contract(status: PrimitiveLoweringStatus) -> bool:
        return has_native_llvm_jit_proof(status) and "blocked" not in status.jit_lowering.lower()

    def has_native_backend_contract(status: PrimitiveLoweringStatus) -> bool:
        return has_llvm_backend_contract(status) and has_jit_backend_contract(status)

    def has_mlir_runtime_contract(status: PrimitiveLoweringStatus) -> bool:
        return status.has_lowering_rule and status.mlir_runtime_verification.startswith(
            "verified:"
        )

    def mlir_runtime_blocker(status: PrimitiveLoweringStatus) -> str | None:
        if has_mlir_runtime_contract(status):
            return None
        if not status.has_lowering_rule:
            return "blocked: no MLIR-runtime lowering rule"
        if not status.mlir_runtime_verification.startswith("verified:"):
            return "blocked: no verified MLIR-runtime provenance"
        return "blocked: MLIR-runtime contract incomplete"

    def primitive_readiness(status: PrimitiveLoweringStatus) -> dict[str, bool | str]:
        registry_contract = has_registry_contract(status)
        forward_contract = has_forward_contract(status)
        reverse_contract = has_reverse_contract(status)
        adjoint_contract = has_adjoint_contract(status)
        transform_contract = has_transform_contract(status)
        mlir_runtime_contract = has_mlir_runtime_contract(status)
        rust_backend_contract = has_rust_backend_contract(status)
        llvm_backend_contract = has_llvm_backend_contract(status)
        jit_backend_contract = has_jit_backend_contract(status)
        native_backend_contract = has_native_backend_contract(status)
        if native_backend_contract and mlir_runtime_contract and transform_contract:
            verdict = "native_executable"
        elif mlir_runtime_contract:
            verdict = "mlir_runtime_verified"
        elif transform_contract:
            verdict = "transform_interchange_only"
        elif registry_contract and forward_contract:
            verdict = "forward_interchange_only"
        elif registry_contract:
            verdict = "registry_contract_only"
        else:
            verdict = "registry_incomplete"
        return {
            "adjoint_contract": adjoint_contract,
            "forward_contract": forward_contract,
            "jit_backend_contract": jit_backend_contract,
            "llvm_backend_contract": llvm_backend_contract,
            "mlir_runtime_contract": mlir_runtime_contract,
            "native_backend_contract": native_backend_contract,
            "registry_contract": registry_contract,
            "reverse_contract": reverse_contract,
            "rust_backend_contract": rust_backend_contract,
            "transform_contract": transform_contract,
            "verdict": verdict,
        }

    primitive_readiness_by_key = {
        status.identity.key: primitive_readiness(status) for status in plan.statuses
    }
    primitive_readiness_verdict_counts: dict[str, int] = {}
    for readiness in primitive_readiness_by_key.values():
        verdict = str(readiness["verdict"])
        primitive_readiness_verdict_counts[verdict] = (
            primitive_readiness_verdict_counts.get(verdict, 0) + 1
        )
    hard_gap_order = (
        "registry_contract",
        "forward_contract",
        "reverse_contract",
        "adjoint_contract",
        "transform_contract",
        "mlir_runtime_contract",
        "rust_backend_contract",
        "llvm_backend_contract",
        "jit_backend_contract",
        "native_backend_contract",
    )
    primitive_hard_gaps = {
        identity: [gap for gap in hard_gap_order if readiness.get(gap) is False]
        for identity, readiness in primitive_readiness_by_key.items()
    }
    primitive_next_hard_gap = {
        identity: gaps[0] for identity, gaps in primitive_hard_gaps.items() if gaps
    }
    primitive_hard_gap_primitives: dict[str, list[str]] = {}
    primitive_hard_gap_counts: dict[str, int] = {}
    for identity, gaps in primitive_hard_gaps.items():
        for gap in gaps:
            primitive_hard_gap_primitives.setdefault(gap, []).append(identity)
            primitive_hard_gap_counts[gap] = primitive_hard_gap_counts.get(gap, 0) + 1
    primitive_hard_gap_priority = [
        gap for gap in hard_gap_order if gap in primitive_hard_gap_primitives
    ]
    primitive_hard_gap_frontier = {
        gap: {
            "count": len(primitive_hard_gap_primitives[gap]),
            "next_primitive": primitive_hard_gap_primitives[gap][0],
            "primitives": primitive_hard_gap_primitives[gap],
        }
        for gap in primitive_hard_gap_priority
    }

    metadata = {
        "claim_boundary": plan.claim_boundary,
        "dialect": plan.dialect,
        "executable_backend": plan.executable_backend,
        "effects": {
            status.identity.key: status.effect
            for status in plan.statuses
            if exposes_nondifferentiable_policy(status)
        },
        "nondifferentiable_policies": {
            status.identity.key: status.nondifferentiable_policy
            for status in plan.statuses
            if exposes_nondifferentiable_policy(status)
        },
        "nondifferentiable_boundaries": {
            status.identity.key: status.nondifferentiable_boundary
            for status in plan.statuses
            if status.nondifferentiable_boundary != "not_declared"
        },
        "nondifferentiable_boundary_policies": {
            status.identity.key: status.nondifferentiable_boundary_policy
            for status in plan.statuses
            if status.nondifferentiable_boundary_policy != "not_declared"
        },
        "boundary_contract_primitives": [
            status.identity.key
            for status in plan.statuses
            if status.nondifferentiable_boundary != "not_declared"
            and status.nondifferentiable_boundary_policy == "fail_closed"
        ],
        "mlir_runtime_lowering_primitives": [
            status.identity.key for status in plan.statuses if status.has_lowering_rule
        ],
        "primitive_identities": [status.identity.key for status in plan.statuses],
        "jvp_rule_primitives": [status.identity.key for status in plan.statuses if status.has_jvp],
        "vjp_rule_primitives": [status.identity.key for status in plan.statuses if status.has_vjp],
        "batching_rule_primitives": [
            status.identity.key for status in plan.statuses if status.has_batching_rule
        ],
        "shape_rule_primitives": [
            status.identity.key for status in plan.statuses if status.has_shape_rule
        ],
        "dtype_rule_primitives": [
            status.identity.key for status in plan.statuses if status.has_dtype_rule
        ],
        "static_argument_primitives": [
            status.identity.key for status in plan.statuses if status.has_static_argument_rule
        ],
        "static_derivative_factories": {
            status.identity.key: status.static_derivative_factory
            for status in plan.statuses
            if status.static_derivative_factory not in {"not_declared", "not_required"}
        },
        "static_derivative_signatures": {
            status.identity.key: status.static_signature
            for status in plan.statuses
            if status.static_signature != "none"
        },
        "registry_contract_primitives": [
            status.identity.key for status in plan.statuses if has_registry_contract(status)
        ],
        "reverse_contract_primitives": [
            status.identity.key for status in plan.statuses if has_reverse_contract(status)
        ],
        "reverse_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not status.has_vjp
        ],
        "forward_contract_primitives": [
            status.identity.key for status in plan.statuses if has_forward_contract(status)
        ],
        "forward_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not has_forward_contract(status)
        ],
        "adjoint_contract_primitives": [
            status.identity.key for status in plan.statuses if has_adjoint_contract(status)
        ],
        "adjoint_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not has_adjoint_contract(status)
        ],
        "transform_contract_primitives": [
            status.identity.key for status in plan.statuses if has_transform_contract(status)
        ],
        "transform_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not has_transform_contract(status)
        ],
        "native_backend_contract_primitives": [
            status.identity.key for status in plan.statuses if has_native_backend_contract(status)
        ],
        "native_backend_incomplete_primitives": [
            status.identity.key
            for status in plan.statuses
            if not has_native_backend_contract(status)
        ],
        "rust_backend_contract_primitives": [
            status.identity.key for status in plan.statuses if has_rust_backend_contract(status)
        ],
        "rust_backend_incomplete_primitives": [
            status.identity.key
            for status in plan.statuses
            if not has_rust_backend_contract(status)
        ],
        "rust_backend_blockers": {
            status.identity.key: status.rust_lowering
            for status in plan.statuses
            if "blocked" in status.rust_lowering.lower()
        },
        "rust_backend_signatures": {
            status.identity.key: status.lowering_metadata["rust_backend_signature"]
            for status in plan.statuses
            if has_rust_backend_contract(status)
        },
        "rust_backend_functions": {
            status.identity.key: status.lowering_metadata["rust_backend_functions"]
            for status in plan.statuses
            if has_rust_backend_contract(status)
        },
        "rust_backend_verification_primitives": {
            status.identity.key: status.lowering_metadata["rust_backend_verification"]
            for status in plan.statuses
            if has_rust_backend_contract(status)
        },
        "llvm_backend_contract_primitives": [
            status.identity.key for status in plan.statuses if has_llvm_backend_contract(status)
        ],
        "llvm_backend_incomplete_primitives": [
            status.identity.key
            for status in plan.statuses
            if not has_llvm_backend_contract(status)
        ],
        "llvm_backend_blockers": {
            status.identity.key: status.llvm_lowering
            for status in plan.statuses
            if "blocked" in status.llvm_lowering.lower()
        },
        "jit_backend_contract_primitives": [
            status.identity.key for status in plan.statuses if has_jit_backend_contract(status)
        ],
        "jit_backend_incomplete_primitives": [
            status.identity.key for status in plan.statuses if not has_jit_backend_contract(status)
        ],
        "jit_backend_blockers": {
            status.identity.key: status.jit_lowering
            for status in plan.statuses
            if "blocked" in status.jit_lowering.lower()
        },
        "mlir_runtime_contract_primitives": [
            status.identity.key for status in plan.statuses if has_mlir_runtime_contract(status)
        ],
        "mlir_runtime_incomplete_primitives": [
            status.identity.key
            for status in plan.statuses
            if not has_mlir_runtime_contract(status)
        ],
        "mlir_runtime_blockers": {
            status.identity.key: blocker
            for status in plan.statuses
            for blocker in (mlir_runtime_blocker(status),)
            if blocker is not None
        },
        "mlir_runtime_verification_primitives": {
            status.identity.key: status.mlir_runtime_verification
            for status in plan.statuses
            if status.mlir_runtime_verification.startswith("verified:")
        },
        "primitive_readiness": primitive_readiness_by_key,
        "primitive_readiness_verdict_counts": primitive_readiness_verdict_counts,
        "primitive_hard_gaps": primitive_hard_gaps,
        "primitive_next_hard_gap": primitive_next_hard_gap,
        "primitive_hard_gap_counts": primitive_hard_gap_counts,
        "primitive_hard_gap_primitives": primitive_hard_gap_primitives,
        "primitive_hard_gap_priority": primitive_hard_gap_priority,
        "primitive_hard_gap_frontier": primitive_hard_gap_frontier,
        "transform": plan.transform,
        "uncontracted_primitives": [
            status.identity.key
            for status in plan.statuses
            if status.nondifferentiable_policy == "not_declared"
            or status.nondifferentiable_boundary == "not_declared"
        ],
    }
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=plan.dialect,
        resource_counts={
            "primitives": len(plan.statuses),
            "jvp_rules": sum(status.has_jvp for status in plan.statuses),
            "vjp_rules": sum(status.has_vjp for status in plan.statuses),
            "batching_rules": sum(status.has_batching_rule for status in plan.statuses),
            "shape_rules": sum(status.has_shape_rule for status in plan.statuses),
            "dtype_rules": sum(status.has_dtype_rule for status in plan.statuses),
            "effects": sum(exposes_nondifferentiable_policy(status) for status in plan.statuses),
            "nondifferentiable_policies": sum(
                exposes_nondifferentiable_policy(status) for status in plan.statuses
            ),
            "nondifferentiable_boundaries": sum(
                status.nondifferentiable_boundary != "not_declared" for status in plan.statuses
            ),
            "nondifferentiable_boundary_policies": sum(
                status.nondifferentiable_boundary_policy != "not_declared"
                for status in plan.statuses
            ),
            "boundary_contracts": sum(
                status.nondifferentiable_boundary != "not_declared"
                and status.nondifferentiable_boundary_policy == "fail_closed"
                for status in plan.statuses
            ),
            "mlir_runtime_lowerings": sum(status.has_lowering_rule for status in plan.statuses),
            "static_argument_rules": sum(
                status.has_static_argument_rule for status in plan.statuses
            ),
            "static_derivative_factories": sum(
                status.static_derivative_factory not in {"not_declared", "not_required"}
                for status in plan.statuses
            ),
            "static_derivative_signatures": sum(
                status.static_signature != "none" for status in plan.statuses
            ),
            "registry_contracts": sum(has_registry_contract(status) for status in plan.statuses),
            "forward_contracts": sum(has_forward_contract(status) for status in plan.statuses),
            "forward_incomplete_primitives": sum(
                not has_forward_contract(status) for status in plan.statuses
            ),
            "reverse_contracts": sum(has_reverse_contract(status) for status in plan.statuses),
            "reverse_incomplete_primitives": sum(not status.has_vjp for status in plan.statuses),
            "adjoint_contracts": sum(has_adjoint_contract(status) for status in plan.statuses),
            "adjoint_incomplete_primitives": sum(
                not has_adjoint_contract(status) for status in plan.statuses
            ),
            "transform_contracts": sum(has_transform_contract(status) for status in plan.statuses),
            "transform_incomplete_primitives": sum(
                not has_transform_contract(status) for status in plan.statuses
            ),
            "native_backend_contracts": sum(
                has_native_backend_contract(status) for status in plan.statuses
            ),
            "native_backend_incomplete_primitives": sum(
                not has_native_backend_contract(status) for status in plan.statuses
            ),
            "rust_backend_contracts": sum(
                has_rust_backend_contract(status) for status in plan.statuses
            ),
            "rust_backend_incomplete_primitives": sum(
                not has_rust_backend_contract(status) for status in plan.statuses
            ),
            "rust_backend_blockers": sum(
                "blocked" in status.rust_lowering.lower() for status in plan.statuses
            ),
            "rust_backend_verifications": sum(
                has_rust_backend_contract(status) for status in plan.statuses
            ),
            "llvm_backend_contracts": sum(
                has_llvm_backend_contract(status) for status in plan.statuses
            ),
            "llvm_backend_incomplete_primitives": sum(
                not has_llvm_backend_contract(status) for status in plan.statuses
            ),
            "llvm_backend_blockers": sum(
                "blocked" in status.llvm_lowering.lower() for status in plan.statuses
            ),
            "jit_backend_contracts": sum(
                has_jit_backend_contract(status) for status in plan.statuses
            ),
            "jit_backend_incomplete_primitives": sum(
                not has_jit_backend_contract(status) for status in plan.statuses
            ),
            "jit_backend_blockers": sum(
                "blocked" in status.jit_lowering.lower() for status in plan.statuses
            ),
            "mlir_runtime_contracts": sum(
                has_mlir_runtime_contract(status) for status in plan.statuses
            ),
            "mlir_runtime_incomplete_primitives": sum(
                not has_mlir_runtime_contract(status) for status in plan.statuses
            ),
            "mlir_runtime_blockers": sum(
                mlir_runtime_blocker(status) is not None for status in plan.statuses
            ),
            "mlir_runtime_verifications": sum(
                status.mlir_runtime_verification.startswith("verified:")
                for status in plan.statuses
            ),
            "primitive_readiness_verdicts": len(plan.statuses),
            "primitive_readiness_registry_incomplete": primitive_readiness_verdict_counts.get(
                "registry_incomplete", 0
            ),
            "primitive_readiness_forward_interchange_only": (
                primitive_readiness_verdict_counts.get("forward_interchange_only", 0)
            ),
            "primitive_readiness_transform_interchange_only": (
                primitive_readiness_verdict_counts.get("transform_interchange_only", 0)
            ),
            "primitive_readiness_mlir_runtime_verified": (
                primitive_readiness_verdict_counts.get("mlir_runtime_verified", 0)
            ),
            "primitive_readiness_native_executable": primitive_readiness_verdict_counts.get(
                "native_executable", 0
            ),
            "primitive_hard_gaps": sum(len(gaps) for gaps in primitive_hard_gaps.values()),
            "primitive_next_hard_gaps": len(primitive_next_hard_gap),
            "primitive_hard_gap_priority_classes": len(primitive_hard_gap_priority),
            "primitive_hard_gap_frontier_classes": len(primitive_hard_gap_frontier),
            "uncontracted_primitives": sum(
                status.nondifferentiable_policy == "not_declared"
                or status.nondifferentiable_boundary == "not_declared"
                for status in plan.statuses
            ),
            "executable_backends": int(plan.executable_backend != "none"),
        },
        metadata=metadata,
    )


@dataclass(frozen=True)
class DifferentiableMLIRCompileConfig:
    """Configuration for differentiable primitive MLIR-style lowering."""

    dialect: str = "scpn_diff"
    target: str = "mlir"
    include_numeric_payload: bool = True
    include_metadata: bool = True

    def __post_init__(self) -> None:
        if not self.dialect or not self.dialect.replace("_", "").isalnum():
            raise ValueError("dialect must be a non-empty MLIR-safe identifier")
        if self.target not in {"mlir"}:
            raise ValueError(
                "target must be 'mlir'; executable LLVM/JIT lowering is not yet available"
            )
        if not isinstance(self.include_numeric_payload, bool):
            raise ValueError("include_numeric_payload must be a boolean")
        if not isinstance(self.include_metadata, bool):
            raise ValueError("include_metadata must be a boolean")


@dataclass(frozen=True)
class CompilerADExecutableConfig:
    """Configuration for verified executable primitive AD kernels."""

    backend: str = "mlir_runtime"
    atol: float = 1.0e-10
    rtol: float = 1.0e-10
    verify: bool = True
    mlir_config: DifferentiableMLIRCompileConfig = field(
        default_factory=DifferentiableMLIRCompileConfig
    )

    def __post_init__(self) -> None:
        if self.backend not in {"mlir_runtime", "native_llvm_jit"}:
            raise ValueError("backend must be 'mlir_runtime' or 'native_llvm_jit'")
        if not np.isfinite(self.atol) or self.atol < 0.0:
            raise ValueError("atol must be finite and non-negative")
        if not np.isfinite(self.rtol) or self.rtol < 0.0:
            raise ValueError("rtol must be finite and non-negative")
        if not isinstance(self.verify, bool):
            raise ValueError("verify must be a boolean")
        if not isinstance(self.mlir_config, DifferentiableMLIRCompileConfig):
            raise ValueError("mlir_config must be a DifferentiableMLIRCompileConfig")


@dataclass(frozen=True)
class CompilerADKernelVerification:
    """Runtime verification evidence for an executable primitive AD kernel."""

    value_close: bool
    jvp_close: bool | None
    vjp_close: bool | None
    max_abs_error: float
    samples: int
    gradient_close: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.value_close, bool):
            raise ValueError("value_close must be a boolean")
        if self.jvp_close is not None and not isinstance(self.jvp_close, bool):
            raise ValueError("jvp_close must be a boolean or None")
        if self.vjp_close is not None and not isinstance(self.vjp_close, bool):
            raise ValueError("vjp_close must be a boolean or None")
        if self.gradient_close is not None and not isinstance(self.gradient_close, bool):
            raise ValueError("gradient_close must be a boolean or None")
        if not np.isfinite(self.max_abs_error) or self.max_abs_error < 0.0:
            raise ValueError("max_abs_error must be finite and non-negative")
        if self.samples < 1:
            raise ValueError("samples must be positive")

    @property
    def passed(self) -> bool:
        """Return whether all executed verification checks passed."""

        checks = (self.value_close, self.jvp_close, self.vjp_close, self.gradient_close)
        return all(check is not False for check in checks)


@dataclass(frozen=True)
class ExecutableCompilerADKernel:
    """Executable compiler-backed primitive AD kernel with MLIR provenance."""

    rule_name: str
    backend: str
    mlir_module: MLIRModule
    value_kernel: Callable[[np.ndarray], np.ndarray]
    jvp_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] | None
    vjp_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] | None
    verification: CompilerADKernelVerification
    llvm_gradient_ir: str | None = None
    claim_boundary: str = (
        "verified executable MLIR-runtime primitive AD kernel; "
        "native LLVM/JIT code generation remains fail-closed"
    )

    def __post_init__(self) -> None:
        if not self.rule_name:
            raise ValueError("rule_name must be non-empty")
        if self.backend not in {"mlir_runtime", "native_llvm_jit"}:
            raise ValueError("backend must be 'mlir_runtime' or 'native_llvm_jit'")
        if not isinstance(self.mlir_module, MLIRModule):
            raise ValueError("mlir_module must be an MLIRModule")
        if not callable(self.value_kernel):
            raise ValueError("value_kernel must be callable")
        if self.jvp_kernel is not None and not callable(self.jvp_kernel):
            raise ValueError("jvp_kernel must be callable")
        if self.vjp_kernel is not None and not callable(self.vjp_kernel):
            raise ValueError("vjp_kernel must be callable")
        if not isinstance(self.verification, CompilerADKernelVerification):
            raise ValueError("verification must be CompilerADKernelVerification")
        if not self.verification.passed:
            raise ValueError("executable compiler AD kernel verification failed")
        if self.llvm_gradient_ir is not None and not self.llvm_gradient_ir.strip():
            raise ValueError("llvm_gradient_ir must be non-empty or None")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def value(self, values: np.ndarray) -> np.ndarray:
        """Execute the compiled value kernel."""

        return self.value_kernel(values)

    def jvp(self, values: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """Execute the compiled JVP kernel."""

        if self.jvp_kernel is None:
            raise ValueError(f"kernel {self.rule_name} has no JVP rule")
        return self.jvp_kernel(values, tangent)

    def vjp(self, values: np.ndarray, cotangent: np.ndarray) -> np.ndarray:
        """Execute the compiled VJP kernel."""

        if self.vjp_kernel is None:
            raise ValueError(f"kernel {self.rule_name} has no VJP rule")
        return self.vjp_kernel(values, cotangent)

    def gradient(self, values: np.ndarray) -> np.ndarray:
        """Execute the compiled scalar-output gradient kernel."""

        if self.vjp_kernel is None:
            raise ValueError(f"kernel {self.rule_name} has no VJP rule")
        checked_values = _as_finite_vector("values", values)
        output = self.value_kernel(checked_values)
        if output.size != 1:
            raise ValueError(f"kernel {self.rule_name} gradient requires scalar output")
        return self.vjp_kernel(checked_values, np.ones(1, dtype=np.float64))


def make_executable_ad_kernel_batching_rule(
    kernel: ExecutableCompilerADKernel,
    *,
    method: str = "auto",
) -> PrimitiveBatchingRule:
    """Create a primitive-specific batching rule backed by an executable AD kernel.

    ``method="auto"`` dispatches one-argument calls to ``value`` and two-argument
    calls to ``jvp`` or ``vjp`` by matching the second slice against the input and
    output dimensions. If those dimensions are equal, callers must request
    ``method="jvp"`` or ``method="vjp"`` explicitly so transform nesting remains
    fail-closed rather than guessing.
    """

    if not isinstance(kernel, ExecutableCompilerADKernel):
        raise ValueError("kernel must be an ExecutableCompilerADKernel")
    if method not in {"auto", "value", "jvp", "vjp", "gradient"}:
        raise ValueError("method must be 'auto', 'value', 'jvp', 'vjp', or 'gradient'")

    def batching_rule(
        function: Callable[..., object],
        args: tuple[object, ...],
        axes: tuple[int | None, ...],
        out_axes: int,
    ) -> object:
        del function
        batched_args, batch_size = _prepare_executable_kernel_batch_args(args, axes)
        outputs = []
        for item in range(batch_size):
            call_args = tuple(
                _slice_executable_kernel_batch_arg(arg, axis, item) for arg, axis in batched_args
            )
            outputs.append(_execute_kernel_batch_slice(kernel, method, call_args))
        return _stack_executable_kernel_batch_outputs(outputs, out_axes)

    return batching_rule


def _prepare_executable_kernel_batch_args(
    args: tuple[object, ...],
    axes: tuple[int | None, ...],
) -> tuple[tuple[tuple[object, int | None], ...], int]:
    if not args:
        raise ValueError("executable AD kernel batching requires at least one argument")
    if len(args) != len(axes):
        raise ValueError("executable AD kernel batching axes must match argument count")
    batched: list[tuple[object, int | None]] = []
    batch_size: int | None = None
    for index, (arg, axis) in enumerate(zip(args, axes, strict=True)):
        if axis is None:
            batched.append((arg, None))
            continue
        if not isinstance(axis, int):
            raise ValueError("executable AD kernel batching axes must be integers or None")
        array = _as_executable_kernel_batch_array(f"argument {index}", arg)
        axis_index = _normalise_executable_kernel_batch_axis(
            f"axes[{index}]",
            axis,
            array.ndim,
        )
        size = int(array.shape[axis_index])
        if size <= 0:
            raise ValueError("executable AD kernel batching axes must be non-empty")
        if batch_size is None:
            batch_size = size
        elif size != batch_size:
            raise ValueError("executable AD kernel batching axes must have the same length")
        batched.append((array, axis_index))
    if batch_size is None:
        raise ValueError("executable AD kernel batching requires at least one mapped axis")
    return tuple(batched), batch_size


def _slice_executable_kernel_batch_arg(arg: object, axis: int | None, item: int) -> object:
    if axis is None:
        return arg
    return np.take(cast(np.ndarray, arg), item, axis=axis)


def _as_executable_kernel_batch_array(name: str, value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind in {"b", "O", "S", "U"}:
        raise ValueError(f"executable AD kernel batching {name} must be numeric")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"executable AD kernel batching {name} cannot map over a scalar")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"executable AD kernel batching {name} must contain only finite values")
    return array


def _normalise_executable_kernel_batch_axis(name: str, axis: int, ndim: int) -> int:
    if ndim == 0:
        raise ValueError(f"{name} cannot map over a scalar")
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(f"{name} is out of bounds for argument rank {ndim}")
    return axis


def _execute_kernel_batch_slice(
    kernel: ExecutableCompilerADKernel,
    method: str,
    args: tuple[object, ...],
) -> np.ndarray:
    if method == "value":
        if len(args) != 1:
            raise ValueError("executable AD kernel value batching requires one argument")
        return kernel.value(_as_finite_vector("values", args[0]))
    if method == "gradient":
        if len(args) != 1:
            raise ValueError("executable AD kernel gradient batching requires one argument")
        return kernel.gradient(_as_finite_vector("values", args[0]))
    if method == "jvp":
        if len(args) != 2:
            raise ValueError("executable AD kernel JVP batching requires values and tangent")
        return kernel.jvp(
            _as_finite_vector("values", args[0]),
            _as_finite_vector("tangent", args[1]),
        )
    if method == "vjp":
        if len(args) != 2:
            raise ValueError("executable AD kernel VJP batching requires values and cotangent")
        return kernel.vjp(
            _as_finite_vector("values", args[0]),
            _as_finite_vector("cotangent", args[1]),
        )
    if len(args) == 1:
        return kernel.value(_as_finite_vector("values", args[0]))
    if len(args) != 2:
        raise ValueError("automatic executable AD kernel batching supports one or two arguments")
    values = _as_finite_vector("values", args[0])
    tangent_or_cotangent = _as_finite_vector("tangent_or_cotangent", args[1])
    output_size = int(kernel.value(values).size)
    input_size = int(values.size)
    jvp_matches = kernel.jvp_kernel is not None and tangent_or_cotangent.size == input_size
    vjp_matches = kernel.vjp_kernel is not None and tangent_or_cotangent.size == output_size
    if jvp_matches and vjp_matches:
        raise ValueError(
            "ambiguous executable AD kernel batching method; specify method='jvp' or method='vjp'"
        )
    if jvp_matches:
        return kernel.jvp(values, tangent_or_cotangent)
    if vjp_matches:
        return kernel.vjp(values, tangent_or_cotangent)
    raise ValueError(
        "automatic executable AD kernel batching could not match the second argument "
        "to tangent or cotangent dimensions"
    )


def _stack_executable_kernel_batch_outputs(
    outputs: Sequence[np.ndarray],
    out_axes: int,
) -> np.ndarray:
    if not outputs:
        raise ValueError("executable AD kernel batching outputs must be non-empty")
    arrays = [np.asarray(output, dtype=np.float64) for output in outputs]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("executable AD kernel batching outputs must have consistent shapes")
    result_rank = arrays[0].ndim + 1
    axis = out_axes
    if axis < 0:
        axis += result_rank
    if axis < 0 or axis >= result_rank:
        raise ValueError("executable AD kernel batching out_axes is out of bounds")
    return np.stack(arrays, axis=axis)


def compile_kuramoto_to_mlir(
    problem: KuramotoProblem | np.ndarray,
    config: MLIRCompileConfig,
    omega: np.ndarray | None = None,
) -> MLIRModule:
    """Compile a Kuramoto problem into deterministic MLIR-style text.

    ``problem`` may be a validated :class:`KuramotoProblem` or a raw coupling
    matrix when ``omega`` is supplied. Raw arrays are validated through the
    public Kuramoto facade before IR generation.
    """

    if isinstance(problem, KuramotoProblem):
        validated = problem
    else:
        if omega is None:
            raise ValueError("omega is required when problem is a raw coupling matrix")
        validated = build_kuramoto_problem(problem, omega)

    coupling_terms = _coupling_terms(validated.K_nm)
    lines = [
        f'module attributes {{scpn.module = "kuramoto_xy", scpn.dialect = "{config.dialect}", '
        f"scpn.n_oscillators = {validated.n_oscillators}, "
        f"scpn.trotter_steps = {config.trotter_steps}, "
        f"scpn.trotter_order = {config.trotter_order}}} {{",
        "  func.func @main() {",
    ]
    for index, value in enumerate(validated.omega):
        lines.append(f"    scpn.omega %{index} {{value = {_fmt_float(float(value))}}}")
    for term_index, (left, right, value) in enumerate(coupling_terms):
        lines.append(
            "    scpn.coupling "
            f"%c{term_index} {{i = {left}, j = {right}, value = {_fmt_float(value)}}}"
        )
    lines.append(
        "    scpn.trotter_evolve "
        f"{{time = {_fmt_float(config.time)}, steps = {config.trotter_steps}, "
        f"order = {config.trotter_order}}}"
    )
    lines.append("    return")
    lines.append("  }")
    if config.include_metadata and validated.metadata:
        encoded = json.dumps(dict(validated.metadata), sort_keys=True, separators=(",", ":"))
        lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    resource_counts = {
        "n_oscillators": validated.n_oscillators,
        "omega_terms": validated.n_oscillators,
        "coupling_terms": len(coupling_terms),
        "trotter_steps": config.trotter_steps,
        "trotter_order": config.trotter_order,
    }
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=config.dialect,
        resource_counts=resource_counts,
        metadata={
            "claim_boundary": "textual MLIR-style IR export; no provider lowering or hardware execution",
            "problem": validated.to_metadata(),
        },
    )


def compile_custom_derivative_rule_to_mlir(
    rule: CustomDerivativeRule,
    values: np.ndarray,
    config: DifferentiableMLIRCompileConfig | None = None,
) -> MLIRModule:
    """Lower an exact custom derivative rule to deterministic MLIR-style text.

    This emits an auditable differentiable-primitive interchange artifact with
    value and Jacobian shape metadata. When numeric payloads are enabled, the
    current value and exact custom Jacobian are embedded as deterministic
    attributes. The function deliberately does not claim executable LLVM or JIT
    code generation.
    """

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("differentiable MLIR lowering requires a CustomDerivativeRule")
    compile_config = DifferentiableMLIRCompileConfig() if config is None else config
    jacobian_result = value_and_custom_jacobian(rule, values)
    parameter_count = jacobian_result.jacobian.shape[1]
    output_count = jacobian_result.value.size
    lines = [
        f'module attributes {{scpn.module = "differentiable_primitive", '
        f'scpn.dialect = "{compile_config.dialect}", '
        f'scpn.rule = "{_escape_mlir_string(rule.name)}", '
        f"scpn.n_parameters = {parameter_count}, "
        f"scpn.n_outputs = {output_count}}} {{",
        "  func.func @main() {",
    ]
    for index, (name, trainable) in enumerate(
        zip(jacobian_result.parameter_names, jacobian_result.trainable, strict=True)
    ):
        lines.append(
            "    scpn_diff.parameter "
            f'%p{index} {{name = "{_escape_mlir_string(name)}", trainable = {_fmt_bool(trainable)}}}'
        )
    if compile_config.include_numeric_payload:
        for index, value in enumerate(jacobian_result.value):
            lines.append(f"    scpn_diff.value %{index} {{value = {_fmt_float(float(value))}}}")
        for row in range(output_count):
            for column in range(parameter_count):
                value = float(jacobian_result.jacobian[row, column])
                if abs(value) > 1.0e-15:
                    lines.append(
                        "    scpn_diff.jacobian "
                        f"{{row = {row}, col = {column}, value = {_fmt_float(value)}}}"
                    )
    lines.append(
        "    scpn_diff.custom_rule "
        f"{{jvp = {_fmt_bool(rule.jvp_rule is not None)}, "
        f"vjp = {_fmt_bool(rule.vjp_rule is not None)}, "
        'execution = "interchange_only"}}'
    )
    lines.append("    return")
    lines.append("  }")
    if compile_config.include_metadata:
        metadata = {
            "method": jacobian_result.method,
            "parameter_names": list(jacobian_result.parameter_names),
            "trainable": list(jacobian_result.trainable),
            "target": compile_config.target,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=compile_config.dialect,
        resource_counts={
            "parameters": parameter_count,
            "outputs": output_count,
            "jacobian_nnz": int(np.count_nonzero(jacobian_result.jacobian)),
            "trainable_parameters": int(sum(jacobian_result.trainable)),
        },
        metadata={
            "claim_boundary": "textual differentiable MLIR-style IR export; no executable LLVM or JIT lowering",
            "rule": rule.name,
            "target": compile_config.target,
            "sha256_source": "module.text",
        },
    )


def compile_custom_derivative_rule_to_executable(
    rule: CustomDerivativeRule,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile a custom derivative rule into a verified executable AD kernel.

    The executable backend is the dependency-free SCPN MLIR runtime adapter:
    it couples deterministic differentiable MLIR provenance with normalized
    runtime callables for value/JVP/VJP execution and verifies those kernels
    against the source custom derivative rule before returning. Native LLVM/JIT
    kernels use primitive-specific lowering entrypoints.
    """

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("executable AD lowering requires a CustomDerivativeRule")
    compile_config = CompilerADExecutableConfig() if config is None else config
    if compile_config.backend != "mlir_runtime":
        raise ValueError(
            "compile_custom_derivative_rule_to_executable requires backend='mlir_runtime'; "
            "use a primitive-specific native LLVM/JIT lowering entrypoint"
        )
    values = _as_finite_vector("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _as_finite_vector(
            "value kernel output", rule.value_fn(_as_finite_vector("values", raw_values))
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        if rule.jvp_rule is None:
            raise ValueError(f"rule {rule.name} has no JVP rule")
        checked_values = _as_finite_vector("values", raw_values)
        checked_tangent = _as_finite_vector("tangent", raw_tangent)
        if checked_tangent.shape != checked_values.shape:
            raise ValueError("tangent shape must match values shape")
        return _as_finite_vector(
            "JVP kernel output", rule.jvp_rule(checked_values, checked_tangent)
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        if rule.vjp_rule is None:
            raise ValueError(f"rule {rule.name} has no VJP rule")
        checked_values = _as_finite_vector("values", raw_values)
        checked_cotangent = _as_finite_vector("cotangent", raw_cotangent)
        return _as_finite_vector(
            "VJP kernel output", rule.vjp_rule(checked_values, checked_cotangent)
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    llvm_gradient_ir = (
        _compile_scalar_gradient_llvm_ir(rule, values, vjp_kernel)
        if verification.gradient_close is True and rule.vjp_rule is not None
        else None
    )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_gradient_ir,
    )


def compile_registered_primitive_to_executable(
    registry: CustomDerivativeRegistry,
    identity: PrimitiveIdentity | str,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile a registered primitive identity into an executable AD kernel."""

    if not isinstance(registry, CustomDerivativeRegistry):
        raise ValueError("registry must be a CustomDerivativeRegistry")
    primitive_identity = PrimitiveIdentity.parse(identity)
    transform = registry.transform_snapshot().get(primitive_identity)
    rule = registry.require(primitive_identity)
    if transform is not None and transform.lowering_rule is not None:
        lowering_rule = cast(Callable[..., Any], transform.lowering_rule)
        try:
            lowered = lowering_rule(
                rule,
                sample_values,
                config,
                sample_tangent=sample_tangent,
                sample_cotangent=sample_cotangent,
            )
        except TypeError:
            lowered = lowering_rule(rule)
        if not isinstance(lowered, ExecutableCompilerADKernel):
            raise ValueError("registered lowering_rule must return an ExecutableCompilerADKernel")
        return lowered
    return compile_custom_derivative_rule_to_executable(
        rule,
        sample_values,
        config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )


def make_program_ad_linalg_matrix_power_executable_lowering_rule(
    power: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[[CustomDerivativeRule], ExecutableCompilerADKernel]:
    """Build a verified executable lowering rule for a fixed matrix_power signature."""

    direct_rule = program_ad_linalg_matrix_power_derivative_rule(power)
    values = _as_finite_vector("sample_values", sample_values)
    compile_config = CompilerADExecutableConfig() if config is None else config

    def lowering_rule(_registered_rule: CustomDerivativeRule) -> ExecutableCompilerADKernel:
        if not isinstance(_registered_rule, CustomDerivativeRule):
            raise ValueError("registered_rule must be a CustomDerivativeRule")
        return compile_custom_derivative_rule_to_executable(
            direct_rule,
            values,
            compile_config,
            sample_tangent=sample_tangent,
        )

    return lowering_rule


def make_program_ad_linalg_multi_dot_executable_lowering_rule(
    operand_shapes: Sequence[Sequence[int]],
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    *,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[[CustomDerivativeRule], ExecutableCompilerADKernel]:
    """Build a verified executable lowering rule for a fixed multi_dot signature."""

    direct_rule = program_ad_linalg_multi_dot_derivative_rule(operand_shapes)
    values = _as_finite_vector("sample_values", sample_values)
    compile_config = CompilerADExecutableConfig() if config is None else config

    def lowering_rule(_registered_rule: CustomDerivativeRule) -> ExecutableCompilerADKernel:
        if not isinstance(_registered_rule, CustomDerivativeRule):
            raise ValueError("registered_rule must be a CustomDerivativeRule")
        return compile_custom_derivative_rule_to_executable(
            direct_rule,
            values,
            compile_config,
            sample_tangent=sample_tangent,
        )

    return lowering_rule


def _verify_executable_ad_kernel(
    rule: CustomDerivativeRule,
    values: np.ndarray,
    value_kernel: Callable[[np.ndarray], np.ndarray],
    jvp_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    vjp_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
    config: CompilerADExecutableConfig,
    *,
    sample_tangent: Sequence[float] | np.ndarray | None,
    sample_cotangent: Sequence[float] | np.ndarray | None,
) -> CompilerADKernelVerification:
    if not config.verify:
        return CompilerADKernelVerification(
            value_close=True,
            jvp_close=None,
            vjp_close=None,
            max_abs_error=0.0,
            samples=1,
            gradient_close=None,
        )
    errors: list[float] = []
    expected_value = _as_finite_vector("rule value", rule.value_fn(values))
    kernel_value = value_kernel(values)
    value_close = bool(
        np.allclose(kernel_value, expected_value, atol=config.atol, rtol=config.rtol)
    )
    errors.append(_max_abs_error(kernel_value, expected_value))
    jvp_close: bool | None = None
    if rule.jvp_rule is not None and jvp_kernel is not None:
        tangent = (
            np.ones_like(values)
            if sample_tangent is None
            else _as_finite_vector("sample_tangent", sample_tangent)
        )
        if tangent.shape != values.shape:
            raise ValueError("sample_tangent shape must match sample_values shape")
        expected_jvp = _as_finite_vector("rule JVP", rule.jvp_rule(values, tangent))
        kernel_jvp = jvp_kernel(values, tangent)
        jvp_close = bool(np.allclose(kernel_jvp, expected_jvp, atol=config.atol, rtol=config.rtol))
        errors.append(_max_abs_error(kernel_jvp, expected_jvp))
    vjp_close: bool | None = None
    gradient_close: bool | None = None
    if rule.vjp_rule is not None and vjp_kernel is not None:
        cotangent = (
            np.ones_like(expected_value)
            if sample_cotangent is None
            else _as_finite_vector("sample_cotangent", sample_cotangent)
        )
        if cotangent.shape != expected_value.shape:
            raise ValueError("sample_cotangent shape must match value output shape")
        expected_vjp = _as_finite_vector("rule VJP", rule.vjp_rule(values, cotangent))
        kernel_vjp = vjp_kernel(values, cotangent)
        vjp_close = bool(np.allclose(kernel_vjp, expected_vjp, atol=config.atol, rtol=config.rtol))
        errors.append(_max_abs_error(kernel_vjp, expected_vjp))
        if expected_value.size == 1:
            unit_cotangent = np.ones(1, dtype=np.float64)
            expected_gradient = _as_finite_vector(
                "rule scalar gradient", rule.vjp_rule(values, unit_cotangent)
            )
            kernel_gradient = vjp_kernel(values, unit_cotangent)
            gradient_close = bool(
                np.allclose(kernel_gradient, expected_gradient, atol=config.atol, rtol=config.rtol)
            )
            errors.append(_max_abs_error(kernel_gradient, expected_gradient))
    verification = CompilerADKernelVerification(
        value_close=value_close,
        jvp_close=jvp_close,
        vjp_close=vjp_close,
        max_abs_error=max(errors),
        samples=1,
        gradient_close=gradient_close,
    )
    if not verification.passed:
        raise ValueError("executable compiler AD kernel verification failed")
    return verification


def _compile_scalar_gradient_llvm_ir(
    rule: CustomDerivativeRule,
    values: np.ndarray,
    vjp_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> str:
    value = _as_finite_vector("rule value", rule.value_fn(values))
    if value.size != 1:
        return ""
    gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
    function_name = _safe_llvm_symbol(f"{rule.name}_gradient")
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule.name)}"',
        '; source = "verified_mlir_runtime_vjp_cotangent_one"',
        '; execution = "mlir_runtime_gradient_adapter"',
        '; native_llvm_jit = "blocked_until_native_codegen_backend_exists"',
        f"define void @{function_name}(double* %out) {{",
        "entry:",
    ]
    for index, component in enumerate(gradient):
        lines.append(f"  %slot{index} = getelementptr double, double* %out, i64 {index}")
        lines.append(f"  store double {_fmt_float(float(component))}, double* %slot{index}")
    lines.append("  ret void")
    lines.append("}")
    return "\n".join(lines) + "\n"


def _load_llvmlite_binding() -> Any:
    try:
        llvm = importlib.import_module("llvmlite.binding")
    except ModuleNotFoundError as exc:
        raise ValueError(
            "native_llvm_jit backend requires llvmlite.binding to be installed"
        ) from exc

    for initializer in (
        llvm.initialize_native_target,
        llvm.initialize_native_asmprinter,
    ):
        try:
            initializer()
        except RuntimeError as exc:
            if "already" not in str(exc).lower():
                raise
    return llvm


def _compile_scalar_quadratic_native_llvm_ir(
    rule_name: str,
    quadratic: float,
    linear: float,
    constant: float,
) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    doubled_quadratic = 2.0 * quadratic
    quadratic_literal = _fmt_llvm_double(quadratic)
    linear_literal = _fmt_llvm_double(linear)
    constant_literal = _fmt_llvm_double(constant)
    doubled_quadratic_literal = _fmt_llvm_double(doubled_quadratic)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; source = "native_scalar_quadratic_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
            "  %x2 = fmul double %x, %x",
            f"  %ax2 = fmul double {quadratic_literal}, %x2",
            f"  %bx = fmul double {linear_literal}, %x",
            "  %sum = fadd double %ax2, %bx",
            f"  %value = fadd double %sum, {constant_literal}",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %value, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
            f"  %ax = fmul double {doubled_quadratic_literal}, %x",
            f"  %grad = fadd double %ax, {linear_literal}",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %grad, double* %out0",
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_jvp(double* %values, "
                "double* %tangent, double* %out) {"
            ),
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
            f"  %ax = fmul double {doubled_quadratic_literal}, %x",
            f"  %grad = fadd double %ax, {linear_literal}",
            "  %tangent0ptr = getelementptr double, double* %tangent, i64 0",
            "  %tangent0 = load double, double* %tangent0ptr",
            "  %jvp = fmul double %grad, %tangent0",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %jvp, double* %out0",
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_vjp(double* %values, "
                "double* %cotangent, double* %out) {"
            ),
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
            f"  %ax = fmul double {doubled_quadratic_literal}, %x",
            f"  %grad = fadd double %ax, {linear_literal}",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
            "  %vjp = fmul double %grad, %cotangent0",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %vjp, double* %out0",
            "  ret void",
            "}",
            "",
        ]
    )


def _fmt_llvm_double(value: float) -> str:
    text = _fmt_float(float(value))
    if "." not in text and "e" not in text.lower():
        return f"{text}.0"
    return text


def _scalar_unary_native_intrinsics(primitive: str) -> tuple[str, ...]:
    if primitive == "sin":
        return ("sin", "cos")
    if primitive == "cos":
        return ("sin", "cos")
    if primitive == "exp":
        return ("exp",)
    raise ValueError("native scalar unary LLVM/JIT primitive must be one of sin, cos, exp")


def _scalar_unary_native_value_lines(primitive: str) -> tuple[str, ...]:
    if primitive == "sin":
        return ("%value = call double @llvm.sin.f64(double %x)",)
    if primitive == "cos":
        return ("%value = call double @llvm.cos.f64(double %x)",)
    if primitive == "exp":
        return ("%value = call double @llvm.exp.f64(double %x)",)
    raise ValueError("native scalar unary LLVM/JIT primitive must be one of sin, cos, exp")


def _scalar_unary_native_gradient_lines(primitive: str) -> tuple[str, ...]:
    if primitive == "sin":
        return ("%grad = call double @llvm.cos.f64(double %x)",)
    if primitive == "cos":
        return (
            "%sin = call double @llvm.sin.f64(double %x)",
            "%grad = fsub double -0.0, %sin",
        )
    if primitive == "exp":
        return ("%grad = call double @llvm.exp.f64(double %x)",)
    raise ValueError("native scalar unary LLVM/JIT primitive must be one of sin, cos, exp")


def _compile_scalar_unary_elementwise_native_llvm_ir(
    rule_name: str,
    primitive: str,
) -> str:
    checked_primitive = primitive.strip().lower()
    intrinsics = _scalar_unary_native_intrinsics(checked_primitive)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        f'; primitive = "{_escape_mlir_string(checked_primitive)}"',
        '; source = "native_scalar_unary_elementwise_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
    ]
    for intrinsic in intrinsics:
        lines.append(f"declare double @llvm.{intrinsic}.f64(double)")
    lines.extend(
        [
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %x = load double, double* %xptr",
        ]
    )
    lines.extend(f"  {line}" for line in _scalar_unary_native_value_lines(checked_primitive))
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %value, double* %out0",
            "  ret void",
            "}",
            "",
        ]
    )
    for function_name, operand_name, result_name in (
        ("gradient", None, "grad"),
        ("jvp", "tangent", "jvp"),
        ("vjp", "cotangent", "vjp"),
    ):
        if operand_name is None:
            lines.append(
                f"define void @{base_symbol}_{function_name}(double* %values, double* %out) {{"
            )
        else:
            lines.append(
                f"define void @{base_symbol}_{function_name}(double* %values, "
                f"double* %{operand_name}, double* %out) {{"
            )
        lines.extend(
            [
                "entry:",
                "  %xptr = getelementptr double, double* %values, i64 0",
                "  %x = load double, double* %xptr",
            ]
        )
        lines.extend(
            f"  {line}" for line in _scalar_unary_native_gradient_lines(checked_primitive)
        )
        if operand_name is not None:
            lines.extend(
                [
                    f"  %{operand_name}0ptr = getelementptr double, double* %{operand_name}, i64 0",
                    f"  %{operand_name}0 = load double, double* %{operand_name}0ptr",
                    f"  %{result_name} = fmul double %grad, %{operand_name}0",
                ]
            )
        lines.extend(
            [
                "  %out0 = getelementptr double, double* %out, i64 0",
                f"  store double %{result_name}, double* %out0",
                "  ret void",
                "}",
                "",
            ]
        )
    return "\n".join(lines)


def _scalar_binary_native_value_line(primitive: str) -> str:
    if primitive == "add":
        return "%value = fadd double %x, %y"
    if primitive == "subtract":
        return "%value = fsub double %x, %y"
    if primitive == "multiply":
        return "%value = fmul double %x, %y"
    raise ValueError(
        "native scalar binary LLVM/JIT primitive must be one of add, subtract, multiply"
    )


def _scalar_binary_native_gradient_lines(primitive: str) -> tuple[str, ...]:
    if primitive == "add":
        return ("%grad_x = fadd double 1.0, 0.0", "%grad_y = fadd double 1.0, 0.0")
    if primitive == "subtract":
        return ("%grad_x = fadd double 1.0, 0.0", "%grad_y = fsub double -0.0, 1.0")
    if primitive == "multiply":
        return ("%grad_x = fadd double %y, 0.0", "%grad_y = fadd double %x, 0.0")
    raise ValueError(
        "native scalar binary LLVM/JIT primitive must be one of add, subtract, multiply"
    )


def _compile_scalar_binary_elementwise_native_llvm_ir(
    rule_name: str,
    primitive: str,
) -> str:
    checked_primitive = primitive.strip().lower()
    value_line = _scalar_binary_native_value_line(checked_primitive)
    gradient_lines = _scalar_binary_native_gradient_lines(checked_primitive)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        f'; primitive = "{_escape_mlir_string(checked_primitive)}"',
        '; source = "native_scalar_binary_elementwise_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
        "  %xptr = getelementptr double, double* %values, i64 0",
        "  %yptr = getelementptr double, double* %values, i64 1",
        "  %x = load double, double* %xptr",
        "  %y = load double, double* %yptr",
        f"  {value_line}",
        "  %out0 = getelementptr double, double* %out, i64 0",
        "  store double %value, double* %out0",
        "  ret void",
        "}",
        "",
        f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
        "entry:",
        "  %xptr = getelementptr double, double* %values, i64 0",
        "  %yptr = getelementptr double, double* %values, i64 1",
        "  %x = load double, double* %xptr",
        "  %y = load double, double* %yptr",
    ]
    lines.extend(f"  {line}" for line in gradient_lines)
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  store double %grad_x, double* %out0",
            "  store double %grad_y, double* %out1",
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_jvp(double* %values, "
                "double* %tangent, double* %out) {"
            ),
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %yptr = getelementptr double, double* %values, i64 1",
            "  %x = load double, double* %xptr",
            "  %y = load double, double* %yptr",
        ]
    )
    lines.extend(f"  {line}" for line in gradient_lines)
    lines.extend(
        [
            "  %tangent0ptr = getelementptr double, double* %tangent, i64 0",
            "  %tangent1ptr = getelementptr double, double* %tangent, i64 1",
            "  %tangent0 = load double, double* %tangent0ptr",
            "  %tangent1 = load double, double* %tangent1ptr",
            "  %jvp_x = fmul double %grad_x, %tangent0",
            "  %jvp_y = fmul double %grad_y, %tangent1",
            "  %jvp = fadd double %jvp_x, %jvp_y",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %jvp, double* %out0",
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_vjp(double* %values, "
                "double* %cotangent, double* %out) {"
            ),
            "entry:",
            "  %xptr = getelementptr double, double* %values, i64 0",
            "  %yptr = getelementptr double, double* %values, i64 1",
            "  %x = load double, double* %xptr",
            "  %y = load double, double* %yptr",
        ]
    )
    lines.extend(f"  {line}" for line in gradient_lines)
    lines.extend(
        [
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
            "  %vjp_x = fmul double %grad_x, %cotangent0",
            "  %vjp_y = fmul double %grad_y, %cotangent0",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  store double %vjp_x, double* %out0",
            "  store double %vjp_y, double* %out1",
            "  ret void",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_vector_dot_dimension(dimension: int | np.integer) -> int:
    checked = int(dimension)
    if checked < 1:
        raise ValueError("native vector dot dimension must be positive")
    return checked


def _compile_vector_dot_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "dot"',
        '; source = "native_vector_dot_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_sum = "0.0"
    for index in range(checked_dimension):
        right_index = checked_dimension + index
        lines.extend(
            [
                f"  %xptr{index} = getelementptr double, double* %values, i64 {index}",
                f"  %yptr{index} = getelementptr double, double* %values, i64 {right_index}",
                f"  %x{index} = load double, double* %xptr{index}",
                f"  %y{index} = load double, double* %yptr{index}",
                f"  %prod{index} = fmul double %x{index}, %y{index}",
                f"  %sum{index} = fadd double {previous_sum}, %prod{index}",
            ]
        )
        previous_sum = f"%sum{index}"
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_sum}, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for index in range(checked_dimension):
        right_index = checked_dimension + index
        lines.extend(
            [
                f"  %xptr{index} = getelementptr double, double* %values, i64 {index}",
                f"  %yptr{index} = getelementptr double, double* %values, i64 {right_index}",
                f"  %x{index} = load double, double* %xptr{index}",
                f"  %y{index} = load double, double* %yptr{index}",
                f"  %outxptr{index} = getelementptr double, double* %out, i64 {index}",
                f"  %outyptr{index} = getelementptr double, double* %out, i64 {right_index}",
                f"  store double %y{index}, double* %outxptr{index}",
                f"  store double %x{index}, double* %outyptr{index}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    previous_jvp_sum = "0.0"
    for index in range(checked_dimension):
        right_index = checked_dimension + index
        lines.extend(
            [
                f"  %xptr_jvp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %yptr_jvp{index} = getelementptr double, double* %values, i64 {right_index}",
                f"  %txptr{index} = getelementptr double, double* %tangent, i64 {index}",
                f"  %typtr{index} = getelementptr double, double* %tangent, i64 {right_index}",
                f"  %x_jvp{index} = load double, double* %xptr_jvp{index}",
                f"  %y_jvp{index} = load double, double* %yptr_jvp{index}",
                f"  %tx{index} = load double, double* %txptr{index}",
                f"  %ty{index} = load double, double* %typtr{index}",
                f"  %left_jvp{index} = fmul double %y_jvp{index}, %tx{index}",
                f"  %right_jvp{index} = fmul double %x_jvp{index}, %ty{index}",
                f"  %term_jvp{index} = fadd double %left_jvp{index}, %right_jvp{index}",
                f"  %sum_jvp{index} = fadd double {previous_jvp_sum}, %term_jvp{index}",
            ]
        )
        previous_jvp_sum = f"%sum_jvp{index}"
    lines.extend(
        [
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_jvp_sum}, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
        ]
    )
    for index in range(checked_dimension):
        right_index = checked_dimension + index
        lines.extend(
            [
                f"  %xptr_vjp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %yptr_vjp{index} = getelementptr double, double* %values, i64 {right_index}",
                f"  %x_vjp{index} = load double, double* %xptr_vjp{index}",
                f"  %y_vjp{index} = load double, double* %yptr_vjp{index}",
                f"  %vjp_x{index} = fmul double %y_vjp{index}, %cotangent0",
                f"  %vjp_y{index} = fmul double %x_vjp{index}, %cotangent0",
                f"  %outxptr_vjp{index} = getelementptr double, double* %out, i64 {index}",
                f"  %outyptr_vjp{index} = getelementptr double, double* %out, i64 {right_index}",
                f"  store double %vjp_x{index}, double* %outxptr_vjp{index}",
                f"  store double %vjp_y{index}, double* %outyptr_vjp{index}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_vector_squared_norm_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "squared_norm"',
        '; source = "native_vector_squared_norm_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_sum = "0.0"
    for index in range(checked_dimension):
        lines.extend(
            [
                f"  %xptr{index} = getelementptr double, double* %values, i64 {index}",
                f"  %x{index} = load double, double* %xptr{index}",
                f"  %square{index} = fmul double %x{index}, %x{index}",
                f"  %sum{index} = fadd double {previous_sum}, %square{index}",
            ]
        )
        previous_sum = f"%sum{index}"
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_sum}, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for index in range(checked_dimension):
        lines.extend(
            [
                f"  %xptr_gradient{index} = getelementptr double, double* %values, i64 {index}",
                f"  %x_gradient{index} = load double, double* %xptr_gradient{index}",
                f"  %grad{index} = fmul double 2.0, %x_gradient{index}",
                f"  %out_gradient{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double %grad{index}, double* %out_gradient{index}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    previous_jvp_sum = "0.0"
    for index in range(checked_dimension):
        lines.extend(
            [
                f"  %xptr_jvp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %tptr{index} = getelementptr double, double* %tangent, i64 {index}",
                f"  %x_jvp{index} = load double, double* %xptr_jvp{index}",
                f"  %t{index} = load double, double* %tptr{index}",
                f"  %prod_jvp{index} = fmul double %x_jvp{index}, %t{index}",
                f"  %term_jvp{index} = fmul double 2.0, %prod_jvp{index}",
                f"  %sum_jvp{index} = fadd double {previous_jvp_sum}, %term_jvp{index}",
            ]
        )
        previous_jvp_sum = f"%sum_jvp{index}"
    lines.extend(
        [
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_jvp_sum}, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
        ]
    )
    for index in range(checked_dimension):
        lines.extend(
            [
                f"  %xptr_vjp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %x_vjp{index} = load double, double* %xptr_vjp{index}",
                f"  %scaled_vjp{index} = fmul double 2.0, %x_vjp{index}",
                f"  %vjp{index} = fmul double %scaled_vjp{index}, %cotangent0",
                f"  %outptr_vjp{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double %vjp{index}, double* %outptr_vjp{index}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _validate_matrix_quadratic_form_dimension(dimension: int | np.integer) -> int:
    checked = int(dimension)
    if checked < 1:
        raise ValueError("native matrix quadratic form dimension must be positive")
    return checked


def _matrix_quadratic_form_value_count(dimension: int) -> int:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    return checked_dimension * checked_dimension + checked_dimension


def _matrix_quadratic_form_matrix_index(dimension: int, row: int, column: int) -> int:
    return row * dimension + column


def _matrix_quadratic_form_vector_index(dimension: int, index: int) -> int:
    return dimension * dimension + index


def _compile_matrix_quadratic_form_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_quadratic_form"',
        '; source = "native_matrix_quadratic_form_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_value_sum = "0.0"
    for row in range(checked_dimension):
        row_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_value{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %xptr_left_value{term} = getelementptr double, double* %values, i64 {row_vector_index}",
                    f"  %xptr_right_value{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %a_value{term} = load double, double* %aptr_value{term}",
                    f"  %x_left_value{term} = load double, double* %xptr_left_value{term}",
                    f"  %x_right_value{term} = load double, double* %xptr_right_value{term}",
                    f"  %left_value{term} = fmul double %a_value{term}, %x_left_value{term}",
                    f"  %term_value{term} = fmul double %left_value{term}, %x_right_value{term}",
                    f"  %sum_value{term} = fadd double {previous_value_sum}, %term_value{term}",
                ]
            )
            previous_value_sum = f"%sum_value{term}"
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_value_sum}, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        row_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %xptr_matrix_left{term} = getelementptr double, double* %values, i64 {row_vector_index}",
                    f"  %xptr_matrix_right{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %x_matrix_left{term} = load double, double* %xptr_matrix_left{term}",
                    f"  %x_matrix_right{term} = load double, double* %xptr_matrix_right{term}",
                    f"  %grad_matrix{term} = fmul double %x_matrix_left{term}, %x_matrix_right{term}",
                    f"  %out_matrix{term} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double %grad_matrix{term}, double* %out_matrix{term}",
                ]
            )
    for row in range(checked_dimension):
        previous_row_sum = "0.0"
        previous_column_sum = "0.0"
        for column in range(checked_dimension):
            row_matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_matrix_index = _matrix_quadratic_form_matrix_index(
                checked_dimension, column, row
            )
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %a_row_ptr_grad{term} = getelementptr double, double* %values, i64 {row_matrix_index}",
                    f"  %a_col_ptr_grad{term} = getelementptr double, double* %values, i64 {column_matrix_index}",
                    f"  %x_ptr_grad{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %a_row_grad{term} = load double, double* %a_row_ptr_grad{term}",
                    f"  %a_col_grad{term} = load double, double* %a_col_ptr_grad{term}",
                    f"  %x_grad{term} = load double, double* %x_ptr_grad{term}",
                    f"  %row_term_grad{term} = fmul double %a_row_grad{term}, %x_grad{term}",
                    f"  %column_term_grad{term} = fmul double %a_col_grad{term}, %x_grad{term}",
                    f"  %row_sum_grad{term} = fadd double {previous_row_sum}, %row_term_grad{term}",
                    f"  %column_sum_grad{term} = fadd double {previous_column_sum}, %column_term_grad{term}",
                ]
            )
            previous_row_sum = f"%row_sum_grad{term}"
            previous_column_sum = f"%column_sum_grad{term}"
        output_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        lines.extend(
            [
                f"  %grad_vector{row} = fadd double {previous_row_sum}, {previous_column_sum}",
                f"  %out_vector{row} = getelementptr double, double* %out, i64 {output_index}",
                f"  store double %grad_vector{row}, double* %out_vector{row}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    lines.extend(
        [
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    previous_jvp_sum = "0.0"
    for row in range(checked_dimension):
        row_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %da_ptr_jvp{term} = getelementptr double, double* %tangent, i64 {matrix_index}",
                    f"  %x_left_ptr_jvp{term} = getelementptr double, double* %values, i64 {row_vector_index}",
                    f"  %x_right_ptr_jvp{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %da_jvp{term} = load double, double* %da_ptr_jvp{term}",
                    f"  %x_left_jvp{term} = load double, double* %x_left_ptr_jvp{term}",
                    f"  %x_right_jvp{term} = load double, double* %x_right_ptr_jvp{term}",
                    f"  %matrix_left_jvp{term} = fmul double %da_jvp{term}, %x_left_jvp{term}",
                    f"  %matrix_term_jvp{term} = fmul double %matrix_left_jvp{term}, %x_right_jvp{term}",
                    f"  %matrix_sum_jvp{term} = fadd double {previous_jvp_sum}, %matrix_term_jvp{term}",
                ]
            )
            previous_jvp_sum = f"%matrix_sum_jvp{term}"
    for row in range(checked_dimension):
        previous_row_sum = "0.0"
        previous_column_sum = "0.0"
        for column in range(checked_dimension):
            row_matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_matrix_index = _matrix_quadratic_form_matrix_index(
                checked_dimension, column, row
            )
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %a_row_ptr_jvp{term} = getelementptr double, double* %values, i64 {row_matrix_index}",
                    f"  %a_col_ptr_jvp{term} = getelementptr double, double* %values, i64 {column_matrix_index}",
                    f"  %x_ptr_jvp{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %a_row_jvp{term} = load double, double* %a_row_ptr_jvp{term}",
                    f"  %a_col_jvp{term} = load double, double* %a_col_ptr_jvp{term}",
                    f"  %x_grad_jvp{term} = load double, double* %x_ptr_jvp{term}",
                    f"  %row_term_jvp{term} = fmul double %a_row_jvp{term}, %x_grad_jvp{term}",
                    f"  %column_term_jvp{term} = fmul double %a_col_jvp{term}, %x_grad_jvp{term}",
                    f"  %row_sum_vector_jvp{term} = fadd double {previous_row_sum}, %row_term_jvp{term}",
                    f"  %column_sum_vector_jvp{term} = fadd double {previous_column_sum}, %column_term_jvp{term}",
                ]
            )
            previous_row_sum = f"%row_sum_vector_jvp{term}"
            previous_column_sum = f"%column_sum_vector_jvp{term}"
        vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        lines.extend(
            [
                f"  %grad_vector_jvp{row} = fadd double {previous_row_sum}, {previous_column_sum}",
                f"  %dx_ptr_jvp{row} = getelementptr double, double* %tangent, i64 {vector_index}",
                f"  %dx_jvp{row} = load double, double* %dx_ptr_jvp{row}",
                f"  %vector_term_jvp{row} = fmul double %grad_vector_jvp{row}, %dx_jvp{row}",
                f"  %vector_sum_jvp{row} = fadd double {previous_jvp_sum}, %vector_term_jvp{row}",
            ]
        )
        previous_jvp_sum = f"%vector_sum_jvp{row}"
    lines.extend(
        [
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_jvp_sum}, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
        ]
    )
    for row in range(checked_dimension):
        row_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %xptr_vjp_left{term} = getelementptr double, double* %values, i64 {row_vector_index}",
                    f"  %xptr_vjp_right{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %x_vjp_left{term} = load double, double* %xptr_vjp_left{term}",
                    f"  %x_vjp_right{term} = load double, double* %xptr_vjp_right{term}",
                    f"  %grad_matrix_vjp{term} = fmul double %x_vjp_left{term}, %x_vjp_right{term}",
                    f"  %vjp_matrix{term} = fmul double %grad_matrix_vjp{term}, %cotangent0",
                    f"  %out_matrix_vjp{term} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double %vjp_matrix{term}, double* %out_matrix_vjp{term}",
                ]
            )
    for row in range(checked_dimension):
        previous_row_sum = "0.0"
        previous_column_sum = "0.0"
        for column in range(checked_dimension):
            row_matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            column_matrix_index = _matrix_quadratic_form_matrix_index(
                checked_dimension, column, row
            )
            column_vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %a_row_ptr_vjp{term} = getelementptr double, double* %values, i64 {row_matrix_index}",
                    f"  %a_col_ptr_vjp{term} = getelementptr double, double* %values, i64 {column_matrix_index}",
                    f"  %x_ptr_vjp{term} = getelementptr double, double* %values, i64 {column_vector_index}",
                    f"  %a_row_vjp{term} = load double, double* %a_row_ptr_vjp{term}",
                    f"  %a_col_vjp{term} = load double, double* %a_col_ptr_vjp{term}",
                    f"  %x_grad_vjp{term} = load double, double* %x_ptr_vjp{term}",
                    f"  %row_term_vjp{term} = fmul double %a_row_vjp{term}, %x_grad_vjp{term}",
                    f"  %column_term_vjp{term} = fmul double %a_col_vjp{term}, %x_grad_vjp{term}",
                    f"  %row_sum_vjp{term} = fadd double {previous_row_sum}, %row_term_vjp{term}",
                    f"  %column_sum_vjp{term} = fadd double {previous_column_sum}, %column_term_vjp{term}",
                ]
            )
            previous_row_sum = f"%row_sum_vjp{term}"
            previous_column_sum = f"%column_sum_vjp{term}"
        output_index = _matrix_quadratic_form_vector_index(checked_dimension, row)
        lines.extend(
            [
                f"  %grad_vector_vjp{row} = fadd double {previous_row_sum}, {previous_column_sum}",
                f"  %vjp_vector{row} = fmul double %grad_vector_vjp{row}, %cotangent0",
                f"  %out_vector_vjp{row} = getelementptr double, double* %out, i64 {output_index}",
                f"  store double %vjp_vector{row}, double* %out_vector_vjp{row}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_matrix_vector_product_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    value_count = _matrix_quadratic_form_value_count(checked_dimension)
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_vector_product"',
        '; source = "native_matrix_vector_product_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f"; value_count = {value_count}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    for row in range(checked_dimension):
        previous_row_sum = "0.0"
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_value{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %xptr_value{term} = getelementptr double, double* %values, i64 {vector_index}",
                    f"  %a_value{term} = load double, double* %aptr_value{term}",
                    f"  %x_value{term} = load double, double* %xptr_value{term}",
                    f"  %prod_value{term} = fmul double %a_value{term}, %x_value{term}",
                    f"  %sum_value{term} = fadd double {previous_row_sum}, %prod_value{term}",
                ]
            )
            previous_row_sum = f"%sum_value{term}"
        lines.extend(
            [
                f"  %out_value{row} = getelementptr double, double* %out, i64 {row}",
                f"  store double {previous_row_sum}, double* %out_value{row}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %xptr_gradient{term} = getelementptr double, double* %values, i64 {vector_index}",
                    f"  %x_gradient{term} = load double, double* %xptr_gradient{term}",
                    f"  %out_matrix_gradient{term} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double %x_gradient{term}, double* %out_matrix_gradient{term}",
                ]
            )
    for column in range(checked_dimension):
        previous_column_sum = "0.0"
        for row in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_gradient{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %a_gradient{term} = load double, double* %aptr_gradient{term}",
                    f"  %sum_gradient{term} = fadd double {previous_column_sum}, %a_gradient{term}",
                ]
            )
            previous_column_sum = f"%sum_gradient{term}"
        output_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
        lines.extend(
            [
                f"  %out_vector_gradient{column} = getelementptr double, double* %out, i64 {output_index}",
                f"  store double {previous_column_sum}, double* %out_vector_gradient{column}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        previous_jvp_sum = "0.0"
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_jvp{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %xptr_jvp{term} = getelementptr double, double* %values, i64 {vector_index}",
                    f"  %taptr_jvp{term} = getelementptr double, double* %tangent, i64 {matrix_index}",
                    f"  %txptr_jvp{term} = getelementptr double, double* %tangent, i64 {vector_index}",
                    f"  %a_jvp{term} = load double, double* %aptr_jvp{term}",
                    f"  %x_jvp{term} = load double, double* %xptr_jvp{term}",
                    f"  %ta_jvp{term} = load double, double* %taptr_jvp{term}",
                    f"  %tx_jvp{term} = load double, double* %txptr_jvp{term}",
                    f"  %left_jvp{term} = fmul double %ta_jvp{term}, %x_jvp{term}",
                    f"  %right_jvp{term} = fmul double %a_jvp{term}, %tx_jvp{term}",
                    f"  %term_jvp{term} = fadd double %left_jvp{term}, %right_jvp{term}",
                    f"  %sum_jvp{term} = fadd double {previous_jvp_sum}, %term_jvp{term}",
                ]
            )
            previous_jvp_sum = f"%sum_jvp{term}"
        lines.extend(
            [
                f"  %out_jvp{row} = getelementptr double, double* %out, i64 {row}",
                f"  store double {previous_jvp_sum}, double* %out_jvp{row}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        cotangent_ptr = f"%cotangent_ptr{row}"
        cotangent_value = f"%cotangent{row}"
        lines.extend(
            [
                f"  {cotangent_ptr} = getelementptr double, double* %cotangent, i64 {row}",
                f"  {cotangent_value} = load double, double* {cotangent_ptr}",
            ]
        )
        for column in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            vector_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %xptr_vjp{term} = getelementptr double, double* %values, i64 {vector_index}",
                    f"  %x_vjp{term} = load double, double* %xptr_vjp{term}",
                    f"  %matrix_grad_vjp{term} = fmul double {cotangent_value}, %x_vjp{term}",
                    f"  %out_matrix_vjp{term} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double %matrix_grad_vjp{term}, double* %out_matrix_vjp{term}",
                ]
            )
    for column in range(checked_dimension):
        previous_vector_sum = "0.0"
        for row in range(checked_dimension):
            matrix_index = _matrix_quadratic_form_matrix_index(checked_dimension, row, column)
            term = f"{row}_{column}"
            lines.extend(
                [
                    f"  %aptr_vjp{term} = getelementptr double, double* %values, i64 {matrix_index}",
                    f"  %cotangent_ptr_vjp{term} = getelementptr double, double* %cotangent, i64 {row}",
                    f"  %a_vjp{term} = load double, double* %aptr_vjp{term}",
                    f"  %cotangent_vjp{term} = load double, double* %cotangent_ptr_vjp{term}",
                    f"  %vector_term_vjp{term} = fmul double %a_vjp{term}, %cotangent_vjp{term}",
                    f"  %vector_sum_vjp{term} = fadd double {previous_vector_sum}, %vector_term_vjp{term}",
                ]
            )
            previous_vector_sum = f"%vector_sum_vjp{term}"
        output_index = _matrix_quadratic_form_vector_index(checked_dimension, column)
        lines.extend(
            [
                f"  %out_vector_vjp{column} = getelementptr double, double* %out, i64 {output_index}",
                f"  store double {previous_vector_sum}, double* %out_vector_vjp{column}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_matrix_matrix_product_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    value_count = 2 * matrix_size
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_matrix_product"',
        '; source = "native_matrix_matrix_product_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f"; value_count = {value_count}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            previous_sum = "0.0"
            for inner in range(checked_dimension):
                left_index = row * checked_dimension + inner
                right_index = matrix_size + inner * checked_dimension + column
                term = f"{row}_{column}_{inner}"
                lines.extend(
                    [
                        f"  %leftptr_value{term} = getelementptr double, double* %values, i64 {left_index}",
                        f"  %rightptr_value{term} = getelementptr double, double* %values, i64 {right_index}",
                        f"  %left_value{term} = load double, double* %leftptr_value{term}",
                        f"  %right_value{term} = load double, double* %rightptr_value{term}",
                        f"  %prod_value{term} = fmul double %left_value{term}, %right_value{term}",
                        f"  %sum_value{term} = fadd double {previous_sum}, %prod_value{term}",
                    ]
                )
                previous_sum = f"%sum_value{term}"
            output_index = row * checked_dimension + column
            lines.extend(
                [
                    f"  %out_value{row}_{column} = getelementptr double, double* %out, i64 {output_index}",
                    f"  store double {previous_sum}, double* %out_value{row}_{column}",
                ]
            )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for inner in range(checked_dimension):
            previous_left_sum = "0.0"
            for column in range(checked_dimension):
                left_right_index = matrix_size + inner * checked_dimension + column
                term = f"{row}_{inner}_{column}"
                lines.extend(
                    [
                        f"  %rightptr_gradient{term} = getelementptr double, double* %values, i64 {left_right_index}",
                        f"  %right_gradient{term} = load double, double* %rightptr_gradient{term}",
                        f"  %left_sum_gradient{term} = fadd double {previous_left_sum}, %right_gradient{term}",
                    ]
                )
                previous_left_sum = f"%left_sum_gradient{term}"
            left_output_index = row * checked_dimension + inner
            lines.extend(
                [
                    f"  %out_left_gradient{row}_{inner} = getelementptr double, double* %out, i64 {left_output_index}",
                    f"  store double {previous_left_sum}, double* %out_left_gradient{row}_{inner}",
                ]
            )
    for inner in range(checked_dimension):
        for column in range(checked_dimension):
            previous_right_sum = "0.0"
            for row in range(checked_dimension):
                right_left_index = row * checked_dimension + inner
                term = f"{inner}_{column}_{row}"
                lines.extend(
                    [
                        f"  %leftptr_gradient{term} = getelementptr double, double* %values, i64 {right_left_index}",
                        f"  %left_gradient{term} = load double, double* %leftptr_gradient{term}",
                        f"  %right_sum_gradient{term} = fadd double {previous_right_sum}, %left_gradient{term}",
                    ]
                )
                previous_right_sum = f"%right_sum_gradient{term}"
            right_output_index = matrix_size + inner * checked_dimension + column
            lines.extend(
                [
                    f"  %out_right_gradient{inner}_{column} = getelementptr double, double* %out, i64 {right_output_index}",
                    f"  store double {previous_right_sum}, double* %out_right_gradient{inner}_{column}",
                ]
            )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            previous_jvp_sum = "0.0"
            for inner in range(checked_dimension):
                left_index = row * checked_dimension + inner
                right_index = matrix_size + inner * checked_dimension + column
                term = f"{row}_{column}_{inner}"
                lines.extend(
                    [
                        f"  %leftptr_jvp{term} = getelementptr double, double* %values, i64 {left_index}",
                        f"  %rightptr_jvp{term} = getelementptr double, double* %values, i64 {right_index}",
                        f"  %tleftptr_jvp{term} = getelementptr double, double* %tangent, i64 {left_index}",
                        f"  %trightptr_jvp{term} = getelementptr double, double* %tangent, i64 {right_index}",
                        f"  %left_jvp{term} = load double, double* %leftptr_jvp{term}",
                        f"  %right_jvp{term} = load double, double* %rightptr_jvp{term}",
                        f"  %tleft_jvp{term} = load double, double* %tleftptr_jvp{term}",
                        f"  %tright_jvp{term} = load double, double* %trightptr_jvp{term}",
                        f"  %left_term_jvp{term} = fmul double %tleft_jvp{term}, %right_jvp{term}",
                        f"  %right_term_jvp{term} = fmul double %left_jvp{term}, %tright_jvp{term}",
                        f"  %term_jvp{term} = fadd double %left_term_jvp{term}, %right_term_jvp{term}",
                        f"  %sum_jvp{term} = fadd double {previous_jvp_sum}, %term_jvp{term}",
                    ]
                )
                previous_jvp_sum = f"%sum_jvp{term}"
            output_index = row * checked_dimension + column
            lines.extend(
                [
                    f"  %out_jvp{row}_{column} = getelementptr double, double* %out, i64 {output_index}",
                    f"  store double {previous_jvp_sum}, double* %out_jvp{row}_{column}",
                ]
            )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for inner in range(checked_dimension):
            previous_left_sum = "0.0"
            for column in range(checked_dimension):
                cotangent_index = row * checked_dimension + column
                right_index = matrix_size + inner * checked_dimension + column
                term = f"{row}_{inner}_{column}"
                lines.extend(
                    [
                        f"  %cotangent_left_ptr_vjp{term} = getelementptr double, double* %cotangent, i64 {cotangent_index}",
                        f"  %rightptr_vjp{term} = getelementptr double, double* %values, i64 {right_index}",
                        f"  %cotangent_left_vjp{term} = load double, double* %cotangent_left_ptr_vjp{term}",
                        f"  %right_vjp{term} = load double, double* %rightptr_vjp{term}",
                        f"  %left_term_vjp{term} = fmul double %cotangent_left_vjp{term}, %right_vjp{term}",
                        f"  %left_sum_vjp{term} = fadd double {previous_left_sum}, %left_term_vjp{term}",
                    ]
                )
                previous_left_sum = f"%left_sum_vjp{term}"
            left_output_index = row * checked_dimension + inner
            lines.extend(
                [
                    f"  %out_left_vjp{row}_{inner} = getelementptr double, double* %out, i64 {left_output_index}",
                    f"  store double {previous_left_sum}, double* %out_left_vjp{row}_{inner}",
                ]
            )
    for inner in range(checked_dimension):
        for column in range(checked_dimension):
            previous_right_sum = "0.0"
            for row in range(checked_dimension):
                left_index = row * checked_dimension + inner
                cotangent_index = row * checked_dimension + column
                term = f"{inner}_{column}_{row}"
                lines.extend(
                    [
                        f"  %leftptr_vjp{term} = getelementptr double, double* %values, i64 {left_index}",
                        f"  %cotangent_right_ptr_vjp{term} = getelementptr double, double* %cotangent, i64 {cotangent_index}",
                        f"  %left_vjp{term} = load double, double* %leftptr_vjp{term}",
                        f"  %cotangent_right_vjp{term} = load double, double* %cotangent_right_ptr_vjp{term}",
                        f"  %right_term_vjp{term} = fmul double %left_vjp{term}, %cotangent_right_vjp{term}",
                        f"  %right_sum_vjp{term} = fadd double {previous_right_sum}, %right_term_vjp{term}",
                    ]
                )
                previous_right_sum = f"%right_sum_vjp{term}"
            right_output_index = matrix_size + inner * checked_dimension + column
            lines.extend(
                [
                    f"  %out_right_vjp{inner}_{column} = getelementptr double, double* %out, i64 {right_output_index}",
                    f"  store double {previous_right_sum}, double* %out_right_vjp{inner}_{column}",
                ]
            )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_matrix_trace_native_llvm_ir(rule_name: str, dimension: int) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_trace"',
        '; source = "native_matrix_trace_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f"; value_count = {matrix_size}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_sum = "0.0"
    for index in range(checked_dimension):
        diagonal_index = index * checked_dimension + index
        lines.extend(
            [
                f"  %diagptr_value{index} = getelementptr double, double* %values, i64 {diagonal_index}",
                f"  %diag_value{index} = load double, double* %diagptr_value{index}",
                f"  %sum_value{index} = fadd double {previous_sum}, %diag_value{index}",
            ]
        )
        previous_sum = f"%sum_value{index}"
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_sum}, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            matrix_index = row * checked_dimension + column
            value = "1.0" if row == column else "0.0"
            lines.extend(
                [
                    f"  %out_gradient{row}_{column} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double {value}, double* %out_gradient{row}_{column}",
                ]
            )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    previous_jvp_sum = "0.0"
    for index in range(checked_dimension):
        diagonal_index = index * checked_dimension + index
        lines.extend(
            [
                f"  %diagptr_jvp{index} = getelementptr double, double* %tangent, i64 {diagonal_index}",
                f"  %diag_jvp{index} = load double, double* %diagptr_jvp{index}",
                f"  %sum_jvp{index} = fadd double {previous_jvp_sum}, %diag_jvp{index}",
            ]
        )
        previous_jvp_sum = f"%sum_jvp{index}"
    lines.extend(
        [
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_jvp_sum}, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
        ]
    )
    for row in range(checked_dimension):
        for column in range(checked_dimension):
            matrix_index = row * checked_dimension + column
            value = "%cotangent0" if row == column else "0.0"
            lines.extend(
                [
                    f"  %out_vjp{row}_{column} = getelementptr double, double* %out, i64 {matrix_index}",
                    f"  store double {value}, double* %out_vjp{row}_{column}",
                ]
            )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_matrix_frobenius_norm_squared_native_llvm_ir(
    rule_name: str,
    dimension: int,
) -> str:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    lines = [
        f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
        '; primitive = "matrix_frobenius_norm_squared"',
        '; source = "native_matrix_frobenius_norm_squared_ad_codegen"',
        '; execution = "native_llvm_mcjit"',
        f"; dimension = {checked_dimension}",
        f"; value_count = {matrix_size}",
        f'target triple = "{_escape_mlir_string(triple)}"',
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        "entry:",
    ]
    previous_value_sum = "0.0"
    for index in range(matrix_size):
        lines.extend(
            [
                f"  %valueptr_value{index} = getelementptr double, double* %values, i64 {index}",
                f"  %value_value{index} = load double, double* %valueptr_value{index}",
                f"  %square_value{index} = fmul double %value_value{index}, %value_value{index}",
                f"  %sum_value{index} = fadd double {previous_value_sum}, %square_value{index}",
            ]
        )
        previous_value_sum = f"%sum_value{index}"
    lines.extend(
        [
            "  %out0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_value_sum}, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
        ]
    )
    for index in range(matrix_size):
        lines.extend(
            [
                f"  %valueptr_gradient{index} = getelementptr double, double* %values, i64 {index}",
                f"  %value_gradient{index} = load double, double* %valueptr_gradient{index}",
                f"  %gradient{index} = fmul double 2.0, %value_gradient{index}",
                f"  %out_gradient{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double %gradient{index}, double* %out_gradient{index}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
        ]
    )
    previous_jvp_sum = "0.0"
    for index in range(matrix_size):
        lines.extend(
            [
                f"  %valueptr_jvp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %tangentptr_jvp{index} = getelementptr double, double* %tangent, i64 {index}",
                f"  %value_jvp{index} = load double, double* %valueptr_jvp{index}",
                f"  %tangent_jvp{index} = load double, double* %tangentptr_jvp{index}",
                f"  %product_jvp{index} = fmul double %value_jvp{index}, %tangent_jvp{index}",
                f"  %scaled_jvp{index} = fmul double 2.0, %product_jvp{index}",
                f"  %sum_jvp{index} = fadd double {previous_jvp_sum}, %scaled_jvp{index}",
            ]
        )
        previous_jvp_sum = f"%sum_jvp{index}"
    lines.extend(
        [
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            f"  store double {previous_jvp_sum}, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent0 = load double, double* %cotangent0ptr",
        ]
    )
    for index in range(matrix_size):
        lines.extend(
            [
                f"  %valueptr_vjp{index} = getelementptr double, double* %values, i64 {index}",
                f"  %value_vjp{index} = load double, double* %valueptr_vjp{index}",
                f"  %gradient_vjp{index} = fmul double 2.0, %value_vjp{index}",
                f"  %scaled_vjp{index} = fmul double %cotangent0, %gradient_vjp{index}",
                f"  %out_vjp{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double %scaled_vjp{index}, double* %out_vjp{index}",
            ]
        )
    lines.extend(["  ret void", "}", ""])
    return "\n".join(lines)


def _compile_matrix_2x2_determinant_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_determinant"',
            '; source = "native_matrix_2x2_determinant_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 4",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a10ptr = getelementptr double, double* %values, i64 2",
            "  %a11ptr = getelementptr double, double* %values, i64 3",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a10 = load double, double* %a10ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %main_diag = fmul double %a00, %a11",
            "  %off_diag = fmul double %a01, %a10",
            "  %det = fsub double %main_diag, %off_diag",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  store double %det, double* %out0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr_gradient = getelementptr double, double* %values, i64 0",
            "  %a01ptr_gradient = getelementptr double, double* %values, i64 1",
            "  %a10ptr_gradient = getelementptr double, double* %values, i64 2",
            "  %a11ptr_gradient = getelementptr double, double* %values, i64 3",
            "  %a00_gradient = load double, double* %a00ptr_gradient",
            "  %a01_gradient = load double, double* %a01ptr_gradient",
            "  %a10_gradient = load double, double* %a10ptr_gradient",
            "  %a11_gradient = load double, double* %a11ptr_gradient",
            "  %neg_a10_gradient = fsub double 0.0, %a10_gradient",
            "  %neg_a01_gradient = fsub double 0.0, %a01_gradient",
            "  %out_gradient0 = getelementptr double, double* %out, i64 0",
            "  %out_gradient1 = getelementptr double, double* %out, i64 1",
            "  %out_gradient2 = getelementptr double, double* %out, i64 2",
            "  %out_gradient3 = getelementptr double, double* %out, i64 3",
            "  store double %a11_gradient, double* %out_gradient0",
            "  store double %neg_a10_gradient, double* %out_gradient1",
            "  store double %neg_a01_gradient, double* %out_gradient2",
            "  store double %a00_gradient, double* %out_gradient3",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %a00ptr_jvp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_jvp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 3",
            "  %t00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %t01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %t10ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %t11ptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a10_jvp = load double, double* %a10ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %t00_jvp = load double, double* %t00ptr_jvp",
            "  %t01_jvp = load double, double* %t01ptr_jvp",
            "  %t10_jvp = load double, double* %t10ptr_jvp",
            "  %t11_jvp = load double, double* %t11ptr_jvp",
            "  %term0_jvp = fmul double %t00_jvp, %a11_jvp",
            "  %term1_jvp = fmul double %a00_jvp, %t11_jvp",
            "  %term2_jvp = fmul double %t01_jvp, %a10_jvp",
            "  %term3_jvp = fmul double %a01_jvp, %t10_jvp",
            "  %sum0_jvp = fadd double %term0_jvp, %term1_jvp",
            "  %sum1_jvp = fsub double %sum0_jvp, %term2_jvp",
            "  %sum2_jvp = fsub double %sum1_jvp, %term3_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  store double %sum2_jvp, double* %out_jvp0",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 3",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a10_vjp = load double, double* %a10ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %cotangent0 = load double, double* %cotangent0ptr",
            "  %neg_a10_vjp = fsub double 0.0, %a10_vjp",
            "  %neg_a01_vjp = fsub double 0.0, %a01_vjp",
            "  %scaled_vjp0 = fmul double %cotangent0, %a11_vjp",
            "  %scaled_vjp1 = fmul double %cotangent0, %neg_a10_vjp",
            "  %scaled_vjp2 = fmul double %cotangent0, %neg_a01_vjp",
            "  %scaled_vjp3 = fmul double %cotangent0, %a00_vjp",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  store double %scaled_vjp0, double* %out_vjp0",
            "  store double %scaled_vjp1, double* %out_vjp1",
            "  store double %scaled_vjp2, double* %out_vjp2",
            "  store double %scaled_vjp3, double* %out_vjp3",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_matrix_2x2_inverse_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_inverse"',
            '; source = "native_matrix_2x2_inverse_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 4",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a10ptr = getelementptr double, double* %values, i64 2",
            "  %a11ptr = getelementptr double, double* %values, i64 3",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a10 = load double, double* %a10ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %main_diag = fmul double %a00, %a11",
            "  %off_diag = fmul double %a01, %a10",
            "  %det = fsub double %main_diag, %off_diag",
            "  %neg_a01 = fsub double 0.0, %a01",
            "  %neg_a10 = fsub double 0.0, %a10",
            "  %inv00 = fdiv double %a11, %det",
            "  %inv01 = fdiv double %neg_a01, %det",
            "  %inv10 = fdiv double %neg_a10, %det",
            "  %inv11 = fdiv double %a00, %det",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  %out2 = getelementptr double, double* %out, i64 2",
            "  %out3 = getelementptr double, double* %out, i64 3",
            "  store double %inv00, double* %out0",
            "  store double %inv01, double* %out1",
            "  store double %inv10, double* %out2",
            "  store double %inv11, double* %out3",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [4 x double]",
            "  %cotangent0 = getelementptr [4 x double], [4 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [4 x double], [4 x double]* %cotangent, i64 0, i64 1",
            "  %cotangent2 = getelementptr [4 x double], [4 x double]* %cotangent, i64 0, i64 2",
            "  %cotangent3 = getelementptr [4 x double], [4 x double]* %cotangent, i64 0, i64 3",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            "  store double 1.0, double* %cotangent2",
            "  store double 1.0, double* %cotangent3",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %a00ptr_jvp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_jvp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 3",
            "  %t00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %t01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %t10ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %t11ptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a10_jvp = load double, double* %a10ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %t00_jvp = load double, double* %t00ptr_jvp",
            "  %t01_jvp = load double, double* %t01ptr_jvp",
            "  %t10_jvp = load double, double* %t10ptr_jvp",
            "  %t11_jvp = load double, double* %t11ptr_jvp",
            "  %main_diag_jvp = fmul double %a00_jvp, %a11_jvp",
            "  %off_diag_jvp = fmul double %a01_jvp, %a10_jvp",
            "  %det_jvp = fsub double %main_diag_jvp, %off_diag_jvp",
            "  %det2_jvp = fmul double %det_jvp, %det_jvp",
            "  %term_detdot0 = fmul double %t00_jvp, %a11_jvp",
            "  %term_detdot1 = fmul double %a00_jvp, %t11_jvp",
            "  %term_detdot2 = fmul double %t01_jvp, %a10_jvp",
            "  %term_detdot3 = fmul double %a01_jvp, %t10_jvp",
            "  %detdot_sum0 = fadd double %term_detdot0, %term_detdot1",
            "  %detdot_sum1 = fsub double %detdot_sum0, %term_detdot2",
            "  %detdot = fsub double %detdot_sum1, %term_detdot3",
            "  %num00_left = fmul double %t11_jvp, %det_jvp",
            "  %num00_right = fmul double %a11_jvp, %detdot",
            "  %num00 = fsub double %num00_left, %num00_right",
            "  %neg_t01_jvp = fsub double 0.0, %t01_jvp",
            "  %num01_left = fmul double %neg_t01_jvp, %det_jvp",
            "  %num01_right = fmul double %a01_jvp, %detdot",
            "  %num01 = fadd double %num01_left, %num01_right",
            "  %neg_t10_jvp = fsub double 0.0, %t10_jvp",
            "  %num10_left = fmul double %neg_t10_jvp, %det_jvp",
            "  %num10_right = fmul double %a10_jvp, %detdot",
            "  %num10 = fadd double %num10_left, %num10_right",
            "  %num11_left = fmul double %t00_jvp, %det_jvp",
            "  %num11_right = fmul double %a00_jvp, %detdot",
            "  %num11 = fsub double %num11_left, %num11_right",
            "  %jvp00 = fdiv double %num00, %det2_jvp",
            "  %jvp01 = fdiv double %num01, %det2_jvp",
            "  %jvp10 = fdiv double %num10, %det2_jvp",
            "  %jvp11 = fdiv double %num11, %det2_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  %out_jvp2 = getelementptr double, double* %out, i64 2",
            "  %out_jvp3 = getelementptr double, double* %out, i64 3",
            "  store double %jvp00, double* %out_jvp0",
            "  store double %jvp01, double* %out_jvp1",
            "  store double %jvp10, double* %out_jvp2",
            "  store double %jvp11, double* %out_jvp3",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 3",
            "  %g00ptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %g01ptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %g10ptr_vjp = getelementptr double, double* %cotangent, i64 2",
            "  %g11ptr_vjp = getelementptr double, double* %cotangent, i64 3",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a10_vjp = load double, double* %a10ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %g00_vjp = load double, double* %g00ptr_vjp",
            "  %g01_vjp = load double, double* %g01ptr_vjp",
            "  %g10_vjp = load double, double* %g10ptr_vjp",
            "  %g11_vjp = load double, double* %g11ptr_vjp",
            "  %main_diag_vjp = fmul double %a00_vjp, %a11_vjp",
            "  %off_diag_vjp = fmul double %a01_vjp, %a10_vjp",
            "  %det_vjp = fsub double %main_diag_vjp, %off_diag_vjp",
            "  %neg_a01_vjp = fsub double 0.0, %a01_vjp",
            "  %neg_a10_vjp = fsub double 0.0, %a10_vjp",
            "  %y00 = fdiv double %a11_vjp, %det_vjp",
            "  %y01 = fdiv double %neg_a01_vjp, %det_vjp",
            "  %y10 = fdiv double %neg_a10_vjp, %det_vjp",
            "  %y11 = fdiv double %a00_vjp, %det_vjp",
            "  %m00_left = fmul double %y00, %g00_vjp",
            "  %m00_right = fmul double %y10, %g10_vjp",
            "  %m00 = fadd double %m00_left, %m00_right",
            "  %m01_left = fmul double %y00, %g01_vjp",
            "  %m01_right = fmul double %y10, %g11_vjp",
            "  %m01 = fadd double %m01_left, %m01_right",
            "  %m10_left = fmul double %y01, %g00_vjp",
            "  %m10_right = fmul double %y11, %g10_vjp",
            "  %m10 = fadd double %m10_left, %m10_right",
            "  %m11_left = fmul double %y01, %g01_vjp",
            "  %m11_right = fmul double %y11, %g11_vjp",
            "  %m11 = fadd double %m11_left, %m11_right",
            "  %h00_left = fmul double %m00, %y00",
            "  %h00_right = fmul double %m01, %y01",
            "  %h00_sum = fadd double %h00_left, %h00_right",
            "  %h00 = fsub double 0.0, %h00_sum",
            "  %h01_left = fmul double %m00, %y10",
            "  %h01_right = fmul double %m01, %y11",
            "  %h01_sum = fadd double %h01_left, %h01_right",
            "  %h01 = fsub double 0.0, %h01_sum",
            "  %h10_left = fmul double %m10, %y00",
            "  %h10_right = fmul double %m11, %y01",
            "  %h10_sum = fadd double %h10_left, %h10_right",
            "  %h10 = fsub double 0.0, %h10_sum",
            "  %h11_left = fmul double %m10, %y10",
            "  %h11_right = fmul double %m11, %y11",
            "  %h11_sum = fadd double %h11_left, %h11_right",
            "  %h11 = fsub double 0.0, %h11_sum",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  store double %h00, double* %out_vjp0",
            "  store double %h01, double* %out_vjp1",
            "  store double %h10, double* %out_vjp2",
            "  store double %h11, double* %out_vjp3",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_matrix_2x2_solve_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_solve"',
            '; source = "native_matrix_2x2_solve_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 6",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a10ptr = getelementptr double, double* %values, i64 2",
            "  %a11ptr = getelementptr double, double* %values, i64 3",
            "  %b0ptr = getelementptr double, double* %values, i64 4",
            "  %b1ptr = getelementptr double, double* %values, i64 5",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a10 = load double, double* %a10ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %b0 = load double, double* %b0ptr",
            "  %b1 = load double, double* %b1ptr",
            "  %main_diag = fmul double %a00, %a11",
            "  %off_diag = fmul double %a01, %a10",
            "  %det = fsub double %main_diag, %off_diag",
            "  %num0_left = fmul double %a11, %b0",
            "  %num0_right = fmul double %a01, %b1",
            "  %num0 = fsub double %num0_left, %num0_right",
            "  %num1_left = fmul double %a00, %b1",
            "  %num1_right = fmul double %a10, %b0",
            "  %num1 = fsub double %num1_left, %num1_right",
            "  %x0 = fdiv double %num0, %det",
            "  %x1 = fdiv double %num1, %det",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  store double %x0, double* %out0",
            "  store double %x1, double* %out1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [2 x double]",
            "  %cotangent0 = getelementptr [2 x double], [2 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [2 x double], [2 x double]* %cotangent, i64 0, i64 1",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %a00ptr_jvp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_jvp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 3",
            "  %b0ptr_jvp = getelementptr double, double* %values, i64 4",
            "  %b1ptr_jvp = getelementptr double, double* %values, i64 5",
            "  %da00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %da01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %da10ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %da11ptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %db0ptr_jvp = getelementptr double, double* %tangent, i64 4",
            "  %db1ptr_jvp = getelementptr double, double* %tangent, i64 5",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a10_jvp = load double, double* %a10ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %b0_jvp = load double, double* %b0ptr_jvp",
            "  %b1_jvp = load double, double* %b1ptr_jvp",
            "  %da00_jvp = load double, double* %da00ptr_jvp",
            "  %da01_jvp = load double, double* %da01ptr_jvp",
            "  %da10_jvp = load double, double* %da10ptr_jvp",
            "  %da11_jvp = load double, double* %da11ptr_jvp",
            "  %db0_jvp = load double, double* %db0ptr_jvp",
            "  %db1_jvp = load double, double* %db1ptr_jvp",
            "  %main_diag_jvp = fmul double %a00_jvp, %a11_jvp",
            "  %off_diag_jvp = fmul double %a01_jvp, %a10_jvp",
            "  %det_jvp = fsub double %main_diag_jvp, %off_diag_jvp",
            "  %num0_left_jvp = fmul double %a11_jvp, %b0_jvp",
            "  %num0_right_jvp = fmul double %a01_jvp, %b1_jvp",
            "  %num0_jvp = fsub double %num0_left_jvp, %num0_right_jvp",
            "  %num1_left_jvp = fmul double %a00_jvp, %b1_jvp",
            "  %num1_right_jvp = fmul double %a10_jvp, %b0_jvp",
            "  %num1_jvp = fsub double %num1_left_jvp, %num1_right_jvp",
            "  %x0_jvp = fdiv double %num0_jvp, %det_jvp",
            "  %x1_jvp = fdiv double %num1_jvp, %det_jvp",
            "  %dax0_left = fmul double %da00_jvp, %x0_jvp",
            "  %dax0_right = fmul double %da01_jvp, %x1_jvp",
            "  %dax0 = fadd double %dax0_left, %dax0_right",
            "  %dax1_left = fmul double %da10_jvp, %x0_jvp",
            "  %dax1_right = fmul double %da11_jvp, %x1_jvp",
            "  %dax1 = fadd double %dax1_left, %dax1_right",
            "  %r0 = fsub double %db0_jvp, %dax0",
            "  %r1 = fsub double %db1_jvp, %dax1",
            "  %dx0_left = fmul double %a11_jvp, %r0",
            "  %dx0_right = fmul double %a01_jvp, %r1",
            "  %dx0_num = fsub double %dx0_left, %dx0_right",
            "  %dx1_left = fmul double %a00_jvp, %r1",
            "  %dx1_right = fmul double %a10_jvp, %r0",
            "  %dx1_num = fsub double %dx1_left, %dx1_right",
            "  %dx0 = fdiv double %dx0_num, %det_jvp",
            "  %dx1 = fdiv double %dx1_num, %det_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  store double %dx0, double* %out_jvp0",
            "  store double %dx1, double* %out_jvp1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a10ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 3",
            "  %b0ptr_vjp = getelementptr double, double* %values, i64 4",
            "  %b1ptr_vjp = getelementptr double, double* %values, i64 5",
            "  %c0ptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %c1ptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a10_vjp = load double, double* %a10ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %b0_vjp = load double, double* %b0ptr_vjp",
            "  %b1_vjp = load double, double* %b1ptr_vjp",
            "  %c0_vjp = load double, double* %c0ptr_vjp",
            "  %c1_vjp = load double, double* %c1ptr_vjp",
            "  %main_diag_vjp = fmul double %a00_vjp, %a11_vjp",
            "  %off_diag_vjp = fmul double %a01_vjp, %a10_vjp",
            "  %det_vjp = fsub double %main_diag_vjp, %off_diag_vjp",
            "  %num0_left_vjp = fmul double %a11_vjp, %b0_vjp",
            "  %num0_right_vjp = fmul double %a01_vjp, %b1_vjp",
            "  %num0_vjp = fsub double %num0_left_vjp, %num0_right_vjp",
            "  %num1_left_vjp = fmul double %a00_vjp, %b1_vjp",
            "  %num1_right_vjp = fmul double %a10_vjp, %b0_vjp",
            "  %num1_vjp = fsub double %num1_left_vjp, %num1_right_vjp",
            "  %x0_vjp = fdiv double %num0_vjp, %det_vjp",
            "  %x1_vjp = fdiv double %num1_vjp, %det_vjp",
            "  %p0_left = fmul double %a11_vjp, %c0_vjp",
            "  %p0_right = fmul double %a10_vjp, %c1_vjp",
            "  %p0_num = fsub double %p0_left, %p0_right",
            "  %p1_left = fmul double %a00_vjp, %c1_vjp",
            "  %p1_right = fmul double %a01_vjp, %c0_vjp",
            "  %p1_num = fsub double %p1_left, %p1_right",
            "  %p0 = fdiv double %p0_num, %det_vjp",
            "  %p1 = fdiv double %p1_num, %det_vjp",
            "  %adj_a00_raw = fmul double %p0, %x0_vjp",
            "  %adj_a01_raw = fmul double %p0, %x1_vjp",
            "  %adj_a10_raw = fmul double %p1, %x0_vjp",
            "  %adj_a11_raw = fmul double %p1, %x1_vjp",
            "  %adj_a00 = fsub double 0.0, %adj_a00_raw",
            "  %adj_a01 = fsub double 0.0, %adj_a01_raw",
            "  %adj_a10 = fsub double 0.0, %adj_a10_raw",
            "  %adj_a11 = fsub double 0.0, %adj_a11_raw",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  %out_vjp4 = getelementptr double, double* %out, i64 4",
            "  %out_vjp5 = getelementptr double, double* %out, i64 5",
            "  store double %adj_a00, double* %out_vjp0",
            "  store double %adj_a01, double* %out_vjp1",
            "  store double %adj_a10, double* %out_vjp2",
            "  store double %adj_a11, double* %out_vjp3",
            "  store double %p0, double* %out_vjp4",
            "  store double %p1, double* %out_vjp5",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_symmetric_2x2_cholesky_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "symmetric_2x2_cholesky"',
            '; source = "native_symmetric_2x2_cholesky_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 3",
            "; input_layout = upper_triangle",
            "; output_layout = lower_triangle",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            "declare double @llvm.sqrt.f64(double)",
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a11ptr = getelementptr double, double* %values, i64 2",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %l00 = call double @llvm.sqrt.f64(double %a00)",
            "  %l10 = fdiv double %a01, %l00",
            "  %l10_sq = fmul double %l10, %l10",
            "  %schur = fsub double %a11, %l10_sq",
            "  %l11 = call double @llvm.sqrt.f64(double %schur)",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  %out2 = getelementptr double, double* %out, i64 2",
            "  store double %l00, double* %out0",
            "  store double %l10, double* %out1",
            "  store double %l11, double* %out2",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [3 x double]",
            "  %cotangent0 = getelementptr [3 x double], [3 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [3 x double], [3 x double]* %cotangent, i64 0, i64 1",
            "  %cotangent2 = getelementptr [3 x double], [3 x double]* %cotangent, i64 0, i64 2",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            "  store double 1.0, double* %cotangent2",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %a00ptr_jvp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_jvp = getelementptr double, double* %values, i64 1",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %t00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %t01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %t11ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %t00_jvp = load double, double* %t00ptr_jvp",
            "  %t01_jvp = load double, double* %t01ptr_jvp",
            "  %t11_jvp = load double, double* %t11ptr_jvp",
            "  %l00_jvp = call double @llvm.sqrt.f64(double %a00_jvp)",
            "  %l10_jvp = fdiv double %a01_jvp, %l00_jvp",
            "  %l10_sq_jvp = fmul double %l10_jvp, %l10_jvp",
            "  %schur_jvp = fsub double %a11_jvp, %l10_sq_jvp",
            "  %l11_jvp = call double @llvm.sqrt.f64(double %schur_jvp)",
            "  %two_l00_jvp = fmul double 2.0, %l00_jvp",
            "  %tangent_l00 = fdiv double %t00_jvp, %two_l00_jvp",
            "  %t01_over_l00 = fdiv double %t01_jvp, %l00_jvp",
            "  %l10_tangent_l00 = fmul double %l10_jvp, %tangent_l00",
            "  %l10_tangent_l00_over_l00 = fdiv double %l10_tangent_l00, %l00_jvp",
            "  %tangent_l10 = fsub double %t01_over_l00, %l10_tangent_l00_over_l00",
            "  %two_l10_jvp = fmul double 2.0, %l10_jvp",
            "  %schur_tangent_term = fmul double %two_l10_jvp, %tangent_l10",
            "  %schur_tangent = fsub double %t11_jvp, %schur_tangent_term",
            "  %two_l11_jvp = fmul double 2.0, %l11_jvp",
            "  %tangent_l11 = fdiv double %schur_tangent, %two_l11_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  %out_jvp2 = getelementptr double, double* %out, i64 2",
            "  store double %tangent_l00, double* %out_jvp0",
            "  store double %tangent_l10, double* %out_jvp1",
            "  store double %tangent_l11, double* %out_jvp2",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %cotangent0ptr = getelementptr double, double* %cotangent, i64 0",
            "  %cotangent1ptr = getelementptr double, double* %cotangent, i64 1",
            "  %cotangent2ptr = getelementptr double, double* %cotangent, i64 2",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %cotangent0 = load double, double* %cotangent0ptr",
            "  %cotangent1 = load double, double* %cotangent1ptr",
            "  %cotangent2 = load double, double* %cotangent2ptr",
            "  %l00_vjp = call double @llvm.sqrt.f64(double %a00_vjp)",
            "  %l10_vjp = fdiv double %a01_vjp, %l00_vjp",
            "  %l10_sq_vjp = fmul double %l10_vjp, %l10_vjp",
            "  %schur_vjp = fsub double %a11_vjp, %l10_sq_vjp",
            "  %l11_vjp = call double @llvm.sqrt.f64(double %schur_vjp)",
            "  %two_l11_vjp = fmul double 2.0, %l11_vjp",
            "  %adjoint_schur = fdiv double %cotangent2, %two_l11_vjp",
            "  %two_l10_vjp = fmul double 2.0, %l10_vjp",
            "  %l10_schur_adjoint = fmul double %two_l10_vjp, %adjoint_schur",
            "  %adjoint_l10 = fsub double %cotangent1, %l10_schur_adjoint",
            "  %l00_sq_vjp = fmul double %l00_vjp, %l00_vjp",
            "  %adjoint_l10_a01 = fmul double %adjoint_l10, %a01_vjp",
            "  %adjoint_l10_a01_over_l00_sq = fdiv double %adjoint_l10_a01, %l00_sq_vjp",
            "  %adjoint_l00 = fsub double %cotangent0, %adjoint_l10_a01_over_l00_sq",
            "  %two_l00_vjp = fmul double 2.0, %l00_vjp",
            "  %out0_value = fdiv double %adjoint_l00, %two_l00_vjp",
            "  %out1_value = fdiv double %adjoint_l10, %l00_vjp",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  %out2 = getelementptr double, double* %out, i64 2",
            "  store double %out0_value, double* %out0",
            "  store double %out1_value, double* %out1",
            "  store double %adjoint_schur, double* %out2",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_symmetric_2x2_eigenvalues_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "symmetric_2x2_eigenvalues"',
            '; source = "native_symmetric_2x2_eigenvalues_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 3",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            "declare double @llvm.sqrt.f64(double)",
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %a00ptr = getelementptr double, double* %values, i64 0",
            "  %a01ptr = getelementptr double, double* %values, i64 1",
            "  %a11ptr = getelementptr double, double* %values, i64 2",
            "  %a00 = load double, double* %a00ptr",
            "  %a01 = load double, double* %a01ptr",
            "  %a11 = load double, double* %a11ptr",
            "  %trace = fadd double %a00, %a11",
            "  %centre = fmul double 5.0e-1, %trace",
            "  %delta = fsub double %a00, %a11",
            "  %half_delta = fmul double 5.0e-1, %delta",
            "  %half_delta_square = fmul double %half_delta, %half_delta",
            "  %offdiag_square = fmul double %a01, %a01",
            "  %radius_square = fadd double %half_delta_square, %offdiag_square",
            "  %radius = call double @llvm.sqrt.f64(double %radius_square)",
            "  %lower = fsub double %centre, %radius",
            "  %upper = fadd double %centre, %radius",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  store double %lower, double* %out0",
            "  store double %upper, double* %out1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [2 x double]",
            "  %cotangent0 = getelementptr [2 x double], [2 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [2 x double], [2 x double]* %cotangent, i64 0, i64 1",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %a00ptr_jvp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_jvp = getelementptr double, double* %values, i64 1",
            "  %a11ptr_jvp = getelementptr double, double* %values, i64 2",
            "  %t00ptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %t01ptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %t11ptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %a00_jvp = load double, double* %a00ptr_jvp",
            "  %a01_jvp = load double, double* %a01ptr_jvp",
            "  %a11_jvp = load double, double* %a11ptr_jvp",
            "  %t00_jvp = load double, double* %t00ptr_jvp",
            "  %t01_jvp = load double, double* %t01ptr_jvp",
            "  %t11_jvp = load double, double* %t11ptr_jvp",
            "  %trace_tangent_jvp = fadd double %t00_jvp, %t11_jvp",
            "  %centre_tangent_jvp = fmul double 5.0e-1, %trace_tangent_jvp",
            "  %delta_jvp = fsub double %a00_jvp, %a11_jvp",
            "  %half_delta_jvp = fmul double 5.0e-1, %delta_jvp",
            "  %tangent_delta_jvp = fsub double %t00_jvp, %t11_jvp",
            "  %tangent_half_delta_jvp = fmul double 5.0e-1, %tangent_delta_jvp",
            "  %half_delta_square_jvp = fmul double %half_delta_jvp, %half_delta_jvp",
            "  %offdiag_square_jvp = fmul double %a01_jvp, %a01_jvp",
            "  %radius_square_jvp = fadd double %half_delta_square_jvp, %offdiag_square_jvp",
            "  %radius_jvp = call double @llvm.sqrt.f64(double %radius_square_jvp)",
            "  %radius_term0_jvp = fmul double %half_delta_jvp, %tangent_half_delta_jvp",
            "  %radius_term1_jvp = fmul double %a01_jvp, %t01_jvp",
            "  %radius_tangent_num_jvp = fadd double %radius_term0_jvp, %radius_term1_jvp",
            "  %radius_tangent_jvp = fdiv double %radius_tangent_num_jvp, %radius_jvp",
            "  %lower_jvp = fsub double %centre_tangent_jvp, %radius_tangent_jvp",
            "  %upper_jvp = fadd double %centre_tangent_jvp, %radius_tangent_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  store double %lower_jvp, double* %out_jvp0",
            "  store double %upper_jvp, double* %out_jvp1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %a00ptr_vjp = getelementptr double, double* %values, i64 0",
            "  %a01ptr_vjp = getelementptr double, double* %values, i64 1",
            "  %a11ptr_vjp = getelementptr double, double* %values, i64 2",
            "  %lower_cotangent_ptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %upper_cotangent_ptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %a00_vjp = load double, double* %a00ptr_vjp",
            "  %a01_vjp = load double, double* %a01ptr_vjp",
            "  %a11_vjp = load double, double* %a11ptr_vjp",
            "  %lower_cotangent_vjp = load double, double* %lower_cotangent_ptr_vjp",
            "  %upper_cotangent_vjp = load double, double* %upper_cotangent_ptr_vjp",
            "  %delta_vjp = fsub double %a00_vjp, %a11_vjp",
            "  %half_delta_vjp = fmul double 5.0e-1, %delta_vjp",
            "  %half_delta_square_vjp = fmul double %half_delta_vjp, %half_delta_vjp",
            "  %offdiag_square_vjp = fmul double %a01_vjp, %a01_vjp",
            "  %radius_square_vjp = fadd double %half_delta_square_vjp, %offdiag_square_vjp",
            "  %radius_vjp = call double @llvm.sqrt.f64(double %radius_square_vjp)",
            "  %two_radius_vjp = fmul double 2.0, %radius_vjp",
            "  %half_term_vjp = fdiv double %half_delta_vjp, %two_radius_vjp",
            "  %offdiag_term_vjp = fdiv double %a01_vjp, %radius_vjp",
            "  %lower_a00_factor = fsub double 5.0e-1, %half_term_vjp",
            "  %upper_a00_factor = fadd double 5.0e-1, %half_term_vjp",
            "  %lower_a11_factor = fadd double 5.0e-1, %half_term_vjp",
            "  %upper_a11_factor = fsub double 5.0e-1, %half_term_vjp",
            "  %adj_a00_lower = fmul double %lower_cotangent_vjp, %lower_a00_factor",
            "  %adj_a00_upper = fmul double %upper_cotangent_vjp, %upper_a00_factor",
            "  %adj_a00 = fadd double %adj_a00_lower, %adj_a00_upper",
            "  %cotangent_diff = fsub double %upper_cotangent_vjp, %lower_cotangent_vjp",
            "  %adj_a01 = fmul double %cotangent_diff, %offdiag_term_vjp",
            "  %adj_a11_lower = fmul double %lower_cotangent_vjp, %lower_a11_factor",
            "  %adj_a11_upper = fmul double %upper_cotangent_vjp, %upper_a11_factor",
            "  %adj_a11 = fadd double %adj_a11_lower, %adj_a11_upper",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  store double %adj_a00, double* %out_vjp0",
            "  store double %adj_a01, double* %out_vjp1",
            "  store double %adj_a11, double* %out_vjp2",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_matrix_2x2_eigenvalues_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_eigenvalues"',
            '; source = "native_matrix_2x2_eigenvalues_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 4",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            "declare double @llvm.sqrt.f64(double)",
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %aptr = getelementptr double, double* %values, i64 0",
            "  %bptr = getelementptr double, double* %values, i64 1",
            "  %cptr = getelementptr double, double* %values, i64 2",
            "  %dptr = getelementptr double, double* %values, i64 3",
            "  %a = load double, double* %aptr",
            "  %b = load double, double* %bptr",
            "  %c = load double, double* %cptr",
            "  %d = load double, double* %dptr",
            "  %trace = fadd double %a, %d",
            "  %delta = fsub double %a, %d",
            "  %delta_square = fmul double %delta, %delta",
            "  %bc = fmul double %b, %c",
            "  %four_bc = fmul double 4.0, %bc",
            "  %discriminant = fadd double %delta_square, %four_bc",
            "  %root = call double @llvm.sqrt.f64(double %discriminant)",
            "  %lower_num = fsub double %trace, %root",
            "  %upper_num = fadd double %trace, %root",
            "  %lower = fmul double 5.0e-1, %lower_num",
            "  %upper = fmul double 5.0e-1, %upper_num",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  store double %lower, double* %out0",
            "  store double %upper, double* %out1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [2 x double]",
            "  %cotangent0 = getelementptr [2 x double], [2 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [2 x double], [2 x double]* %cotangent, i64 0, i64 1",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %aptr_jvp = getelementptr double, double* %values, i64 0",
            "  %bptr_jvp = getelementptr double, double* %values, i64 1",
            "  %cptr_jvp = getelementptr double, double* %values, i64 2",
            "  %dptr_jvp = getelementptr double, double* %values, i64 3",
            "  %taptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %tbptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %tcptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %tdptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %a_jvp = load double, double* %aptr_jvp",
            "  %b_jvp = load double, double* %bptr_jvp",
            "  %c_jvp = load double, double* %cptr_jvp",
            "  %d_jvp = load double, double* %dptr_jvp",
            "  %ta_jvp = load double, double* %taptr_jvp",
            "  %tb_jvp = load double, double* %tbptr_jvp",
            "  %tc_jvp = load double, double* %tcptr_jvp",
            "  %td_jvp = load double, double* %tdptr_jvp",
            "  %trace_tangent_jvp = fadd double %ta_jvp, %td_jvp",
            "  %delta_jvp = fsub double %a_jvp, %d_jvp",
            "  %delta_tangent_jvp = fsub double %ta_jvp, %td_jvp",
            "  %delta_square_jvp = fmul double %delta_jvp, %delta_jvp",
            "  %bc_jvp = fmul double %b_jvp, %c_jvp",
            "  %four_bc_jvp = fmul double 4.0, %bc_jvp",
            "  %discriminant_jvp = fadd double %delta_square_jvp, %four_bc_jvp",
            "  %root_jvp = call double @llvm.sqrt.f64(double %discriminant_jvp)",
            "  %disc_tangent_delta = fmul double 2.0, %delta_jvp",
            "  %disc_tangent_delta_scaled = fmul double %disc_tangent_delta, %delta_tangent_jvp",
            "  %tb_c = fmul double %tb_jvp, %c_jvp",
            "  %b_tc = fmul double %b_jvp, %tc_jvp",
            "  %offdiag_tangent_sum = fadd double %tb_c, %b_tc",
            "  %offdiag_disc_tangent = fmul double 4.0, %offdiag_tangent_sum",
            "  %disc_tangent = fadd double %disc_tangent_delta_scaled, %offdiag_disc_tangent",
            "  %two_root_jvp = fmul double 2.0, %root_jvp",
            "  %root_tangent = fdiv double %disc_tangent, %two_root_jvp",
            "  %lower_tangent_num = fsub double %trace_tangent_jvp, %root_tangent",
            "  %upper_tangent_num = fadd double %trace_tangent_jvp, %root_tangent",
            "  %lower_tangent = fmul double 5.0e-1, %lower_tangent_num",
            "  %upper_tangent = fmul double 5.0e-1, %upper_tangent_num",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  store double %lower_tangent, double* %out_jvp0",
            "  store double %upper_tangent, double* %out_jvp1",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %aptr_vjp = getelementptr double, double* %values, i64 0",
            "  %bptr_vjp = getelementptr double, double* %values, i64 1",
            "  %cptr_vjp = getelementptr double, double* %values, i64 2",
            "  %dptr_vjp = getelementptr double, double* %values, i64 3",
            "  %lower_cotangent_ptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %upper_cotangent_ptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %a_vjp = load double, double* %aptr_vjp",
            "  %b_vjp = load double, double* %bptr_vjp",
            "  %c_vjp = load double, double* %cptr_vjp",
            "  %d_vjp = load double, double* %dptr_vjp",
            "  %lower_cotangent_vjp = load double, double* %lower_cotangent_ptr_vjp",
            "  %upper_cotangent_vjp = load double, double* %upper_cotangent_ptr_vjp",
            "  %delta_vjp = fsub double %a_vjp, %d_vjp",
            "  %delta_square_vjp = fmul double %delta_vjp, %delta_vjp",
            "  %bc_vjp = fmul double %b_vjp, %c_vjp",
            "  %four_bc_vjp = fmul double 4.0, %bc_vjp",
            "  %discriminant_vjp = fadd double %delta_square_vjp, %four_bc_vjp",
            "  %root_vjp = call double @llvm.sqrt.f64(double %discriminant_vjp)",
            "  %cotangent_sum = fadd double %lower_cotangent_vjp, %upper_cotangent_vjp",
            "  %alpha = fmul double 5.0e-1, %cotangent_sum",
            "  %cotangent_diff = fsub double %upper_cotangent_vjp, %lower_cotangent_vjp",
            "  %four_root_vjp = fmul double 4.0, %root_vjp",
            "  %beta = fdiv double %cotangent_diff, %four_root_vjp",
            "  %two_delta = fmul double 2.0, %delta_vjp",
            "  %a_disc_term = fmul double %two_delta, %beta",
            "  %adj_a = fadd double %alpha, %a_disc_term",
            "  %adj_d = fsub double %alpha, %a_disc_term",
            "  %four_c = fmul double 4.0, %c_vjp",
            "  %four_b = fmul double 4.0, %b_vjp",
            "  %adj_b = fmul double %four_c, %beta",
            "  %adj_c = fmul double %four_b, %beta",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  store double %adj_a, double* %out_vjp0",
            "  store double %adj_b, double* %out_vjp1",
            "  store double %adj_c, double* %out_vjp2",
            "  store double %adj_d, double* %out_vjp3",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_matrix_2x2_eigensystem_native_llvm_ir(rule_name: str) -> str:
    llvm = _load_llvmlite_binding()
    triple = llvm.get_default_triple()
    base_symbol = _safe_llvm_symbol(rule_name)
    return "\n".join(
        [
            f'; scpn.compiler_ad = "{_escape_mlir_string(rule_name)}"',
            '; primitive = "matrix_2x2_eigensystem"',
            '; source = "native_matrix_2x2_eigensystem_ad_codegen"',
            '; execution = "native_llvm_mcjit"',
            "; dimension = 2",
            "; value_count = 4",
            "; output_count = 6",
            f'target triple = "{_escape_mlir_string(triple)}"',
            "",
            "declare double @llvm.sqrt.f64(double)",
            "",
            f"define void @{base_symbol}_value(double* %values, double* %out) {{",
            "entry:",
            "  %aptr = getelementptr double, double* %values, i64 0",
            "  %bptr = getelementptr double, double* %values, i64 1",
            "  %cptr = getelementptr double, double* %values, i64 2",
            "  %dptr = getelementptr double, double* %values, i64 3",
            "  %a = load double, double* %aptr",
            "  %b = load double, double* %bptr",
            "  %c = load double, double* %cptr",
            "  %d = load double, double* %dptr",
            "  %trace = fadd double %a, %d",
            "  %delta = fsub double %a, %d",
            "  %delta_square = fmul double %delta, %delta",
            "  %bc = fmul double %b, %c",
            "  %four_bc = fmul double 4.0, %bc",
            "  %discriminant = fadd double %delta_square, %four_bc",
            "  %root = call double @llvm.sqrt.f64(double %discriminant)",
            "  %lower_num = fsub double %trace, %root",
            "  %upper_num = fadd double %trace, %root",
            "  %lower = fmul double 5.0e-1, %lower_num",
            "  %upper = fmul double 5.0e-1, %upper_num",
            "  %neg_delta = fsub double 0.0, %delta",
            "  %q_lower_num = fsub double %neg_delta, %root",
            "  %q_upper_num = fadd double %neg_delta, %root",
            "  %q_lower = fmul double 5.0e-1, %q_lower_num",
            "  %q_upper = fmul double 5.0e-1, %q_upper_num",
            "  %b_square_lower = fmul double %b, %b",
            "  %q_lower_square = fmul double %q_lower, %q_lower",
            "  %norm_lower_square = fadd double %b_square_lower, %q_lower_square",
            "  %norm_lower = call double @llvm.sqrt.f64(double %norm_lower_square)",
            "  %b_square_upper = fmul double %b, %b",
            "  %q_upper_square = fmul double %q_upper, %q_upper",
            "  %norm_upper_square = fadd double %b_square_upper, %q_upper_square",
            "  %norm_upper = call double @llvm.sqrt.f64(double %norm_upper_square)",
            "  %v_lower0 = fdiv double %b, %norm_lower",
            "  %v_lower1 = fdiv double %q_lower, %norm_lower",
            "  %v_upper0 = fdiv double %b, %norm_upper",
            "  %v_upper1 = fdiv double %q_upper, %norm_upper",
            "  %out0 = getelementptr double, double* %out, i64 0",
            "  %out1 = getelementptr double, double* %out, i64 1",
            "  %out2 = getelementptr double, double* %out, i64 2",
            "  %out3 = getelementptr double, double* %out, i64 3",
            "  %out4 = getelementptr double, double* %out, i64 4",
            "  %out5 = getelementptr double, double* %out, i64 5",
            "  store double %lower, double* %out0",
            "  store double %upper, double* %out1",
            "  store double %v_lower0, double* %out2",
            "  store double %v_upper0, double* %out3",
            "  store double %v_lower1, double* %out4",
            "  store double %v_upper1, double* %out5",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
            "entry:",
            "  %cotangent = alloca [6 x double]",
            "  %cotangent0 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 0",
            "  %cotangent1 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 1",
            "  %cotangent2 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 2",
            "  %cotangent3 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 3",
            "  %cotangent4 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 4",
            "  %cotangent5 = getelementptr [6 x double], [6 x double]* %cotangent, i64 0, i64 5",
            "  store double 1.0, double* %cotangent0",
            "  store double 1.0, double* %cotangent1",
            "  store double 1.0, double* %cotangent2",
            "  store double 1.0, double* %cotangent3",
            "  store double 1.0, double* %cotangent4",
            "  store double 1.0, double* %cotangent5",
            f"  call void @{base_symbol}_vjp(double* %values, double* %cotangent0, double* %out)",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            "entry:",
            "  %aptr_jvp = getelementptr double, double* %values, i64 0",
            "  %bptr_jvp = getelementptr double, double* %values, i64 1",
            "  %cptr_jvp = getelementptr double, double* %values, i64 2",
            "  %dptr_jvp = getelementptr double, double* %values, i64 3",
            "  %taptr_jvp = getelementptr double, double* %tangent, i64 0",
            "  %tbptr_jvp = getelementptr double, double* %tangent, i64 1",
            "  %tcptr_jvp = getelementptr double, double* %tangent, i64 2",
            "  %tdptr_jvp = getelementptr double, double* %tangent, i64 3",
            "  %a_jvp = load double, double* %aptr_jvp",
            "  %b_jvp = load double, double* %bptr_jvp",
            "  %c_jvp = load double, double* %cptr_jvp",
            "  %d_jvp = load double, double* %dptr_jvp",
            "  %ta_jvp = load double, double* %taptr_jvp",
            "  %tb_jvp = load double, double* %tbptr_jvp",
            "  %tc_jvp = load double, double* %tcptr_jvp",
            "  %td_jvp = load double, double* %tdptr_jvp",
            "  %trace_tangent_jvp = fadd double %ta_jvp, %td_jvp",
            "  %delta_jvp = fsub double %a_jvp, %d_jvp",
            "  %delta_tangent_jvp = fsub double %ta_jvp, %td_jvp",
            "  %delta_square_jvp = fmul double %delta_jvp, %delta_jvp",
            "  %bc_jvp = fmul double %b_jvp, %c_jvp",
            "  %four_bc_jvp = fmul double 4.0, %bc_jvp",
            "  %discriminant_jvp = fadd double %delta_square_jvp, %four_bc_jvp",
            "  %root_jvp = call double @llvm.sqrt.f64(double %discriminant_jvp)",
            "  %disc_tangent_delta = fmul double 2.0, %delta_jvp",
            "  %disc_tangent_delta_scaled = fmul double %disc_tangent_delta, %delta_tangent_jvp",
            "  %tb_c = fmul double %tb_jvp, %c_jvp",
            "  %b_tc = fmul double %b_jvp, %tc_jvp",
            "  %offdiag_tangent_sum = fadd double %tb_c, %b_tc",
            "  %offdiag_disc_tangent = fmul double 4.0, %offdiag_tangent_sum",
            "  %disc_tangent = fadd double %disc_tangent_delta_scaled, %offdiag_disc_tangent",
            "  %two_root_jvp = fmul double 2.0, %root_jvp",
            "  %root_tangent = fdiv double %disc_tangent, %two_root_jvp",
            "  %lower_tangent_num = fsub double %trace_tangent_jvp, %root_tangent",
            "  %upper_tangent_num = fadd double %trace_tangent_jvp, %root_tangent",
            "  %lower_tangent = fmul double 5.0e-1, %lower_tangent_num",
            "  %upper_tangent = fmul double 5.0e-1, %upper_tangent_num",
            "  %neg_delta_jvp = fsub double 0.0, %delta_jvp",
            "  %q_lower_num_jvp = fsub double %neg_delta_jvp, %root_jvp",
            "  %q_upper_num_jvp = fadd double %neg_delta_jvp, %root_jvp",
            "  %q_lower_jvp = fmul double 5.0e-1, %q_lower_num_jvp",
            "  %q_upper_jvp = fmul double 5.0e-1, %q_upper_num_jvp",
            "  %q_lower_tangent = fsub double %lower_tangent, %ta_jvp",
            "  %q_upper_tangent = fsub double %upper_tangent, %ta_jvp",
            "  %b_square_lower_jvp = fmul double %b_jvp, %b_jvp",
            "  %q_lower_square_jvp = fmul double %q_lower_jvp, %q_lower_jvp",
            "  %norm_lower_square_jvp = fadd double %b_square_lower_jvp, %q_lower_square_jvp",
            "  %norm_lower_jvp = call double @llvm.sqrt.f64(double %norm_lower_square_jvp)",
            "  %b_square_upper_jvp = fmul double %b_jvp, %b_jvp",
            "  %q_upper_square_jvp = fmul double %q_upper_jvp, %q_upper_jvp",
            "  %norm_upper_square_jvp = fadd double %b_square_upper_jvp, %q_upper_square_jvp",
            "  %norm_upper_jvp = call double @llvm.sqrt.f64(double %norm_upper_square_jvp)",
            "  %v_lower0_jvp = fdiv double %b_jvp, %norm_lower_jvp",
            "  %v_lower1_jvp = fdiv double %q_lower_jvp, %norm_lower_jvp",
            "  %v_upper0_jvp = fdiv double %b_jvp, %norm_upper_jvp",
            "  %v_upper1_jvp = fdiv double %q_upper_jvp, %norm_upper_jvp",
            "  %vl_dot_term0 = fmul double %v_lower0_jvp, %tb_jvp",
            "  %vl_dot_term1 = fmul double %v_lower1_jvp, %q_lower_tangent",
            "  %vl_dot = fadd double %vl_dot_term0, %vl_dot_term1",
            "  %vl_proj0 = fmul double %v_lower0_jvp, %vl_dot",
            "  %vl_proj1 = fmul double %v_lower1_jvp, %vl_dot",
            "  %vl_raw0 = fsub double %tb_jvp, %vl_proj0",
            "  %vl_raw1 = fsub double %q_lower_tangent, %vl_proj1",
            "  %vl_tangent0 = fdiv double %vl_raw0, %norm_lower_jvp",
            "  %vl_tangent1 = fdiv double %vl_raw1, %norm_lower_jvp",
            "  %vu_dot_term0 = fmul double %v_upper0_jvp, %tb_jvp",
            "  %vu_dot_term1 = fmul double %v_upper1_jvp, %q_upper_tangent",
            "  %vu_dot = fadd double %vu_dot_term0, %vu_dot_term1",
            "  %vu_proj0 = fmul double %v_upper0_jvp, %vu_dot",
            "  %vu_proj1 = fmul double %v_upper1_jvp, %vu_dot",
            "  %vu_raw0 = fsub double %tb_jvp, %vu_proj0",
            "  %vu_raw1 = fsub double %q_upper_tangent, %vu_proj1",
            "  %vu_tangent0 = fdiv double %vu_raw0, %norm_upper_jvp",
            "  %vu_tangent1 = fdiv double %vu_raw1, %norm_upper_jvp",
            "  %out_jvp0 = getelementptr double, double* %out, i64 0",
            "  %out_jvp1 = getelementptr double, double* %out, i64 1",
            "  %out_jvp2 = getelementptr double, double* %out, i64 2",
            "  %out_jvp3 = getelementptr double, double* %out, i64 3",
            "  %out_jvp4 = getelementptr double, double* %out, i64 4",
            "  %out_jvp5 = getelementptr double, double* %out, i64 5",
            "  store double %lower_tangent, double* %out_jvp0",
            "  store double %upper_tangent, double* %out_jvp1",
            "  store double %vl_tangent0, double* %out_jvp2",
            "  store double %vu_tangent0, double* %out_jvp3",
            "  store double %vl_tangent1, double* %out_jvp4",
            "  store double %vu_tangent1, double* %out_jvp5",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            "entry:",
            "  %aptr_vjp = getelementptr double, double* %values, i64 0",
            "  %bptr_vjp = getelementptr double, double* %values, i64 1",
            "  %cptr_vjp = getelementptr double, double* %values, i64 2",
            "  %dptr_vjp = getelementptr double, double* %values, i64 3",
            "  %clptr_vjp = getelementptr double, double* %cotangent, i64 0",
            "  %cuptr_vjp = getelementptr double, double* %cotangent, i64 1",
            "  %cvl0ptr_vjp = getelementptr double, double* %cotangent, i64 2",
            "  %cvu0ptr_vjp = getelementptr double, double* %cotangent, i64 3",
            "  %cvl1ptr_vjp = getelementptr double, double* %cotangent, i64 4",
            "  %cvu1ptr_vjp = getelementptr double, double* %cotangent, i64 5",
            "  %a_vjp = load double, double* %aptr_vjp",
            "  %b_vjp = load double, double* %bptr_vjp",
            "  %c_vjp = load double, double* %cptr_vjp",
            "  %d_vjp = load double, double* %dptr_vjp",
            "  %cl_vjp = load double, double* %clptr_vjp",
            "  %cu_vjp = load double, double* %cuptr_vjp",
            "  %cvl0_vjp = load double, double* %cvl0ptr_vjp",
            "  %cvu0_vjp = load double, double* %cvu0ptr_vjp",
            "  %cvl1_vjp = load double, double* %cvl1ptr_vjp",
            "  %cvu1_vjp = load double, double* %cvu1ptr_vjp",
            "  %delta_vjp = fsub double %a_vjp, %d_vjp",
            "  %delta_square_vjp = fmul double %delta_vjp, %delta_vjp",
            "  %bc_vjp = fmul double %b_vjp, %c_vjp",
            "  %four_bc_vjp = fmul double 4.0, %bc_vjp",
            "  %discriminant_vjp = fadd double %delta_square_vjp, %four_bc_vjp",
            "  %root_vjp = call double @llvm.sqrt.f64(double %discriminant_vjp)",
            "  %neg_delta_vjp = fsub double 0.0, %delta_vjp",
            "  %q_lower_num_vjp = fsub double %neg_delta_vjp, %root_vjp",
            "  %q_upper_num_vjp = fadd double %neg_delta_vjp, %root_vjp",
            "  %q_lower_vjp = fmul double 5.0e-1, %q_lower_num_vjp",
            "  %q_upper_vjp = fmul double 5.0e-1, %q_upper_num_vjp",
            "  %b_square_lower_vjp = fmul double %b_vjp, %b_vjp",
            "  %q_lower_square_vjp = fmul double %q_lower_vjp, %q_lower_vjp",
            "  %norm_lower_square_vjp = fadd double %b_square_lower_vjp, %q_lower_square_vjp",
            "  %norm_lower_vjp = call double @llvm.sqrt.f64(double %norm_lower_square_vjp)",
            "  %b_square_upper_vjp = fmul double %b_vjp, %b_vjp",
            "  %q_upper_square_vjp = fmul double %q_upper_vjp, %q_upper_vjp",
            "  %norm_upper_square_vjp = fadd double %b_square_upper_vjp, %q_upper_square_vjp",
            "  %norm_upper_vjp = call double @llvm.sqrt.f64(double %norm_upper_square_vjp)",
            "  %vl0_vjp = fdiv double %b_vjp, %norm_lower_vjp",
            "  %vl1_vjp = fdiv double %q_lower_vjp, %norm_lower_vjp",
            "  %vu0_vjp = fdiv double %b_vjp, %norm_upper_vjp",
            "  %vu1_vjp = fdiv double %q_upper_vjp, %norm_upper_vjp",
            "  %vl_eta_dot0 = fmul double %vl0_vjp, %cvl0_vjp",
            "  %vl_eta_dot1 = fmul double %vl1_vjp, %cvl1_vjp",
            "  %vl_eta_dot = fadd double %vl_eta_dot0, %vl_eta_dot1",
            "  %vl_eta_proj0 = fmul double %vl0_vjp, %vl_eta_dot",
            "  %vl_eta_proj1 = fmul double %vl1_vjp, %vl_eta_dot",
            "  %vl_gu0_num = fsub double %cvl0_vjp, %vl_eta_proj0",
            "  %vl_gu1_num = fsub double %cvl1_vjp, %vl_eta_proj1",
            "  %vl_gu0 = fdiv double %vl_gu0_num, %norm_lower_vjp",
            "  %vl_gu1 = fdiv double %vl_gu1_num, %norm_lower_vjp",
            "  %vu_eta_dot0 = fmul double %vu0_vjp, %cvu0_vjp",
            "  %vu_eta_dot1 = fmul double %vu1_vjp, %cvu1_vjp",
            "  %vu_eta_dot = fadd double %vu_eta_dot0, %vu_eta_dot1",
            "  %vu_eta_proj0 = fmul double %vu0_vjp, %vu_eta_dot",
            "  %vu_eta_proj1 = fmul double %vu1_vjp, %vu_eta_dot",
            "  %vu_gu0_num = fsub double %cvu0_vjp, %vu_eta_proj0",
            "  %vu_gu1_num = fsub double %cvu1_vjp, %vu_eta_proj1",
            "  %vu_gu0 = fdiv double %vu_gu0_num, %norm_upper_vjp",
            "  %vu_gu1 = fdiv double %vu_gu1_num, %norm_upper_vjp",
            "  %glambda_lower = fadd double %cl_vjp, %vl_gu1",
            "  %glambda_upper = fadd double %cu_vjp, %vu_gu1",
            "  %cotangent_sum = fadd double %glambda_lower, %glambda_upper",
            "  %alpha = fmul double 5.0e-1, %cotangent_sum",
            "  %cotangent_diff = fsub double %glambda_upper, %glambda_lower",
            "  %four_root_vjp = fmul double 4.0, %root_vjp",
            "  %beta = fdiv double %cotangent_diff, %four_root_vjp",
            "  %two_delta = fmul double 2.0, %delta_vjp",
            "  %a_disc_term = fmul double %two_delta, %beta",
            "  %adj_a_eig = fadd double %alpha, %a_disc_term",
            "  %adj_d_eig = fsub double %alpha, %a_disc_term",
            "  %four_c = fmul double 4.0, %c_vjp",
            "  %four_b = fmul double 4.0, %b_vjp",
            "  %adj_b_eig = fmul double %four_c, %beta",
            "  %adj_c_eig = fmul double %four_b, %beta",
            "  %gu1_sum = fadd double %vl_gu1, %vu_gu1",
            "  %adj_a_chart = fsub double %adj_a_eig, %gu1_sum",
            "  %gu0_sum = fadd double %vl_gu0, %vu_gu0",
            "  %adj_b_chart = fadd double %adj_b_eig, %gu0_sum",
            "  %out_vjp0 = getelementptr double, double* %out, i64 0",
            "  %out_vjp1 = getelementptr double, double* %out, i64 1",
            "  %out_vjp2 = getelementptr double, double* %out, i64 2",
            "  %out_vjp3 = getelementptr double, double* %out, i64 3",
            "  store double %adj_a_chart, double* %out_vjp0",
            "  store double %adj_b_chart, double* %out_vjp1",
            "  store double %adj_c_eig, double* %out_vjp2",
            "  store double %adj_d_eig, double* %out_vjp3",
            "  ret void",
            "}",
            "",
        ]
    )


def _compile_native_llvm_jit_functions(
    llvm_ir: str,
    base_symbol: str,
) -> Mapping[str, Any]:
    llvm = _load_llvmlite_binding()
    module = llvm.parse_assembly(llvm_ir)
    module.verify()
    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    backing_module = llvm.parse_assembly("")
    engine = llvm.create_mcjit_compiler(backing_module, target_machine)
    engine.add_module(module)
    engine.finalize_object()
    engine.run_static_constructors()

    double_pointer = ctypes.POINTER(ctypes.c_double)
    unary_function = ctypes.CFUNCTYPE(None, double_pointer, double_pointer)
    binary_function = ctypes.CFUNCTYPE(None, double_pointer, double_pointer, double_pointer)
    batch_value_gradient_function = ctypes.CFUNCTYPE(
        None,
        double_pointer,
        ctypes.c_int64,
        double_pointer,
        double_pointer,
    )
    batch_binary_function = ctypes.CFUNCTYPE(
        None,
        double_pointer,
        double_pointer,
        ctypes.c_int64,
        double_pointer,
    )
    functions: dict[str, Any] = {"engine": engine}
    for name, signature in (
        ("value", unary_function),
        ("gradient", unary_function),
        ("jvp", binary_function),
        ("vjp", binary_function),
    ):
        address = engine.get_function_address(f"{base_symbol}_{name}")
        if address == 0:
            raise ValueError(f"native_llvm_jit symbol {base_symbol}_{name} was not emitted")
        functions[name] = signature(address)
    batch_address = engine.get_function_address(f"{base_symbol}_batch_value_gradient")
    if batch_address != 0:
        functions["batch_value_gradient"] = batch_value_gradient_function(batch_address)
    for name in ("batch_jvp", "batch_vjp"):
        address = engine.get_function_address(f"{base_symbol}_{name}")
        if address != 0:
            functions[name] = batch_binary_function(address)
    return MappingProxyType(functions)


def _call_native_scalar_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != 1:
        raise ValueError("native scalar LLVM/JIT kernel requires one value")
    output = np.zeros(1, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_scalar_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != 1:
        raise ValueError("native scalar LLVM/JIT kernel requires one value")
    if checked_vector.size != 1:
        raise ValueError(f"native scalar LLVM/JIT kernel requires one {label} value")
    output = np.zeros(1, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_scalar_pair_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    output_size: int,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != 2:
        raise ValueError("native scalar binary LLVM/JIT kernel requires two values")
    if output_size not in {1, 2}:
        raise ValueError("native scalar binary LLVM/JIT output_size must be one or two")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_scalar_pair_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    output_size: int,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != 2:
        raise ValueError("native scalar binary LLVM/JIT kernel requires two values")
    expected_vector_size = 2 if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native scalar binary LLVM/JIT kernel requires {expected_vector_size} "
            f"{label} value(s)"
        )
    if output_size not in {1, 2}:
        raise ValueError("native scalar binary LLVM/JIT output_size must be one or two")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_vector_dot_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != 2 * checked_dimension:
        raise ValueError("native vector dot LLVM/JIT kernel requires 2 * dimension values")
    if output_size not in {1, 2 * checked_dimension}:
        raise ValueError("native vector dot LLVM/JIT output_size must be one or 2 * dimension")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_vector_dot_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != 2 * checked_dimension:
        raise ValueError("native vector dot LLVM/JIT kernel requires 2 * dimension values")
    expected_vector_size = 2 * checked_dimension if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native vector dot LLVM/JIT kernel requires {expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, 2 * checked_dimension}:
        raise ValueError("native vector dot LLVM/JIT output_size must be one or 2 * dimension")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_vector_squared_norm_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != checked_dimension:
        raise ValueError("native vector squared norm LLVM/JIT kernel requires dimension values")
    if output_size not in {1, checked_dimension}:
        raise ValueError(
            "native vector squared norm LLVM/JIT output_size must be one or dimension"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_vector_squared_norm_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_vector_dot_dimension(dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != checked_dimension:
        raise ValueError("native vector squared norm LLVM/JIT kernel requires dimension values")
    expected_vector_size = checked_dimension if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native vector squared norm LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, checked_dimension}:
        raise ValueError(
            "native vector squared norm LLVM/JIT output_size must be one or dimension"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_quadratic_form_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix quadratic form LLVM/JIT kernel requires "
            "dimension * dimension + dimension values"
        )
    if output_size not in {1, expected_value_count}:
        raise ValueError(
            "native matrix quadratic form LLVM/JIT output_size must be one or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_quadratic_form_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix quadratic form LLVM/JIT kernel requires "
            "dimension * dimension + dimension values"
        )
    expected_vector_size = expected_value_count if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native matrix quadratic form LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, expected_value_count}:
        raise ValueError(
            "native matrix quadratic form LLVM/JIT output_size must be one or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_vector_product_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix-vector product LLVM/JIT kernel requires "
            "dimension * dimension + dimension values"
        )
    if output_size not in {checked_dimension, expected_value_count}:
        raise ValueError(
            "native matrix-vector product LLVM/JIT output_size must be dimension or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_vector_product_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix-vector product LLVM/JIT kernel requires "
            "dimension * dimension + dimension values"
        )
    expected_vector_size = expected_value_count if label == "tangent" else checked_dimension
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native matrix-vector product LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {checked_dimension, expected_value_count}:
        raise ValueError(
            "native matrix-vector product LLVM/JIT output_size must be dimension or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_matrix_product_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    expected_value_count = 2 * matrix_size
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix-matrix product LLVM/JIT kernel requires "
            "2 * dimension * dimension values"
        )
    if output_size not in {matrix_size, expected_value_count}:
        raise ValueError(
            "native matrix-matrix product LLVM/JIT output_size must be matrix-sized or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_matrix_product_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    expected_value_count = 2 * matrix_size
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != expected_value_count:
        raise ValueError(
            "native matrix-matrix product LLVM/JIT kernel requires "
            "2 * dimension * dimension values"
        )
    expected_vector_size = expected_value_count if label == "tangent" else matrix_size
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native matrix-matrix product LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {matrix_size, expected_value_count}:
        raise ValueError(
            "native matrix-matrix product LLVM/JIT output_size must be matrix-sized or input-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_trace_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != matrix_size:
        raise ValueError(
            "native matrix trace LLVM/JIT kernel requires dimension * dimension values"
        )
    if output_size not in {1, matrix_size}:
        raise ValueError("native matrix trace LLVM/JIT output_size must be one or matrix-sized")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_trace_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != matrix_size:
        raise ValueError(
            "native matrix trace LLVM/JIT kernel requires dimension * dimension values"
        )
    expected_vector_size = matrix_size if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native matrix trace LLVM/JIT kernel requires {expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, matrix_size}:
        raise ValueError("native matrix trace LLVM/JIT output_size must be one or matrix-sized")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_frobenius_norm_squared_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != matrix_size:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT kernel requires dimension * dimension values"
        )
    if output_size not in {1, matrix_size}:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT output_size must be one or matrix-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_frobenius_norm_squared_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    dimension: int,
    output_size: int,
) -> np.ndarray:
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != matrix_size:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT kernel requires dimension * dimension values"
        )
    expected_vector_size = matrix_size if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, matrix_size}:
        raise ValueError(
            "native matrix Frobenius-squared LLVM/JIT output_size must be one or matrix-sized"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_determinant_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    output_size: int,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if checked_values.size != 4:
        raise ValueError("native 2x2 determinant LLVM/JIT kernel requires four matrix values")
    if output_size not in {1, 4}:
        raise ValueError("native 2x2 determinant LLVM/JIT output_size must be one or four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_determinant_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    output_size: int,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_values.size != 4:
        raise ValueError("native 2x2 determinant LLVM/JIT kernel requires four matrix values")
    expected_vector_size = 4 if label == "tangent" else 1
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native 2x2 determinant LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {1, 4}:
        raise ValueError("native 2x2 determinant LLVM/JIT output_size must be one or four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_matrix_2x2_inverse_values(
    label: str,
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 4:
        raise ValueError("native 2x2 inverse LLVM/JIT kernel requires four matrix values")
    determinant = checked_values[0] * checked_values[3] - checked_values[1] * checked_values[2]
    if not np.isfinite(determinant) or abs(float(determinant)) <= 1.0e-12:
        raise ValueError("native 2x2 inverse LLVM/JIT kernel requires a nonsingular matrix")
    return checked_values


def _call_native_matrix_2x2_inverse_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_matrix_2x2_inverse_values("values", values)
    if output_size != 4:
        raise ValueError("native 2x2 inverse LLVM/JIT output_size must be four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_inverse_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
) -> np.ndarray:
    checked_values = _as_native_matrix_2x2_inverse_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_vector.size != 4:
        raise ValueError(f"native 2x2 inverse LLVM/JIT kernel requires four {label} value(s)")
    output = np.zeros(4, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_matrix_2x2_solve_values(
    label: str,
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 6:
        raise ValueError(
            "native 2x2 solve LLVM/JIT kernel requires four matrix and two vector values"
        )
    determinant = checked_values[0] * checked_values[3] - checked_values[1] * checked_values[2]
    if not np.isfinite(determinant) or abs(float(determinant)) <= 1.0e-12:
        raise ValueError("native 2x2 solve LLVM/JIT kernel requires a nonsingular matrix")
    return checked_values


def _call_native_matrix_2x2_solve_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_matrix_2x2_solve_values("values", values)
    if output_size not in {2, 6}:
        raise ValueError("native 2x2 solve LLVM/JIT output_size must be two or six")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_solve_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_matrix_2x2_solve_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    expected_vector_size = 6 if label == "tangent" else 2
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            f"native 2x2 solve LLVM/JIT kernel requires {expected_vector_size} {label} value(s)"
        )
    if output_size not in {2, 6}:
        raise ValueError("native 2x2 solve LLVM/JIT output_size must be two or six")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_symmetric_2x2_cholesky_values(
    label: str,
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 3:
        raise ValueError(
            "native symmetric 2x2 Cholesky LLVM/JIT kernel requires upper-triangle values"
        )
    a00, a01, a11 = (float(item) for item in checked_values)
    schur = a11 - (a01 * a01) / a00 if a00 > 0.0 else float("nan")
    if not np.isfinite(schur) or a00 <= 1.0e-24 or schur <= 1.0e-24:
        raise ValueError(
            f"native symmetric 2x2 Cholesky LLVM/JIT kernel requires positive definite {label}"
        )
    return checked_values


def _call_native_symmetric_2x2_cholesky_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_symmetric_2x2_cholesky_values("values", values)
    if output_size != 3:
        raise ValueError("native symmetric 2x2 Cholesky LLVM/JIT output_size must be three")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_symmetric_2x2_cholesky_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_symmetric_2x2_cholesky_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    if checked_vector.size != 3:
        raise ValueError(
            f"native symmetric 2x2 Cholesky LLVM/JIT kernel requires three {label} values"
        )
    if output_size != 3:
        raise ValueError("native symmetric 2x2 Cholesky LLVM/JIT output_size must be three")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_symmetric_2x2_eigenvalues_values(
    label: str,
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 3:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT kernel requires upper-triangle values"
        )
    half_delta = 0.5 * (checked_values[0] - checked_values[2])
    radius_square = half_delta * half_delta + checked_values[1] * checked_values[1]
    if not np.isfinite(radius_square) or float(radius_square) <= 1.0e-24:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT kernel requires distinct eigenvalues"
        )
    return checked_values


def _call_native_symmetric_2x2_eigenvalues_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_symmetric_2x2_eigenvalues_values("values", values)
    if output_size not in {2, 3}:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT output_size must be two or three"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_symmetric_2x2_eigenvalues_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_symmetric_2x2_eigenvalues_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    expected_vector_size = 3 if label == "tangent" else 2
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {2, 3}:
        raise ValueError(
            "native symmetric 2x2 eigenvalue LLVM/JIT output_size must be two or three"
        )
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_matrix_2x2_eigenvalues_values(
    label: str,
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    checked_values = np.ascontiguousarray(_as_finite_vector(label, values), dtype=np.float64)
    if checked_values.size != 4:
        raise ValueError(
            "native matrix 2x2 eigenvalue LLVM/JIT kernel requires row-major matrix values"
        )
    a00, a01, a10, a11 = checked_values
    discriminant = (a00 - a11) * (a00 - a11) + 4.0 * a01 * a10
    if not np.isfinite(discriminant) or float(discriminant) <= 1.0e-24:
        raise ValueError(
            "native matrix 2x2 eigenvalue LLVM/JIT kernel requires real distinct eigenvalues"
        )
    return checked_values


def _call_native_matrix_2x2_eigenvalues_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_matrix_2x2_eigenvalues_values("values", values)
    if output_size not in {2, 4}:
        raise ValueError("native matrix 2x2 eigenvalue LLVM/JIT output_size must be two or four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_eigenvalues_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_matrix_2x2_eigenvalues_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    expected_vector_size = 4 if label == "tangent" else 2
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native matrix 2x2 eigenvalue LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {2, 4}:
        raise ValueError("native matrix 2x2 eigenvalue LLVM/JIT output_size must be two or four")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _as_native_matrix_2x2_eigensystem_values(
    label: str,
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    checked_values = _as_native_matrix_2x2_eigenvalues_values(label, values)
    if abs(float(checked_values[1])) <= 1.0e-12:
        raise ValueError(
            "native matrix 2x2 eigensystem LLVM/JIT kernel requires a non-zero "
            "upper off-diagonal eigenvector chart"
        )
    return checked_values


def _call_native_matrix_2x2_eigensystem_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_matrix_2x2_eigensystem_values("values", values)
    if output_size not in {4, 6}:
        raise ValueError("native matrix 2x2 eigensystem LLVM/JIT output_size must be four or six")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def _call_native_matrix_2x2_eigensystem_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    tangent_or_cotangent: np.ndarray,
    label: str,
    output_size: int,
) -> np.ndarray:
    checked_values = _as_native_matrix_2x2_eigensystem_values("values", values)
    checked_vector = np.ascontiguousarray(
        _as_finite_vector(label, tangent_or_cotangent), dtype=np.float64
    )
    expected_vector_size = 4 if label == "tangent" else 6
    if checked_vector.size != expected_vector_size:
        raise ValueError(
            "native matrix 2x2 eigensystem LLVM/JIT kernel requires "
            f"{expected_vector_size} {label} value(s)"
        )
    if output_size not in {4, 6}:
        raise ValueError("native matrix 2x2 eigensystem LLVM/JIT output_size must be four or six")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    return output


def compile_scalar_quadratic_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    quadratic: float,
    linear: float,
    constant: float,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile scalar quadratic value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native scalar quadratic AD requires backend='native_llvm_jit'")
    coefficients = np.asarray([quadratic, linear, constant], dtype=np.float64)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("quadratic, linear, and constant coefficients must be finite")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 1:
        raise ValueError("native scalar quadratic AD requires exactly one sample value")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_scalar_quadratic_native_llvm_ir(
        rule.name,
        float(coefficients[0]),
        float(coefficients[1]),
        float(coefficients[2]),
    )
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_scalar_unary(native_functions["value"], raw_values)

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_scalar_binary(
            native_functions["jvp"], raw_values, raw_tangent, "tangent"
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_scalar_binary(
            native_functions["vjp"], raw_values, raw_cotangent, "cotangent"
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_scalar_unary(native_functions["gradient"], values)
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT gradient kernel verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT scalar quadratic value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_scalar_quadratic_native_llvm_jit_lowering_rule(
    *,
    quadratic: float,
    linear: float,
    constant: float,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for scalar quadratic native LLVM/JIT AD kernels."""

    coefficients = np.asarray([quadratic, linear, constant], dtype=np.float64)
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("quadratic, linear, and constant coefficients must be finite")
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native scalar quadratic lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_scalar_quadratic_ad_to_native_llvm_jit(
            rule,
            quadratic=float(coefficients[0]),
            linear=float(coefficients[1]),
            constant=float(coefficients[2]),
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    primitive: str,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile scalar unary elementwise value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_primitive = primitive.strip().lower()
    _scalar_unary_native_intrinsics(checked_primitive)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native scalar unary AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 1:
        raise ValueError("native scalar unary AD requires exactly one sample value")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_scalar_unary_elementwise_native_llvm_ir(
        rule.name,
        checked_primitive,
    )
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_scalar_unary(native_functions["value"], raw_values)

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_scalar_binary(
            native_functions["jvp"], raw_values, raw_tangent, "tangent"
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_scalar_binary(
            native_functions["vjp"], raw_values, raw_cotangent, "cotangent"
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_scalar_unary(native_functions["gradient"], values)
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT scalar unary gradient verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT scalar unary value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_scalar_unary_elementwise_native_llvm_jit_lowering_rule(
    *,
    primitive: str,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for scalar unary elementwise native LLVM/JIT kernels."""

    checked_primitive = primitive.strip().lower()
    _scalar_unary_native_intrinsics(checked_primitive)
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native scalar unary lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_scalar_unary_elementwise_ad_to_native_llvm_jit(
            rule,
            primitive=checked_primitive,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    primitive: str,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile scalar binary elementwise value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_primitive = primitive.strip().lower()
    _scalar_binary_native_value_line(checked_primitive)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native scalar binary AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 2:
        raise ValueError("native scalar binary AD requires exactly two sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_scalar_binary_elementwise_native_llvm_ir(
        rule.name,
        checked_primitive,
    )
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_scalar_pair_unary(native_functions["value"], raw_values, 1)

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_scalar_pair_binary(
            native_functions["jvp"], raw_values, raw_tangent, "tangent", 1
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_scalar_pair_binary(
            native_functions["vjp"], raw_values, raw_cotangent, "cotangent", 2
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_scalar_pair_unary(native_functions["gradient"], values, 2)
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT scalar binary gradient verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT scalar binary value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_scalar_binary_elementwise_native_llvm_jit_lowering_rule(
    *,
    primitive: str,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for scalar binary elementwise native LLVM/JIT kernels."""

    checked_primitive = primitive.strip().lower()
    _scalar_binary_native_value_line(checked_primitive)
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native scalar binary lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_scalar_binary_elementwise_ad_to_native_llvm_jit(
            rule,
            primitive=checked_primitive,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def compile_vector_dot_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile vector dot-product value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_vector_dot_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native vector dot AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 2 * checked_dimension:
        raise ValueError("native vector dot AD requires exactly 2 * dimension sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_vector_dot_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_vector_dot_unary(
            native_functions["value"], raw_values, checked_dimension, 1
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_vector_dot_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_vector_dot_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            2 * checked_dimension,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_vector_dot_unary(
            native_functions["gradient"], values, checked_dimension, 2 * checked_dimension
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT vector dot gradient verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT vector dot value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_vector_dot_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for vector dot-product native LLVM/JIT kernels."""

    checked_dimension = _validate_vector_dot_dimension(dimension)
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native vector dot lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_vector_dot_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_vector_dot_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT vector dot contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_vector_dot_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native vector dot primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 2 * checked_dimension:
        raise ValueError(
            "native vector dot primitive transform requires exactly 2 * dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_vector_dot_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:dot;dimension:{checked_dimension};layout:x_then_y"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_vector_dot_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_vector_dot",
            "mlir_runtime_verification": "verified: native LLVM/JIT vector dot JVP",
            "rust": "available: Rust PyO3 vector dot value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine vector_dot value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "vector_dot_value,vector_dot_jvp,vector_dot_vjp,vector_dot_gradient"
            ),
            "llvm": "available: native LLVM MCJIT vector dot AD kernel",
            "jit": "available: native LLVM MCJIT vector dot AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT vector dot value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_vector_dot_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_bilinear_vector_dot",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_bilinear_real_domain",
        effect="pure",
    )


def compile_vector_squared_norm_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile vector squared-norm value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_vector_dot_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native vector squared norm AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != checked_dimension:
        raise ValueError("native vector squared norm AD requires exactly dimension sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_vector_squared_norm_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_vector_squared_norm_unary(
            native_functions["value"], raw_values, checked_dimension, 1
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_vector_squared_norm_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_vector_squared_norm_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            checked_dimension,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_vector_squared_norm_unary(
            native_functions["gradient"], values, checked_dimension, checked_dimension
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT vector squared norm gradient verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT vector squared norm value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_vector_squared_norm_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for vector squared-norm native LLVM/JIT kernels."""

    checked_dimension = _validate_vector_dot_dimension(dimension)
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native vector squared norm lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_vector_squared_norm_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_vector_squared_norm_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT squared-norm contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_vector_dot_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native vector squared norm primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != checked_dimension:
        raise ValueError(
            "native vector squared norm primitive transform requires exactly dimension "
            "sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_vector_squared_norm_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:squared_norm;dimension:{checked_dimension}"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_vector_squared_norm_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_vector_squared_norm",
            "mlir_runtime_verification": "verified: native LLVM/JIT vector squared norm JVP",
            "rust": "available: Rust PyO3 vector squared norm value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine vector_squared_norm value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "vector_squared_norm_value,vector_squared_norm_jvp,"
                "vector_squared_norm_vjp,vector_squared_norm_gradient"
            ),
            "llvm": "available: native LLVM MCJIT vector squared norm AD kernel",
            "jit": "available: native LLVM MCJIT vector squared norm AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT vector squared norm value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_vector_squared_norm_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_vector_squared_norm",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_vector_squared_norm_real_domain",
        effect="pure",
    )


def compile_matrix_vector_product_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix-vector value/JVP/VJP kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix-vector product AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix-vector product AD requires dimension * dimension + dimension "
            "sample values"
        )
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_vector_product_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_vector_product_unary(
            native_functions["value"], raw_values, checked_dimension, checked_dimension
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_vector_product_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            checked_dimension,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_vector_product_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            expected_value_count,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_matrix_vector_product_unary(
            native_functions["gradient"], values, checked_dimension, expected_value_count
        )
        reference_gradient = vjp_kernel(values, np.ones(checked_dimension, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT matrix-vector product gradient verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT matrix-vector product value/JVP/VJP kernel; "
            "gradient() remains fail-closed for vector-output kernels"
        ),
    )


def make_matrix_vector_product_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix-vector native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native matrix-vector product lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_vector_product_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_vector_product_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT matrix-vector contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix-vector product primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix-vector product primitive transform requires "
            "dimension * dimension + dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_vector_product_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:matvec;dimension:{checked_dimension};layout:matrix_then_vector"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel),
        lowering_rule=make_matrix_vector_product_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_vector_product",
            "mlir_runtime_verification": "verified: native LLVM/JIT matrix-vector JVP",
            "rust": (
                "available: Rust PyO3 matrix-vector product value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_vector_product "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_vector_product_value,matrix_vector_product_jvp,"
                "matrix_vector_product_vjp,matrix_vector_product_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix-vector AD kernel",
            "jit": "available: native LLVM MCJIT matrix-vector AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix-vector value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_vector_product_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_vector_product",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (checked_dimension,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_vector_product_real_domain",
        effect="pure",
    )


def compile_matrix_matrix_product_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix-matrix value/JVP/VJP kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    expected_value_count = 2 * matrix_size
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix-matrix product AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix-matrix product AD requires 2 * dimension * dimension sample values"
        )
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_matrix_product_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_matrix_product_unary(
            native_functions["value"], raw_values, checked_dimension, matrix_size
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_matrix_product_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            matrix_size,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_matrix_product_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            expected_value_count,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_matrix_matrix_product_unary(
            native_functions["gradient"], values, checked_dimension, expected_value_count
        )
        reference_gradient = vjp_kernel(values, np.ones(matrix_size, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT matrix-matrix product gradient verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT matrix-matrix product value/JVP/VJP kernel; "
            "gradient() remains fail-closed for matrix-output kernels"
        ),
    )


def make_matrix_matrix_product_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix-matrix native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native matrix-matrix product lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_matrix_product_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_matrix_product_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT matrix-matrix contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    expected_value_count = 2 * matrix_size
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix-matrix product primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix-matrix product primitive transform requires "
            "2 * dimension * dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_matrix_product_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:matmul;dimension:{checked_dimension};layout:left_then_right"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel),
        lowering_rule=make_matrix_matrix_product_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_matrix_product",
            "mlir_runtime_verification": "verified: native LLVM/JIT matrix-matrix JVP",
            "rust": (
                "available: Rust PyO3 matrix-matrix product value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_matrix_product "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_matrix_product_value,matrix_matrix_product_jvp,"
                "matrix_matrix_product_vjp,matrix_matrix_product_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix-matrix AD kernel",
            "jit": "available: native LLVM MCJIT matrix-matrix AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix-matrix value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_matrix_product_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_matrix_product",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (checked_dimension, checked_dimension),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_matrix_product_real_domain",
        effect="pure",
    )


def compile_matrix_trace_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix trace value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix trace AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != matrix_size:
        raise ValueError("native matrix trace AD requires dimension * dimension sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_trace_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_trace_unary(
            native_functions["value"], raw_values, checked_dimension, 1
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_trace_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_trace_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            matrix_size,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_matrix_trace_unary(
            native_functions["gradient"], values, checked_dimension, matrix_size
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT matrix trace gradient verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT matrix trace value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_matrix_trace_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix trace native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native matrix trace lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_trace_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_trace_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT matrix-trace contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix trace primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != matrix_size:
        raise ValueError(
            "native matrix trace primitive transform requires dimension * dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_trace_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = f"primitive:trace;dimension:{checked_dimension};layout:row_major"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_trace_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_trace",
            "mlir_runtime_verification": "verified: native LLVM/JIT matrix trace JVP",
            "rust": "available: Rust PyO3 matrix trace value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_trace value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_trace_value,matrix_trace_jvp,matrix_trace_vjp,matrix_trace_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix trace AD kernel",
            "jit": "available: native LLVM MCJIT matrix trace AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix trace value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_matrix_trace_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_trace",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_trace_real_domain",
        effect="pure",
    )


def compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix Frobenius-squared value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix Frobenius-squared AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != matrix_size:
        raise ValueError(
            "native matrix Frobenius-squared AD requires dimension * dimension sample values"
        )
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_frobenius_norm_squared_native_llvm_ir(
        rule.name,
        checked_dimension,
    )
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_frobenius_norm_squared_unary(
            native_functions["value"],
            raw_values,
            checked_dimension,
            1,
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_frobenius_norm_squared_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_frobenius_norm_squared_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            matrix_size,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_matrix_frobenius_norm_squared_unary(
            native_functions["gradient"],
            values,
            checked_dimension,
            matrix_size,
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT matrix Frobenius-squared gradient verification failed"
            )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT matrix Frobenius-squared value/JVP/VJP/gradient "
            "kernel; unregistered primitives remain fail-closed"
        ),
    )


def make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix Frobenius-squared native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native matrix Frobenius-squared lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT Frobenius-squared contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    matrix_size = checked_dimension * checked_dimension
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix Frobenius-squared primitive transform requires "
            "backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != matrix_size:
        raise ValueError(
            "native matrix Frobenius-squared primitive transform requires dimension * "
            "dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = (
        f"primitive:frobenius_norm_squared;dimension:{checked_dimension};layout:row_major"
    )
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_frobenius_norm_squared",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT matrix Frobenius-squared JVP"
            ),
            "rust": (
                "available: Rust PyO3 matrix Frobenius-squared value/JVP/VJP/gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_frobenius_norm_squared "
                "value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_frobenius_norm_squared_value,matrix_frobenius_norm_squared_jvp,"
                "matrix_frobenius_norm_squared_vjp,matrix_frobenius_norm_squared_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix Frobenius-squared AD kernel",
            "jit": "available: native LLVM MCJIT matrix Frobenius-squared AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix Frobenius-squared value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_matrix_frobenius_norm_squared_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_frobenius_norm_squared",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_frobenius_norm_squared_real_domain",
        effect="pure",
    )


def compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile exact 2x2 determinant value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native 2x2 determinant AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 4:
        raise ValueError("native 2x2 determinant AD requires four sample values")
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_determinant_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_determinant_unary(
            native_functions["value"],
            raw_values,
            1,
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_determinant_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            1,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_determinant_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            4,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_matrix_2x2_determinant_unary(
            native_functions["gradient"],
            values,
            4,
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT 2x2 determinant gradient verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT 2x2 determinant value/JVP/VJP/gradient kernel; "
            "unregistered primitives remain fail-closed"
        ),
    )


def make_matrix_2x2_determinant_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for exact 2x2 determinant native LLVM/JIT kernels."""

    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native 2x2 determinant lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_determinant_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT 2x2 determinant contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native 2x2 determinant primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != 4:
        raise ValueError("native 2x2 determinant primitive transform requires four sample values")
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_determinant_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:determinant;dimension:2;layout:row_major"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_2x2_determinant_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_determinant",
            "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 determinant JVP",
            "rust": "available: Rust PyO3 2x2 determinant value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_determinant "
                "value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_2x2_determinant_value,matrix_2x2_determinant_jvp,"
                "matrix_2x2_determinant_vjp,matrix_2x2_determinant_gradient"
            ),
            "llvm": "available: native LLVM MCJIT 2x2 determinant AD kernel",
            "jit": "available: native LLVM MCJIT 2x2 determinant AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT 2x2 determinant value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_matrix_2x2_determinant_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_polynomial_matrix_2x2_determinant",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="polynomial_matrix_2x2_determinant_real_domain",
        effect="pure",
    )


def compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile exact nonsingular 2x2 inverse value/JVP/VJP kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native 2x2 inverse AD requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_inverse_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_inverse_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_inverse_unary(
            native_functions["value"],
            raw_values,
            4,
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_inverse_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_inverse_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_sum_gradient = _call_native_matrix_2x2_inverse_unary(
            native_functions["gradient"],
            values,
            4,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(4, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT 2x2 inverse sum-gradient provenance verification failed"
            )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT 2x2 inverse value/JVP/VJP kernel with "
            "sum-output gradient provenance; public gradient remains scalar-output "
            "fail-closed and singular matrices remain fail-closed"
        ),
    )


def make_matrix_2x2_inverse_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for exact nonsingular 2x2 inverse native LLVM/JIT kernels."""

    captured_values = (
        None
        if sample_values is None
        else _as_native_matrix_2x2_inverse_values("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native 2x2 inverse lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_inverse_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT nonsingular 2x2 inverse contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native 2x2 inverse primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_native_matrix_2x2_inverse_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_inverse_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:inverse;dimension:2;layout:row_major"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_2x2_inverse_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_inverse",
            "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 inverse JVP",
            "rust": (
                "available: Rust PyO3 nonsingular 2x2 inverse value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_inverse "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_2x2_inverse_value,matrix_2x2_inverse_jvp,"
                "matrix_2x2_inverse_vjp,matrix_2x2_inverse_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT 2x2 inverse AD kernel",
            "jit": "available: native LLVM MCJIT 2x2 inverse AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT 2x2 inverse value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_2x2_inverse_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "singular_matrix_2x2_inverse",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (4,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="nonsingular_matrix_2x2_inverse_real_domain",
        effect="pure",
    )


def compile_matrix_2x2_solve_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile exact nonsingular 2x2 linear-solve value/JVP/VJP kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native 2x2 solve AD requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_solve_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_solve_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_solve_unary(
            native_functions["value"],
            raw_values,
            2,
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_solve_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            2,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_solve_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            6,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_sum_gradient = _call_native_matrix_2x2_solve_unary(
            native_functions["gradient"],
            values,
            6,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(2, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT 2x2 solve sum-gradient provenance verification failed"
            )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT 2x2 solve value/JVP/VJP kernel with "
            "sum-output gradient provenance; public gradient remains scalar-output "
            "fail-closed and singular matrices remain fail-closed"
        ),
    )


def make_matrix_2x2_solve_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for exact nonsingular 2x2 solve native LLVM/JIT kernels."""

    captured_values = (
        None
        if sample_values is None
        else _as_native_matrix_2x2_solve_values("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native 2x2 solve lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_solve_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_solve_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT nonsingular 2x2 solve contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native 2x2 solve primitive transform requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_solve_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_solve_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:solve;dimension:2;layout:row_major"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_2x2_solve_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_solve",
            "mlir_runtime_verification": "verified: native LLVM/JIT 2x2 solve JVP",
            "rust": (
                "available: Rust PyO3 nonsingular 2x2 solve value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_solve value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_2x2_solve_value,matrix_2x2_solve_jvp,"
                "matrix_2x2_solve_vjp,matrix_2x2_solve_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT 2x2 solve AD kernel",
            "jit": "available: native LLVM MCJIT 2x2 solve AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": "verified: native LLVM MCJIT 2x2 solve value/JVP/VJP",
            "static_derivative_factory": "native_matrix_2x2_solve_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "singular_matrix_2x2_solve",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (2,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="nonsingular_matrix_2x2_solve_real_domain",
        effect="pure",
    )


def compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile SPD symmetric 2x2 Cholesky value/JVP/VJP kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native symmetric 2x2 Cholesky AD requires backend='native_llvm_jit'")
    values = _as_native_symmetric_2x2_cholesky_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_symmetric_2x2_cholesky_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_symmetric_2x2_cholesky_unary(
            native_functions["value"],
            raw_values,
            3,
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_symmetric_2x2_cholesky_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            3,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_symmetric_2x2_cholesky_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            3,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_sum_gradient = _call_native_symmetric_2x2_cholesky_unary(
            native_functions["gradient"],
            values,
            3,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(3, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT symmetric 2x2 Cholesky sum-gradient provenance verification failed"
            )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT SPD symmetric 2x2 Cholesky value/JVP/VJP "
            "kernel with sum-output gradient provenance; public gradient remains "
            "scalar-output fail-closed and non-SPD matrices remain fail-closed"
        ),
    )


def make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for SPD symmetric 2x2 Cholesky native LLVM/JIT kernels."""

    captured_values = (
        None
        if sample_values is None
        else _as_native_symmetric_2x2_cholesky_values("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native symmetric 2x2 Cholesky lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT SPD 2x2 Cholesky contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native symmetric 2x2 Cholesky primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_native_symmetric_2x2_cholesky_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:cholesky;dimension:2;layout:upper_triangle"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_symmetric_2x2_cholesky",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT SPD symmetric 2x2 Cholesky JVP"
            ),
            "rust": (
                "available: Rust PyO3 SPD symmetric 2x2 Cholesky value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine symmetric_2x2_cholesky "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "symmetric_2x2_cholesky_value,symmetric_2x2_cholesky_jvp,"
                "symmetric_2x2_cholesky_vjp,symmetric_2x2_cholesky_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT SPD symmetric 2x2 Cholesky AD kernel",
            "jit": "available: native LLVM MCJIT SPD symmetric 2x2 Cholesky AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT SPD symmetric 2x2 Cholesky value/JVP/VJP"
            ),
            "static_derivative_factory": "native_symmetric_2x2_cholesky_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "non_spd_symmetric_2x2_cholesky",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (3,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="symmetric_2x2_spd_cholesky_real_domain",
        effect="pure",
    )


def compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile distinct symmetric 2x2 eigenvalue value/JVP/VJP kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native symmetric 2x2 eigenvalue AD requires backend='native_llvm_jit'")
    values = _as_native_symmetric_2x2_eigenvalues_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_symmetric_2x2_eigenvalues_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_symmetric_2x2_eigenvalues_unary(
            native_functions["value"],
            raw_values,
            2,
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_symmetric_2x2_eigenvalues_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            2,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_symmetric_2x2_eigenvalues_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            3,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_sum_gradient = _call_native_symmetric_2x2_eigenvalues_unary(
            native_functions["gradient"],
            values,
            3,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(2, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT symmetric 2x2 eigenvalue sum-gradient provenance verification failed"
            )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT distinct symmetric 2x2 eigenvalues value/JVP/VJP "
            "kernel with sum-output gradient provenance; public gradient remains "
            "scalar-output fail-closed and repeated eigenvalues remain fail-closed"
        ),
    )


def make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for distinct symmetric 2x2 eigenvalue native LLVM/JIT kernels."""

    captured_values = (
        None
        if sample_values is None
        else _as_native_symmetric_2x2_eigenvalues_values("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native symmetric 2x2 eigenvalue lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT distinct symmetric eigvalsh contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native symmetric 2x2 eigenvalue primitive transform requires "
            "backend='native_llvm_jit'"
        )
    values = _as_native_symmetric_2x2_eigenvalues_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = "primitive:eigvalsh;dimension:2;layout:upper_triangle"
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_symmetric_2x2_eigenvalues",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT distinct symmetric 2x2 eigenvalue JVP"
            ),
            "rust": (
                "available: Rust PyO3 distinct symmetric 2x2 eigenvalue "
                "value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine symmetric_2x2_eigenvalues "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "symmetric_2x2_eigenvalues_value,symmetric_2x2_eigenvalues_jvp,"
                "symmetric_2x2_eigenvalues_vjp,symmetric_2x2_eigenvalues_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT distinct symmetric 2x2 eigenvalue AD kernel",
            "jit": "available: native LLVM MCJIT distinct symmetric 2x2 eigenvalue AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT distinct symmetric 2x2 eigenvalue value/JVP/VJP"
            ),
            "static_derivative_factory": "native_symmetric_2x2_eigenvalues_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "repeated_symmetric_2x2_eigenvalue",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (2,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="distinct_symmetric_2x2_eigenvalues_real_domain",
        effect="pure",
    )


def compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile real-simple nonsymmetric 2x2 eigenvalue value/JVP/VJP kernels."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix 2x2 eigenvalue AD requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_eigenvalues_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_eigenvalues_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_eigenvalues_unary(
            native_functions["value"],
            raw_values,
            2,
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_eigenvalues_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            2,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_eigenvalues_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            4,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_sum_gradient = _call_native_matrix_2x2_eigenvalues_unary(
            native_functions["gradient"],
            values,
            4,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(2, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT matrix 2x2 eigenvalue sum-gradient provenance verification failed"
            )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalues "
            "value/JVP/VJP kernel with sum-output gradient provenance; public "
            "gradient remains scalar-output fail-closed and complex or repeated "
            "eigenvalues remain fail-closed"
        ),
    )


def make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for real-simple nonsymmetric 2x2 eigenvalue kernels."""

    captured_values = (
        None
        if sample_values is None
        else _as_native_matrix_2x2_eigenvalues_values("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native matrix 2x2 eigenvalue lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT eigenvalue contract.

    The contract is intentionally narrow: row-major real nonsymmetric 2x2
    matrices whose spectra are real and distinct. Complex spectra and repeated
    eigenvalues remain fail-closed.
    """

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix 2x2 eigenvalue primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_native_matrix_2x2_eigenvalues_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel),
        lowering_rule=make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_eigenvalues",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT real-simple nonsymmetric 2x2 eigenvalue JVP"
            ),
            "rust": (
                "available: Rust PyO3 real-simple nonsymmetric 2x2 eigenvalue "
                "value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_eigenvalues "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": (
                "primitive:eigvals;dimension:2;layout:row_major;domain:real_simple"
            ),
            "rust_backend_functions": (
                "matrix_2x2_eigenvalues_value,matrix_2x2_eigenvalues_jvp,"
                "matrix_2x2_eigenvalues_vjp,matrix_2x2_eigenvalues_sum_gradient"
            ),
            "llvm": (
                "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue AD kernel"
            ),
            "jit": (
                "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue AD kernel"
            ),
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT real-simple nonsymmetric 2x2 eigenvalue value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_2x2_eigenvalues_llvm_jit",
            "static_signature": (
                "primitive:eigvals;dimension:2;layout:row_major;domain:real_simple"
            ),
            "nondifferentiable_boundary": "nonreal_or_repeated_matrix_2x2_eigenvalue",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (2,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="real_simple_matrix_2x2_eigenvalues_domain",
        effect="pure",
    )


def compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile real-simple nonsymmetric 2x2 eigensystem value/JVP/VJP kernels."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix 2x2 eigensystem AD requires backend='native_llvm_jit'")
    values = _as_native_matrix_2x2_eigensystem_values("sample_values", sample_values)
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_2x2_eigensystem_native_llvm_ir(rule.name)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_eigensystem_unary(
            native_functions["value"],
            raw_values,
            6,
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_eigensystem_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            6,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_2x2_eigensystem_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            4,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_sum_gradient = _call_native_matrix_2x2_eigensystem_unary(
            native_functions["gradient"],
            values,
            4,
        )
        reference_sum_gradient = vjp_kernel(values, np.ones(6, dtype=np.float64))
        if not np.allclose(
            native_sum_gradient,
            reference_sum_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError(
                "native LLVM/JIT matrix 2x2 eigensystem sum-gradient provenance verification failed"
            )
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem "
            "value/JVP/VJP kernel with sum-output gradient provenance; complex "
            "spectra, repeated eigenvalues, and zero upper off-diagonal eigenvector "
            "charts remain fail-closed"
        ),
    )


def make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule(
    *,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for real-simple nonsymmetric 2x2 eigensystem kernels."""

    captured_values = (
        None
        if sample_values is None
        else _as_native_matrix_2x2_eigensystem_values("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native matrix 2x2 eigensystem lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit(
            rule,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT eigensystem contract.

    The contract is intentionally narrow: row-major real nonsymmetric 2x2
    matrices whose spectra are real and distinct and whose eigenvector chart
    uses a non-zero upper off-diagonal entry. All other eigensystem domains
    remain fail-closed.
    """

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix 2x2 eigensystem primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_native_matrix_2x2_eigensystem_values("sample_values", sample_values)
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit(
        rule,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel),
        lowering_rule=make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule(
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_2x2_eigensystem",
            "mlir_runtime_verification": (
                "verified: native LLVM/JIT real-simple nonsymmetric 2x2 eigensystem JVP"
            ),
            "rust": (
                "available: Rust PyO3 real-simple nonsymmetric 2x2 eigensystem "
                "value/JVP/VJP/sum-gradient kernel"
            ),
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_2x2_eigensystem "
                "value/JVP/VJP/sum-gradient parity"
            ),
            "rust_backend_signature": (
                "primitive:eig;dimension:2;layout:row_major;domain:real_simple_upper_chart"
            ),
            "rust_backend_functions": (
                "matrix_2x2_eigensystem_value,matrix_2x2_eigensystem_jvp,"
                "matrix_2x2_eigensystem_vjp,matrix_2x2_eigensystem_sum_gradient"
            ),
            "llvm": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem AD kernel",
            "jit": "available: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT real-simple nonsymmetric 2x2 eigensystem value/JVP/VJP"
            ),
            "static_derivative_factory": "native_matrix_2x2_eigensystem_llvm_jit",
            "static_signature": (
                "primitive:eig;dimension:2;layout:row_major;domain:real_simple_upper_chart"
            ),
            "nondifferentiable_boundary": (
                "nonreal_repeated_or_zero_upper_chart_matrix_2x2_eigensystem"
            ),
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (6,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="real_simple_upper_chart_matrix_2x2_eigensystem_domain",
        effect="pure",
    )


def compile_matrix_quadratic_form_ad_to_native_llvm_jit(
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> ExecutableCompilerADKernel:
    """Compile matrix quadratic-form value/JVP/VJP/gradient kernels to LLVM MCJIT."""

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError("native matrix quadratic form AD requires backend='native_llvm_jit'")
    values = _as_finite_vector("sample_values", sample_values)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix quadratic form AD requires dimension * dimension + dimension "
            "sample values"
        )
    mlir_module = compile_custom_derivative_rule_to_mlir(
        rule,
        values,
        compile_config.mlir_config,
    )
    llvm_ir = _compile_matrix_quadratic_form_native_llvm_ir(rule.name, checked_dimension)
    native_functions = _compile_native_llvm_jit_functions(
        llvm_ir,
        _safe_llvm_symbol(rule.name),
    )

    def value_kernel(raw_values: np.ndarray) -> np.ndarray:
        return _call_native_matrix_quadratic_form_unary(
            native_functions["value"], raw_values, checked_dimension, 1
        )

    def jvp_kernel(raw_values: np.ndarray, raw_tangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_quadratic_form_binary(
            native_functions["jvp"],
            raw_values,
            raw_tangent,
            "tangent",
            checked_dimension,
            1,
        )

    def vjp_kernel(raw_values: np.ndarray, raw_cotangent: np.ndarray) -> np.ndarray:
        return _call_native_matrix_quadratic_form_binary(
            native_functions["vjp"],
            raw_values,
            raw_cotangent,
            "cotangent",
            checked_dimension,
            expected_value_count,
        )

    verification = _verify_executable_ad_kernel(
        rule,
        values,
        value_kernel,
        jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel if rule.vjp_rule is not None else None,
        compile_config,
        sample_tangent=sample_tangent,
        sample_cotangent=sample_cotangent,
    )
    if rule.vjp_rule is not None:
        native_gradient = _call_native_matrix_quadratic_form_unary(
            native_functions["gradient"], values, checked_dimension, expected_value_count
        )
        reference_gradient = vjp_kernel(values, np.ones(1, dtype=np.float64))
        if not np.allclose(
            native_gradient,
            reference_gradient,
            atol=compile_config.atol,
            rtol=compile_config.rtol,
        ):
            raise ValueError("native LLVM/JIT matrix quadratic form gradient verification failed")
    return ExecutableCompilerADKernel(
        rule_name=rule.name,
        backend=compile_config.backend,
        mlir_module=mlir_module,
        value_kernel=value_kernel,
        jvp_kernel=jvp_kernel if rule.jvp_rule is not None else None,
        vjp_kernel=vjp_kernel if rule.vjp_rule is not None else None,
        verification=verification,
        llvm_gradient_ir=llvm_ir,
        claim_boundary=(
            "verified native LLVM MCJIT matrix quadratic form "
            "value/JVP/VJP/gradient kernel; unregistered primitives remain fail-closed"
        ),
    )


def make_matrix_quadratic_form_native_llvm_jit_lowering_rule(
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray | None = None,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> Callable[..., ExecutableCompilerADKernel]:
    """Create a lowering rule for matrix quadratic-form native LLVM/JIT kernels."""

    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    captured_values = (
        None if sample_values is None else _as_finite_vector("sample_values", sample_values)
    )
    captured_tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    captured_cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )

    def lowering_rule(
        rule: CustomDerivativeRule,
        runtime_sample_values: Sequence[float] | np.ndarray | None = None,
        runtime_config: CompilerADExecutableConfig | None = None,
        *,
        sample_tangent: Sequence[float] | np.ndarray | None = None,
        sample_cotangent: Sequence[float] | np.ndarray | None = None,
    ) -> ExecutableCompilerADKernel:
        effective_values = runtime_sample_values
        if effective_values is None:
            effective_values = captured_values
        if effective_values is None:
            raise ValueError("native matrix quadratic form lowering requires sample_values")
        effective_config = runtime_config if runtime_config is not None else config
        effective_tangent = sample_tangent if sample_tangent is not None else captured_tangent
        effective_cotangent = (
            sample_cotangent if sample_cotangent is not None else captured_cotangent
        )
        return compile_matrix_quadratic_form_ad_to_native_llvm_jit(
            rule,
            dimension=checked_dimension,
            sample_values=effective_values,
            config=effective_config,
            sample_tangent=effective_tangent,
            sample_cotangent=effective_cotangent,
        )

    return lowering_rule


def make_matrix_quadratic_form_native_llvm_jit_primitive_transform(
    identity: PrimitiveIdentity | str,
    rule: CustomDerivativeRule,
    *,
    dimension: int | np.integer,
    sample_values: Sequence[float] | np.ndarray,
    config: CompilerADExecutableConfig | None = None,
    sample_tangent: Sequence[float] | np.ndarray | None = None,
    sample_cotangent: Sequence[float] | np.ndarray | None = None,
) -> PrimitiveTransformRule:
    """Create a complete Rust/PyO3 + native LLVM/JIT quadratic-form contract."""

    primitive_identity = PrimitiveIdentity.parse(identity)
    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("rule must be a CustomDerivativeRule")
    checked_dimension = _validate_matrix_quadratic_form_dimension(dimension)
    compile_config = (
        CompilerADExecutableConfig(backend="native_llvm_jit") if config is None else config
    )
    if compile_config.backend != "native_llvm_jit":
        raise ValueError(
            "native matrix quadratic form primitive transform requires backend='native_llvm_jit'"
        )
    values = _as_finite_vector("sample_values", sample_values)
    expected_value_count = _matrix_quadratic_form_value_count(checked_dimension)
    if values.size != expected_value_count:
        raise ValueError(
            "native matrix quadratic form primitive transform requires dimension * "
            "dimension + dimension sample values"
        )
    tangent = (
        None if sample_tangent is None else _as_finite_vector("sample_tangent", sample_tangent)
    )
    cotangent = (
        None
        if sample_cotangent is None
        else _as_finite_vector("sample_cotangent", sample_cotangent)
    )
    kernel = compile_matrix_quadratic_form_ad_to_native_llvm_jit(
        rule,
        dimension=checked_dimension,
        sample_values=values,
        config=compile_config,
        sample_tangent=tangent,
        sample_cotangent=cotangent,
    )
    static_signature = (
        f"primitive:quadratic_form;dimension:{checked_dimension};layout:matrix_then_vector"
    )
    return PrimitiveTransformRule(
        identity=primitive_identity,
        derivative_rule=rule,
        batching_rule=make_executable_ad_kernel_batching_rule(kernel, method="value"),
        lowering_rule=make_matrix_quadratic_form_native_llvm_jit_lowering_rule(
            dimension=checked_dimension,
            sample_values=values,
            config=compile_config,
            sample_tangent=tangent,
            sample_cotangent=cotangent,
        ),
        lowering_metadata={
            "mlir": "available: executable scpn_diff MLIR-runtime primitive kernel",
            "mlir_op": "scpn_diff.native_matrix_quadratic_form",
            "mlir_runtime_verification": ("verified: native LLVM/JIT matrix quadratic form JVP"),
            "rust": "available: Rust PyO3 matrix quadratic form value/JVP/VJP/gradient kernel",
            "rust_backend": "rust_pyo3",
            "rust_backend_verification": (
                "verified: scpn_quantum_engine matrix_quadratic_form value/JVP/VJP/gradient parity"
            ),
            "rust_backend_signature": static_signature,
            "rust_backend_functions": (
                "matrix_quadratic_form_value,matrix_quadratic_form_jvp,"
                "matrix_quadratic_form_vjp,matrix_quadratic_form_gradient"
            ),
            "llvm": "available: native LLVM MCJIT matrix quadratic form AD kernel",
            "jit": "available: native LLVM MCJIT matrix quadratic form AD kernel",
            "native_backend": "native_llvm_jit",
            "native_backend_verification": (
                "verified: native LLVM MCJIT matrix quadratic form value/JVP/VJP/gradient"
            ),
            "static_derivative_factory": "native_matrix_quadratic_form_llvm_jit",
            "static_signature": static_signature,
            "nondifferentiable_boundary": "none_smooth_matrix_quadratic_form",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=lambda _args: (1,),
        dtype_rule=lambda _args: "float64",
        static_argument_rule=lambda args: args,
        nondifferentiable_policy="smooth_matrix_quadratic_form_real_domain",
        effect="pure",
    )


def _safe_llvm_symbol(value: str) -> str:
    symbol = "".join(
        character if character.isalnum() or character == "_" else "_" for character in value
    )
    if not symbol or symbol[0].isdigit():
        symbol = f"_{symbol}"
    return symbol


def compile_whole_program_ad_trace_to_mlir(
    result: WholeProgramADResult,
    config: DifferentiableMLIRCompileConfig | None = None,
) -> MLIRModule:
    """Lower a whole-program AD execution trace to MLIR-style interchange text.

    The emitted module is an audit artifact for Python whole-program gradients
    and polyglot compiler planning. It deliberately records Rust and LLVM/JIT
    executable differentiation as blocked unless a real backend is provided.
    """

    if not isinstance(result, WholeProgramADResult):
        raise ValueError("whole-program MLIR lowering requires a WholeProgramADResult")
    compile_config = DifferentiableMLIRCompileConfig() if config is None else config
    lines = [
        f'module attributes {{scpn.module = "whole_program_ad", '
        f'scpn.dialect = "{compile_config.dialect}", '
        f"scpn.n_parameters = {result.gradient.size}, "
        f"scpn.trace_events = {len(result.trace_events)}, "
        f"scpn.control_flow = {_fmt_bool(result.control_flow_observed)}, "
        f"scpn.numpy = {_fmt_bool(result.numpy_observed)}}} {{",
        "  func.func @main() {",
    ]
    if compile_config.include_numeric_payload:
        lines.append(f"    scpn_diff.value %objective {{value = {_fmt_float(result.value)}}}")
        for index, (name, trainable, gradient) in enumerate(
            zip(result.parameter_names, result.trainable, result.gradient, strict=True)
        ):
            lines.append(
                "    scpn_diff.parameter "
                f'%p{index} {{name = "{_escape_mlir_string(name)}", '
                f"trainable = {_fmt_bool(trainable)}, "
                f"gradient = {_fmt_float(float(gradient))}}}"
            )
    for index, event in enumerate(result.trace_events):
        lines.append(
            "    scpn_diff.trace_event "
            f'{{index = {index}, file = "{_escape_mlir_string(event.filename)}", '
            f'line = {event.line_number}, function = "{_escape_mlir_string(event.function_name)}", '
            f'source = "{_escape_mlir_string(event.source)}"}}'
        )
    for index, instruction in enumerate(result.bytecode_instructions):
        lines.append(
            "    scpn_diff.bytecode "
            f"{{index = {index}, offset = {instruction.offset}, "
            f'op = "{_escape_mlir_string(instruction.opname)}", '
            f'arg = "{_escape_mlir_string(instruction.argrepr)}"}}'
        )
    for index, feature in enumerate(result.source_ir_features):
        lines.append(
            "    scpn_diff.source_semantics "
            f'{{index = {index}, kind = "{_escape_mlir_string(feature.kind)}", '
            f'detail = "{_escape_mlir_string(feature.detail)}", line = {feature.line_number}}}'
        )
    lines.append(
        "    scpn_diff.whole_program_ad "
        f'{{method = "{_escape_mlir_string(result.method)}", '
        'execution = "python_whole_program_ad_interchange"}}'
    )
    lines.append("    return")
    lines.append("  }")
    if compile_config.include_metadata:
        metadata = {
            "claim_boundary": result.claim_boundary,
            "method": result.method,
            "polyglot_targets": result.polyglot_targets,
            "semantics_report": None
            if result.semantics_report is None
            else {
                "aliasing_observed": result.semantics_report.aliasing_observed,
                "bytecode_frontend": result.semantics_report.bytecode_frontend,
                "control_flow_observed": result.semantics_report.control_flow_observed,
                "differentiation_semantics": result.semantics_report.differentiation_semantics,
                "graph_capture": result.semantics_report.graph_capture,
                "loop_observed": result.semantics_report.loop_observed,
                "mutation_observed": result.semantics_report.mutation_observed,
                "numpy_observed": result.semantics_report.numpy_observed,
                "source_frontend": result.semantics_report.source_frontend,
            },
            "target": compile_config.target,
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        lines.append(f'  scpn.metadata {{json = "{_escape_mlir_string(encoded)}"}}')
    lines.append("}")
    text = "\n".join(lines) + "\n"
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=compile_config.dialect,
        resource_counts={
            "parameters": int(result.gradient.size),
            "bytecode_instructions": len(result.bytecode_instructions),
            "source_ir_features": len(result.source_ir_features),
            "ir_nodes": len(result.ir_nodes),
            "trace_events": len(result.trace_events),
            "trainable_parameters": int(sum(result.trainable)),
            "gradient_nnz": int(np.count_nonzero(result.gradient)),
        },
        metadata={
            "claim_boundary": "whole-program AD trace interchange; no executable Rust, LLVM, or JIT lowering",
            "target": compile_config.target,
            "polyglot_targets": dict(result.polyglot_targets),
            "semantics_report": None
            if result.semantics_report is None
            else {
                "aliasing_observed": result.semantics_report.aliasing_observed,
                "bytecode_frontend": result.semantics_report.bytecode_frontend,
                "control_flow_observed": result.semantics_report.control_flow_observed,
                "differentiation_semantics": result.semantics_report.differentiation_semantics,
                "graph_capture": result.semantics_report.graph_capture,
                "loop_observed": result.semantics_report.loop_observed,
                "mutation_observed": result.semantics_report.mutation_observed,
                "numpy_observed": result.semantics_report.numpy_observed,
                "source_frontend": result.semantics_report.source_frontend,
            },
            "sha256_source": "module.text",
        },
    )


@dataclass(frozen=True)
class ExecutableWholeProgramADBatchResult:
    """Batched replay result from an executable whole-program AD kernel."""

    values: NDArray[np.float64]
    gradients: NDArray[np.float64]
    parameter_names: tuple[str, ...]
    row_signatures: tuple[tuple[str, ...], ...]
    mlir_sha256: str
    backend: str = "program_ad_trace_replay"
    claim_boundary: str = (
        "batched executable replay of supported captured scalar program AD IR; "
        "each row must preserve the compiled branch/signature contract"
    )

    def __post_init__(self) -> None:
        values = _as_finite_vector("batch values", self.values)
        gradients = np.asarray(self.gradients, dtype=np.float64)
        if gradients.ndim != 2:
            raise ValueError("batch gradients must be two-dimensional")
        if gradients.shape[0] != values.size:
            raise ValueError("batch gradients row count must match batch values")
        if gradients.shape[1] != len(self.parameter_names):
            raise ValueError("batch gradients column count must match parameter_names")
        if not np.all(np.isfinite(gradients)):
            raise ValueError("batch gradients must contain only finite values")
        if len(self.row_signatures) != values.size:
            raise ValueError("row_signatures count must match batch values")
        for signature in self.row_signatures:
            if any(not isinstance(item, str) or not item for item in signature):
                raise ValueError("row_signatures entries must be non-empty strings")
        if not self.mlir_sha256:
            raise ValueError("mlir_sha256 must be non-empty")
        if self.backend not in {"program_ad_trace_replay", "native_llvm_jit"}:
            raise ValueError("backend must be 'program_ad_trace_replay' or 'native_llvm_jit'")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "gradients", gradients.copy())


@dataclass(frozen=True)
class _NativeWholeProgramADCacheEntry:
    mlir_module: MLIRModule
    llvm_ir: str
    native_functions: Mapping[str, Any]
    verification: CompilerADKernelVerification
    supported_ops: tuple[str, ...]
    lowering_report: WholeProgramADNativeLoweringReport


@dataclass(frozen=True)
class WholeProgramADNativeLoweringReport:
    """Fail-closed native lowering audit for one captured program AD trace."""

    supported: bool
    lowerable_ops: tuple[str, ...]
    unsupported_ops: tuple[str, ...]
    control_flow_ops: tuple[str, ...]
    effect_kinds: tuple[str, ...]
    operation_count: int
    lowerable_operation_count: int
    unsupported_operation_count: int
    fail_closed_reason: str

    def __post_init__(self) -> None:
        for field_name, values in (
            ("lowerable_ops", self.lowerable_ops),
            ("unsupported_ops", self.unsupported_ops),
            ("control_flow_ops", self.control_flow_ops),
            ("effect_kinds", self.effect_kinds),
        ):
            if any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{field_name} entries must be non-empty strings")
        if self.operation_count < 1:
            raise ValueError("operation_count must be positive")
        if self.lowerable_operation_count < 0:
            raise ValueError("lowerable_operation_count must be non-negative")
        if self.unsupported_operation_count < 0:
            raise ValueError("unsupported_operation_count must be non-negative")
        if self.operation_count != (
            self.lowerable_operation_count + self.unsupported_operation_count
        ):
            raise ValueError("operation counts must partition the native lowering report")
        if self.supported != (self.unsupported_operation_count == 0):
            raise ValueError("supported must match unsupported_operation_count")
        if not self.fail_closed_reason:
            raise ValueError("fail_closed_reason must be non-empty")

    def as_metadata(self) -> Mapping[str, object]:
        """Return deterministic MLIR-serialisable native lowering metadata."""

        return MappingProxyType(
            {
                "supported": self.supported,
                "lowerable_ops": self.lowerable_ops,
                "unsupported_ops": self.unsupported_ops,
                "control_flow_ops": self.control_flow_ops,
                "effect_kinds": self.effect_kinds,
                "operation_count": self.operation_count,
                "lowerable_operation_count": self.lowerable_operation_count,
                "unsupported_operation_count": self.unsupported_operation_count,
                "fail_closed_reason": self.fail_closed_reason,
            }
        )


_NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK = threading.RLock()
_NATIVE_WHOLE_PROGRAM_AD_CACHE: dict[str, _NativeWholeProgramADCacheEntry] = {}
_NATIVE_WHOLE_PROGRAM_AD_CACHE_MAXSIZE = 32


@dataclass(frozen=True)
class NativeWholeProgramADKernel:
    """Native LLVM/JIT kernel for a supported scalar program AD trace."""

    objective: Callable[[Any], object]
    source_result: WholeProgramADResult
    parameters: tuple[Parameter, ...]
    mlir_module: MLIRModule
    llvm_ir: str
    native_functions: Mapping[str, Any]
    verification: CompilerADKernelVerification
    parameter_names: tuple[str, ...]
    parameter_shape: tuple[int, ...]
    trace_signature: tuple[str, ...]
    supported_ops: tuple[str, ...]
    lowering_report: WholeProgramADNativeLoweringReport
    cache_key: str
    cache_hit: bool
    backend: str = "native_llvm_jit"
    claim_boundary: str = (
        "native LLVM/JIT execution for supported scalar program AD traces with "
        "stable executed branch signatures and finite supported primitive domains; "
        "compiled batch value/gradient, JVP, and VJP execution for matching row "
        "signatures; "
        "unsupported control flow, mutation-dependent path changes, and unsupported "
        "operations fail closed"
    )

    def __post_init__(self) -> None:
        if not callable(self.objective):
            raise ValueError("objective must be callable")
        if not isinstance(self.source_result, WholeProgramADResult):
            raise ValueError("source_result must be a WholeProgramADResult")
        if not isinstance(self.mlir_module, MLIRModule):
            raise ValueError("mlir_module must be an MLIRModule")
        if not self.llvm_ir.strip():
            raise ValueError("llvm_ir must be non-empty")
        for name in (
            "value",
            "gradient",
            "jvp",
            "vjp",
            "batch_value_gradient",
            "batch_jvp",
            "batch_vjp",
            "engine",
        ):
            if name not in self.native_functions:
                raise ValueError(f"native_functions missing {name}")
        for name in (
            "value",
            "gradient",
            "jvp",
            "vjp",
            "batch_value_gradient",
            "batch_jvp",
            "batch_vjp",
        ):
            if not callable(self.native_functions[name]):
                raise ValueError(f"native function {name} must be callable")
        if not isinstance(self.verification, CompilerADKernelVerification):
            raise ValueError("verification must be CompilerADKernelVerification")
        if not self.verification.passed:
            raise ValueError("native whole-program AD kernel verification failed")
        if not self.parameters or any(
            not isinstance(parameter, Parameter) for parameter in self.parameters
        ):
            raise ValueError("parameters must be a non-empty tuple of Parameter objects")
        if self.parameter_names != tuple(parameter.name for parameter in self.parameters):
            raise ValueError("parameter_names must match parameters")
        if self.parameter_names != self.source_result.parameter_names:
            raise ValueError("parameter_names must match source_result")
        if self.parameter_shape != (len(self.parameters),):
            raise ValueError("parameter_shape must match parameter count")
        if self.source_result.gradient.shape != self.parameter_shape:
            raise ValueError("source_result gradient shape must match parameter_shape")
        if self.trace_signature != _whole_program_replay_signature(self.source_result):
            raise ValueError("trace_signature must match source_result")
        if any(not isinstance(item, str) or not item for item in self.supported_ops):
            raise ValueError("supported_ops entries must be non-empty strings")
        if not isinstance(self.lowering_report, WholeProgramADNativeLoweringReport):
            raise ValueError("lowering_report must be a WholeProgramADNativeLoweringReport")
        if not self.lowering_report.supported:
            raise ValueError("lowering_report must describe a supported native trace")
        if self.supported_ops != self.lowering_report.lowerable_ops:
            raise ValueError("supported_ops must match lowering_report lowerable_ops")
        if len(self.cache_key) != 64:
            raise ValueError("cache_key must be a sha256 hex digest")
        if not isinstance(self.cache_hit, bool):
            raise ValueError("cache_hit must be a bool")
        if self.backend != "native_llvm_jit":
            raise ValueError("backend must be 'native_llvm_jit'")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def _checked_values(self, values: Sequence[float] | np.ndarray) -> NDArray[np.float64]:
        checked = _as_finite_vector("values", values)
        if checked.shape != self.parameter_shape:
            raise ValueError(
                "values shape must match native whole-program AD parameter shape "
                f"{self.parameter_shape}"
            )
        return np.ascontiguousarray(checked, dtype=np.float64)

    def _checked_batch_values(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> NDArray[np.float64]:
        checked = np.asarray(values, dtype=np.float64)
        if checked.ndim != 2:
            raise ValueError("batch values must be two-dimensional")
        if checked.shape[0] < 1:
            raise ValueError("batch values must contain at least one row")
        if checked.shape[1:] != self.parameter_shape:
            raise ValueError(
                "batch values shape must be (batch, parameters) with parameter shape "
                f"{self.parameter_shape}"
            )
        if not np.all(np.isfinite(checked)):
            raise ValueError("batch values must contain only finite values")
        return np.ascontiguousarray(checked, dtype=np.float64)

    def _checked_batch_tangents(
        self,
        tangents: Sequence[Sequence[float]] | np.ndarray,
        row_count: int,
    ) -> NDArray[np.float64]:
        checked = np.asarray(tangents, dtype=np.float64)
        if checked.ndim != 2:
            raise ValueError("batch tangents must be two-dimensional")
        if checked.shape != (row_count, self.parameter_shape[0]):
            raise ValueError("batch tangents shape must match batch values shape")
        if not np.all(np.isfinite(checked)):
            raise ValueError("batch tangents must contain only finite values")
        return np.ascontiguousarray(checked, dtype=np.float64)

    @staticmethod
    def _checked_batch_cotangents(
        cotangents: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
        row_count: int,
    ) -> NDArray[np.float64]:
        checked = np.asarray(cotangents, dtype=np.float64)
        if checked.ndim == 2 and checked.shape[1:] == (1,):
            checked = checked.reshape(-1)
        if checked.ndim != 1:
            raise ValueError("batch cotangents must be one-dimensional")
        if checked.shape != (row_count,):
            raise ValueError("batch cotangents row count must match batch values")
        if not np.all(np.isfinite(checked)):
            raise ValueError("batch cotangents must contain only finite values")
        return np.ascontiguousarray(checked, dtype=np.float64)

    def _validate_trace_signature(self, values: NDArray[np.float64]) -> None:
        if not _whole_program_native_requires_runtime_recapture(self.source_result):
            return
        result = whole_program_value_and_grad(
            self.objective,
            values,
            parameters=self.parameters,
            trace=False,
        )
        signature = _whole_program_native_replay_signature(result)
        if signature != _whole_program_native_replay_signature(self.source_result):
            raise ValueError(
                "native whole-program AD kernel branch signature changed; "
                "recompile with representative sample values"
            )

    def value(self, values: Sequence[float] | np.ndarray) -> float:
        """Execute the native scalar value kernel."""

        checked = self._checked_values(values)
        self._validate_trace_signature(checked)
        output = _call_native_whole_program_unary(
            self.native_functions["value"],
            checked,
            1,
        )
        return float(output[0])

    def gradient(self, values: Sequence[float] | np.ndarray) -> NDArray[np.float64]:
        """Execute the native scalar-output gradient kernel."""

        checked = self._checked_values(values)
        self._validate_trace_signature(checked)
        return _call_native_whole_program_unary(
            self.native_functions["gradient"],
            checked,
            self.parameter_shape[0],
        )

    def value_and_grad(
        self,
        values: Sequence[float] | np.ndarray,
    ) -> tuple[float, NDArray[np.float64]]:
        """Execute native value and gradient kernels."""

        checked = self._checked_values(values)
        self._validate_trace_signature(checked)
        value = _call_native_whole_program_unary(
            self.native_functions["value"],
            checked,
            1,
        )
        gradient = _call_native_whole_program_unary(
            self.native_functions["gradient"],
            checked,
            self.parameter_shape[0],
        )
        return float(value[0]), gradient

    def jvp(
        self,
        values: Sequence[float] | np.ndarray,
        tangent: Sequence[float] | np.ndarray,
    ) -> float:
        """Execute the native scalar JVP kernel."""

        checked_values = self._checked_values(values)
        checked_tangent = _as_finite_vector("tangent", tangent)
        if checked_tangent.shape != self.parameter_shape:
            raise ValueError("tangent shape must match parameter_shape")
        self._validate_trace_signature(checked_values)
        output = _call_native_whole_program_binary(
            self.native_functions["jvp"],
            checked_values,
            checked_tangent,
            1,
        )
        return float(output[0])

    def vjp(
        self,
        values: Sequence[float] | np.ndarray,
        cotangent: Sequence[float] | np.ndarray,
    ) -> NDArray[np.float64]:
        """Execute the native scalar VJP kernel."""

        checked_cotangent = _as_finite_vector("cotangent", cotangent)
        if checked_cotangent.shape != (1,):
            raise ValueError("cotangent must contain exactly one scalar")
        checked_values = self._checked_values(values)
        self._validate_trace_signature(checked_values)
        return _call_native_whole_program_binary(
            self.native_functions["vjp"],
            checked_values,
            checked_cotangent,
            self.parameter_shape[0],
        )

    def batch_value_and_grad(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> ExecutableWholeProgramADBatchResult:
        """Execute native value and gradient kernels over a two-dimensional batch."""

        batch = self._checked_batch_values(values)
        for row in batch:
            self._validate_trace_signature(row)
        row_values, row_gradients = _call_native_whole_program_batch_value_gradient(
            self.native_functions["batch_value_gradient"],
            batch,
            self.parameter_shape[0],
        )
        return ExecutableWholeProgramADBatchResult(
            values=row_values,
            gradients=row_gradients,
            parameter_names=self.parameter_names,
            row_signatures=(self.trace_signature,) * batch.shape[0],
            mlir_sha256=self.mlir_module.sha256,
            backend=self.backend,
            claim_boundary=(
                "compiled batched native LLVM/JIT value/gradient execution for supported "
                "scalar program AD traces preserving compiled branch signatures and finite "
                "primitive domains"
            ),
        )

    def batch_value(self, values: Sequence[Sequence[float]] | np.ndarray) -> NDArray[np.float64]:
        """Execute native value kernels over a two-dimensional batch."""

        return self.batch_value_and_grad(values).values

    def batch_gradient(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> NDArray[np.float64]:
        """Execute native gradient kernels over a two-dimensional batch."""

        return self.batch_value_and_grad(values).gradients

    def batch_jvp(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
        tangents: Sequence[Sequence[float]] | np.ndarray,
    ) -> NDArray[np.float64]:
        """Execute the compiled native JVP kernel over a two-dimensional batch."""

        batch = self._checked_batch_values(values)
        checked_tangents = self._checked_batch_tangents(tangents, batch.shape[0])
        for row in batch:
            self._validate_trace_signature(row)
        return _call_native_whole_program_batch_jvp(
            self.native_functions["batch_jvp"],
            batch,
            checked_tangents,
            self.parameter_shape[0],
        )

    def batch_vjp(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
        cotangents: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    ) -> NDArray[np.float64]:
        """Execute the compiled native VJP kernel over a two-dimensional batch."""

        batch = self._checked_batch_values(values)
        checked_cotangents = self._checked_batch_cotangents(cotangents, batch.shape[0])
        for row in batch:
            self._validate_trace_signature(row)
        return _call_native_whole_program_batch_vjp(
            self.native_functions["batch_vjp"],
            batch,
            checked_cotangents,
            self.parameter_shape[0],
        )


@dataclass(frozen=True)
class ExecutableWholeProgramADKernel:
    """Executable replay kernel for a supported captured program AD trace.

    The kernel is intentionally bounded: it replays the original Python
    objective through the supported operator-intercepted program AD IR, checks
    the one-dimensional parameter shape, checks the captured control/signature
    surface, and computes gradients through reverse-mode adjoint replay. It is
    executable and deterministic for the supported captured trace contract; it
    does not claim arbitrary source compilation or native LLVM/JIT lowering for
    arbitrary Python programs.
    """

    objective: Callable[[Any], object]
    source_result: WholeProgramADResult
    parameters: tuple[Parameter, ...]
    mlir_module: MLIRModule
    parameter_names: tuple[str, ...]
    parameter_shape: tuple[int, ...]
    branch_signature: tuple[str, ...]
    backend: str = "program_ad_trace_replay"
    claim_boundary: str = (
        "executable replay of supported captured scalar program AD IR with "
        "deterministic MLIR provenance; branch/signature changes fail closed; "
        "no arbitrary source compiler or native LLVM/JIT claim"
    )

    def __post_init__(self) -> None:
        if not callable(self.objective):
            raise ValueError("objective must be callable")
        if not isinstance(self.source_result, WholeProgramADResult):
            raise ValueError("source_result must be a WholeProgramADResult")
        if not isinstance(self.mlir_module, MLIRModule):
            raise ValueError("mlir_module must be an MLIRModule")
        if not self.parameters or any(
            not isinstance(parameter, Parameter) for parameter in self.parameters
        ):
            raise ValueError("parameters must be a non-empty tuple of Parameter objects")
        if self.parameter_names != tuple(parameter.name for parameter in self.parameters):
            raise ValueError("parameter_names must match parameters")
        if self.parameter_names != self.source_result.parameter_names:
            raise ValueError("parameter_names must match source_result")
        if self.parameter_shape != (len(self.parameters),):
            raise ValueError("parameter_shape must match parameter count")
        if self.source_result.gradient.shape != self.parameter_shape:
            raise ValueError("source_result gradient shape must match parameter_shape")
        if any(not isinstance(item, str) or not item for item in self.branch_signature):
            raise ValueError("branch_signature entries must be non-empty strings")
        if self.branch_signature != _whole_program_replay_signature(self.source_result):
            raise ValueError("branch_signature must match source_result")
        if self.backend != "program_ad_trace_replay":
            raise ValueError("backend must be 'program_ad_trace_replay'")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")

    def _checked_values(self, values: Sequence[float] | np.ndarray) -> NDArray[np.float64]:
        checked = _as_finite_vector("values", values)
        if checked.shape != self.parameter_shape:
            raise ValueError(
                "values shape must match executable whole-program AD parameter shape "
                f"{self.parameter_shape}"
            )
        return checked

    def _checked_batch_values(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> NDArray[np.float64]:
        checked = np.asarray(values, dtype=np.float64)
        if checked.ndim != 2:
            raise ValueError("batch values must be two-dimensional")
        if checked.shape[0] < 1:
            raise ValueError("batch values must contain at least one row")
        if checked.shape[1:] != self.parameter_shape:
            raise ValueError(
                "batch values shape must be (batch, parameters) with parameter shape "
                f"{self.parameter_shape}"
            )
        if not np.all(np.isfinite(checked)):
            raise ValueError("batch values must contain only finite values")
        return cast(NDArray[np.float64], checked.copy())

    def _recapture(self, values: Sequence[float] | np.ndarray) -> WholeProgramADResult:
        checked = self._checked_values(values)
        result = whole_program_value_and_grad(
            self.objective,
            checked,
            parameters=self.parameters,
            trace=False,
        )
        signature = _whole_program_replay_signature(result)
        if signature != self.branch_signature:
            raise ValueError(
                "whole-program executable AD kernel branch signature changed; "
                "recompile with representative sample values"
            )
        return result

    def value_and_grad(
        self,
        values: Sequence[float] | np.ndarray,
    ) -> tuple[float, NDArray[np.float64]]:
        """Execute value replay and reverse-mode adjoint gradient replay."""

        result = self._recapture(values)
        return result.value, program_adjoint_gradient(result)

    def value(self, values: Sequence[float] | np.ndarray) -> float:
        """Execute value replay for the captured program AD trace."""

        return self.value_and_grad(values)[0]

    def gradient(self, values: Sequence[float] | np.ndarray) -> NDArray[np.float64]:
        """Execute reverse-mode adjoint replay for the captured program AD trace."""

        return self.value_and_grad(values)[1]

    def batch_value_and_grad(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> ExecutableWholeProgramADBatchResult:
        """Execute same-branch batched value and reverse-adjoint gradient replay."""

        batch = self._checked_batch_values(values)
        row_values: list[float] = []
        row_gradients: list[NDArray[np.float64]] = []
        row_signatures: list[tuple[str, ...]] = []
        for row_index, row in enumerate(batch):
            result = self._recapture(row)
            signature = _whole_program_replay_signature(result)
            if signature != self.branch_signature:
                raise ValueError(
                    f"whole-program executable AD batch row {row_index} branch signature changed"
                )
            row_values.append(result.value)
            row_gradients.append(program_adjoint_gradient(result))
            row_signatures.append(signature)
        return ExecutableWholeProgramADBatchResult(
            values=np.asarray(row_values, dtype=np.float64),
            gradients=np.vstack(row_gradients).astype(np.float64, copy=False),
            parameter_names=self.parameter_names,
            row_signatures=tuple(row_signatures),
            mlir_sha256=self.mlir_module.sha256,
        )

    def batch_value(self, values: Sequence[Sequence[float]] | np.ndarray) -> NDArray[np.float64]:
        """Execute batched value replay for rows preserving the compiled branch path."""

        return self.batch_value_and_grad(values).values

    def batch_gradient(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
    ) -> NDArray[np.float64]:
        """Execute batched reverse-adjoint replay for rows preserving the branch path."""

        return self.batch_value_and_grad(values).gradients


def compile_whole_program_ad_trace_to_executable(
    objective: Callable[[Any], object],
    sample_values: Sequence[float] | np.ndarray,
    parameters: Sequence[Parameter] | None = None,
    config: DifferentiableMLIRCompileConfig | None = None,
    *,
    trace: bool = True,
) -> ExecutableWholeProgramADKernel:
    """Compile a supported captured program AD trace to an executable replay kernel.

    This is the executable compiler boundary for whole-program AD today: it
    captures the supported scalar program IR, verifies reverse adjoint replay is
    available, emits deterministic MLIR provenance, then returns a fail-closed
    replay kernel. Shape drift, non-finite inputs, and branch/signature drift
    raise errors instead of silently changing the differentiated program.
    """

    if not callable(objective):
        raise ValueError("whole-program executable AD objective must be callable")
    checked_sample = _as_finite_vector("sample_values", sample_values)
    source_result = whole_program_value_and_grad(
        objective,
        checked_sample,
        parameters=parameters,
        trace=trace,
    )
    program_adjoint_gradient(source_result)
    compile_config = DifferentiableMLIRCompileConfig() if config is None else config
    mlir_module = compile_whole_program_ad_trace_to_mlir(source_result, compile_config)
    replay_parameters = tuple(
        Parameter(name, trainable=trainable)
        for name, trainable in zip(
            source_result.parameter_names,
            source_result.trainable,
            strict=True,
        )
    )
    return ExecutableWholeProgramADKernel(
        objective=objective,
        source_result=source_result,
        parameters=replay_parameters,
        mlir_module=mlir_module,
        parameter_names=source_result.parameter_names,
        parameter_shape=source_result.gradient.shape,
        branch_signature=_whole_program_replay_signature(source_result),
    )


def compile_whole_program_ad_trace_to_native_llvm_jit(
    objective: Callable[[Any], object],
    sample_values: Sequence[float] | np.ndarray,
    parameters: Sequence[Parameter] | None = None,
    config: DifferentiableMLIRCompileConfig | None = None,
    *,
    trace: bool = True,
) -> NativeWholeProgramADKernel:
    """Compile a supported scalar program AD trace to native LLVM/JIT kernels."""

    if not callable(objective):
        raise ValueError("whole-program native AD objective must be callable")
    checked_sample = _as_finite_vector("sample_values", sample_values)
    source_result = whole_program_value_and_grad(
        objective,
        checked_sample,
        parameters=parameters,
        trace=trace,
    )
    lowering_report = analyse_whole_program_ad_native_lowering(source_result)
    if not lowering_report.supported:
        raise ValueError(
            f"native whole-program AD lowering failed closed: {lowering_report.fail_closed_reason}"
        )
    program_adjoint_gradient(source_result)
    base_symbol = f"whole_program_ad_{source_result.gradient.size}_{source_result.evaluations}"
    base_symbol = f"{base_symbol}_{source_result.method}"
    base_symbol = _safe_llvm_symbol(base_symbol)
    llvm_ir = _compile_whole_program_ad_native_llvm_ir(source_result, base_symbol)
    compile_config = DifferentiableMLIRCompileConfig() if config is None else config
    cache_key = _native_whole_program_ad_cache_key(
        source_result,
        checked_sample,
        compile_config,
        llvm_ir,
    )
    cache_hit = False
    with _NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK:
        cached_entry = _NATIVE_WHOLE_PROGRAM_AD_CACHE.get(cache_key)
    if cached_entry is None:
        native_functions = _compile_native_llvm_jit_functions(llvm_ir, base_symbol)
        verification = _verify_native_whole_program_ad_kernel(
            source_result,
            native_functions,
            checked_sample,
        )
        mlir_module = _annotate_whole_program_native_mlir(
            compile_whole_program_ad_trace_to_mlir(source_result, compile_config),
            llvm_ir,
            source_result,
        )
        cached_entry = _NativeWholeProgramADCacheEntry(
            mlir_module=mlir_module,
            llvm_ir=llvm_ir,
            native_functions=native_functions,
            verification=verification,
            supported_ops=_whole_program_native_supported_ops(source_result),
            lowering_report=lowering_report,
        )
        _store_native_whole_program_ad_cache_entry(cache_key, cached_entry)
    else:
        cache_hit = True
    mlir_module = _with_native_whole_program_cache_metadata(
        cached_entry.mlir_module,
        cache_key=cache_key,
        cache_hit=cache_hit,
    )
    replay_parameters = tuple(
        Parameter(name, trainable=trainable)
        for name, trainable in zip(
            source_result.parameter_names,
            source_result.trainable,
            strict=True,
        )
    )
    return NativeWholeProgramADKernel(
        objective=objective,
        source_result=source_result,
        parameters=replay_parameters,
        mlir_module=mlir_module,
        llvm_ir=cached_entry.llvm_ir,
        native_functions=cached_entry.native_functions,
        verification=cached_entry.verification,
        parameter_names=source_result.parameter_names,
        parameter_shape=source_result.gradient.shape,
        trace_signature=_whole_program_replay_signature(source_result),
        supported_ops=cached_entry.supported_ops,
        lowering_report=cached_entry.lowering_report,
        cache_key=cache_key,
        cache_hit=cache_hit,
    )


def _whole_program_has_control_flow(result: WholeProgramADResult) -> bool:
    return any(node.op.startswith(("branch:", "loop:", "control:")) for node in result.ir_nodes)


def _whole_program_has_unsupported_native_control_flow(result: WholeProgramADResult) -> bool:
    return any(node.op.startswith(("loop:", "control:")) for node in result.ir_nodes)


_WHOLE_PROGRAM_NATIVE_STRUCTURAL_OPS = frozenset({"parameter", "constant"})
_WHOLE_PROGRAM_NATIVE_UNARY_OPS = frozenset(
    {
        "sin",
        "cos",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "tan",
        "tanh",
        "arcsin",
        "arccos",
        "reciprocal",
        "square",
        "abs",
        "neg",
        "negative",
    }
)
_WHOLE_PROGRAM_NATIVE_BINARY_OPS = frozenset(
    {
        "add",
        "sub",
        "subtract",
        "mul",
        "multiply",
        "div",
        "divide",
        "truediv",
        "pow",
        "power",
        "maximum",
        "minimum",
        "where",
    }
)


def analyse_whole_program_ad_native_lowering(
    result: WholeProgramADResult,
) -> WholeProgramADNativeLoweringReport:
    """Return the fail-closed native LLVM/JIT lowering audit for a program AD trace."""

    if not isinstance(result, WholeProgramADResult):
        raise ValueError("native lowering analysis requires a WholeProgramADResult")
    if not result.ir_nodes:
        raise ValueError("native lowering analysis requires captured IR nodes")
    lowerable_count = 0
    unsupported: list[str] = []
    lowerable: list[str] = []
    control_flow: list[str] = []
    for node in result.ir_nodes:
        if node.op.startswith(("branch:", "loop:", "control:")):
            control_flow.append(node.op)
        if _whole_program_native_node_is_lowerable(node.op):
            lowerable_count += 1
            lowerable.append(node.op)
        else:
            unsupported.append(node.op)
    unsupported_ops = tuple(dict.fromkeys(unsupported))
    lowerable_ops = tuple(dict.fromkeys(lowerable))
    effect_kinds: tuple[str, ...]
    if result.program_ir is None:
        effect_kinds = ()
    else:
        effect_kinds = tuple(dict.fromkeys(effect.kind for effect in result.program_ir.effects))
    if unsupported_ops:
        reason = "unsupported native ops: " + ", ".join(unsupported_ops)
    else:
        reason = "supported native LLVM/JIT lowering surface"
    return WholeProgramADNativeLoweringReport(
        supported=not unsupported_ops,
        lowerable_ops=lowerable_ops,
        unsupported_ops=unsupported_ops,
        control_flow_ops=tuple(dict.fromkeys(control_flow)),
        effect_kinds=effect_kinds,
        operation_count=len(result.ir_nodes),
        lowerable_operation_count=lowerable_count,
        unsupported_operation_count=len(result.ir_nodes) - lowerable_count,
        fail_closed_reason=reason,
    )


def _whole_program_native_node_is_lowerable(op: str) -> bool:
    if op in _WHOLE_PROGRAM_NATIVE_STRUCTURAL_OPS:
        return True
    if op in _WHOLE_PROGRAM_NATIVE_UNARY_OPS:
        return True
    if op in _WHOLE_PROGRAM_NATIVE_BINARY_OPS:
        return True
    return bool(op.startswith("branch:"))


def _whole_program_native_requires_runtime_recapture(result: WholeProgramADResult) -> bool:
    return _whole_program_has_control_flow(result) or any(
        node.op in {"maximum", "minimum", "where"} for node in result.ir_nodes
    )


def _whole_program_native_replay_signature(result: WholeProgramADResult) -> tuple[str, ...]:
    where_branch_ops = {
        branch_op
        for node in result.ir_nodes
        if node.op == "where" and node.inputs
        for branch_op in (_whole_program_native_where_branch_op(node.inputs[0]),)
        if branch_op is not None
    }
    control_signature = tuple(
        f"{node.index}:{node.op}:{','.join(node.inputs)}"
        for node in result.ir_nodes
        if node.op.startswith(("branch:", "loop:", "control:")) and node.op not in where_branch_ops
    )
    if control_signature:
        return control_signature
    return tuple(
        f"{node.index}:{node.op}:{','.join(_whole_program_native_signature_inputs(node))}"
        for node in result.ir_nodes
        if not (node.op.startswith("branch:") and node.op in where_branch_ops)
    )


def _whole_program_native_signature_inputs(node: Any) -> tuple[str, ...]:
    if node.op != "where" or not node.inputs:
        return tuple(node.inputs)
    return (
        _whole_program_native_where_predicate_body(node.inputs[0]),
        *tuple(node.inputs[1:]),
    )


def _whole_program_native_where_branch_op(predicate: str) -> str | None:
    if ":truth:" not in predicate:
        return None
    label, truth = predicate.rsplit(":truth:", 1)
    if truth == "1":
        return f"branch:{label}:True"
    if truth == "0":
        return f"branch:{label}:False"
    return None


def _whole_program_native_supported_ops(result: WholeProgramADResult) -> tuple[str, ...]:
    return analyse_whole_program_ad_native_lowering(result).lowerable_ops


def _native_whole_program_ad_cache_key(
    result: WholeProgramADResult,
    sample_values: NDArray[np.float64],
    config: DifferentiableMLIRCompileConfig,
    llvm_ir: str,
) -> str:
    payload = {
        "format": "native_whole_program_ad_cache.v1",
        "parameter_names": result.parameter_names,
        "trainable": result.trainable,
        "trace_signature": _whole_program_replay_signature(result),
        "method": result.method,
        "evaluations": result.evaluations,
        "ir_nodes": [
            {
                "index": node.index,
                "op": node.op,
                "inputs": node.inputs,
                "value": _fmt_llvm_float(node.value),
            }
            for node in result.ir_nodes
        ],
        "sample_values": [_fmt_llvm_float(value) for value in sample_values],
        "config": _jsonable_cache_payload(config),
        "llvm_ir_sha256": hashlib.sha256(llvm_ir.encode("utf-8")).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _jsonable_cache_payload(value: object) -> object:
    if is_dataclass(value):
        return _jsonable_cache_payload(vars(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable_cache_payload(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable_cache_payload(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable_cache_payload(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable_cache_payload(value.item())
    if isinstance(value, float):
        return _fmt_llvm_float(value)
    if isinstance(value, str | int | bool) or value is None:
        return value
    return repr(value)


def _store_native_whole_program_ad_cache_entry(
    cache_key: str,
    entry: _NativeWholeProgramADCacheEntry,
) -> None:
    with _NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK:
        if cache_key in _NATIVE_WHOLE_PROGRAM_AD_CACHE:
            return
        if len(_NATIVE_WHOLE_PROGRAM_AD_CACHE) >= _NATIVE_WHOLE_PROGRAM_AD_CACHE_MAXSIZE:
            oldest_key = next(iter(_NATIVE_WHOLE_PROGRAM_AD_CACHE))
            del _NATIVE_WHOLE_PROGRAM_AD_CACHE[oldest_key]
        _NATIVE_WHOLE_PROGRAM_AD_CACHE[cache_key] = entry


def native_whole_program_ad_compile_cache_stats() -> Mapping[str, object]:
    """Return bounded process-local native whole-program AD compile-cache metadata."""

    with _NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK:
        return MappingProxyType(
            {
                "entries": len(_NATIVE_WHOLE_PROGRAM_AD_CACHE),
                "max_size": _NATIVE_WHOLE_PROGRAM_AD_CACHE_MAXSIZE,
                "keys": tuple(_NATIVE_WHOLE_PROGRAM_AD_CACHE.keys()),
            }
        )


def clear_native_whole_program_ad_compile_cache() -> int:
    """Clear verified native whole-program AD compile-cache entries and return count."""

    with _NATIVE_WHOLE_PROGRAM_AD_CACHE_LOCK:
        removed = len(_NATIVE_WHOLE_PROGRAM_AD_CACHE)
        _NATIVE_WHOLE_PROGRAM_AD_CACHE.clear()
        return removed


def _with_native_whole_program_cache_metadata(
    module: MLIRModule,
    *,
    cache_key: str,
    cache_hit: bool,
) -> MLIRModule:
    metadata = dict(module.metadata)
    metadata["native_compile_cache_key"] = cache_key
    metadata["native_compile_cache_hit"] = cache_hit
    resource_counts = dict(module.resource_counts)
    resource_counts["native_compile_cache_hit"] = int(cache_hit)
    return MLIRModule(
        text=module.text,
        sha256=module.sha256,
        dialect=module.dialect,
        resource_counts=resource_counts,
        metadata=metadata,
    )


def _compile_whole_program_ad_native_llvm_ir(
    result: WholeProgramADResult,
    base_symbol: str,
) -> str:
    if result.gradient.ndim != 1 or result.gradient.size < 1:
        raise ValueError("native whole-program AD lowering requires parameters")
    lowering_report = analyse_whole_program_ad_native_lowering(result)
    if not lowering_report.supported:
        raise ValueError(
            f"native whole-program AD lowering failed closed: {lowering_report.fail_closed_reason}"
        )
    computation_lines, final_value, final_derivatives = _emit_whole_program_native_computation(
        result,
        values_pointer="%values",
    )
    (
        batch_computation_lines,
        batch_final_value,
        batch_final_derivatives,
    ) = _emit_whole_program_native_computation(
        result,
        values_pointer="%row_values",
    )

    lines = [
        "declare double @llvm.sin.f64(double)",
        "declare double @llvm.cos.f64(double)",
        "declare double @llvm.exp.f64(double)",
        "declare double @llvm.log.f64(double)",
        "declare double @llvm.sqrt.f64(double)",
        "declare double @llvm.pow.f64(double, double)",
        "declare double @llvm.asin.f64(double)",
        "declare double @llvm.acos.f64(double)",
        "",
        f"define void @{base_symbol}_value(double* %values, double* %out) {{",
        *computation_lines,
        "  %value_out_ptr = getelementptr double, double* %out, i64 0",
        f"  store double {final_value}, double* %value_out_ptr",
        "  ret void",
        "}",
        "",
        f"define void @{base_symbol}_gradient(double* %values, double* %out) {{",
        *computation_lines,
    ]
    for index, derivative in enumerate(final_derivatives):
        lines.extend(
            [
                f"  %gradient_out_ptr_{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double {derivative}, double* %gradient_out_ptr_{index}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_jvp(double* %values, double* %tangent, double* %out) {{",
            *computation_lines,
        ]
    )
    jvp_accumulator = _fmt_llvm_float(0.0)
    for index, derivative in enumerate(final_derivatives):
        tangent_ptr = f"%jvp_tangent_ptr_{index}"
        tangent_value = f"%jvp_tangent_{index}"
        term = f"%jvp_term_{index}"
        accumulator = f"%jvp_acc_{index}"
        lines.extend(
            [
                f"  {tangent_ptr} = getelementptr double, double* %tangent, i64 {index}",
                f"  {tangent_value} = load double, double* {tangent_ptr}",
                f"  {term} = fmul double {derivative}, {tangent_value}",
                f"  {accumulator} = fadd double {jvp_accumulator}, {term}",
            ]
        )
        jvp_accumulator = accumulator
    lines.extend(
        [
            "  %jvp_out_ptr = getelementptr double, double* %out, i64 0",
            f"  store double {jvp_accumulator}, double* %jvp_out_ptr",
            "  ret void",
            "}",
            "",
            f"define void @{base_symbol}_vjp(double* %values, double* %cotangent, double* %out) {{",
            *computation_lines,
            "  %vjp_cotangent_ptr = getelementptr double, double* %cotangent, i64 0",
            "  %vjp_cotangent = load double, double* %vjp_cotangent_ptr",
        ]
    )
    for index, derivative in enumerate(final_derivatives):
        vjp_value = f"%vjp_value_{index}"
        lines.extend(
            [
                f"  {vjp_value} = fmul double {derivative}, %vjp_cotangent",
                f"  %vjp_out_ptr_{index} = getelementptr double, double* %out, i64 {index}",
                f"  store double {vjp_value}, double* %vjp_out_ptr_{index}",
            ]
        )
    lines.extend(
        [
            "  ret void",
            "}",
            "",
            (
                f"define void @{base_symbol}_batch_value_gradient(double* %values, "
                "i64 %rows, double* %value_out, double* %gradient_out) {"
            ),
            "entry:",
            "  br label %batch_loop",
            "batch_loop:",
            "  %batch_i = phi i64 [0, %entry], [%batch_next, %batch_continue]",
            "  %batch_done = icmp eq i64 %batch_i, %rows",
            "  br i1 %batch_done, label %batch_exit, label %batch_body",
            "batch_body:",
            (
                f"  %batch_row_offset = mul i64 %batch_i, "
                f"{_fmt_llvm_int(len(result.parameter_names))}"
            ),
            "  %row_values = getelementptr double, double* %values, i64 %batch_row_offset",
            *batch_computation_lines,
            "  %batch_value_out_ptr = getelementptr double, double* %value_out, i64 %batch_i",
            f"  store double {batch_final_value}, double* %batch_value_out_ptr",
            (
                f"  %batch_gradient_row_offset = mul i64 %batch_i, "
                f"{_fmt_llvm_int(len(result.parameter_names))}"
            ),
        ]
    )
    for index, derivative in enumerate(batch_final_derivatives):
        lines.extend(
            [
                f"  %batch_gradient_offset_{index} = add i64 %batch_gradient_row_offset, {index}",
                (
                    f"  %batch_gradient_out_ptr_{index} = getelementptr double, "
                    f"double* %gradient_out, i64 %batch_gradient_offset_{index}"
                ),
                f"  store double {derivative}, double* %batch_gradient_out_ptr_{index}",
            ]
        )
    lines.extend(
        [
            "  br label %batch_continue",
            "batch_continue:",
            "  %batch_next = add i64 %batch_i, 1",
            "  br label %batch_loop",
            "batch_exit:",
            "  ret void",
            "}",
            "",
        ]
    )
    lines.extend(
        _emit_whole_program_native_batch_jvp(
            result,
            base_symbol,
            batch_computation_lines,
            batch_final_derivatives,
        )
    )
    lines.extend(
        _emit_whole_program_native_batch_vjp(
            result,
            base_symbol,
            batch_computation_lines,
            batch_final_derivatives,
        )
    )
    return "\n".join(lines)


def _emit_whole_program_native_batch_jvp(
    result: WholeProgramADResult,
    base_symbol: str,
    computation_lines: Sequence[str],
    final_derivatives: Sequence[str],
) -> list[str]:
    parameter_count = len(result.parameter_names)
    lines = [
        (
            f"define void @{base_symbol}_batch_jvp(double* %values, double* %tangents, "
            "i64 %rows, double* %out) {"
        ),
        "entry:",
        "  br label %batch_jvp_loop",
        "batch_jvp_loop:",
        "  %batch_jvp_i = phi i64 [0, %entry], [%batch_jvp_next, %batch_jvp_continue]",
        "  %batch_jvp_done = icmp eq i64 %batch_jvp_i, %rows",
        "  br i1 %batch_jvp_done, label %batch_jvp_exit, label %batch_jvp_body",
        "batch_jvp_body:",
        f"  %batch_jvp_row_offset = mul i64 %batch_jvp_i, {_fmt_llvm_int(parameter_count)}",
        "  %row_values = getelementptr double, double* %values, i64 %batch_jvp_row_offset",
        "  %row_tangents = getelementptr double, double* %tangents, i64 %batch_jvp_row_offset",
        *computation_lines,
    ]
    accumulator = _fmt_llvm_float(0.0)
    for index, derivative in enumerate(final_derivatives):
        tangent_ptr = f"%batch_jvp_tangent_ptr_{index}"
        tangent_value = f"%batch_jvp_tangent_{index}"
        term = f"%batch_jvp_term_{index}"
        next_accumulator = f"%batch_jvp_acc_{index}"
        lines.extend(
            [
                f"  {tangent_ptr} = getelementptr double, double* %row_tangents, i64 {index}",
                f"  {tangent_value} = load double, double* {tangent_ptr}",
                f"  {term} = fmul double {derivative}, {tangent_value}",
                f"  {next_accumulator} = fadd double {accumulator}, {term}",
            ]
        )
        accumulator = next_accumulator
    lines.extend(
        [
            "  %batch_jvp_out_ptr = getelementptr double, double* %out, i64 %batch_jvp_i",
            f"  store double {accumulator}, double* %batch_jvp_out_ptr",
            "  br label %batch_jvp_continue",
            "batch_jvp_continue:",
            "  %batch_jvp_next = add i64 %batch_jvp_i, 1",
            "  br label %batch_jvp_loop",
            "batch_jvp_exit:",
            "  ret void",
            "}",
            "",
        ]
    )
    return lines


def _emit_whole_program_native_batch_vjp(
    result: WholeProgramADResult,
    base_symbol: str,
    computation_lines: Sequence[str],
    final_derivatives: Sequence[str],
) -> list[str]:
    parameter_count = len(result.parameter_names)
    lines = [
        (
            f"define void @{base_symbol}_batch_vjp(double* %values, double* %cotangents, "
            "i64 %rows, double* %out) {"
        ),
        "entry:",
        "  br label %batch_vjp_loop",
        "batch_vjp_loop:",
        "  %batch_vjp_i = phi i64 [0, %entry], [%batch_vjp_next, %batch_vjp_continue]",
        "  %batch_vjp_done = icmp eq i64 %batch_vjp_i, %rows",
        "  br i1 %batch_vjp_done, label %batch_vjp_exit, label %batch_vjp_body",
        "batch_vjp_body:",
        f"  %batch_vjp_row_offset = mul i64 %batch_vjp_i, {_fmt_llvm_int(parameter_count)}",
        "  %row_values = getelementptr double, double* %values, i64 %batch_vjp_row_offset",
        *computation_lines,
        "  %batch_vjp_cotangent_ptr = getelementptr double, double* %cotangents, i64 %batch_vjp_i",
        "  %batch_vjp_cotangent = load double, double* %batch_vjp_cotangent_ptr",
        f"  %batch_vjp_gradient_row_offset = mul i64 %batch_vjp_i, {_fmt_llvm_int(parameter_count)}",
    ]
    for index, derivative in enumerate(final_derivatives):
        vjp_value = f"%batch_vjp_value_{index}"
        lines.extend(
            [
                f"  {vjp_value} = fmul double {derivative}, %batch_vjp_cotangent",
                f"  %batch_vjp_offset_{index} = add i64 %batch_vjp_gradient_row_offset, {index}",
                (
                    f"  %batch_vjp_out_ptr_{index} = getelementptr double, "
                    f"double* %out, i64 %batch_vjp_offset_{index}"
                ),
                f"  store double {vjp_value}, double* %batch_vjp_out_ptr_{index}",
            ]
        )
    lines.extend(
        [
            "  br label %batch_vjp_continue",
            "batch_vjp_continue:",
            "  %batch_vjp_next = add i64 %batch_vjp_i, 1",
            "  br label %batch_vjp_loop",
            "batch_vjp_exit:",
            "  ret void",
            "}",
            "",
        ]
    )
    return lines


def _emit_whole_program_native_computation(
    result: WholeProgramADResult,
    *,
    values_pointer: str,
) -> tuple[list[str], str, tuple[str, ...]]:
    parameter_count = int(result.gradient.size)
    lines: list[str] = []
    names = set(result.parameter_names)
    for node in result.ir_nodes:
        value_name = _whole_program_native_value_name(node.index)
        if node.op == "parameter":
            if len(node.inputs) != 1 or node.inputs[0] not in names:
                raise ValueError("native whole-program AD parameter node is malformed")
            parameter_index = result.parameter_names.index(node.inputs[0])
            lines.extend(
                [
                    (
                        f"  %param_ptr_{node.index} = getelementptr double, "
                        f"double* {values_pointer}, i64 {parameter_index}"
                    ),
                    f"  {value_name} = load double, double* %param_ptr_{node.index}",
                ]
            )
            for derivative_index in range(parameter_count):
                seed = (
                    1.0
                    if result.trainable[parameter_index] and derivative_index == parameter_index
                    else 0.0
                )
                lines.append(
                    f"  {_whole_program_native_derivative_name(node.index, derivative_index)} = "
                    f"fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(seed)}"
                )
            continue
        if node.op == "constant":
            lines.append(
                f"  {value_name} = fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(node.value)}"
            )
            for derivative_index in range(parameter_count):
                lines.append(
                    f"  {_whole_program_native_derivative_name(node.index, derivative_index)} = "
                    f"fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(0.0)}"
                )
            continue
        if node.op.startswith("branch:"):
            if node.inputs:
                raise ValueError("native whole-program AD branch node must be signature-only")
            lines.append(
                f"  {value_name} = fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(node.value)}"
            )
            for derivative_index in range(parameter_count):
                lines.append(
                    f"  {_whole_program_native_derivative_name(node.index, derivative_index)} = "
                    f"fadd double {_fmt_llvm_float(0.0)}, {_fmt_llvm_float(0.0)}"
                )
            continue
        _emit_whole_program_native_operation(lines, result, node)
    if not result.ir_nodes:
        raise ValueError("native whole-program AD lowering requires IR nodes")
    final_index = result.ir_nodes[-1].index
    return (
        lines,
        _whole_program_native_value_name(final_index),
        tuple(
            _whole_program_native_derivative_name(final_index, index)
            for index in range(parameter_count)
        ),
    )


def _emit_whole_program_native_operation(
    lines: list[str],
    result: WholeProgramADResult,
    node: Any,
) -> None:
    parameter_count = int(result.gradient.size)
    value_name = _whole_program_native_value_name(node.index)
    inputs = tuple(node.inputs)
    if node.op in {
        "sin",
        "cos",
        "exp",
        "expm1",
        "log",
        "log1p",
        "sqrt",
        "tan",
        "tanh",
        "arcsin",
        "arccos",
        "reciprocal",
        "square",
        "abs",
        "neg",
        "negative",
    }:
        if len(inputs) != 1:
            raise ValueError(f"native operation {node.op} expects one input")
        argument = _whole_program_native_operand(inputs[0])
        if node.op == "sin":
            lines.append(f"  {value_name} = call double @llvm.sin.f64(double {argument})")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = call double @llvm.cos.f64(double {argument})")
        elif node.op == "cos":
            cos_value = value_name
            sin_value = f"%sin_{node.index}"
            lines.append(f"  {cos_value} = call double @llvm.cos.f64(double {argument})")
            lines.append(f"  {sin_value} = call double @llvm.sin.f64(double {argument})")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fsub double {_fmt_llvm_float(0.0)}, {sin_value}")
        elif node.op == "exp":
            lines.append(f"  {value_name} = call double @llvm.exp.f64(double {argument})")
            local_factor = value_name
        elif node.op == "expm1":
            exp_value = f"%exp_{node.index}"
            lines.append(f"  {exp_value} = call double @llvm.exp.f64(double {argument})")
            lines.append(f"  {value_name} = fsub double {exp_value}, {_fmt_llvm_float(1.0)}")
            local_factor = exp_value
        elif node.op == "log":
            lines.append(f"  {value_name} = call double @llvm.log.f64(double {argument})")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {argument}")
        elif node.op == "log1p":
            shifted = f"%log1p_shifted_{node.index}"
            lines.append(f"  {shifted} = fadd double {_fmt_llvm_float(1.0)}, {argument}")
            lines.append(f"  {value_name} = call double @llvm.log.f64(double {shifted})")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {shifted}")
        elif node.op == "sqrt":
            lines.append(f"  {value_name} = call double @llvm.sqrt.f64(double {argument})")
            local_factor = f"%factor_{node.index}"
            denominator = f"%sqrt_denominator_{node.index}"
            lines.append(f"  {denominator} = fmul double {_fmt_llvm_float(2.0)}, {value_name}")
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {denominator}")
        elif node.op == "tan":
            sin_value = f"%tan_sin_{node.index}"
            cos_value = f"%tan_cos_{node.index}"
            cos_squared = f"%tan_cos_squared_{node.index}"
            lines.extend(
                [
                    f"  {sin_value} = call double @llvm.sin.f64(double {argument})",
                    f"  {cos_value} = call double @llvm.cos.f64(double {argument})",
                    f"  {value_name} = fdiv double {sin_value}, {cos_value}",
                    f"  {cos_squared} = fmul double {cos_value}, {cos_value}",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {cos_squared}")
        elif node.op == "tanh":
            doubled = f"%tanh_doubled_{node.index}"
            exp_value = f"%tanh_exp_{node.index}"
            numerator = f"%tanh_numerator_{node.index}"
            denominator = f"%tanh_denominator_{node.index}"
            squared = f"%tanh_squared_{node.index}"
            lines.extend(
                [
                    f"  {doubled} = fmul double {_fmt_llvm_float(2.0)}, {argument}",
                    f"  {exp_value} = call double @llvm.exp.f64(double {doubled})",
                    f"  {numerator} = fsub double {exp_value}, {_fmt_llvm_float(1.0)}",
                    f"  {denominator} = fadd double {exp_value}, {_fmt_llvm_float(1.0)}",
                    f"  {value_name} = fdiv double {numerator}, {denominator}",
                    f"  {squared} = fmul double {value_name}, {value_name}",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fsub double {_fmt_llvm_float(1.0)}, {squared}")
        elif node.op == "arcsin":
            argument_squared = f"%arcsin_argument_squared_{node.index}"
            radicand = f"%arcsin_radicand_{node.index}"
            root = f"%arcsin_root_{node.index}"
            lines.extend(
                [
                    f"  {value_name} = call double @llvm.asin.f64(double {argument})",
                    f"  {argument_squared} = fmul double {argument}, {argument}",
                    f"  {radicand} = fsub double {_fmt_llvm_float(1.0)}, {argument_squared}",
                    f"  {root} = call double @llvm.sqrt.f64(double {radicand})",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(1.0)}, {root}")
        elif node.op == "arccos":
            argument_squared = f"%arccos_argument_squared_{node.index}"
            radicand = f"%arccos_radicand_{node.index}"
            root = f"%arccos_root_{node.index}"
            positive_factor = f"%arccos_positive_factor_{node.index}"
            lines.extend(
                [
                    f"  {value_name} = call double @llvm.acos.f64(double {argument})",
                    f"  {argument_squared} = fmul double {argument}, {argument}",
                    f"  {radicand} = fsub double {_fmt_llvm_float(1.0)}, {argument_squared}",
                    f"  {root} = call double @llvm.sqrt.f64(double {radicand})",
                    f"  {positive_factor} = fdiv double {_fmt_llvm_float(1.0)}, {root}",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(
                f"  {local_factor} = fsub double {_fmt_llvm_float(0.0)}, {positive_factor}"
            )
        elif node.op == "reciprocal":
            denominator = f"%reciprocal_denominator_{node.index}"
            lines.extend(
                [
                    f"  {value_name} = fdiv double {_fmt_llvm_float(1.0)}, {argument}",
                    f"  {denominator} = fmul double {argument}, {argument}",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {_fmt_llvm_float(-1.0)}, {denominator}")
        elif node.op == "square":
            lines.append(f"  {value_name} = fmul double {argument}, {argument}")
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fmul double {_fmt_llvm_float(2.0)}, {argument}")
        elif node.op == "abs":
            squared = f"%abs_squared_{node.index}"
            lines.extend(
                [
                    f"  {squared} = fmul double {argument}, {argument}",
                    f"  {value_name} = call double @llvm.sqrt.f64(double {squared})",
                ]
            )
            local_factor = f"%factor_{node.index}"
            lines.append(f"  {local_factor} = fdiv double {argument}, {value_name}")
        else:
            lines.append(f"  {value_name} = fsub double {_fmt_llvm_float(0.0)}, {argument}")
            local_factor = _fmt_llvm_float(-1.0)
        for derivative_index in range(parameter_count):
            argument_derivative = _whole_program_native_derivative_operand(
                inputs[0], derivative_index
            )
            lines.append(
                f"  {_whole_program_native_derivative_name(node.index, derivative_index)} = "
                f"fmul double {local_factor}, {argument_derivative}"
            )
        return
    if node.op not in {
        "add",
        "sub",
        "subtract",
        "mul",
        "multiply",
        "div",
        "divide",
        "truediv",
        "pow",
        "power",
        "maximum",
        "minimum",
        "where",
    }:
        raise ValueError(f"native whole-program AD lowering does not support op {node.op}")
    if node.op == "where":
        if len(inputs) != 3:
            raise ValueError(
                "native operation where expects predicate, true value, and false value"
            )
        predicate = _emit_whole_program_native_where_predicate(lines, node.index, inputs[0])
        true_value = _whole_program_native_operand(inputs[1])
        false_value = _whole_program_native_operand(inputs[2])
        lines.append(
            f"  {value_name} = select i1 {predicate}, double {true_value}, double {false_value}"
        )
        for derivative_index in range(parameter_count):
            true_derivative = _whole_program_native_derivative_operand(inputs[1], derivative_index)
            false_derivative = _whole_program_native_derivative_operand(
                inputs[2], derivative_index
            )
            derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
            lines.append(
                f"  {derivative_name} = select i1 {predicate}, "
                f"double {true_derivative}, double {false_derivative}"
            )
        return
    if len(inputs) != 2:
        raise ValueError(f"native operation {node.op} expects two inputs")
    left = _whole_program_native_operand(inputs[0])
    right = _whole_program_native_operand(inputs[1])
    if node.op == "add":
        lines.append(f"  {value_name} = fadd double {left}, {right}")
    elif node.op in {"sub", "subtract"}:
        lines.append(f"  {value_name} = fsub double {left}, {right}")
    elif node.op in {"mul", "multiply"}:
        lines.append(f"  {value_name} = fmul double {left}, {right}")
    elif node.op in {"div", "divide", "truediv"}:
        lines.append(f"  {value_name} = fdiv double {left}, {right}")
    elif node.op == "maximum":
        predicate = f"%select_pred_{node.index}"
        lines.append(f"  {predicate} = fcmp ogt double {left}, {right}")
        lines.append(f"  {value_name} = select i1 {predicate}, double {left}, double {right}")
    elif node.op == "minimum":
        predicate = f"%select_pred_{node.index}"
        lines.append(f"  {predicate} = fcmp olt double {left}, {right}")
        lines.append(f"  {value_name} = select i1 {predicate}, double {left}, double {right}")
    else:
        exponent = _whole_program_native_constant(inputs[1])
        if exponent is None:
            raise ValueError("native whole-program AD pow lowering requires constant exponent")
        lines.append(f"  {value_name} = call double @llvm.pow.f64(double {left}, double {right})")
        pow_factor = f"%pow_factor_{node.index}"
        exponent_minus_one = _fmt_llvm_float(exponent - 1.0)
        lines.append(
            f"  {pow_factor} = call double @llvm.pow.f64(double {left}, double {exponent_minus_one})"
        )
        scaled_factor = f"%pow_scaled_factor_{node.index}"
        lines.append(f"  {scaled_factor} = fmul double {_fmt_llvm_float(exponent)}, {pow_factor}")
    for derivative_index in range(parameter_count):
        left_derivative = _whole_program_native_derivative_operand(inputs[0], derivative_index)
        right_derivative = _whole_program_native_derivative_operand(inputs[1], derivative_index)
        derivative_name = _whole_program_native_derivative_name(node.index, derivative_index)
        if node.op == "add":
            lines.append(
                f"  {derivative_name} = fadd double {left_derivative}, {right_derivative}"
            )
        elif node.op in {"sub", "subtract"}:
            lines.append(
                f"  {derivative_name} = fsub double {left_derivative}, {right_derivative}"
            )
        elif node.op in {"mul", "multiply"}:
            left_term = f"%d{node.index}_{derivative_index}_left_mul"
            right_term = f"%d{node.index}_{derivative_index}_right_mul"
            lines.extend(
                [
                    f"  {left_term} = fmul double {left_derivative}, {right}",
                    f"  {right_term} = fmul double {left}, {right_derivative}",
                    f"  {derivative_name} = fadd double {left_term}, {right_term}",
                ]
            )
        elif node.op in {"div", "divide", "truediv"}:
            left_term = f"%d{node.index}_{derivative_index}_left_div"
            right_term = f"%d{node.index}_{derivative_index}_right_div"
            numerator = f"%d{node.index}_{derivative_index}_div_num"
            denominator = f"%d{node.index}_{derivative_index}_div_den"
            lines.extend(
                [
                    f"  {left_term} = fmul double {left_derivative}, {right}",
                    f"  {right_term} = fmul double {left}, {right_derivative}",
                    f"  {numerator} = fsub double {left_term}, {right_term}",
                    f"  {denominator} = fmul double {right}, {right}",
                    f"  {derivative_name} = fdiv double {numerator}, {denominator}",
                ]
            )
        elif node.op in {"maximum", "minimum"}:
            predicate = f"%select_pred_{node.index}"
            lines.append(
                f"  {derivative_name} = select i1 {predicate}, "
                f"double {left_derivative}, double {right_derivative}"
            )
        else:
            lines.append(f"  {derivative_name} = fmul double {scaled_factor}, {left_derivative}")


def _emit_whole_program_native_where_predicate(
    lines: list[str],
    node_index: int,
    predicate: str,
) -> str:
    predicate_body = _whole_program_native_where_predicate_body(predicate)
    parts = predicate_body.split(":")
    if len(parts) != 3:
        raise ValueError("native whole-program AD where predicate is malformed")
    left_token, op, right_token = parts
    predicate_name = f"%where_pred_{node_index}"
    llvm_predicate = {
        "gt": "ogt",
        "ge": "oge",
        "lt": "olt",
        "le": "ole",
        "eq": "oeq",
        "ne": "one",
    }.get(op)
    if llvm_predicate is None:
        raise ValueError(f"native whole-program AD where predicate op {op} is unsupported")
    left = _whole_program_native_operand(left_token)
    right = _whole_program_native_operand(right_token)
    lines.append(f"  {predicate_name} = fcmp {llvm_predicate} double {left}, {right}")
    return predicate_name


def _whole_program_native_where_predicate_body(predicate: str) -> str:
    if ":truth:" not in predicate:
        raise ValueError("native whole-program AD where predicate requires recorded truth")
    body, truth = predicate.rsplit(":truth:", 1)
    if truth not in {"0", "1"}:
        raise ValueError("native whole-program AD where predicate truth must be 0 or 1")
    return body


def _whole_program_native_operand(token: str) -> str:
    if _whole_program_native_is_ir_value(token):
        return _whole_program_native_value_name(int(token[1:]))
    constant = _whole_program_native_constant(token)
    if constant is None:
        raise ValueError(f"native whole-program AD cannot lower operand {token}")
    return _fmt_llvm_float(constant)


def _whole_program_native_derivative_operand(token: str, derivative_index: int) -> str:
    if _whole_program_native_is_ir_value(token):
        return _whole_program_native_derivative_name(int(token[1:]), derivative_index)
    constant = _whole_program_native_constant(token)
    if constant is None:
        raise ValueError(f"native whole-program AD cannot lower derivative operand {token}")
    return _fmt_llvm_float(0.0)


def _whole_program_native_constant(token: str) -> float | None:
    try:
        return float(token)
    except (TypeError, ValueError):
        return None


def _whole_program_native_is_ir_value(token: str) -> bool:
    return token.startswith("%") and token[1:].isdigit()


def _whole_program_native_value_name(index: int) -> str:
    return f"%n{index}"


def _whole_program_native_derivative_name(node_index: int, derivative_index: int) -> str:
    return f"%d{node_index}_{derivative_index}"


def _fmt_llvm_float(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("LLVM numeric constants must be finite")
    return format(float(value), ".17e")


def _fmt_llvm_int(value: int) -> str:
    if int(value) < 1:
        raise ValueError("LLVM integer constants must be positive")
    return str(int(value))


def _call_native_whole_program_unary(
    function: Callable[[Any, Any], None],
    values: np.ndarray,
    output_size: int,
) -> NDArray[np.float64]:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    if output_size < 1:
        raise ValueError("native whole-program AD output_size must be positive")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("native whole-program AD output must be finite")
    return cast(NDArray[np.float64], output)


def _call_native_whole_program_binary(
    function: Callable[[Any, Any, Any], None],
    values: np.ndarray,
    vector: np.ndarray,
    output_size: int,
) -> NDArray[np.float64]:
    checked_values = np.ascontiguousarray(_as_finite_vector("values", values), dtype=np.float64)
    checked_vector = np.ascontiguousarray(_as_finite_vector("vector", vector), dtype=np.float64)
    if output_size < 1:
        raise ValueError("native whole-program AD output_size must be positive")
    output = np.zeros(output_size, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_vector.ctypes.data_as(double_pointer),
        output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("native whole-program AD output must be finite")
    return cast(NDArray[np.float64], output)


def _call_native_whole_program_batch_value_gradient(
    function: Callable[[Any, int, Any, Any], None],
    values: np.ndarray,
    parameter_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    checked_values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if checked_values.ndim != 2:
        raise ValueError("native whole-program AD batch values must be two-dimensional")
    if checked_values.shape[0] < 1:
        raise ValueError("native whole-program AD batch values must contain at least one row")
    if checked_values.shape[1] != parameter_count:
        raise ValueError("native whole-program AD batch parameter count mismatch")
    if not np.all(np.isfinite(checked_values)):
        raise ValueError("native whole-program AD batch values must be finite")
    rows = int(checked_values.shape[0])
    value_output = np.zeros(rows, dtype=np.float64)
    gradient_output = np.zeros((rows, parameter_count), dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        rows,
        value_output.ctypes.data_as(double_pointer),
        gradient_output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(value_output)) or not np.all(np.isfinite(gradient_output)):
        raise ValueError("native whole-program AD batch output must be finite")
    return (
        cast(NDArray[np.float64], value_output),
        cast(NDArray[np.float64], gradient_output),
    )


def _call_native_whole_program_batch_jvp(
    function: Callable[[Any, Any, int, Any], None],
    values: np.ndarray,
    tangents: np.ndarray,
    parameter_count: int,
) -> NDArray[np.float64]:
    checked_values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    checked_tangents = np.ascontiguousarray(np.asarray(tangents, dtype=np.float64))
    if checked_values.ndim != 2:
        raise ValueError("native whole-program AD batch values must be two-dimensional")
    if checked_values.shape[0] < 1:
        raise ValueError("native whole-program AD batch values must contain at least one row")
    if checked_values.shape != checked_tangents.shape:
        raise ValueError("native whole-program AD batch tangents must match values shape")
    if checked_values.shape[1] != parameter_count:
        raise ValueError("native whole-program AD batch parameter count mismatch")
    if not np.all(np.isfinite(checked_values)) or not np.all(np.isfinite(checked_tangents)):
        raise ValueError("native whole-program AD batch JVP inputs must be finite")
    rows = int(checked_values.shape[0])
    output = np.zeros(rows, dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_tangents.ctypes.data_as(double_pointer),
        rows,
        output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("native whole-program AD batch JVP output must be finite")
    return cast(NDArray[np.float64], output)


def _call_native_whole_program_batch_vjp(
    function: Callable[[Any, Any, int, Any], None],
    values: np.ndarray,
    cotangents: np.ndarray,
    parameter_count: int,
) -> NDArray[np.float64]:
    checked_values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    checked_cotangents = np.ascontiguousarray(np.asarray(cotangents, dtype=np.float64))
    if checked_values.ndim != 2:
        raise ValueError("native whole-program AD batch values must be two-dimensional")
    if checked_values.shape[0] < 1:
        raise ValueError("native whole-program AD batch values must contain at least one row")
    if checked_values.shape[1] != parameter_count:
        raise ValueError("native whole-program AD batch parameter count mismatch")
    if checked_cotangents.shape != (checked_values.shape[0],):
        raise ValueError("native whole-program AD batch cotangent row count mismatch")
    if not np.all(np.isfinite(checked_values)) or not np.all(np.isfinite(checked_cotangents)):
        raise ValueError("native whole-program AD batch VJP inputs must be finite")
    rows = int(checked_values.shape[0])
    output = np.zeros((rows, parameter_count), dtype=np.float64)
    double_pointer = ctypes.POINTER(ctypes.c_double)
    function(
        checked_values.ctypes.data_as(double_pointer),
        checked_cotangents.ctypes.data_as(double_pointer),
        rows,
        output.ctypes.data_as(double_pointer),
    )
    if not np.all(np.isfinite(output)):
        raise ValueError("native whole-program AD batch VJP output must be finite")
    return cast(NDArray[np.float64], output)


def _verify_native_whole_program_ad_kernel(
    result: WholeProgramADResult,
    native_functions: Mapping[str, Any],
    sample_values: NDArray[np.float64],
) -> CompilerADKernelVerification:
    value = _call_native_whole_program_unary(native_functions["value"], sample_values, 1)
    gradient = _call_native_whole_program_unary(
        native_functions["gradient"],
        sample_values,
        int(result.gradient.size),
    )
    tangent = np.ones(result.gradient.size, dtype=np.float64)
    jvp = _call_native_whole_program_binary(native_functions["jvp"], sample_values, tangent, 1)
    cotangent = np.ones(1, dtype=np.float64)
    vjp = _call_native_whole_program_binary(
        native_functions["vjp"],
        sample_values,
        cotangent,
        int(result.gradient.size),
    )
    batch_values, batch_gradients = _call_native_whole_program_batch_value_gradient(
        native_functions["batch_value_gradient"],
        sample_values.reshape(1, -1),
        int(result.gradient.size),
    )
    batch_jvp = _call_native_whole_program_batch_jvp(
        native_functions["batch_jvp"],
        sample_values.reshape(1, -1),
        tangent.reshape(1, -1),
        int(result.gradient.size),
    )
    batch_vjp = _call_native_whole_program_batch_vjp(
        native_functions["batch_vjp"],
        sample_values.reshape(1, -1),
        cotangent,
        int(result.gradient.size),
    )
    expected_value = np.array([result.value], dtype=np.float64)
    expected_jvp = np.array([float(np.dot(result.gradient, tangent))], dtype=np.float64)
    expected_vjp = result.gradient.copy()
    errors = (
        _max_abs_error(value, expected_value),
        _max_abs_error(gradient, result.gradient),
        _max_abs_error(jvp, expected_jvp),
        _max_abs_error(vjp, expected_vjp),
        _max_abs_error(batch_values, expected_value),
        _max_abs_error(batch_gradients, result.gradient.reshape(1, -1)),
        _max_abs_error(batch_jvp, expected_jvp),
        _max_abs_error(batch_vjp, result.gradient.reshape(1, -1)),
    )
    return CompilerADKernelVerification(
        value_close=bool(np.allclose(value, expected_value, rtol=1.0e-10, atol=1.0e-10)),
        jvp_close=bool(
            np.allclose(jvp, expected_jvp, rtol=1.0e-10, atol=1.0e-10)
            and np.allclose(batch_jvp, expected_jvp, rtol=1.0e-10, atol=1.0e-10)
        ),
        vjp_close=bool(
            np.allclose(vjp, expected_vjp, rtol=1.0e-10, atol=1.0e-10)
            and np.allclose(batch_vjp, result.gradient.reshape(1, -1), rtol=1.0e-10, atol=1.0e-10)
        ),
        gradient_close=bool(
            np.allclose(gradient, result.gradient, rtol=1.0e-10, atol=1.0e-10)
            and np.allclose(
                batch_gradients, result.gradient.reshape(1, -1), rtol=1.0e-10, atol=1.0e-10
            )
            and np.allclose(batch_values, expected_value, rtol=1.0e-10, atol=1.0e-10)
        ),
        max_abs_error=max(errors),
        samples=1,
    )


def _annotate_whole_program_native_mlir(
    module: MLIRModule,
    llvm_ir: str,
    result: WholeProgramADResult,
) -> MLIRModule:
    llvm_sha256 = hashlib.sha256(llvm_ir.encode("utf-8")).hexdigest()
    lowering_report = analyse_whole_program_ad_native_lowering(result)
    if not module.text.endswith("}\n"):
        raise ValueError("whole-program MLIR module must end with a module terminator")
    text = (
        module.text[:-2]
        + "  scpn_diff.native_llvm_jit "
        + '{execution = "native_llvm_jit", '
        + f'gradient = "forward_kernel", llvm_sha256 = "{llvm_sha256}"}}\n'
        + "}\n"
    )
    resource_counts = dict(module.resource_counts)
    resource_counts["native_whole_program_kernels"] = 1
    resource_counts["native_whole_program_batch_kernels"] = 1
    resource_counts["native_whole_program_batch_transform_kernels"] = 2
    resource_counts["native_supported_ops"] = len(lowering_report.lowerable_ops)
    resource_counts["native_lowerable_ops"] = len(lowering_report.lowerable_ops)
    resource_counts["native_unsupported_ops"] = len(lowering_report.unsupported_ops)
    resource_counts["native_supported_elementary_ops"] = sum(
        1
        for op in lowering_report.lowerable_ops
        if op
        in {
            "sin",
            "cos",
            "exp",
            "expm1",
            "log",
            "log1p",
            "sqrt",
            "tan",
            "tanh",
            "arcsin",
            "arccos",
            "reciprocal",
            "square",
            "abs",
        }
    )
    metadata = dict(module.metadata)
    polyglot_targets = dict(metadata.get("polyglot_targets", {}))
    polyglot_targets["llvm"] = (
        "available: native_llvm_jit scalar program AD with expanded elementary ops "
        "and stable executed branch signatures"
    )
    polyglot_targets["jit"] = (
        "available: native_llvm_jit scalar program AD with expanded elementary ops "
        "and stable executed branch signatures"
    )
    metadata.update(
        {
            "claim_boundary": (
                "whole-program AD trace interchange plus native LLVM/JIT lowering "
                "for supported scalar traces with expanded elementary ops and stable "
                "executed branch signatures"
            ),
            "llvm_ir_sha256": llvm_sha256,
            "native_backend": "native_llvm_jit",
            "native_supported_ops": lowering_report.lowerable_ops,
            "native_lowering_report": lowering_report.as_metadata(),
            "native_lowerable_ops": lowering_report.lowerable_ops,
            "native_unsupported_ops": lowering_report.unsupported_ops,
            "native_fail_closed_reason": lowering_report.fail_closed_reason,
            "polyglot_targets": polyglot_targets,
        }
    )
    return MLIRModule(
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        dialect=module.dialect,
        resource_counts=resource_counts,
        metadata=metadata,
    )


def _whole_program_replay_signature(result: WholeProgramADResult) -> tuple[str, ...]:
    """Return a stable non-numeric signature for supported program AD replay."""

    control_signature = tuple(
        f"{node.index}:{node.op}:{','.join(node.inputs)}"
        for node in result.ir_nodes
        if node.op.startswith(("branch:", "loop:", "control:"))
    )
    if control_signature:
        return control_signature
    return tuple(f"{node.index}:{node.op}:{','.join(node.inputs)}" for node in result.ir_nodes)


def _coupling_terms(K_nm: np.ndarray) -> tuple[tuple[int, int, float], ...]:
    terms: list[tuple[int, int, float]] = []
    n_oscillators = K_nm.shape[0]
    for left in range(n_oscillators):
        for right in range(left + 1, n_oscillators):
            value = float(K_nm[left, right])
            if abs(value) > 1e-15:
                terms.append((left, right, value))
    return tuple(terms)


def _as_finite_vector(name: str, value: object) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(NDArray[np.float64], array.copy())


def _max_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _fmt_float(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("MLIR numeric attributes must be finite")
    return format(value, ".17g")


def _fmt_bool(value: bool) -> str:
    return "true" if value else "false"


def _escape_mlir_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


__all__ = [
    "CompilerADTransformPlan",
    "CompilerADExecutableConfig",
    "CompilerADKernelVerification",
    "DifferentiableMLIRCompileConfig",
    "ExecutableCompilerADKernel",
    "ExecutableWholeProgramADBatchResult",
    "ExecutableWholeProgramADKernel",
    "MLIRCompileConfig",
    "NativeWholeProgramADKernel",
    "PrimitiveLoweringStatus",
    "WholeProgramADNativeLoweringReport",
    "MLIRModule",
    "analyse_whole_program_ad_native_lowering",
    "build_compiler_ad_transform_plan",
    "compile_compiler_ad_transform_plan_to_mlir",
    "compile_custom_derivative_rule_to_mlir",
    "compile_custom_derivative_rule_to_executable",
    "compile_registered_primitive_to_executable",
    "compile_whole_program_ad_trace_to_executable",
    "compile_whole_program_ad_trace_to_native_llvm_jit",
    "compile_matrix_2x2_determinant_ad_to_native_llvm_jit",
    "compile_matrix_2x2_eigenvalues_ad_to_native_llvm_jit",
    "compile_matrix_2x2_eigensystem_ad_to_native_llvm_jit",
    "compile_matrix_2x2_inverse_ad_to_native_llvm_jit",
    "compile_matrix_2x2_solve_ad_to_native_llvm_jit",
    "compile_matrix_frobenius_norm_squared_ad_to_native_llvm_jit",
    "compile_matrix_matrix_product_ad_to_native_llvm_jit",
    "compile_matrix_quadratic_form_ad_to_native_llvm_jit",
    "compile_matrix_trace_ad_to_native_llvm_jit",
    "compile_matrix_vector_product_ad_to_native_llvm_jit",
    "compile_scalar_binary_elementwise_ad_to_native_llvm_jit",
    "compile_scalar_quadratic_ad_to_native_llvm_jit",
    "compile_scalar_unary_elementwise_ad_to_native_llvm_jit",
    "compile_vector_dot_ad_to_native_llvm_jit",
    "compile_vector_squared_norm_ad_to_native_llvm_jit",
    "compile_symmetric_2x2_cholesky_ad_to_native_llvm_jit",
    "compile_symmetric_2x2_eigenvalues_ad_to_native_llvm_jit",
    "compile_whole_program_ad_trace_to_mlir",
    "compile_kuramoto_to_mlir",
    "make_executable_ad_kernel_batching_rule",
    "make_matrix_2x2_determinant_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_determinant_native_llvm_jit_primitive_transform",
    "make_matrix_2x2_eigenvalues_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_eigenvalues_native_llvm_jit_primitive_transform",
    "make_matrix_2x2_eigensystem_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_eigensystem_native_llvm_jit_primitive_transform",
    "make_matrix_2x2_inverse_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_inverse_native_llvm_jit_primitive_transform",
    "make_matrix_2x2_solve_native_llvm_jit_lowering_rule",
    "make_matrix_2x2_solve_native_llvm_jit_primitive_transform",
    "make_matrix_frobenius_norm_squared_native_llvm_jit_lowering_rule",
    "make_matrix_frobenius_norm_squared_native_llvm_jit_primitive_transform",
    "make_matrix_matrix_product_native_llvm_jit_lowering_rule",
    "make_matrix_matrix_product_native_llvm_jit_primitive_transform",
    "make_matrix_quadratic_form_native_llvm_jit_lowering_rule",
    "make_matrix_quadratic_form_native_llvm_jit_primitive_transform",
    "make_matrix_trace_native_llvm_jit_lowering_rule",
    "make_matrix_trace_native_llvm_jit_primitive_transform",
    "make_matrix_vector_product_native_llvm_jit_lowering_rule",
    "make_matrix_vector_product_native_llvm_jit_primitive_transform",
    "make_scalar_binary_elementwise_native_llvm_jit_lowering_rule",
    "make_scalar_quadratic_native_llvm_jit_lowering_rule",
    "make_scalar_unary_elementwise_native_llvm_jit_lowering_rule",
    "make_symmetric_2x2_cholesky_native_llvm_jit_lowering_rule",
    "make_symmetric_2x2_cholesky_native_llvm_jit_primitive_transform",
    "make_symmetric_2x2_eigenvalues_native_llvm_jit_lowering_rule",
    "make_symmetric_2x2_eigenvalues_native_llvm_jit_primitive_transform",
    "make_vector_dot_native_llvm_jit_lowering_rule",
    "make_vector_dot_native_llvm_jit_primitive_transform",
    "make_vector_squared_norm_native_llvm_jit_lowering_rule",
    "make_vector_squared_norm_native_llvm_jit_primitive_transform",
]
