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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from ..differentiable import (
    CustomDerivativeRegistry,
    CustomDerivativeRule,
    PrimitiveIdentity,
    WholeProgramADResult,
    program_ad_linalg_matrix_power_derivative_rule,
    program_ad_linalg_multi_dot_derivative_rule,
    value_and_custom_jacobian,
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
        if "blocked" not in self.rust_lowering.lower():
            raise ValueError("rust_lowering must remain blocked until Rust AD lowering exists")
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
        return "blocked" not in status.rust_lowering.lower()

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
    "MLIRCompileConfig",
    "PrimitiveLoweringStatus",
    "MLIRModule",
    "build_compiler_ad_transform_plan",
    "compile_compiler_ad_transform_plan_to_mlir",
    "compile_custom_derivative_rule_to_mlir",
    "compile_custom_derivative_rule_to_executable",
    "compile_registered_primitive_to_executable",
    "compile_matrix_quadratic_form_ad_to_native_llvm_jit",
    "compile_scalar_binary_elementwise_ad_to_native_llvm_jit",
    "compile_scalar_quadratic_ad_to_native_llvm_jit",
    "compile_scalar_unary_elementwise_ad_to_native_llvm_jit",
    "compile_vector_dot_ad_to_native_llvm_jit",
    "compile_whole_program_ad_trace_to_mlir",
    "compile_kuramoto_to_mlir",
    "make_matrix_quadratic_form_native_llvm_jit_lowering_rule",
    "make_scalar_binary_elementwise_native_llvm_jit_lowering_rule",
    "make_scalar_quadratic_native_llvm_jit_lowering_rule",
    "make_scalar_unary_elementwise_native_llvm_jit_lowering_rule",
    "make_vector_dot_native_llvm_jit_lowering_rule",
]
