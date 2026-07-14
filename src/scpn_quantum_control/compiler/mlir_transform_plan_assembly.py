# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR transform plan assembly module
# scpn-quantum-control -- MLIR transform-plan assembly
"""Compiler-AD transform-plan assembly and deterministic MLIR interchange."""

from __future__ import annotations

import hashlib
import json

from ..differentiable import CustomDerivativeRegistry
from .mlir_native_primitives import _escape_mlir_string, _fmt_bool
from .mlir_records import CompilerADTransformPlan, MLIRModule, PrimitiveLoweringStatus


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
        return "blocked: no MLIR-runtime lowering rule"

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
