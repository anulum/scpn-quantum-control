# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- MLIR textual compiler surface
"""Deterministic MLIR-style export for Kuramoto-XY workloads.

The module emits a conservative textual interchange layer for the SCPN
Kuramoto-XY compiler. It does not require an MLIR Python runtime and does not
claim lowering to LLVM, QIR, or provider pulses. The value is a stable,
auditable IR boundary for compiler passes and external tooling.
"""

from __future__ import annotations

import hashlib
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
    has_shape_rule: bool = False
    has_dtype_rule: bool = False
    has_static_argument_rule: bool = False
    has_lowering_rule: bool = False
    lowering_metadata: Mapping[str, str] = field(default_factory=dict)
    static_derivative_factory: str = "not_declared"
    static_signature: str = "none"
    nondifferentiable_policy: str = "not_declared"
    effect: str = "pure"
    mlir_lowering: str = "available: scpn_diff dialect interchange"
    rust_lowering: str = "blocked: no Rust differentiable primitive backend"
    llvm_lowering: str = "blocked: no LLVM/JIT differentiable primitive backend"

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
        if not isinstance(self.effect, str) or not self.effect:
            raise ValueError("effect must be non-empty")
        for label, status in (
            ("mlir_lowering", self.mlir_lowering),
            ("rust_lowering", self.rust_lowering),
            ("llvm_lowering", self.llvm_lowering),
        ):
            if not isinstance(status, str) or not status:
                raise ValueError(f"{label} must be non-empty")


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
        if self.executable_backend != "none":
            raise ValueError("executable_backend must be 'none' until a real backend exists")
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
                effect="pure" if transform_rule is None else transform_rule.effect,
                mlir_lowering=metadata.get("mlir", default_mlir_status),
                rust_lowering=metadata.get(
                    "rust", "blocked: no Rust differentiable primitive backend"
                ),
                llvm_lowering=metadata.get(
                    "llvm", "blocked: no LLVM/JIT differentiable primitive backend"
                ),
            )
        )
    return CompilerADTransformPlan(tuple(statuses), dialect=dialect, transform=transform)


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
    for index, status in enumerate(plan.statuses):
        lines.append(
            "    scpn_diff.primitive "
            f'%p{index} {{identity = "{_escape_mlir_string(status.identity.key)}", '
            f'rule = "{_escape_mlir_string(status.rule_name)}", '
            f'op = "{_escape_mlir_string(status.mlir_op)}", '
            f"jvp = {_fmt_bool(status.has_jvp)}, vjp = {_fmt_bool(status.has_vjp)}, "
            f"shape_rule = {_fmt_bool(status.has_shape_rule)}, "
            f"dtype_rule = {_fmt_bool(status.has_dtype_rule)}, "
            f"static_argument_rule = {_fmt_bool(status.has_static_argument_rule)}, "
            f"lowering_rule = {_fmt_bool(status.has_lowering_rule)}, "
            f'static_derivative_factory = "{_escape_mlir_string(status.static_derivative_factory)}", '
            f'static_signature = "{_escape_mlir_string(status.static_signature)}", '
            f'policy = "{_escape_mlir_string(status.nondifferentiable_policy)}", '
            f'effect = "{_escape_mlir_string(status.effect)}"}}'
        )
        lines.append(
            "    scpn_diff.lowering_status "
            f'{{identity = "{_escape_mlir_string(status.identity.key)}", '
            f'mlir = "{_escape_mlir_string(status.mlir_lowering)}", '
            f'rust = "{_escape_mlir_string(status.rust_lowering)}", '
            f'llvm = "{_escape_mlir_string(status.llvm_lowering)}"}}'
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
        f'{{kind = "{_escape_mlir_string(plan.transform)}", execution = "interchange_only"}}'
    )
    lines.append("    return")
    lines.append("  }")
    metadata = {
        "claim_boundary": plan.claim_boundary,
        "dialect": plan.dialect,
        "executable_backend": plan.executable_backend,
        "effects": {
            status.identity.key: status.effect
            for status in plan.statuses
            if status.nondifferentiable_policy != "not_declared"
        },
        "nondifferentiable_policies": {
            status.identity.key: status.nondifferentiable_policy
            for status in plan.statuses
            if status.nondifferentiable_policy != "not_declared"
        },
        "mlir_runtime_lowering_primitives": [
            status.identity.key for status in plan.statuses if status.has_lowering_rule
        ],
        "primitive_identities": [status.identity.key for status in plan.statuses],
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
        "transform": plan.transform,
        "uncontracted_primitives": [
            status.identity.key
            for status in plan.statuses
            if status.nondifferentiable_policy == "not_declared"
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
            "shape_rules": sum(status.has_shape_rule for status in plan.statuses),
            "dtype_rules": sum(status.has_dtype_rule for status in plan.statuses),
            "effects": sum(
                status.nondifferentiable_policy != "not_declared" for status in plan.statuses
            ),
            "nondifferentiable_policies": sum(
                status.nondifferentiable_policy != "not_declared" for status in plan.statuses
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
            "uncontracted_primitives": sum(
                status.nondifferentiable_policy == "not_declared" for status in plan.statuses
            ),
            "executable_backends": 0,
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
        if self.backend != "mlir_runtime":
            raise ValueError(
                "backend must be 'mlir_runtime'; native LLVM/JIT lowering is not yet available"
            )
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
        if self.backend != "mlir_runtime":
            raise ValueError("backend must be 'mlir_runtime'")
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
    targets remain fail-closed until real code generation is present.
    """

    if not isinstance(rule, CustomDerivativeRule):
        raise ValueError("executable AD lowering requires a CustomDerivativeRule")
    compile_config = CompilerADExecutableConfig() if config is None else config
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
        lowered = transform.lowering_rule(rule)
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
    "compile_whole_program_ad_trace_to_mlir",
    "compile_kuramoto_to_mlir",
]
