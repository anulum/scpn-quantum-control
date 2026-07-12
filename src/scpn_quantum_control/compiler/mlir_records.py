# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR records module
# scpn-quantum-control -- MLIR compile-config and lowering-status value records
"""Value records for the MLIR compilation and compiler-AD lowering surface.

These frozen data contracts describe the MLIR compilation inputs and outputs:
the compile configuration, the emitted MLIR module, the per-primitive lowering
status, the compiler-AD transform plan, the differentiable compile and executable
configurations, and the kernel verification record. They carry validation only;
the compilers, lowering passes, and verification logic that produce them live in
:mod:`scpn_quantum_control.compiler.mlir`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from ..program_ad_registry import PrimitiveIdentity


@dataclass(frozen=True)
class MLIRCompileConfig:
    """Configuration for Kuramoto-XY MLIR-style export."""

    time: float
    trotter_steps: int = 1
    trotter_order: int = 1
    dialect: str = "scpn_kuramoto"
    include_metadata: bool = True

    def __post_init__(self) -> None:
        """Validate public MLIR compile configuration fields."""
        if not np.isfinite(self.time) or self.time <= 0.0:
            raise ValueError("time must be finite and positive")
        if not isinstance(self.trotter_steps, int) or self.trotter_steps < 1:
            raise ValueError("trotter_steps must be a positive integer")
        if self.trotter_order not in {1, 2}:
            raise ValueError("trotter_order must be 1 or 2")
        if not self.dialect or not self.dialect.replace("_", "").isalnum():
            raise ValueError("dialect must be a non-empty MLIR-safe identifier")
        if not isinstance(self.include_metadata, bool):
            raise ValueError("include_metadata must be a boolean")


@dataclass(frozen=True)
class MLIRModule:
    """Textual MLIR module plus deterministic provenance."""

    text: str
    sha256: str
    dialect: str
    resource_counts: Mapping[str, int]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate module text provenance and freeze mapping fields."""
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        expected = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.sha256 != expected:
            raise ValueError("sha256 must match text")
        if not self.dialect or not self.dialect.replace("_", "").isalnum():
            raise ValueError("dialect must be a non-empty MLIR-safe identifier")
        if not isinstance(self.resource_counts, Mapping):
            raise ValueError("resource_counts must be a mapping")
        resource_counts = dict(self.resource_counts)
        if any(not isinstance(key, str) or not key for key in resource_counts):
            raise ValueError("resource_counts keys must be non-empty strings")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in resource_counts.values()
        ):
            raise ValueError("resource_counts values must be non-negative integers")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        metadata = dict(self.metadata)
        if any(not isinstance(key, str) or not key for key in metadata):
            raise ValueError("metadata keys must be non-empty strings")
        object.__setattr__(self, "resource_counts", MappingProxyType(resource_counts))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


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
        """Validate primitive lowering status and backend claim provenance."""
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
        """Validate transform-plan rows and claim-boundary metadata."""
        if not self.statuses:
            raise ValueError("compiler AD transform plan requires at least one primitive")
        if any(not isinstance(status, PrimitiveLoweringStatus) for status in self.statuses):
            raise ValueError("statuses must contain PrimitiveLoweringStatus rows")
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


@dataclass(frozen=True)
class DifferentiableMLIRCompileConfig:
    """Configuration for differentiable primitive MLIR-style lowering."""

    dialect: str = "scpn_diff"
    target: str = "mlir"
    include_numeric_payload: bool = True
    include_metadata: bool = True

    def __post_init__(self) -> None:
        """Validate differentiable MLIR compile configuration fields."""
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
        """Validate executable compiler-AD kernel configuration fields."""
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
        """Validate executable kernel verification evidence fields."""
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
        if not isinstance(self.samples, int) or self.samples < 1:
            raise ValueError("samples must be a positive integer")

    @property
    def passed(self) -> bool:
        """Return whether all executed verification checks passed."""
        checks = (self.value_close, self.jvp_close, self.vjp_close, self.gradient_close)
        return all(check is not False for check in checks)
