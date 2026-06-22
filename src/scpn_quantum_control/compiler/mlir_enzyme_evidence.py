# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Enzyme/MLIR compiler-AD evidence records
"""Evidence records and builders for the Enzyme/MLIR compiler-AD maturity surface.

These frozen value records capture the bounded evidence emitted by the
Enzyme/MLIR compiler-AD lane: toolchain detection status, native execution and
LLVM correctness evidence, benchmark attachments, per-case and aggregate
compiler-AD breadth artifacts, and the maturity-audit result. The
``build_enzyme_mlir_*`` builders construct the benchmark-attachment and
compiler-AD breadth artifact/evidence records from validated inputs. The
toolchain detection and maturity-audit orchestration live in
:mod:`scpn_quantum_control.compiler.mlir`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..phase.qnode_affinity_benchmark import PhaseQNodeAffinityArtifactValidation

ENZYME_MLIR_COMPILER_AD_BREADTH_CASES = frozenset(
    {
        "scalar_forward_mode",
        "scalar_reverse_mode",
        "vector_jvp",
        "vector_vjp",
        "matrix_jvp",
        "matrix_vjp",
        "loop_activity",
        "alias_activity",
        "mlir_lowering",
        "llvm_ir_generation",
        "native_enzyme_execution",
    }
)


ENZYME_MLIR_COMPILER_AD_TRANSFORM_MODES = frozenset({"forward", "reverse", "jvp", "vjp"})


@dataclass(frozen=True)
class EnzymeMLIRToolchainStatus:
    """Detected status for one native compiler-AD command."""

    command: str
    executable: str | None
    available: bool
    version: str | None
    failure_class: str | None
    setup_instructions: str | None

    def __post_init__(self) -> None:
        if not self.command:
            raise ValueError("command must be non-empty")
        if self.available:
            if not self.executable or not self.version:
                raise ValueError("available toolchains require executable and version metadata")
            if self.failure_class is not None or self.setup_instructions is not None:
                raise ValueError("available toolchains must not carry hard-gap metadata")
        else:
            if self.executable is not None or self.version is not None:
                raise ValueError("unavailable toolchains must not carry executable metadata")
            if not self.failure_class or not self.setup_instructions:
                raise ValueError(
                    "unavailable toolchains require failure_class and setup instructions"
                )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready toolchain status metadata."""

        return {
            "command": self.command,
            "executable": self.executable,
            "available": self.available,
            "version": self.version,
            "failure_class": self.failure_class,
            "setup_instructions": self.setup_instructions,
        }


@dataclass(frozen=True)
class EnzymeNativeExecutionEvidence:
    """Validated native Enzyme execution evidence or named runtime hard gap."""

    artifact_id: str
    status: str
    failure_class: str | None
    value_error: float | None
    gradient_error: float | None
    runtime_seconds: float | None
    toolchain: Mapping[str, str]
    setup_instructions: str | None
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        if self.status not in {"success", "hard_gap"}:
            raise ValueError("status must be success or hard_gap")
        toolchain = dict(self.toolchain)
        if any(not key or not value for key, value in toolchain.items()):
            raise ValueError("toolchain metadata must map non-empty strings")
        if self.status == "success":
            if self.failure_class is not None or self.setup_instructions is not None:
                raise ValueError("successful Enzyme evidence must not carry hard-gap metadata")
            for name, value in (
                ("value_error", self.value_error),
                ("gradient_error", self.gradient_error),
                ("runtime_seconds", self.runtime_seconds),
            ):
                if value is None or value < 0.0 or not np.isfinite(value):
                    raise ValueError(f"successful Enzyme evidence requires finite {name}")
        else:
            if not self.failure_class or not self.setup_instructions:
                raise ValueError("hard-gap Enzyme evidence requires failure metadata")
            if any(
                value is not None
                for value in (self.value_error, self.gradient_error, self.runtime_seconds)
            ):
                raise ValueError("hard-gap Enzyme evidence must not carry success metrics")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "toolchain", MappingProxyType(toolchain))

    @property
    def passed(self) -> bool:
        """Return whether native Enzyme execution matched the SCPN reference."""

        return self.status == "success"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready native Enzyme execution evidence."""

        return {
            "artifact_id": self.artifact_id,
            "status": self.status,
            "failure_class": self.failure_class,
            "value_error": self.value_error,
            "gradient_error": self.gradient_error,
            "runtime_seconds": self.runtime_seconds,
            "toolchain": dict(self.toolchain),
            "setup_instructions": self.setup_instructions,
            "claim_boundary": self.claim_boundary,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class MLIRLLVMCorrectnessEvidence:
    """Persistable correctness snapshot for the bounded MLIR/LLVM audit path."""

    artifact_id: str
    checks: Mapping[str, bool]
    toolchain_versions: Mapping[str, str]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not self.artifact_id.strip():
            raise ValueError("artifact_id must be non-empty")
        checks = dict(self.checks)
        if not checks:
            raise ValueError("checks must be non-empty")
        if any(not key or not isinstance(value, bool) for key, value in checks.items()):
            raise ValueError("checks must map non-empty names to booleans")
        toolchain_versions = dict(self.toolchain_versions)
        if any(not key or not value for key, value in toolchain_versions.items()):
            raise ValueError("toolchain_versions must map non-empty strings")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(self, "toolchain_versions", MappingProxyType(toolchain_versions))

    @property
    def passed(self) -> bool:
        """Return whether every bounded MLIR/LLVM correctness check passed."""

        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready MLIR/LLVM correctness evidence."""

        return {
            "artifact_id": self.artifact_id,
            "checks": dict(self.checks),
            "toolchain_versions": dict(self.toolchain_versions),
            "claim_boundary": self.claim_boundary,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class EnzymeMLIRBenchmarkAttachment:
    """Validated isolated benchmark attachment for Enzyme/MLIR compiler AD.

    Parameters
    ----------
    validation:
        Phase-QNode affinity benchmark artefact validation produced from a raw
        benchmark JSON file.
    required_breadth_cases:
        Enzyme/MLIR compiler-AD breadth cases covered by the benchmark
        attachment. The set must exactly match the compiler-AD breadth contract.
    claim_boundary:
        Explicit statement preventing local or partial benchmark evidence from
        becoming a provider, QPU, hardware, or arbitrary performance claim.
    """

    validation: PhaseQNodeAffinityArtifactValidation
    required_breadth_cases: tuple[str, ...]
    claim_boundary: str

    def __post_init__(self) -> None:
        if not isinstance(self.validation, PhaseQNodeAffinityArtifactValidation):
            raise ValueError("validation must be PhaseQNodeAffinityArtifactValidation")
        cases = tuple(sorted(case.strip() for case in self.required_breadth_cases))
        if set(cases) != ENZYME_MLIR_COMPILER_AD_BREADTH_CASES:
            missing = sorted(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES.difference(cases))
            extra = sorted(set(cases).difference(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES))
            details = ", ".join(
                part
                for part in (
                    f"missing={missing}" if missing else "",
                    f"extra={extra}" if extra else "",
                )
                if part
            )
            raise ValueError(f"required_breadth_cases must match Enzyme/MLIR cases: {details}")
        if any(not case for case in cases):
            raise ValueError("required_breadth_cases must contain non-empty entries")
        object.__setattr__(self, "required_breadth_cases", cases)
        claim_boundary = self.claim_boundary.strip()
        if not claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "claim_boundary", claim_boundary)

    @property
    def benchmark_artifact_id(self) -> str:
        """Return the validated benchmark artefact identifier."""

        return self.validation.benchmark_artifact_id

    @property
    def promotion_ready(self) -> bool:
        """Return whether the benchmark attachment satisfies promotion policy."""

        return (
            self.validation.promotion_ready
            and self.validation.evidence_label == "isolated_affinity"
            and self.validation.production_benchmark
            and self.validation.raw_timing_row_count > 0
            and not self.validation.missing_requirements
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready benchmark attachment metadata."""

        return {
            "benchmark_artifact_id": self.benchmark_artifact_id,
            "evidence_label": self.validation.evidence_label,
            "production_benchmark": self.validation.production_benchmark,
            "promotion_ready": self.promotion_ready,
            "raw_timing_row_count": self.validation.raw_timing_row_count,
            "missing_requirements": list(self.validation.missing_requirements),
            "artifact_path": self.validation.artifact_path,
            "artifact_sha256": self.validation.artifact_sha256,
            "required_breadth_cases": list(self.required_breadth_cases),
            "required_breadth_case_count": len(self.required_breadth_cases),
            "validation_claim_boundary": self.validation.claim_boundary,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class EnzymeMLIRCompilerADBreadthCaseEvidence:
    """One raw Enzyme/MLIR compiler-AD breadth case row.

    Parameters
    ----------
    case_id:
        Required Enzyme/MLIR breadth-case identifier.
    status:
        ``success`` for captured passing evidence or ``hard_gap`` for a named
        missing or failed route.
    transform_modes:
        Compiler-AD transform modes covered by this case row.
    frontend_language:
        Frontend or IR surface represented by the row.
    value_error:
        Absolute value error for successful correctness rows.
    gradient_error:
        Absolute derivative error for successful correctness rows.
    runtime_seconds:
        Positive bounded runner runtime for successful rows.
    artifact_refs:
        Stable references to raw logs, JSON, IR, or benchmark artefacts.
    failure_class:
        Named failure class for hard-gap rows.
    setup_instructions:
        Reproduction or remediation instructions for hard-gap rows.
    claim_boundary:
        Boundary preventing a case row from becoming a provider, hardware, or
        performance claim.
    """

    case_id: str
    status: str
    transform_modes: tuple[str, ...]
    frontend_language: str
    value_error: float | None
    gradient_error: float | None
    runtime_seconds: float | None
    artifact_refs: Mapping[str, str]
    failure_class: str | None
    setup_instructions: str | None
    claim_boundary: str

    def __post_init__(self) -> None:
        case_id = self.case_id.strip()
        if case_id not in ENZYME_MLIR_COMPILER_AD_BREADTH_CASES:
            raise ValueError("case_id must be an Enzyme/MLIR compiler-AD breadth case")
        object.__setattr__(self, "case_id", case_id)
        if self.status not in {"success", "hard_gap"}:
            raise ValueError("status must be success or hard_gap")
        modes = tuple(sorted(mode.strip().lower() for mode in self.transform_modes))
        if not modes or any(not mode for mode in modes):
            raise ValueError("transform_modes must contain non-empty entries")
        if not set(modes).issubset(ENZYME_MLIR_COMPILER_AD_TRANSFORM_MODES):
            raise ValueError("transform_modes must be valid Enzyme/MLIR transform modes")
        object.__setattr__(self, "transform_modes", modes)
        frontend_language = self.frontend_language.strip().lower()
        if not frontend_language:
            raise ValueError("frontend_language must be non-empty")
        object.__setattr__(self, "frontend_language", frontend_language)
        refs = dict(self.artifact_refs)
        if not refs or any(not key or not value for key, value in refs.items()):
            raise ValueError("artifact_refs must map non-empty strings")
        object.__setattr__(self, "artifact_refs", MappingProxyType(refs))
        if self.status == "success":
            for name, value in (
                ("value_error", self.value_error),
                ("gradient_error", self.gradient_error),
                ("runtime_seconds", self.runtime_seconds),
            ):
                if value is None or value < 0.0 or not np.isfinite(value):
                    raise ValueError(f"success case rows require finite {name}")
            if self.runtime_seconds is None or self.runtime_seconds <= 0.0:
                raise ValueError("success case rows require positive runtime_seconds")
            if self.failure_class is not None or self.setup_instructions is not None:
                raise ValueError("success case rows must not carry hard-gap metadata")
        else:
            if not self.failure_class or not self.setup_instructions:
                raise ValueError("hard-gap case rows require failure metadata")
            if any(
                value is not None
                for value in (self.value_error, self.gradient_error, self.runtime_seconds)
            ):
                raise ValueError("hard-gap case rows must not carry success metrics")
        claim_boundary = self.claim_boundary.strip()
        if not claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "claim_boundary", claim_boundary)

    @property
    def passed(self) -> bool:
        """Return whether this breadth case has passing raw evidence."""

        return self.status == "success"

    @property
    def max_abs_error(self) -> float:
        """Return the largest correctness error carried by this row."""

        return max(float(self.value_error or 0.0), float(self.gradient_error or 0.0))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready breadth-case evidence metadata."""

        return {
            "case_id": self.case_id,
            "status": self.status,
            "passed": self.passed,
            "transform_modes": list(self.transform_modes),
            "frontend_language": self.frontend_language,
            "value_error": self.value_error,
            "gradient_error": self.gradient_error,
            "runtime_seconds": self.runtime_seconds,
            "artifact_refs": dict(self.artifact_refs),
            "failure_class": self.failure_class,
            "setup_instructions": self.setup_instructions,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class EnzymeMLIRCompilerADBreadthArtifact:
    """Raw Enzyme/MLIR compiler-AD breadth artefact with benchmark linkage."""

    artifact_id: str
    cases: tuple[EnzymeMLIRCompilerADBreadthCaseEvidence, ...]
    isolated_benchmark_evidence: EnzymeMLIRBenchmarkAttachment
    claim_boundary: str

    def __post_init__(self) -> None:
        artifact_id = self.artifact_id.strip()
        if not artifact_id:
            raise ValueError("artifact_id must be non-empty")
        object.__setattr__(self, "artifact_id", artifact_id)
        if not isinstance(self.isolated_benchmark_evidence, EnzymeMLIRBenchmarkAttachment):
            raise ValueError("isolated_benchmark_evidence must be EnzymeMLIRBenchmarkAttachment")
        cases = tuple(self.cases)
        if any(not isinstance(case, EnzymeMLIRCompilerADBreadthCaseEvidence) for case in cases):
            raise ValueError("cases must be EnzymeMLIRCompilerADBreadthCaseEvidence rows")
        case_ids = tuple(case.case_id for case in cases)
        if set(case_ids) != ENZYME_MLIR_COMPILER_AD_BREADTH_CASES or len(case_ids) != len(
            set(case_ids)
        ):
            missing = sorted(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES.difference(case_ids))
            extra = sorted(set(case_ids).difference(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES))
            duplicates = sorted(
                case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
            )
            details = ", ".join(
                part
                for part in (
                    f"missing={missing}" if missing else "",
                    f"extra={extra}" if extra else "",
                    f"duplicates={duplicates}" if duplicates else "",
                )
                if part
            )
            raise ValueError(f"cases must exactly cover Enzyme/MLIR breadth cases: {details}")
        object.__setattr__(self, "cases", tuple(sorted(cases, key=lambda row: row.case_id)))
        claim_boundary = self.claim_boundary.strip()
        if not claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "claim_boundary", claim_boundary)

    @property
    def transform_modes(self) -> tuple[str, ...]:
        """Return sorted transform modes covered across breadth cases."""

        modes = {mode for case in self.cases for mode in case.transform_modes}
        return tuple(sorted(modes))

    @property
    def frontend_languages(self) -> tuple[str, ...]:
        """Return sorted frontend or IR surfaces covered by the artefact."""

        return tuple(sorted({case.frontend_language for case in self.cases}))

    @property
    def failed_case_ids(self) -> tuple[str, ...]:
        """Return sorted breadth-case identifiers that remain hard gaps."""

        return tuple(case.case_id for case in self.cases if not case.passed)

    @property
    def passed_case_ids(self) -> tuple[str, ...]:
        """Return sorted breadth-case identifiers with passing raw evidence."""

        return tuple(case.case_id for case in self.cases if case.passed)

    @property
    def max_abs_error(self) -> float:
        """Return the largest correctness error recorded across passing rows."""

        return max(case.max_abs_error for case in self.cases)

    @property
    def runtime_seconds(self) -> float:
        """Return total bounded runtime across passing rows."""

        return sum(float(case.runtime_seconds or 0.0) for case in self.cases)

    @property
    def promotion_ready(self) -> bool:
        """Return whether the artefact can derive compiler-AD breadth evidence."""

        return (
            self.isolated_benchmark_evidence.promotion_ready
            and all(case.passed for case in self.cases)
            and set(self.transform_modes) == ENZYME_MLIR_COMPILER_AD_TRANSFORM_MODES
            and self.runtime_seconds > 0.0
        )

    def to_breadth_evidence(self) -> EnzymeMLIRCompilerADBreadthEvidence:
        """Derive promotion evidence from a complete raw breadth artefact.

        Raises
        ------
        ValueError
            If any breadth case failed, the transform modes are incomplete, or
            the attached isolated benchmark evidence is not promotion-ready.
        """

        if not self.promotion_ready:
            raise ValueError("compiler AD breadth artifact must be promotion-ready")
        return EnzymeMLIRCompilerADBreadthEvidence(
            artifact_id=self.artifact_id,
            cases={case.case_id: case.passed for case in self.cases},
            transform_modes=self.transform_modes,
            frontend_languages=self.frontend_languages,
            isolated_benchmark_artifact_id=(
                self.isolated_benchmark_evidence.benchmark_artifact_id
            ),
            max_abs_error=self.max_abs_error,
            runtime_seconds=self.runtime_seconds,
            claim_boundary=self.claim_boundary,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready raw breadth artefact metadata."""

        return {
            "schema": "scpn_qc_enzyme_mlir_compiler_ad_breadth_artifact_v1",
            "artifact_id": self.artifact_id,
            "case_count": len(self.cases),
            "cases": [case.to_dict() for case in self.cases],
            "passed_case_ids": list(self.passed_case_ids),
            "failed_case_ids": list(self.failed_case_ids),
            "transform_modes": list(self.transform_modes),
            "frontend_languages": list(self.frontend_languages),
            "max_abs_error": self.max_abs_error,
            "runtime_seconds": self.runtime_seconds,
            "isolated_benchmark_evidence": self.isolated_benchmark_evidence.to_dict(),
            "promotion_ready": self.promotion_ready,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class EnzymeMLIRCompilerADBreadthEvidence:
    """Validated breadth evidence for Enzyme/MLIR compiler AD promotion.

    The evidence records captured compiler-AD coverage across scalar, vector,
    matrix, control-flow, alias/activity, MLIR lowering, LLVM IR generation,
    and native Enzyme execution checks. It is an attachment contract for
    reviewer evidence; it does not run Enzyme, submit provider jobs, or create
    isolated benchmark artefacts.

    Parameters
    ----------
    artifact_id:
        Stable identifier for the captured breadth evidence bundle.
    cases:
        Mapping from required breadth-case names to pass/fail flags. The mapping
        must contain exactly the required case set and every case must pass.
    transform_modes:
        Compiler AD transform modes covered by the evidence. The required set
        is forward, reverse, JVP, and VJP.
    frontend_languages:
        Frontend or IR surfaces covered by the evidence bundle.
    isolated_benchmark_artifact_id:
        Identifier for the matching isolated benchmark artefact.
    max_abs_error:
        Maximum absolute correctness error recorded across breadth cases.
    runtime_seconds:
        Captured runtime for the breadth evidence runner.
    claim_boundary:
        Explicit claim boundary retained in serialized evidence.
    """

    artifact_id: str
    cases: Mapping[str, bool]
    transform_modes: tuple[str, ...]
    frontend_languages: tuple[str, ...]
    isolated_benchmark_artifact_id: str
    max_abs_error: float
    runtime_seconds: float
    claim_boundary: str

    def __post_init__(self) -> None:
        artifact_id = self.artifact_id.strip()
        if not artifact_id:
            raise ValueError("artifact_id must be non-empty")
        object.__setattr__(self, "artifact_id", artifact_id)
        cases = dict(self.cases)
        if set(cases) != ENZYME_MLIR_COMPILER_AD_BREADTH_CASES or not all(cases.values()):
            missing = sorted(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES.difference(cases))
            failed = sorted(key for key, value in cases.items() if not value)
            extra = sorted(set(cases).difference(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES))
            details = ", ".join(
                part
                for part in (
                    f"missing={missing}" if missing else "",
                    f"failed={failed}" if failed else "",
                    f"extra={extra}" if extra else "",
                )
                if part
            )
            raise ValueError(f"compiler AD breadth cases must be complete and passing: {details}")
        if any(not isinstance(value, bool) for value in cases.values()):
            raise ValueError("compiler AD breadth cases must map to booleans")
        object.__setattr__(self, "cases", MappingProxyType(cases))
        modes = tuple(sorted(mode.strip().lower() for mode in self.transform_modes))
        if set(modes) != ENZYME_MLIR_COMPILER_AD_TRANSFORM_MODES:
            raise ValueError("transform_modes must cover forward, reverse, jvp, and vjp")
        object.__setattr__(self, "transform_modes", modes)
        languages = tuple(sorted(language.strip().lower() for language in self.frontend_languages))
        if not languages or any(not language for language in languages):
            raise ValueError("frontend_languages must contain non-empty entries")
        if len(set(languages)) != len(languages):
            raise ValueError("frontend_languages must be unique")
        object.__setattr__(self, "frontend_languages", languages)
        isolated_benchmark_artifact_id = self.isolated_benchmark_artifact_id.strip()
        if not isolated_benchmark_artifact_id:
            raise ValueError("isolated_benchmark_artifact_id must be non-empty")
        object.__setattr__(
            self,
            "isolated_benchmark_artifact_id",
            isolated_benchmark_artifact_id,
        )
        if self.max_abs_error < 0.0 or not np.isfinite(self.max_abs_error):
            raise ValueError("max_abs_error must be finite and non-negative")
        if self.runtime_seconds <= 0.0 or not np.isfinite(self.runtime_seconds):
            raise ValueError("runtime_seconds must be finite and positive")
        claim_boundary = self.claim_boundary.strip()
        if not claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "claim_boundary", claim_boundary)

    @property
    def passed(self) -> bool:
        """Return whether all required compiler-AD breadth cases passed."""

        return (
            set(self.cases) == ENZYME_MLIR_COMPILER_AD_BREADTH_CASES
            and all(self.cases.values())
            and set(self.transform_modes) == ENZYME_MLIR_COMPILER_AD_TRANSFORM_MODES
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready compiler AD breadth evidence metadata."""

        return {
            "artifact_id": self.artifact_id,
            "cases": dict(self.cases),
            "case_count": len(self.cases),
            "transform_modes": list(self.transform_modes),
            "frontend_languages": list(self.frontend_languages),
            "isolated_benchmark_artifact_id": self.isolated_benchmark_artifact_id,
            "max_abs_error": self.max_abs_error,
            "runtime_seconds": self.runtime_seconds,
            "claim_boundary": self.claim_boundary,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class EnzymeMLIRMaturityAuditResult:
    """Provider-exceedance gate for Enzyme/MLIR compiler AD maturity."""

    scpn_mlir_runtime_verified: bool
    native_llvm_jit_surface: str
    toolchain: Mapping[str, EnzymeMLIRToolchainStatus]
    correctness_checks: Mapping[str, bool]
    hard_gaps: tuple[str, ...]
    isolated_benchmark_artifact_id: str | None
    isolated_benchmark_evidence: EnzymeMLIRBenchmarkAttachment | None
    native_enzyme_execution_artifact_id: str | None
    native_enzyme_execution_evidence: EnzymeNativeExecutionEvidence | None = None
    mlir_llvm_correctness_evidence: MLIRLLVMCorrectnessEvidence | None = None
    compiler_ad_breadth_evidence: EnzymeMLIRCompilerADBreadthEvidence | None = None
    compiler_ad_breadth_artifact: EnzymeMLIRCompilerADBreadthArtifact | None = None
    claim_boundary: str = "bounded_enzyme_mlir_compiler_maturity_audit"

    def __post_init__(self) -> None:
        if not isinstance(self.scpn_mlir_runtime_verified, bool):
            raise ValueError("scpn_mlir_runtime_verified must be a bool")
        if not self.native_llvm_jit_surface:
            raise ValueError("native_llvm_jit_surface must be non-empty")
        if not self.toolchain:
            raise ValueError("toolchain status map must be non-empty")
        if any(
            not isinstance(status, EnzymeMLIRToolchainStatus) for status in self.toolchain.values()
        ):
            raise ValueError("toolchain values must be EnzymeMLIRToolchainStatus")
        if not self.correctness_checks:
            raise ValueError("correctness_checks must be non-empty")
        if any(not isinstance(value, bool) for value in self.correctness_checks.values()):
            raise ValueError("correctness checks must be bool values")
        if any(not gap for gap in self.hard_gaps):
            raise ValueError("hard gap entries must be non-empty")
        if self.isolated_benchmark_artifact_id is not None and not (
            self.isolated_benchmark_artifact_id.strip()
        ):
            raise ValueError("isolated_benchmark_artifact_id must be non-empty when provided")
        if self.isolated_benchmark_evidence is not None and not isinstance(
            self.isolated_benchmark_evidence,
            EnzymeMLIRBenchmarkAttachment,
        ):
            raise ValueError("isolated_benchmark_evidence must be EnzymeMLIRBenchmarkAttachment")
        if (
            self.isolated_benchmark_evidence is not None
            and self.isolated_benchmark_artifact_id is not None
            and self.isolated_benchmark_evidence.benchmark_artifact_id
            != self.isolated_benchmark_artifact_id
        ):
            raise ValueError(
                "isolated_benchmark_evidence.benchmark_artifact_id must match "
                "isolated_benchmark_artifact_id"
            )
        if self.native_enzyme_execution_artifact_id is not None and not (
            self.native_enzyme_execution_artifact_id.strip()
        ):
            raise ValueError("native_enzyme_execution_artifact_id must be non-empty when provided")
        if self.native_enzyme_execution_evidence is not None and not isinstance(
            self.native_enzyme_execution_evidence,
            EnzymeNativeExecutionEvidence,
        ):
            raise ValueError(
                "native_enzyme_execution_evidence must be EnzymeNativeExecutionEvidence"
            )
        if (
            self.native_enzyme_execution_evidence is not None
            and self.native_enzyme_execution_artifact_id is not None
            and self.native_enzyme_execution_evidence.artifact_id
            != self.native_enzyme_execution_artifact_id
        ):
            raise ValueError("native_enzyme_execution_artifact_id must match attached evidence")
        if self.mlir_llvm_correctness_evidence is not None and not isinstance(
            self.mlir_llvm_correctness_evidence,
            MLIRLLVMCorrectnessEvidence,
        ):
            raise ValueError("mlir_llvm_correctness_evidence must be MLIRLLVMCorrectnessEvidence")
        if self.compiler_ad_breadth_evidence is not None and not isinstance(
            self.compiler_ad_breadth_evidence,
            EnzymeMLIRCompilerADBreadthEvidence,
        ):
            raise ValueError(
                "compiler_ad_breadth_evidence must be EnzymeMLIRCompilerADBreadthEvidence"
            )
        if self.compiler_ad_breadth_artifact is not None and not isinstance(
            self.compiler_ad_breadth_artifact,
            EnzymeMLIRCompilerADBreadthArtifact,
        ):
            raise ValueError(
                "compiler_ad_breadth_artifact must be EnzymeMLIRCompilerADBreadthArtifact"
            )
        if (
            self.compiler_ad_breadth_evidence is not None
            and self.isolated_benchmark_artifact_id is not None
            and self.compiler_ad_breadth_evidence.isolated_benchmark_artifact_id
            != self.isolated_benchmark_artifact_id
        ):
            raise ValueError(
                "compiler_ad_breadth_evidence.isolated_benchmark_artifact_id must match "
                "isolated_benchmark_artifact_id"
            )
        if (
            self.compiler_ad_breadth_artifact is not None
            and self.compiler_ad_breadth_evidence is not None
            and self.compiler_ad_breadth_artifact.artifact_id
            != self.compiler_ad_breadth_evidence.artifact_id
        ):
            raise ValueError("compiler_ad_breadth_artifact must match attached breadth evidence")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "toolchain", MappingProxyType(dict(self.toolchain)))
        object.__setattr__(
            self,
            "correctness_checks",
            MappingProxyType(dict(self.correctness_checks)),
        )

    @property
    def ready_for_provider_exceedance(self) -> bool:
        """Return whether Enzyme/MLIR can be promoted beyond bounded SCPN evidence."""

        return (
            self.scpn_mlir_runtime_verified
            and all(status.available for status in self.toolchain.values())
            and all(self.correctness_checks.values())
            and self.mlir_llvm_correctness_evidence is not None
            and self.mlir_llvm_correctness_evidence.passed
            and self.isolated_benchmark_artifact_id is not None
            and self.isolated_benchmark_evidence is not None
            and self.isolated_benchmark_evidence.promotion_ready
            and self.native_enzyme_execution_artifact_id is not None
            and self.native_enzyme_execution_evidence is not None
            and self.native_enzyme_execution_evidence.passed
            and self.compiler_ad_breadth_evidence is not None
            and self.compiler_ad_breadth_evidence.passed
            and (
                self.compiler_ad_breadth_artifact is None
                or self.compiler_ad_breadth_artifact.promotion_ready
            )
            and not self.hard_gaps
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready audit metadata for docs, ledgers, and benchmarks."""

        return {
            "scpn_mlir_runtime_verified": self.scpn_mlir_runtime_verified,
            "native_llvm_jit_surface": self.native_llvm_jit_surface,
            "toolchain": {command: status.to_dict() for command, status in self.toolchain.items()},
            "correctness_checks": dict(self.correctness_checks),
            "hard_gaps": list(self.hard_gaps),
            "isolated_benchmark_artifact_id": self.isolated_benchmark_artifact_id,
            "isolated_benchmark_evidence": (
                self.isolated_benchmark_evidence.to_dict()
                if self.isolated_benchmark_evidence is not None
                else None
            ),
            "native_enzyme_execution_artifact_id": self.native_enzyme_execution_artifact_id,
            "native_enzyme_execution_evidence": (
                self.native_enzyme_execution_evidence.to_dict()
                if self.native_enzyme_execution_evidence is not None
                else None
            ),
            "mlir_llvm_correctness_evidence": (
                self.mlir_llvm_correctness_evidence.to_dict()
                if self.mlir_llvm_correctness_evidence is not None
                else None
            ),
            "compiler_ad_breadth_evidence": (
                self.compiler_ad_breadth_evidence.to_dict()
                if self.compiler_ad_breadth_evidence is not None
                else None
            ),
            "compiler_ad_breadth_artifact": (
                self.compiler_ad_breadth_artifact.to_dict()
                if self.compiler_ad_breadth_artifact is not None
                else None
            ),
            "ready_for_provider_exceedance": self.ready_for_provider_exceedance,
            "claim_boundary": self.claim_boundary,
        }


def build_enzyme_mlir_benchmark_attachment(
    *,
    validation: PhaseQNodeAffinityArtifactValidation,
    required_breadth_cases: Sequence[str],
    claim_boundary: str,
) -> EnzymeMLIRBenchmarkAttachment:
    """Build a fail-closed Enzyme/MLIR isolated benchmark attachment.

    Parameters
    ----------
    validation:
        Existing Phase-QNode affinity artefact validation. The resulting
        attachment is promotional only when this validation is already
        ``isolated_affinity`` and promotion-ready.
    required_breadth_cases:
        Breadth cases represented by the benchmark attachment. The set must
        match the Enzyme/MLIR compiler-AD breadth contract exactly.
    claim_boundary:
        Claim boundary stored in the serialized maturity audit.

    Returns
    -------
    EnzymeMLIRBenchmarkAttachment
        Validated benchmark attachment for maturity-audit input.

    Raises
    ------
    ValueError
        If the validation type, breadth cases, or claim boundary are malformed.
    """

    return EnzymeMLIRBenchmarkAttachment(
        validation=validation,
        required_breadth_cases=tuple(required_breadth_cases),
        claim_boundary=claim_boundary,
    )


def build_enzyme_mlir_compiler_ad_breadth_artifact(
    *,
    artifact_id: str,
    cases: Sequence[EnzymeMLIRCompilerADBreadthCaseEvidence],
    isolated_benchmark_evidence: EnzymeMLIRBenchmarkAttachment,
    claim_boundary: str,
) -> EnzymeMLIRCompilerADBreadthArtifact:
    """Build a validated raw Enzyme/MLIR compiler-AD breadth artefact.

    Parameters
    ----------
    artifact_id:
        Stable raw artefact identifier.
    cases:
        Case rows that must exactly cover every required Enzyme/MLIR breadth
        case.
    isolated_benchmark_evidence:
        Promotion-ready isolated benchmark attachment for the same evidence
        chain. Non-promotional benchmark evidence keeps the artefact blocked.
    claim_boundary:
        Claim boundary serialized into the artifact and derived evidence.

    Returns
    -------
    EnzymeMLIRCompilerADBreadthArtifact
        Validated raw breadth artefact.
    """

    return EnzymeMLIRCompilerADBreadthArtifact(
        artifact_id=artifact_id,
        cases=tuple(cases),
        isolated_benchmark_evidence=isolated_benchmark_evidence,
        claim_boundary=claim_boundary,
    )


def build_enzyme_mlir_compiler_ad_breadth_gap_artifact(
    *,
    artifact_id: str,
    observed_cases: Sequence[EnzymeMLIRCompilerADBreadthCaseEvidence],
    isolated_benchmark_evidence: EnzymeMLIRBenchmarkAttachment,
    default_transform_modes: Sequence[str] = ("forward", "reverse", "jvp", "vjp"),
    default_frontend_language: str = "mlir",
    missing_case_failure_class: str = "missing_case_evidence",
    missing_case_setup_instructions: str = (
        "Attach raw Enzyme/MLIR compiler-AD case evidence before promotion."
    ),
    claim_boundary: str = (
        "bounded Enzyme/MLIR compiler-AD breadth artifact; missing cases remain explicit hard gaps"
    ),
) -> EnzymeMLIRCompilerADBreadthArtifact:
    """Build a complete raw breadth artefact from partial observed case rows.

    Parameters
    ----------
    artifact_id:
        Stable raw artefact identifier.
    observed_cases:
        Captured raw case rows. Duplicate case identifiers are rejected. Any
        required Enzyme/MLIR breadth case not present here is filled as a
        ``hard_gap`` row rather than silently omitted.
    isolated_benchmark_evidence:
        Benchmark attachment for the same evidence chain. Non-promotional
        benchmark evidence keeps the resulting artefact blocked.
    default_transform_modes:
        Transform modes assigned to generated hard-gap rows.
    default_frontend_language:
        Frontend or IR surface assigned to generated hard-gap rows.
    missing_case_failure_class:
        Failure class assigned to generated hard-gap rows.
    missing_case_setup_instructions:
        Remediation text assigned to generated hard-gap rows.
    claim_boundary:
        Claim boundary serialized into the artifact and generated hard-gap
        rows.

    Returns
    -------
    EnzymeMLIRCompilerADBreadthArtifact
        Complete 11-case breadth artefact. It is promotion-ready only when all
        rows pass and the benchmark attachment is promotion-ready.

    Raises
    ------
    ValueError
        If observed rows are duplicated or any default hard-gap metadata is
        malformed.
    """

    observed_by_case: dict[str, EnzymeMLIRCompilerADBreadthCaseEvidence] = {}
    for case in observed_cases:
        if not isinstance(case, EnzymeMLIRCompilerADBreadthCaseEvidence):
            raise ValueError("observed_cases must contain EnzymeMLIRCompilerADBreadthCaseEvidence")
        if case.case_id in observed_by_case:
            raise ValueError("observed_cases must not contain duplicate case identifiers")
        observed_by_case[case.case_id] = case
    if not missing_case_failure_class.strip():
        raise ValueError("missing_case_failure_class must be non-empty")
    if not missing_case_setup_instructions.strip():
        raise ValueError("missing_case_setup_instructions must be non-empty")
    rows: list[EnzymeMLIRCompilerADBreadthCaseEvidence] = []
    for case_id in sorted(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES):
        observed = observed_by_case.get(case_id)
        if observed is not None:
            rows.append(observed)
            continue
        rows.append(
            EnzymeMLIRCompilerADBreadthCaseEvidence(
                case_id=case_id,
                status="hard_gap",
                transform_modes=tuple(default_transform_modes),
                frontend_language=default_frontend_language,
                value_error=None,
                gradient_error=None,
                runtime_seconds=None,
                artifact_refs={"missing_case": f"enzyme_mlir_compiler_ad_breadth:{case_id}"},
                failure_class=missing_case_failure_class,
                setup_instructions=missing_case_setup_instructions,
                claim_boundary=claim_boundary,
            )
        )
    return build_enzyme_mlir_compiler_ad_breadth_artifact(
        artifact_id=artifact_id,
        cases=tuple(rows),
        isolated_benchmark_evidence=isolated_benchmark_evidence,
        claim_boundary=claim_boundary,
    )


def build_enzyme_mlir_compiler_ad_breadth_evidence(
    *,
    artifact_id: str,
    cases: Mapping[str, bool],
    transform_modes: Sequence[str],
    frontend_languages: Sequence[str],
    isolated_benchmark_artifact_id: str,
    max_abs_error: float,
    runtime_seconds: float,
    claim_boundary: str,
) -> EnzymeMLIRCompilerADBreadthEvidence:
    """Build validated Enzyme/MLIR compiler AD breadth evidence.

    Parameters
    ----------
    artifact_id:
        Stable identifier for the captured breadth evidence bundle.
    cases:
        Required breadth-case pass/fail map. It must cover every case in the
        Enzyme/MLIR compiler AD breadth contract.
    transform_modes:
        Compiler AD transform modes covered by the captured evidence.
    frontend_languages:
        Frontend or IR surfaces represented by the evidence bundle.
    isolated_benchmark_artifact_id:
        Matching isolated benchmark artefact identifier.
    max_abs_error:
        Maximum absolute correctness error across the captured cases.
    runtime_seconds:
        Positive runtime for the captured evidence runner.
    claim_boundary:
        Explicit boundary that prevents breadth evidence from becoming a
        provider, hardware, or fabricated performance claim.

    Returns
    -------
    EnzymeMLIRCompilerADBreadthEvidence
        Validated evidence ready for maturity-audit attachment.

    Raises
    ------
    ValueError
        If case coverage, transform modes, languages, benchmark linkage,
        errors, runtime, or claim boundary are malformed.
    """

    return EnzymeMLIRCompilerADBreadthEvidence(
        artifact_id=artifact_id,
        cases=cases,
        transform_modes=tuple(transform_modes),
        frontend_languages=tuple(frontend_languages),
        isolated_benchmark_artifact_id=isolated_benchmark_artifact_id,
        max_abs_error=max_abs_error,
        runtime_seconds=runtime_seconds,
        claim_boundary=claim_boundary,
    )
