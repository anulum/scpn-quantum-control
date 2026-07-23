# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — MLIR/Enzyme evidence contract tests
"""Exercise every fail-closed MLIR/Enzyme evidence-record boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, cast

import pytest

from scpn_quantum_control.compiler import mlir_enzyme_evidence as evidence
from scpn_quantum_control.compiler.mlir_enzyme_evidence import (
    ENZYME_MLIR_COMPILER_AD_BREADTH_CASES,
    ENZYME_MLIR_COMPILER_AD_TRANSFORM_MODES,
    EnzymeMLIRBenchmarkAttachment,
    EnzymeMLIRCompilerADBreadthArtifact,
    EnzymeMLIRCompilerADBreadthCaseEvidence,
    EnzymeMLIRCompilerADBreadthEvidence,
    EnzymeMLIRMaturityAuditResult,
    EnzymeMLIRToolchainStatus,
    EnzymeNativeExecutionEvidence,
    MLIRLLVMCorrectnessEvidence,
    build_enzyme_mlir_compiler_ad_breadth_gap_artifact,
)
from scpn_quantum_control.phase.qnode_affinity_benchmark import (
    PhaseQNodeAffinityArtifactValidation,
)


def _toolchain() -> EnzymeMLIRToolchainStatus:
    """Return a valid available toolchain row."""
    return EnzymeMLIRToolchainStatus(
        command="enzyme",
        executable="/opt/bin/enzyme",
        available=True,
        version="1.0",
        failure_class=None,
        setup_instructions=None,
    )


def _native() -> EnzymeNativeExecutionEvidence:
    """Return valid native execution evidence."""
    return EnzymeNativeExecutionEvidence(
        artifact_id="native-001",
        status="success",
        failure_class=None,
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        toolchain={"enzyme": "1.0"},
        setup_instructions=None,
        claim_boundary="bounded native execution",
    )


def _correctness() -> MLIRLLVMCorrectnessEvidence:
    """Return valid MLIR/LLVM correctness evidence."""
    return MLIRLLVMCorrectnessEvidence(
        artifact_id="correctness-001",
        checks={"value": True, "gradient": True},
        toolchain_versions={"llvm": "18.1.3"},
        claim_boundary="bounded MLIR/LLVM correctness",
    )


def _validation(
    *, benchmark_artifact_id: str = "benchmark-001"
) -> PhaseQNodeAffinityArtifactValidation:
    """Return a promotion-ready isolated benchmark validation."""
    return PhaseQNodeAffinityArtifactValidation(
        artifact_path="data/benchmarks/enzyme_mlir.json",
        artifact_sha256="a" * 64,
        benchmark_artifact_id=benchmark_artifact_id,
        evidence_label="isolated_affinity",
        production_benchmark=True,
        promotion_ready=True,
        raw_timing_row_count=2,
        missing_requirements=(),
        claim_boundary="bounded isolated benchmark validation",
    )


def _attachment(*, benchmark_artifact_id: str = "benchmark-001") -> EnzymeMLIRBenchmarkAttachment:
    """Return a valid benchmark attachment covering every breadth case."""
    return EnzymeMLIRBenchmarkAttachment(
        validation=_validation(benchmark_artifact_id=benchmark_artifact_id),
        required_breadth_cases=tuple(sorted(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES)),
        claim_boundary="bounded benchmark attachment",
    )


def _case(case_id: str = "scalar_forward_mode") -> EnzymeMLIRCompilerADBreadthCaseEvidence:
    """Return one valid passing compiler-AD breadth row."""
    return EnzymeMLIRCompilerADBreadthCaseEvidence(
        case_id=case_id,
        status="success",
        transform_modes=tuple(sorted(ENZYME_MLIR_COMPILER_AD_TRANSFORM_MODES)),
        frontend_language="mlir",
        value_error=0.0,
        gradient_error=0.0,
        runtime_seconds=0.01,
        artifact_refs={"raw": f"{case_id}.json"},
        failure_class=None,
        setup_instructions=None,
        claim_boundary="bounded compiler-AD case",
    )


def _artifact(*, artifact_id: str = "breadth-001") -> EnzymeMLIRCompilerADBreadthArtifact:
    """Return a complete passing raw breadth artefact."""
    return EnzymeMLIRCompilerADBreadthArtifact(
        artifact_id=artifact_id,
        cases=tuple(_case(case_id) for case_id in sorted(ENZYME_MLIR_COMPILER_AD_BREADTH_CASES)),
        isolated_benchmark_evidence=_attachment(),
        claim_boundary="bounded raw breadth artifact",
    )


def _breadth(
    *,
    artifact_id: str = "breadth-001",
    isolated_benchmark_artifact_id: str = "benchmark-001",
) -> EnzymeMLIRCompilerADBreadthEvidence:
    """Return complete passing compiler-AD breadth evidence."""
    return EnzymeMLIRCompilerADBreadthEvidence(
        artifact_id=artifact_id,
        cases={case_id: True for case_id in ENZYME_MLIR_COMPILER_AD_BREADTH_CASES},
        transform_modes=tuple(sorted(ENZYME_MLIR_COMPILER_AD_TRANSFORM_MODES)),
        frontend_languages=("llvm_ir", "mlir"),
        isolated_benchmark_artifact_id=isolated_benchmark_artifact_id,
        max_abs_error=0.0,
        runtime_seconds=0.1,
        claim_boundary="bounded compiler-AD breadth evidence",
    )


def _audit(**overrides: Any) -> EnzymeMLIRMaturityAuditResult:
    """Return a structurally valid maturity result with optional fields absent."""
    kwargs: dict[str, Any] = {
        "scpn_mlir_runtime_verified": False,
        "native_llvm_jit_surface": "bounded JIT surface",
        "toolchain": {"enzyme": _toolchain()},
        "correctness_checks": {"runtime": True},
        "hard_gaps": (),
        "isolated_benchmark_artifact_id": None,
        "isolated_benchmark_evidence": None,
        "native_enzyme_execution_artifact_id": None,
        "native_enzyme_execution_evidence": None,
        "mlir_llvm_correctness_evidence": None,
        "compiler_ad_breadth_evidence": None,
        "compiler_ad_breadth_artifact": None,
        "claim_boundary": "bounded maturity audit",
    }
    kwargs.update(overrides)
    return EnzymeMLIRMaturityAuditResult(**kwargs)


def _assert_invalid(factory: Callable[[], object], match: str) -> None:
    """Assert one constructor or replacement fails with its named contract."""
    with pytest.raises(ValueError, match=match):
        factory()


@pytest.mark.parametrize(
    ("match", "factory"),
    [
        ("command", lambda: replace(_toolchain(), command="")),
        ("executable and version", lambda: replace(_toolchain(), executable=None)),
        ("must not carry hard-gap", lambda: replace(_toolchain(), failure_class="unexpected")),
        (
            "must not carry executable",
            lambda: EnzymeMLIRToolchainStatus("enzyme", "/bin/enzyme", False, None, "gap", "fix"),
        ),
        (
            "require failure_class",
            lambda: EnzymeMLIRToolchainStatus("enzyme", None, False, None, None, "fix"),
        ),
    ],
)
def test_toolchain_status_rejects_incoherent_rows(
    match: str, factory: Callable[[], object]
) -> None:
    """Toolchain rows must preserve available versus hard-gap coupling."""
    _assert_invalid(factory, match)


@pytest.mark.parametrize(
    ("match", "factory"),
    [
        ("artifact_id", lambda: replace(_native(), artifact_id=" ")),
        ("status", lambda: replace(_native(), status="unknown")),
        ("toolchain metadata", lambda: replace(_native(), toolchain={"": "1.0"})),
        ("must not carry hard-gap", lambda: replace(_native(), failure_class="unexpected")),
        (
            "must not carry success metrics",
            lambda: replace(
                _native(),
                status="hard_gap",
                failure_class="missing_runtime",
                setup_instructions="install enzyme",
            ),
        ),
        ("claim_boundary", lambda: replace(_native(), claim_boundary="")),
    ],
)
def test_native_execution_rejects_incoherent_evidence(
    match: str, factory: Callable[[], object]
) -> None:
    """Native evidence must keep status, metrics, and hard-gap fields coupled."""
    _assert_invalid(factory, match)


@pytest.mark.parametrize(
    ("match", "factory"),
    [
        ("artifact_id", lambda: replace(_correctness(), artifact_id="")),
        ("checks must map", lambda: replace(_correctness(), checks={"": True})),
        (
            "toolchain_versions",
            lambda: replace(_correctness(), toolchain_versions={"llvm": ""}),
        ),
        ("claim_boundary", lambda: replace(_correctness(), claim_boundary="")),
        (
            "validation must be",
            lambda: replace(_attachment(), validation=cast(Any, object())),
        ),
        (
            "required_breadth_cases",
            lambda: replace(
                _attachment(),
                required_breadth_cases=tuple(
                    sorted((*ENZYME_MLIR_COMPILER_AD_BREADTH_CASES, "unknown_case"))
                ),
            ),
        ),
        ("claim_boundary", lambda: replace(_attachment(), claim_boundary=" ")),
    ],
)
def test_correctness_and_benchmark_records_reject_malformed_fields(
    match: str, factory: Callable[[], object]
) -> None:
    """Correctness and benchmark attachments must reject malformed linkage."""
    _assert_invalid(factory, match)


@pytest.mark.parametrize(
    ("match", "factory"),
    [
        ("case_id", lambda: replace(_case(), case_id="unknown")),
        ("status", lambda: replace(_case(), status="unknown")),
        ("transform_modes must contain", lambda: replace(_case(), transform_modes=())),
        ("valid Enzyme/MLIR", lambda: replace(_case(), transform_modes=("unknown",))),
        ("frontend_language", lambda: replace(_case(), frontend_language=" ")),
        ("artifact_refs", lambda: replace(_case(), artifact_refs={})),
        ("positive runtime", lambda: replace(_case(), runtime_seconds=0.0)),
        ("must not carry hard-gap", lambda: replace(_case(), failure_class="unexpected")),
        (
            "require failure metadata",
            lambda: replace(
                _case(),
                status="hard_gap",
                value_error=None,
                gradient_error=None,
                runtime_seconds=None,
            ),
        ),
        (
            "must not carry success metrics",
            lambda: replace(
                _case(),
                status="hard_gap",
                failure_class="missing_case",
                setup_instructions="capture evidence",
            ),
        ),
        ("claim_boundary", lambda: replace(_case(), claim_boundary=" ")),
    ],
)
def test_breadth_case_rejects_every_invalid_contract_boundary(
    match: str, factory: Callable[[], object]
) -> None:
    """Raw breadth rows must fail closed for malformed status and evidence."""
    _assert_invalid(factory, match)


def test_raw_artifact_rejects_invalid_linkage_and_case_sets() -> None:
    """Raw breadth artefacts must require one typed row per case and a boundary."""
    artifact = _artifact()
    cases = artifact.cases
    invalid: tuple[tuple[str, Callable[[], object]], ...] = (
        ("artifact_id", lambda: replace(artifact, artifact_id=" ")),
        (
            "isolated_benchmark_evidence",
            lambda: replace(artifact, isolated_benchmark_evidence=cast(Any, object())),
        ),
        ("rows", lambda: replace(artifact, cases=cast(Any, (*cases[:-1], object())))),
        ("exactly cover", lambda: replace(artifact, cases=cases[:-1])),
        ("duplicates", lambda: replace(artifact, cases=(*cases, cases[0]))),
        ("claim_boundary", lambda: replace(artifact, claim_boundary=" ")),
    )
    for match, factory in invalid:
        _assert_invalid(factory, match)


def test_promotion_evidence_rejects_every_invalid_contract_boundary() -> None:
    """Promotion evidence must require typed complete finite benchmark-linked data."""
    breadth = _breadth()
    integer_cases = cast(
        Mapping[str, bool],
        {case_id: 1 for case_id in ENZYME_MLIR_COMPILER_AD_BREADTH_CASES},
    )
    extra_cases = {
        **{case_id: True for case_id in ENZYME_MLIR_COMPILER_AD_BREADTH_CASES},
        "unknown": True,
    }
    invalid: tuple[tuple[str, Callable[[], object]], ...] = (
        ("artifact_id", lambda: replace(breadth, artifact_id=" ")),
        ("complete and passing", lambda: replace(breadth, cases=extra_cases)),
        ("map to booleans", lambda: replace(breadth, cases=integer_cases)),
        ("transform_modes", lambda: replace(breadth, transform_modes=("forward",))),
        ("frontend_languages", lambda: replace(breadth, frontend_languages=())),
        (
            "frontend_languages must be unique",
            lambda: replace(breadth, frontend_languages=("mlir", "MLIR")),
        ),
        (
            "isolated_benchmark_artifact_id",
            lambda: replace(breadth, isolated_benchmark_artifact_id=" "),
        ),
        ("max_abs_error", lambda: replace(breadth, max_abs_error=-1.0)),
        ("max_abs_error", lambda: replace(breadth, max_abs_error=float("nan"))),
        ("runtime_seconds", lambda: replace(breadth, runtime_seconds=0.0)),
        ("runtime_seconds", lambda: replace(breadth, runtime_seconds=float("inf"))),
        ("claim_boundary", lambda: replace(breadth, claim_boundary=" ")),
    )
    for match, factory in invalid:
        _assert_invalid(factory, match)


def test_maturity_result_rejects_every_invalid_attachment_boundary() -> None:
    """Maturity results must reject malformed maps, ids, types, and cross-links."""
    invalid: tuple[tuple[str, Callable[[], object]], ...] = (
        ("must be a bool", lambda: _audit(scpn_mlir_runtime_verified=cast(Any, 1))),
        ("native_llvm_jit_surface", lambda: _audit(native_llvm_jit_surface="")),
        ("toolchain status map", lambda: _audit(toolchain={})),
        ("toolchain values", lambda: _audit(toolchain={"enzyme": cast(Any, object())})),
        ("correctness_checks", lambda: _audit(correctness_checks={})),
        ("correctness checks", lambda: _audit(correctness_checks={"runtime": cast(Any, 1)})),
        ("hard gap entries", lambda: _audit(hard_gaps=("",))),
        ("benchmark_artifact_id", lambda: _audit(isolated_benchmark_artifact_id=" ")),
        (
            "isolated_benchmark_evidence must be",
            lambda: _audit(isolated_benchmark_evidence=cast(Any, object())),
        ),
        (
            "benchmark_artifact_id must match",
            lambda: _audit(
                isolated_benchmark_artifact_id="benchmark-001",
                isolated_benchmark_evidence=_attachment(benchmark_artifact_id="benchmark-002"),
            ),
        ),
        (
            "native_enzyme_execution_artifact_id must be non-empty",
            lambda: _audit(native_enzyme_execution_artifact_id=" "),
        ),
        (
            "native_enzyme_execution_evidence must be",
            lambda: _audit(native_enzyme_execution_evidence=cast(Any, object())),
        ),
        (
            "must match attached evidence",
            lambda: _audit(
                native_enzyme_execution_artifact_id="other-native",
                native_enzyme_execution_evidence=_native(),
            ),
        ),
        (
            "mlir_llvm_correctness_evidence must be",
            lambda: _audit(mlir_llvm_correctness_evidence=cast(Any, object())),
        ),
        (
            "compiler_ad_breadth_evidence must be",
            lambda: _audit(compiler_ad_breadth_evidence=cast(Any, object())),
        ),
        (
            "compiler_ad_breadth_artifact must be",
            lambda: _audit(compiler_ad_breadth_artifact=cast(Any, object())),
        ),
        (
            "isolated_benchmark_artifact_id must match",
            lambda: _audit(
                isolated_benchmark_artifact_id="benchmark-001",
                compiler_ad_breadth_evidence=_breadth(
                    isolated_benchmark_artifact_id="benchmark-002"
                ),
            ),
        ),
        (
            "artifact must match",
            lambda: _audit(
                compiler_ad_breadth_evidence=_breadth(artifact_id="breadth-evidence"),
                compiler_ad_breadth_artifact=_artifact(artifact_id="breadth-artifact"),
            ),
        ),
        ("claim_boundary", lambda: _audit(claim_boundary="")),
    )
    for match, factory in invalid:
        _assert_invalid(factory, match)


def test_gap_builder_and_custom_stem_fail_closed() -> None:
    """Gap synthesis must reject malformed rows/defaults and sanitize custom ids."""
    attachment = _attachment()
    invalid: tuple[tuple[str, Callable[[], object]], ...] = (
        (
            "observed_cases must contain",
            lambda: build_enzyme_mlir_compiler_ad_breadth_gap_artifact(
                artifact_id="gap",
                observed_cases=cast(
                    Sequence[EnzymeMLIRCompilerADBreadthCaseEvidence], (object(),)
                ),
                isolated_benchmark_evidence=attachment,
            ),
        ),
        (
            "missing_case_failure_class",
            lambda: build_enzyme_mlir_compiler_ad_breadth_gap_artifact(
                artifact_id="gap",
                observed_cases=(),
                isolated_benchmark_evidence=attachment,
                missing_case_failure_class=" ",
            ),
        ),
        (
            "missing_case_setup_instructions",
            lambda: build_enzyme_mlir_compiler_ad_breadth_gap_artifact(
                artifact_id="gap",
                observed_cases=(),
                isolated_benchmark_evidence=attachment,
                missing_case_setup_instructions=" ",
            ),
        ),
    )
    for match, factory in invalid:
        _assert_invalid(factory, match)

    assert evidence._enzyme_mlir_breadth_artifact_stem("custom-artifact-id") == (
        "custom_artifact_id"
    )
