# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable hardening gate.
"""Per-slice differentiable-programming hardening gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .differentiable_evidence import (
    AcceleratorEvidenceMetadata,
    BenchmarkIsolationMetadata,
)

DEFAULT_CLAIM_LEDGER_VALIDATION_TARGET = "tests/test_differentiable_claim_ledger.py"
DEFAULT_TEST_QUALITY_AUDIT_TARGET = "tools/audit_test_quality.py"


@dataclass(frozen=True)
class DifferentiableHardeningGateCheck:
    """One required verification surface for a differentiable hardening slice."""

    check_id: str
    command: tuple[str, ...]
    passed: bool
    evidence: str

    def __post_init__(self) -> None:
        """Validate that the verification row has executable evidence fields."""
        if not self.check_id:
            raise ValueError("check_id must be non-empty")
        if not self.command:
            raise ValueError("command must be non-empty")
        if not self.evidence:
            raise ValueError("evidence must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready check row."""
        return {
            "check_id": self.check_id,
            "command": list(self.command),
            "passed": self.passed,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class DifferentiableBenchmarkClassificationCase:
    """Expected benchmark-evidence classification for one runner scenario."""

    case_id: str
    expected_classification: str
    expected_failure_class: str | None
    metadata: BenchmarkIsolationMetadata

    @property
    def passed(self) -> bool:
        """Return whether the observed classification matches the contract."""
        return (
            self.metadata.classification == self.expected_classification
            and self.metadata.failure_class == self.expected_failure_class
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready benchmark classification evidence."""
        return {
            "case_id": self.case_id,
            "expected_classification": self.expected_classification,
            "expected_failure_class": self.expected_failure_class,
            "observed_classification": self.metadata.classification,
            "observed_failure_class": self.metadata.failure_class,
            "production_eligible": self.metadata.production_eligible,
            "passed": self.passed,
            "metadata": self.metadata.to_dict(),
        }


@dataclass(frozen=True)
class DifferentiableHardeningSliceGateResult:
    """Auditable verification checklist for one differentiable hardening slice."""

    passed: bool
    checks: tuple[DifferentiableHardeningGateCheck, ...]
    benchmark_classification_cases: tuple[DifferentiableBenchmarkClassificationCase, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready hardening-gate evidence."""
        return {
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
            "benchmark_classification_cases": [
                case.to_dict() for case in self.benchmark_classification_cases
            ],
            "claim_boundary": self.claim_boundary,
        }


def run_differentiable_hardening_slice_gate(
    *,
    module_specific_pytest_targets: Sequence[str],
    changed_python_targets: Sequence[str] = (),
    claim_ledger_validation_target: str = DEFAULT_CLAIM_LEDGER_VALIDATION_TARGET,
    test_quality_audit_target: str = DEFAULT_TEST_QUALITY_AUDIT_TARGET,
) -> DifferentiableHardeningSliceGateResult:
    """Build the required verification checklist for a hardening slice.

    The gate verifies command coverage and benchmark-classification invariants.
    It does not execute the commands and does not promote local benchmark rows
    to production evidence.
    """
    pytest_targets = _normalise_targets(
        module_specific_pytest_targets,
        field_name="module_specific_pytest_targets",
        disallowed_exact={"tests"},
    )
    python_targets = _normalise_targets(
        changed_python_targets,
        field_name="changed_python_targets",
        allow_empty=True,
    )
    ledger_target = _normalise_single_target(
        claim_ledger_validation_target,
        field_name="claim_ledger_validation_target",
    )
    quality_target = _normalise_single_target(
        test_quality_audit_target,
        field_name="test_quality_audit_target",
    )
    checks = (
        DifferentiableHardeningGateCheck(
            check_id="ruff_format",
            command=("./.venv/bin/ruff", "format", *python_targets, *pytest_targets),
            passed=bool(python_targets or pytest_targets),
            evidence="Focused formatter command is scoped to changed source and module-specific tests.",
        ),
        DifferentiableHardeningGateCheck(
            check_id="ruff_check",
            command=("./.venv/bin/ruff", "check", *python_targets, *pytest_targets),
            passed=bool(python_targets or pytest_targets),
            evidence="Focused lint command is scoped to changed source and module-specific tests.",
        ),
        DifferentiableHardeningGateCheck(
            check_id="mypy",
            command=("./.venv/bin/mypy", *python_targets),
            passed=bool(python_targets),
            evidence="Typed source targets are explicit; test-only slices should pass a source surface when public API changes.",
        ),
        DifferentiableHardeningGateCheck(
            check_id="module_specific_pytest",
            command=(
                "PYTHONPATH=src",
                "./.venv/bin/python",
                "-m",
                "pytest",
                *pytest_targets,
                "-q",
            ),
            passed=all(target.startswith("tests/test_") for target in pytest_targets),
            evidence="Pytest targets are module-specific files, not bucket-wide test directories.",
        ),
        DifferentiableHardeningGateCheck(
            check_id="test_quality_audit",
            command=("PYTHONPATH=src", "./.venv/bin/python", quality_target),
            passed=quality_target == DEFAULT_TEST_QUALITY_AUDIT_TARGET,
            evidence="Repository test-quality audit remains part of every differentiable hardening closeout.",
        ),
        DifferentiableHardeningGateCheck(
            check_id="claim_ledger_validation",
            command=(
                "PYTHONPATH=src",
                "./.venv/bin/python",
                "-m",
                "pytest",
                ledger_target,
                "-q",
            ),
            passed=ledger_target.startswith("tests/test_"),
            evidence="Claim-ledger validation stays explicit when a slice affects public differentiable claims.",
        ),
    )
    classification_cases = _benchmark_classification_cases()
    passed = all(check.passed for check in checks) and all(
        case.passed for case in classification_cases
    )
    return DifferentiableHardeningSliceGateResult(
        passed=passed,
        checks=checks,
        benchmark_classification_cases=classification_cases,
        claim_boundary=(
            "This hardening gate validates per-slice verification coverage and "
            "benchmark-evidence classification invariants. It does not execute "
            "the listed commands, does not submit provider or hardware jobs, and "
            "does not upgrade functional_non_isolated benchmark rows to "
            "isolated_affinity evidence."
        ),
    )


def _normalise_targets(
    targets: Sequence[str],
    *,
    field_name: str,
    disallowed_exact: set[str] | None = None,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    normalised = tuple(str(target).strip() for target in targets if str(target).strip())
    if not normalised and not allow_empty:
        raise ValueError(f"{field_name} must contain at least one target")
    disallowed = disallowed_exact or set()
    blocked = sorted(target for target in normalised if target in disallowed)
    if blocked:
        raise ValueError(f"{field_name} contains bucket target(s): {', '.join(blocked)}")
    return normalised


def _normalise_single_target(target: str, *, field_name: str) -> str:
    normalised = str(target).strip()
    if not normalised:
        raise ValueError(f"{field_name} must be non-empty")
    return normalised


def _benchmark_classification_cases() -> tuple[DifferentiableBenchmarkClassificationCase, ...]:
    command = ("python", "scripts/run_differentiable_benchmark_evidence.py")
    cases: list[DifferentiableBenchmarkClassificationCase] = []
    cases.append(
        _classification_case(
            case_id="github_hosted_functional_non_isolated",
            expected_classification="functional_non_isolated",
            expected_failure_class="non_isolated_runner",
            env={
                "RUNNER_ENVIRONMENT": "github-hosted",
                "RUNNER_LABELS": "ubuntu-latest,linux",
                "RUNNER_NAME": "GitHub Actions 42",
            },
            command=command,
            cpu_affinity=None,
            isolation_method=None,
            load_before=(1.0, 1.0, 1.0),
            load_after=(1.0, 1.0, 1.0),
            governor="performance",
            frequency_mhz=3200.0,
            heavy_jobs_running=False,
            accelerator_metadata=AcceleratorEvidenceMetadata.cpu_only(),
        )
    )
    cases.append(
        _classification_case(
            case_id="self_hosted_isolated_missing_context_hard_gap",
            expected_classification="hard_gap",
            expected_failure_class="insufficient_isolation_metadata",
            env={
                "RUNNER_ENVIRONMENT": "self-hosted",
                "RUNNER_LABELS": "self-hosted,linux,isolated-benchmark",
                "RUNNER_NAME": "isolated-qc-runner",
            },
            command=command,
            cpu_affinity="2",
            isolation_method="taskset",
            load_before=None,
            load_after=None,
            governor=None,
            frequency_mhz=None,
            heavy_jobs_running=False,
            accelerator_metadata=AcceleratorEvidenceMetadata.cpu_only(),
        )
    )
    cases.append(
        _classification_case(
            case_id="self_hosted_isolated_affinity",
            expected_classification="isolated_affinity",
            expected_failure_class=None,
            env={
                "RUNNER_ENVIRONMENT": "self-hosted",
                "RUNNER_LABELS": "self-hosted,linux,isolated-benchmark",
                "RUNNER_NAME": "isolated-qc-runner",
            },
            command=("taskset", "-c", "2", *command),
            cpu_affinity="2",
            isolation_method="taskset",
            load_before=(0.05, 0.04, 0.03),
            load_after=(0.06, 0.05, 0.04),
            governor="performance",
            frequency_mhz=3200.0,
            heavy_jobs_running=False,
            accelerator_metadata=AcceleratorEvidenceMetadata.cpu_only(),
        )
    )
    cases.append(
        _classification_case(
            case_id="requested_accelerator_fallback_hard_gap",
            expected_classification="hard_gap",
            expected_failure_class="silent_accelerator_fallback",
            env={
                "RUNNER_ENVIRONMENT": "self-hosted",
                "RUNNER_LABELS": "self-hosted,linux,isolated-benchmark",
                "RUNNER_NAME": "isolated-qc-runner",
            },
            command=("taskset", "-c", "2", *command),
            cpu_affinity="2",
            isolation_method="taskset",
            load_before=(0.05, 0.04, 0.03),
            load_after=(0.06, 0.05, 0.04),
            governor="performance",
            frequency_mhz=3200.0,
            heavy_jobs_running=False,
            accelerator_metadata=AcceleratorEvidenceMetadata(
                requested_backend="cuda",
                detected_backend="cpu",
                device_ids=(),
                device_names=(),
                runtime_versions={"cuda": "12.4"},
                cpu_fallback_detected=True,
                claim_boundary=(
                    "Requested accelerator execution has no visible device evidence; "
                    "the row is a hard gap, not accelerator performance evidence."
                ),
            ),
        )
    )
    return tuple(cases)


def _classification_case(
    *,
    case_id: str,
    expected_classification: str,
    expected_failure_class: str | None,
    env: Mapping[str, str],
    command: Sequence[str],
    cpu_affinity: str | None,
    isolation_method: str | None,
    load_before: tuple[float, float, float] | None,
    load_after: tuple[float, float, float] | None,
    governor: str | None,
    frequency_mhz: float | None,
    heavy_jobs_running: bool,
    accelerator_metadata: AcceleratorEvidenceMetadata,
) -> DifferentiableBenchmarkClassificationCase:
    metadata = BenchmarkIsolationMetadata.from_ci_environment(
        env,
        command=command,
        cpu_affinity=cpu_affinity,
        isolation_method=isolation_method,
        load_before=load_before,
        load_after=load_after,
        governor=governor,
        frequency_mhz=frequency_mhz,
        heavy_jobs_running=heavy_jobs_running,
        accelerator_metadata=accelerator_metadata,
    )
    return DifferentiableBenchmarkClassificationCase(
        case_id=case_id,
        expected_classification=expected_classification,
        expected_failure_class=expected_failure_class,
        metadata=metadata,
    )


__all__ = [
    "DifferentiableBenchmarkClassificationCase",
    "DifferentiableHardeningGateCheck",
    "DifferentiableHardeningSliceGateResult",
    "run_differentiable_hardening_slice_gate",
]
