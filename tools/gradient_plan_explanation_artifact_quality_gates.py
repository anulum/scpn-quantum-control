# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — gradient-plan explanation artefact quality gates
"""Build strict documentation, drift, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
GRADIENT_PLAN_EXPLANATION_ARTIFACT_SOURCE = (
    "src/scpn_quantum_control/gradient_plan_explanation_artifact.py"
)
GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_COHORT = [
    "tests/test_gradient_plan_explanation_artifact.py",
]
GRADIENT_PLAN_EXPLANATION_ARTIFACT_TYPING_RATCHET = [
    GRADIENT_PLAN_EXPLANATION_ARTIFACT_SOURCE,
    "tools/gradient_plan_explanation_artifact_quality_gates.py",
    "tests/test_gradient_plan_explanation_artifact_quality_gate.py",
]
GRADIENT_PLAN_EXPLANATION_ARTIFACT_DOCSTRING_RATCHET = [
    GRADIENT_PLAN_EXPLANATION_ARTIFACT_SOURCE,
    *GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_COHORT,
    "tools/gradient_plan_explanation_artifact_quality_gates.py",
    "tests/test_gradient_plan_explanation_artifact_quality_gate.py",
]
GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-gradient-plan-explanation-artifact-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing, NumPy-docstring, and artefact-drift gates."""
    return [
        (
            "mypy-strict-gradient-plan-explanation-artifact-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *GRADIENT_PLAN_EXPLANATION_ARTIFACT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D gradient-plan-explanation-artifact quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--select",
                "D,D413",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *GRADIENT_PLAN_EXPLANATION_ARTIFACT_DOCSTRING_RATCHET,
            ],
        ),
        (
            "gradient-plan explanation committed artefact drift",
            [
                python,
                "-m",
                "scpn_quantum_control.gradient_plan_explanation_artifact",
                "--check",
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline execution and exact source coverage gates."""
    return [
        (
            "gradient-plan-explanation-artifact focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_COHORT,
            ],
        ),
        (
            "gradient-plan-explanation-artifact exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/gradient_plan_explanation_artifact.py",
            ],
        ),
    ]


__all__ = [
    "GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_COHORT",
    "GRADIENT_PLAN_EXPLANATION_ARTIFACT_COVERAGE_DATA_FILE",
    "GRADIENT_PLAN_EXPLANATION_ARTIFACT_DOCSTRING_RATCHET",
    "GRADIENT_PLAN_EXPLANATION_ARTIFACT_SOURCE",
    "GRADIENT_PLAN_EXPLANATION_ARTIFACT_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
