# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — transform support-matrix artefact quality gates
"""Build strict documentation, drift, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SOURCE = (
    "src/scpn_quantum_control/differentiable_transform_support_matrix_artifact.py"
)
TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_COHORT = [
    "tests/test_differentiable_transform_support_matrix_artifact.py",
    "tests/test_studio_support_matrix_bundle.py",
]
TRANSFORM_SUPPORT_MATRIX_ARTIFACT_TYPING_RATCHET = [
    TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SOURCE,
    "tools/differentiable_transform_support_matrix_artifact_quality_gates.py",
    "tests/test_differentiable_transform_support_matrix_artifact_quality_gate.py",
]
TRANSFORM_SUPPORT_MATRIX_ARTIFACT_DOCSTRING_RATCHET = [
    TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SOURCE,
    *TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_COHORT,
    "tools/differentiable_transform_support_matrix_artifact_quality_gates.py",
    "tests/test_differentiable_transform_support_matrix_artifact_quality_gate.py",
]
TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-transform-support-matrix-artifact-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing, NumPy-docstring, and artefact-drift gates."""
    return [
        (
            "mypy-strict-transform-support-matrix-artifact-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *TRANSFORM_SUPPORT_MATRIX_ARTIFACT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D transform-support-matrix-artifact quality ratchet",
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
                *TRANSFORM_SUPPORT_MATRIX_ARTIFACT_DOCSTRING_RATCHET,
            ],
        ),
        (
            "transform-support-matrix committed artefact drift",
            [
                python,
                "-m",
                "scpn_quantum_control.differentiable_transform_support_matrix_artifact",
                "--check",
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline execution and exact source coverage gates."""
    return [
        (
            "transform-support-matrix-artifact focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_COHORT,
            ],
        ),
        (
            "transform-support-matrix-artifact exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/differentiable_transform_support_matrix_artifact.py",
            ],
        ),
    ]


__all__ = [
    "TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_COHORT",
    "TRANSFORM_SUPPORT_MATRIX_ARTIFACT_COVERAGE_DATA_FILE",
    "TRANSFORM_SUPPORT_MATRIX_ARTIFACT_DOCSTRING_RATCHET",
    "TRANSFORM_SUPPORT_MATRIX_ARTIFACT_SOURCE",
    "TRANSFORM_SUPPORT_MATRIX_ARTIFACT_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
