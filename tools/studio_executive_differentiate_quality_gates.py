# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio executive differentiate quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
STUDIO_EXECUTIVE_DIFFERENTIATE_SOURCE = (
    "src/scpn_quantum_control/studio/executive_differentiate.py"
)
STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_COHORT = ["tests/test_studio_executive_differentiate.py"]
STUDIO_EXECUTIVE_DIFFERENTIATE_TYPING_RATCHET = [
    STUDIO_EXECUTIVE_DIFFERENTIATE_SOURCE,
    *STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_COHORT,
    "tools/studio_executive_differentiate_quality_gates.py",
    "tests/test_studio_executive_differentiate_quality_gate.py",
]
STUDIO_EXECUTIVE_DIFFERENTIATE_DOCSTRING_RATCHET = list(
    STUDIO_EXECUTIVE_DIFFERENTIATE_TYPING_RATCHET
)
STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-studio-executive-differentiate-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-studio-executive-differentiate-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STUDIO_EXECUTIVE_DIFFERENTIATE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D studio-executive-differentiate quality ratchet",
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
                *STUDIO_EXECUTIVE_DIFFERENTIATE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline executive execution and exact source coverage gates."""
    return [
        (
            "studio-executive-differentiate focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_COHORT,
            ],
        ),
        (
            "studio-executive-differentiate exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/studio/executive_differentiate.py",
            ],
        ),
    ]


__all__ = [
    "STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_COHORT",
    "STUDIO_EXECUTIVE_DIFFERENTIATE_COVERAGE_DATA_FILE",
    "STUDIO_EXECUTIVE_DIFFERENTIATE_DOCSTRING_RATCHET",
    "STUDIO_EXECUTIVE_DIFFERENTIATE_SOURCE",
    "STUDIO_EXECUTIVE_DIFFERENTIATE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
