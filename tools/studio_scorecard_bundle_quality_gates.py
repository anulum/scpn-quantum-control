# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio scorecard-bundle quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
STUDIO_SCORECARD_BUNDLE_SOURCE = "src/scpn_quantum_control/studio/scorecard_bundle.py"
STUDIO_SCORECARD_BUNDLE_COVERAGE_COHORT = ["tests/test_studio_scorecard_bundle.py"]
STUDIO_SCORECARD_BUNDLE_TYPING_RATCHET = [
    STUDIO_SCORECARD_BUNDLE_SOURCE,
    *STUDIO_SCORECARD_BUNDLE_COVERAGE_COHORT,
    "tools/studio_scorecard_bundle_quality_gates.py",
    "tests/test_studio_scorecard_bundle_quality_gate.py",
]
STUDIO_SCORECARD_BUNDLE_DOCSTRING_RATCHET = [
    *STUDIO_SCORECARD_BUNDLE_TYPING_RATCHET,
]
STUDIO_SCORECARD_BUNDLE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-studio-scorecard-bundle-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-studio-scorecard-bundle-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STUDIO_SCORECARD_BUNDLE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D Studio scorecard-bundle quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D413,D417,D420",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *STUDIO_SCORECARD_BUNDLE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build public entrypoint execution and exact source-coverage gates."""
    return [
        (
            "Studio scorecard-bundle focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_SCORECARD_BUNDLE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STUDIO_SCORECARD_BUNDLE_COVERAGE_COHORT,
            ],
        ),
        (
            "Studio scorecard-bundle exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_SCORECARD_BUNDLE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/studio/scorecard_bundle.py",
            ],
        ),
    ]


__all__ = [
    "STUDIO_SCORECARD_BUNDLE_COVERAGE_COHORT",
    "STUDIO_SCORECARD_BUNDLE_COVERAGE_DATA_FILE",
    "STUDIO_SCORECARD_BUNDLE_DOCSTRING_RATCHET",
    "STUDIO_SCORECARD_BUNDLE_SOURCE",
    "STUDIO_SCORECARD_BUNDLE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
