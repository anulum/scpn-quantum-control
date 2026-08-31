# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — analog platform catalogue quality gates
"""Build strict documentation and exact coverage gates for analog profiles."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

ANALOG_PLATFORM_CATALOGUE_SOURCE = "src/scpn_quantum_control/analog_mapping/platforms.py"
"""Static analog platform catalogue and feasibility source."""
ANALOG_NATIVE_READINESS_SOURCE = "src/scpn_quantum_control/hardware/analog_native_readiness.py"
"""S10 no-submit primitive-comparison and provider-readiness source."""
ANALOG_NATIVE_READINESS_TEST = "tests/test_analog_native_readiness.py"
"""Public primitive-comparison, provider-row, payload, and rendering tests."""
ANALOG_NATIVE_READINESS_BRANCH_TEST = "tests/test_analog_native_readiness_branches.py"
"""Problem-array validation branch tests."""
ANALOG_NATIVE_READINESS_EXPORTER = "scripts/export_s10_analog_native_readiness.py"
"""Executable S10 JSON and Markdown exporter."""
ANALOG_NATIVE_READINESS_EXPORT_TEST = "tests/test_export_s10_analog_native_readiness.py"
"""Real filesystem exporter test."""
ANALOG_PLATFORM_CATALOGUE_QUALITY_RATCHET = [
    ANALOG_PLATFORM_CATALOGUE_SOURCE,
    "tests/test_analog_mapping_feasibility.py",
    ANALOG_NATIVE_READINESS_SOURCE,
    ANALOG_NATIVE_READINESS_TEST,
    ANALOG_NATIVE_READINESS_BRANCH_TEST,
    ANALOG_NATIVE_READINESS_EXPORTER,
    ANALOG_NATIVE_READINESS_EXPORT_TEST,
    "tools/analog_platform_catalogue_quality_gates.py",
    "tests/test_analog_platform_catalogue_quality_gate.py",
]
"""Complete strict-typing and preview-documentation owner."""

ANALOG_PLATFORM_CATALOGUE_COVERAGE_COHORT = [
    "tests/test_analog_mapping_feasibility.py",
    ANALOG_NATIVE_READINESS_TEST,
    ANALOG_NATIVE_READINESS_BRANCH_TEST,
    ANALOG_NATIVE_READINESS_EXPORT_TEST,
]
"""Real catalogue, readiness, provider-plan, and exporter suites."""

ANALOG_PLATFORM_CATALOGUE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-analog-platform-catalogue-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the analog platform catalogue."""

ANALOG_PLATFORM_CATALOGUE_COVERAGE_INCLUDE = (
    "*/analog_mapping/platforms.py,*/hardware/analog_native_readiness.py"
)
"""Production catalogue and S10 readiness sources at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and preview NumPy-documentation gates."""
    return [
        (
            "mypy-strict-analog-platform-catalogue-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ANALOG_PLATFORM_CATALOGUE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D analog-platform-catalogue quality ratchet",
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
                "lint.explicit-preview-rules = true",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *ANALOG_PLATFORM_CATALOGUE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real execution and exact catalogue-coverage gates."""
    return [
        (
            "analog-platform-catalogue focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ANALOG_PLATFORM_CATALOGUE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ANALOG_PLATFORM_CATALOGUE_COVERAGE_COHORT,
            ],
        ),
        (
            "analog-platform-catalogue exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ANALOG_PLATFORM_CATALOGUE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={ANALOG_PLATFORM_CATALOGUE_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "ANALOG_NATIVE_READINESS_BRANCH_TEST",
    "ANALOG_NATIVE_READINESS_EXPORTER",
    "ANALOG_NATIVE_READINESS_EXPORT_TEST",
    "ANALOG_NATIVE_READINESS_SOURCE",
    "ANALOG_NATIVE_READINESS_TEST",
    "ANALOG_PLATFORM_CATALOGUE_COVERAGE_COHORT",
    "ANALOG_PLATFORM_CATALOGUE_COVERAGE_DATA_FILE",
    "ANALOG_PLATFORM_CATALOGUE_COVERAGE_INCLUDE",
    "ANALOG_PLATFORM_CATALOGUE_QUALITY_RATCHET",
    "ANALOG_PLATFORM_CATALOGUE_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
