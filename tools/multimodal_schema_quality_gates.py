# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — multimodal schema quality gates
"""Build strict documentation and exact coverage gates for schema custody."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

MULTIMODAL_SCHEMA_QUALITY_RATCHET = [
    "src/scpn_quantum_control/forecasting/multimodal_schema.py",
    "src/scpn_quantum_control/forecasting/partial_observation.py",
    "tests/test_multimodal_schema.py",
    "tests/test_partial_observation_forecast.py",
    "tools/multimodal_schema_quality_gates.py",
    "tests/test_multimodal_schema_quality_gate.py",
]
"""Complete strict-typing and preview-documentation owner."""

MULTIMODAL_SCHEMA_COVERAGE_COHORT = [
    "tests/test_multimodal_schema.py",
    "tests/test_partial_observation_forecast.py",
]
"""Real immutable custody and partial-observation objective suites."""

MULTIMODAL_SCHEMA_COVERAGE_DATA_FILE = "/tmp/scpn-qc-multimodal-schema-quality.coverage"
"""Isolated coverage database for the multimodal schema."""

MULTIMODAL_SCHEMA_COVERAGE_INCLUDE = (
    "*/forecasting/multimodal_schema.py,*/forecasting/partial_observation.py"
)
"""Production custody and objective sources enforced at exact branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and preview NumPy-documentation gates."""
    return [
        (
            "mypy-strict-multimodal-schema-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *MULTIMODAL_SCHEMA_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D multimodal-schema quality ratchet",
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
                *MULTIMODAL_SCHEMA_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real execution and exact schema-coverage gates."""
    return [
        (
            "multimodal-schema focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={MULTIMODAL_SCHEMA_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *MULTIMODAL_SCHEMA_COVERAGE_COHORT,
            ],
        ),
        (
            "multimodal-schema exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={MULTIMODAL_SCHEMA_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={MULTIMODAL_SCHEMA_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "MULTIMODAL_SCHEMA_COVERAGE_COHORT",
    "MULTIMODAL_SCHEMA_COVERAGE_DATA_FILE",
    "MULTIMODAL_SCHEMA_COVERAGE_INCLUDE",
    "MULTIMODAL_SCHEMA_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
