# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — provider-gradient audit quality-gate specification
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PROVIDER_GRADIENT_AUDIT_SOURCE = "src/scpn_quantum_control/phase/provider_gradient_audit.py"
PROVIDER_GRADIENT_AUDIT_TEST = "tests/test_phase_provider_gradient_audit.py"
PROVIDER_GRADIENT_AUDIT_TYPING_RATCHET = [
    PROVIDER_GRADIENT_AUDIT_SOURCE,
    PROVIDER_GRADIENT_AUDIT_TEST,
    "tools/provider_gradient_audit_quality_gates.py",
    "tests/test_provider_gradient_audit_quality_gate.py",
    "tools/preflight.py",
    "tests/test_preflight_tool.py",
]
"""Production, tests, and gate surfaces held to strict MyPy."""
PROVIDER_GRADIENT_AUDIT_DOCSTRING_RATCHET = [
    PROVIDER_GRADIENT_AUDIT_SOURCE,
    PROVIDER_GRADIENT_AUDIT_TEST,
    "tools/provider_gradient_audit_quality_gates.py",
    "tests/test_provider_gradient_audit_quality_gate.py",
    "tests/test_preflight_tool.py",
]
"""Whole owner cohort held to complete NumPy docstrings."""
PROVIDER_GRADIENT_AUDIT_COVERAGE_COHORT = [PROVIDER_GRADIENT_AUDIT_TEST]
"""Public readiness-audit suite that owns source branch coverage."""
PROVIDER_GRADIENT_AUDIT_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-provider-gradient-audit-quality.coverage"  # nosec B108
)
PROVIDER_GRADIENT_AUDIT_COVERAGE_INCLUDE = "*/phase/provider_gradient_audit.py"


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-provider-gradient-audit-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PROVIDER_GRADIENT_AUDIT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D provider-gradient-audit quality ratchet",
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
                *PROVIDER_GRADIENT_AUDIT_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real readiness-audit execution and exact source-coverage gates."""
    return [
        (
            "provider-gradient-audit focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PROVIDER_GRADIENT_AUDIT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PROVIDER_GRADIENT_AUDIT_COVERAGE_COHORT,
            ],
        ),
        (
            "provider-gradient-audit exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PROVIDER_GRADIENT_AUDIT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={PROVIDER_GRADIENT_AUDIT_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "PROVIDER_GRADIENT_AUDIT_COVERAGE_COHORT",
    "PROVIDER_GRADIENT_AUDIT_COVERAGE_DATA_FILE",
    "PROVIDER_GRADIENT_AUDIT_COVERAGE_INCLUDE",
    "PROVIDER_GRADIENT_AUDIT_DOCSTRING_RATCHET",
    "PROVIDER_GRADIENT_AUDIT_SOURCE",
    "PROVIDER_GRADIENT_AUDIT_TEST",
    "PROVIDER_GRADIENT_AUDIT_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
