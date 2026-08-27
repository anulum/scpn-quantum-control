# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — identity binding-spec quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

IDENTITY_BINDING_SPEC_QUALITY_RATCHET = [
    "src/scpn_quantum_control/identity/binding_spec.py",
    "tests/test_binding_spec.py",
    "tests/test_binding_spec_branch.py",
    "tools/identity_binding_spec_quality_gates.py",
    "tests/test_identity_binding_spec_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

IDENTITY_BINDING_SPEC_COVERAGE_COHORT = [
    "tests/test_binding_spec.py",
    "tests/test_binding_spec_branch.py",
]
"""Tests that own exact identity binding-spec coverage."""

IDENTITY_BINDING_SPEC_COVERAGE_DATA_FILE = "/tmp/scpn-qc-identity-binding-spec-quality.coverage"  # nosec B108
"""Isolated coverage database for the identity binding-spec owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-identity-binding-spec-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *IDENTITY_BINDING_SPEC_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D identity-binding-spec quality ratchet",
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
                *IDENTITY_BINDING_SPEC_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "identity-binding-spec focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={IDENTITY_BINDING_SPEC_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *IDENTITY_BINDING_SPEC_COVERAGE_COHORT,
            ],
        ),
        (
            "identity-binding-spec exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={IDENTITY_BINDING_SPEC_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/identity/binding_spec.py",
            ],
        ),
    ]


__all__ = [
    "IDENTITY_BINDING_SPEC_COVERAGE_COHORT",
    "IDENTITY_BINDING_SPEC_COVERAGE_DATA_FILE",
    "IDENTITY_BINDING_SPEC_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
