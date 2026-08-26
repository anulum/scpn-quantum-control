# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — polyglot parity certificate quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

POLYGLOT_PARITY_CERTIFICATE_QUALITY_RATCHET = [
    "src/scpn_quantum_control/polyglot_parity_certificate.py",
    "tests/test_polyglot_parity_certificate.py",
    "tools/polyglot_parity_certificate_quality_gates.py",
    "tests/test_polyglot_parity_certificate_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

POLYGLOT_PARITY_CERTIFICATE_COVERAGE_COHORT = ["tests/test_polyglot_parity_certificate.py"]
"""Tests that own exact polyglot parity certificate coverage."""

POLYGLOT_PARITY_CERTIFICATE_COVERAGE_DATA_FILE = ".coverage.polyglot-parity-certificate-quality"
"""Isolated coverage database for the polyglot parity certificate owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Ordered static gates for the certificate owner cohort.

    """
    return [
        (
            "mypy-strict-polyglot-parity-certificate-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *POLYGLOT_PARITY_CERTIFICATE_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D polyglot-parity-certificate quality ratchet",
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
                *POLYGLOT_PARITY_CERTIFICATE_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by preflight.

    Returns
    -------
    list[Gate]
        Focused owner execution followed by the exact report.

    """
    return [
        (
            "polyglot-parity-certificate focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={POLYGLOT_PARITY_CERTIFICATE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *POLYGLOT_PARITY_CERTIFICATE_COVERAGE_COHORT,
            ],
        ),
        (
            "polyglot-parity-certificate exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={POLYGLOT_PARITY_CERTIFICATE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/polyglot_parity_certificate.py",
            ],
        ),
    ]


__all__ = [
    "POLYGLOT_PARITY_CERTIFICATE_COVERAGE_COHORT",
    "POLYGLOT_PARITY_CERTIFICATE_COVERAGE_DATA_FILE",
    "POLYGLOT_PARITY_CERTIFICATE_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
