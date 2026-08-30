# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Kuramoto-variant quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
KURAMOTO_VARIANTS_SOURCE = "src/scpn_quantum_control/phase/kuramoto_variants.py"
KURAMOTO_VARIANTS_MIXED_CONSUMER = "tests/test_commutator_bounds.py"
KURAMOTO_VARIANTS_COVERAGE_COHORT = [
    "tests/test_kuramoto_variants.py",
    KURAMOTO_VARIANTS_MIXED_CONSUMER,
    "tests/test_floquet_kuramoto.py",
    "tests/test_floquet_kuramoto_strict_surface.py",
]
KURAMOTO_VARIANTS_TYPING_RATCHET = [
    KURAMOTO_VARIANTS_SOURCE,
    "src/scpn_quantum_control/phase/floquet_kuramoto.py",
    "tests/test_floquet_kuramoto.py",
    "tests/test_floquet_kuramoto_strict_surface.py",
    "tools/kuramoto_variants_quality_gates.py",
    "tests/test_kuramoto_variants_quality_gate.py",
]
KURAMOTO_VARIANTS_DOCSTRING_RATCHET = [
    KURAMOTO_VARIANTS_SOURCE,
    "tests/test_kuramoto_variants.py",
    "src/scpn_quantum_control/phase/floquet_kuramoto.py",
    "tests/test_floquet_kuramoto.py",
    "tests/test_floquet_kuramoto_strict_surface.py",
    "tools/kuramoto_variants_quality_gates.py",
    "tests/test_kuramoto_variants_quality_gate.py",
]
KURAMOTO_VARIANTS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-kuramoto-variants-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-kuramoto-variants-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KURAMOTO_VARIANTS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D Kuramoto-variants quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D413,D417",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *KURAMOTO_VARIANTS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build connected Kuramoto-variant execution and exact coverage gates."""
    return [
        (
            "Kuramoto-variants focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KURAMOTO_VARIANTS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *KURAMOTO_VARIANTS_COVERAGE_COHORT,
            ],
        ),
        (
            "Kuramoto-variants exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KURAMOTO_VARIANTS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/kuramoto_variants.py,*/phase/floquet_kuramoto.py",
            ],
        ),
    ]


__all__ = [
    "KURAMOTO_VARIANTS_COVERAGE_COHORT",
    "KURAMOTO_VARIANTS_COVERAGE_DATA_FILE",
    "KURAMOTO_VARIANTS_DOCSTRING_RATCHET",
    "KURAMOTO_VARIANTS_MIXED_CONSUMER",
    "KURAMOTO_VARIANTS_SOURCE",
    "KURAMOTO_VARIANTS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
