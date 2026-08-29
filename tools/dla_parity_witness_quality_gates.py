# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA parity-witness quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
DLA_PARITY_WITNESS_SOURCE = "src/scpn_quantum_control/analysis/dla_parity_witness.py"
DLA_PARITY_WITNESS_COVERAGE_COHORT = ["tests/test_observables.py"]
DLA_PARITY_WITNESS_TYPING_RATCHET = [
    DLA_PARITY_WITNESS_SOURCE,
    *DLA_PARITY_WITNESS_COVERAGE_COHORT,
    "tools/dla_parity_witness_quality_gates.py",
    "tests/test_dla_parity_witness_quality_gate.py",
]
DLA_PARITY_WITNESS_DOCSTRING_RATCHET = [*DLA_PARITY_WITNESS_TYPING_RATCHET]
DLA_PARITY_WITNESS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-dla-parity-witness-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-dla-parity-witness-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DLA_PARITY_WITNESS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D DLA-parity-witness quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D107,D413,D417,D420",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *DLA_PARITY_WITNESS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real observable execution and exact source-coverage gates."""
    return [
        (
            "DLA-parity-witness focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DLA_PARITY_WITNESS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DLA_PARITY_WITNESS_COVERAGE_COHORT,
            ],
        ),
        (
            "DLA-parity-witness exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DLA_PARITY_WITNESS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/analysis/dla_parity_witness.py",
            ],
        ),
    ]


__all__ = [
    "DLA_PARITY_WITNESS_COVERAGE_COHORT",
    "DLA_PARITY_WITNESS_COVERAGE_DATA_FILE",
    "DLA_PARITY_WITNESS_DOCSTRING_RATCHET",
    "DLA_PARITY_WITNESS_SOURCE",
    "DLA_PARITY_WITNESS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
