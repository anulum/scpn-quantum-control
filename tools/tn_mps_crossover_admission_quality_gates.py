# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — TN/MPS crossover-admission quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
TN_MPS_CROSSOVER_ADMISSION_SOURCE = (
    "src/scpn_quantum_control/benchmarks/tn_mps_crossover_admission.py"
)
TN_MPS_CROSSOVER_ADMISSION_COVERAGE_COHORT = ["tests/test_tn_mps_crossover_admission.py"]
TN_MPS_CROSSOVER_ADMISSION_TYPING_RATCHET = [
    TN_MPS_CROSSOVER_ADMISSION_SOURCE,
    *TN_MPS_CROSSOVER_ADMISSION_COVERAGE_COHORT,
    "scripts/export_tn_mps_crossover_admission.py",
    "tools/tn_mps_crossover_admission_quality_gates.py",
    "tests/test_tn_mps_crossover_admission_quality_gate.py",
]
TN_MPS_CROSSOVER_ADMISSION_DOCSTRING_RATCHET = [*TN_MPS_CROSSOVER_ADMISSION_TYPING_RATCHET]
TN_MPS_CROSSOVER_ADMISSION_COVERAGE_DATA_FILE = (  # nosec B108
    "/tmp/scpn-qc-tn-mps-crossover-admission-quality.coverage"
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-tn-mps-crossover-admission-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *TN_MPS_CROSSOVER_ADMISSION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D TN/MPS-crossover-admission quality ratchet",
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
                *TN_MPS_CROSSOVER_ADMISSION_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build admission execution and exact source-coverage gates."""
    return [
        (
            "TN/MPS-crossover-admission focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={TN_MPS_CROSSOVER_ADMISSION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *TN_MPS_CROSSOVER_ADMISSION_COVERAGE_COHORT,
            ],
        ),
        (
            "TN/MPS-crossover-admission exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={TN_MPS_CROSSOVER_ADMISSION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/benchmarks/tn_mps_crossover_admission.py",
            ],
        ),
    ]


__all__ = [
    "TN_MPS_CROSSOVER_ADMISSION_COVERAGE_COHORT",
    "TN_MPS_CROSSOVER_ADMISSION_COVERAGE_DATA_FILE",
    "TN_MPS_CROSSOVER_ADMISSION_DOCSTRING_RATCHET",
    "TN_MPS_CROSSOVER_ADMISSION_SOURCE",
    "TN_MPS_CROSSOVER_ADMISSION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
