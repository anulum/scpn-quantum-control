# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — magnetisation-sector quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
MAGNETISATION_SECTORS_SOURCE = "src/scpn_quantum_control/analysis/magnetisation_sectors.py"
SPECTRAL_FORM_FACTOR_SOURCE = "src/scpn_quantum_control/analysis/spectral_form_factor.py"
SPECTRAL_FORM_FACTOR_DIRECT_TEST = "tests/test_spectral_form_factor.py"
SPECTRAL_FORM_FACTOR_BRANCH_TEST = "tests/test_spectral_form_factor_branch.py"
SPECTRAL_FORM_FACTOR_CONNECTED_TEST = "tests/test_analysis_topology_contracts.py"
SYMMETRY_SECTORS_SOURCE = "src/scpn_quantum_control/analysis/symmetry_sectors.py"
SYMMETRY_SECTORS_DIRECT_TEST = "tests/test_symmetry_sectors.py"
SYMMETRY_SECTORS_BRANCH_TEST = "tests/test_symmetry_sectors_branch.py"
MAGNETISATION_SECTORS_COVERAGE_INCLUDE = (
    "*/analysis/magnetisation_sectors.py,*/analysis/spectral_form_factor.py,"
    "*/analysis/symmetry_sectors.py"
)
MAGNETISATION_SECTORS_COVERAGE_COHORT = [
    "tests/test_magnetisation_sectors.py",
    "tests/test_magnetisation_sectors_empty_guard.py",
    "tests/test_open_system_workflow.py",
    "tests/test_rust_new_functions.py",
    "tests/test_sparse_hamiltonian.py",
    SPECTRAL_FORM_FACTOR_DIRECT_TEST,
    SPECTRAL_FORM_FACTOR_BRANCH_TEST,
    SPECTRAL_FORM_FACTOR_CONNECTED_TEST,
    SYMMETRY_SECTORS_DIRECT_TEST,
    SYMMETRY_SECTORS_BRANCH_TEST,
    "tests/test_symmetry_sparse_workflow.py",
]
MAGNETISATION_SECTORS_TYPING_RATCHET = [
    MAGNETISATION_SECTORS_SOURCE,
    SPECTRAL_FORM_FACTOR_SOURCE,
    SPECTRAL_FORM_FACTOR_DIRECT_TEST,
    SPECTRAL_FORM_FACTOR_BRANCH_TEST,
    SYMMETRY_SECTORS_SOURCE,
    SYMMETRY_SECTORS_DIRECT_TEST,
    SYMMETRY_SECTORS_BRANCH_TEST,
    "tools/magnetisation_sectors_quality_gates.py",
    "tests/test_magnetisation_sectors_quality_gate.py",
]
MAGNETISATION_SECTORS_DOCSTRING_RATCHET = [
    MAGNETISATION_SECTORS_SOURCE,
    SPECTRAL_FORM_FACTOR_SOURCE,
    "tests/test_magnetisation_sectors.py",
    "tests/test_magnetisation_sectors_empty_guard.py",
    SPECTRAL_FORM_FACTOR_DIRECT_TEST,
    SPECTRAL_FORM_FACTOR_BRANCH_TEST,
    SYMMETRY_SECTORS_SOURCE,
    SYMMETRY_SECTORS_DIRECT_TEST,
    SYMMETRY_SECTORS_BRANCH_TEST,
    "tools/magnetisation_sectors_quality_gates.py",
    "tests/test_magnetisation_sectors_quality_gate.py",
]
MAGNETISATION_SECTORS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-magnetisation-sectors-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-magnetisation-sectors-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *MAGNETISATION_SECTORS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D magnetisation-sectors quality ratchet",
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
                *MAGNETISATION_SECTORS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline sector execution and exact coverage gates."""
    return [
        (
            "magnetisation-sectors focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={MAGNETISATION_SECTORS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *MAGNETISATION_SECTORS_COVERAGE_COHORT,
            ],
        ),
        (
            "magnetisation-sectors exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={MAGNETISATION_SECTORS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={MAGNETISATION_SECTORS_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "MAGNETISATION_SECTORS_COVERAGE_COHORT",
    "MAGNETISATION_SECTORS_COVERAGE_DATA_FILE",
    "MAGNETISATION_SECTORS_COVERAGE_INCLUDE",
    "MAGNETISATION_SECTORS_DOCSTRING_RATCHET",
    "MAGNETISATION_SECTORS_SOURCE",
    "MAGNETISATION_SECTORS_TYPING_RATCHET",
    "SPECTRAL_FORM_FACTOR_BRANCH_TEST",
    "SPECTRAL_FORM_FACTOR_CONNECTED_TEST",
    "SPECTRAL_FORM_FACTOR_DIRECT_TEST",
    "SPECTRAL_FORM_FACTOR_SOURCE",
    "SYMMETRY_SECTORS_BRANCH_TEST",
    "SYMMETRY_SECTORS_DIRECT_TEST",
    "SYMMETRY_SECTORS_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
