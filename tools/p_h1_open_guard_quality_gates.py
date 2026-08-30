# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — p_h1 open-claim guard quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
P_H1_OPEN_GUARD_SOURCE = "src/scpn_quantum_control/analysis/p_h1_open_guard.py"
PERSISTENT_HOMOLOGY_SOURCE = "src/scpn_quantum_control/analysis/persistent_homology.py"
QUANTUM_PERSISTENT_HOMOLOGY_SOURCE = (
    "src/scpn_quantum_control/analysis/quantum_persistent_homology.py"
)
TCBO_WEIGHTED_COMPLEX_SOURCE = "src/scpn_quantum_control/analysis/tcbo_weighted_complex.py"
P_H1_OPEN_GUARD_EXPORT = "scripts/check_p_h1_open_claim_guard.py"
P_H1_OPEN_GUARD_DIRECT_TEST = "tests/test_p_h1_open_guard.py"
P_H1_OPEN_GUARD_SHARED_CLI_TEST = "tests/test_bench_cli.py"
PERSISTENT_HOMOLOGY_DIRECT_TEST = "tests/test_persistent_homology.py"
QUANTUM_PERSISTENT_HOMOLOGY_DIRECT_TEST = "tests/test_quantum_persistent_homology.py"
TCBO_WEIGHTED_COMPLEX_DIRECT_TEST = "tests/test_tcbo_weighted_complex.py"
PERSISTENT_HOMOLOGY_BRANCH_TEST = "tests/test_persistent_homology_branches.py"
PERSISTENT_HOMOLOGY_CONNECTED_TEST = "tests/test_analysis_topology_contracts.py"
P_H1_OPEN_GUARD_COVERAGE_INCLUDE = (
    "*/analysis/p_h1_open_guard.py,*/scripts/check_p_h1_open_claim_guard.py,"
    "*/analysis/persistent_homology.py,*/analysis/quantum_persistent_homology.py,"
    "*/analysis/tcbo_weighted_complex.py"
)
P_H1_OPEN_GUARD_COVERAGE_COHORT = [
    P_H1_OPEN_GUARD_DIRECT_TEST,
    P_H1_OPEN_GUARD_SHARED_CLI_TEST,
    PERSISTENT_HOMOLOGY_DIRECT_TEST,
    QUANTUM_PERSISTENT_HOMOLOGY_DIRECT_TEST,
    PERSISTENT_HOMOLOGY_BRANCH_TEST,
    PERSISTENT_HOMOLOGY_CONNECTED_TEST,
    TCBO_WEIGHTED_COMPLEX_DIRECT_TEST,
]
P_H1_OPEN_GUARD_TYPING_RATCHET = [
    P_H1_OPEN_GUARD_SOURCE,
    PERSISTENT_HOMOLOGY_SOURCE,
    QUANTUM_PERSISTENT_HOMOLOGY_SOURCE,
    TCBO_WEIGHTED_COMPLEX_SOURCE,
    P_H1_OPEN_GUARD_EXPORT,
    P_H1_OPEN_GUARD_DIRECT_TEST,
    P_H1_OPEN_GUARD_SHARED_CLI_TEST,
    PERSISTENT_HOMOLOGY_DIRECT_TEST,
    QUANTUM_PERSISTENT_HOMOLOGY_DIRECT_TEST,
    TCBO_WEIGHTED_COMPLEX_DIRECT_TEST,
    "tools/p_h1_open_guard_quality_gates.py",
    "tests/test_p_h1_open_guard_quality_gate.py",
]
P_H1_OPEN_GUARD_DOCSTRING_RATCHET = [
    P_H1_OPEN_GUARD_SOURCE,
    PERSISTENT_HOMOLOGY_SOURCE,
    QUANTUM_PERSISTENT_HOMOLOGY_SOURCE,
    TCBO_WEIGHTED_COMPLEX_SOURCE,
    P_H1_OPEN_GUARD_EXPORT,
    P_H1_OPEN_GUARD_DIRECT_TEST,
    PERSISTENT_HOMOLOGY_DIRECT_TEST,
    QUANTUM_PERSISTENT_HOMOLOGY_DIRECT_TEST,
    TCBO_WEIGHTED_COMPLEX_DIRECT_TEST,
    "tools/p_h1_open_guard_quality_gates.py",
    "tests/test_p_h1_open_guard_quality_gate.py",
]
P_H1_OPEN_GUARD_COVERAGE_DATA_FILE = "/tmp/scpn-qc-p-h1-open-guard-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and direct-owner NumPy-docstring gates."""
    return [
        (
            "mypy-strict-p-h1-open-guard-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *P_H1_OPEN_GUARD_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D p_h1 open-guard quality ratchet",
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
                *P_H1_OPEN_GUARD_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build connected guard execution and exact source-coverage gates."""
    return [
        (
            "p_h1 open-guard focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={P_H1_OPEN_GUARD_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *P_H1_OPEN_GUARD_COVERAGE_COHORT,
            ],
        ),
        (
            "p_h1 open-guard exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={P_H1_OPEN_GUARD_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={P_H1_OPEN_GUARD_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "P_H1_OPEN_GUARD_COVERAGE_COHORT",
    "P_H1_OPEN_GUARD_COVERAGE_DATA_FILE",
    "P_H1_OPEN_GUARD_COVERAGE_INCLUDE",
    "P_H1_OPEN_GUARD_DIRECT_TEST",
    "P_H1_OPEN_GUARD_DOCSTRING_RATCHET",
    "P_H1_OPEN_GUARD_EXPORT",
    "P_H1_OPEN_GUARD_SHARED_CLI_TEST",
    "P_H1_OPEN_GUARD_SOURCE",
    "P_H1_OPEN_GUARD_TYPING_RATCHET",
    "PERSISTENT_HOMOLOGY_BRANCH_TEST",
    "PERSISTENT_HOMOLOGY_CONNECTED_TEST",
    "PERSISTENT_HOMOLOGY_DIRECT_TEST",
    "PERSISTENT_HOMOLOGY_SOURCE",
    "QUANTUM_PERSISTENT_HOMOLOGY_DIRECT_TEST",
    "QUANTUM_PERSISTENT_HOMOLOGY_SOURCE",
    "TCBO_WEIGHTED_COMPLEX_DIRECT_TEST",
    "TCBO_WEIGHTED_COMPLEX_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
