# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD adjoint quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PROGRAM_AD_ADJOINT_SOURCE = "src/scpn_quantum_control/program_ad_adjoint.py"
PROGRAM_AD_ADJOINT_COVERAGE_COHORT = [
    "tests/test_adjoint_replay_product.py",
    "tests/test_differentiable_programming_benchmark_program_ir_edges.py",
    "tests/test_program_ad_adjoint_generation.py",
    "tests/test_program_ad_adjoint_generation_docstrings.py",
    "tests/test_program_adjoint_replay.py",
]
PROGRAM_AD_ADJOINT_TYPING_RATCHET = [
    PROGRAM_AD_ADJOINT_SOURCE,
    "tools/program_ad_adjoint_quality_gates.py",
    "tests/test_program_ad_adjoint_quality_gate.py",
]
PROGRAM_AD_ADJOINT_DOCSTRING_RATCHET = [
    PROGRAM_AD_ADJOINT_SOURCE,
    *PROGRAM_AD_ADJOINT_COVERAGE_COHORT,
    "tools/program_ad_adjoint_quality_gates.py",
    "tests/test_program_ad_adjoint_quality_gate.py",
]
PROGRAM_AD_ADJOINT_COVERAGE_DATA_FILE = "/tmp/scpn-qc-program-ad-adjoint-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-program-ad-adjoint-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PROGRAM_AD_ADJOINT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D Program AD adjoint quality ratchet",
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
                *PROGRAM_AD_ADJOINT_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline Program AD adjoint execution and exact coverage gates."""
    return [
        (
            "Program AD adjoint focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PROGRAM_AD_ADJOINT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PROGRAM_AD_ADJOINT_COVERAGE_COHORT,
            ],
        ),
        (
            "Program AD adjoint exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PROGRAM_AD_ADJOINT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/program_ad_adjoint.py",
            ],
        ),
    ]


__all__ = [
    "PROGRAM_AD_ADJOINT_COVERAGE_COHORT",
    "PROGRAM_AD_ADJOINT_COVERAGE_DATA_FILE",
    "PROGRAM_AD_ADJOINT_DOCSTRING_RATCHET",
    "PROGRAM_AD_ADJOINT_SOURCE",
    "PROGRAM_AD_ADJOINT_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
