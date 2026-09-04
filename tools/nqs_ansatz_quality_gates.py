# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — NQS-ansatz quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
NQS_ANSATZ_SOURCE = "src/scpn_quantum_control/phase/nqs_ansatz.py"
JAX_NQS_BASELINE_PRODUCT_SOURCE = "src/scpn_quantum_control/jax_nqs_baseline_product.py"
JAX_NQS_BASELINE_PRODUCT_TEST = "tests/test_jax_nqs_baseline_product.py"
NQS_ANSATZ_COVERAGE_COHORT = ["tests/test_nqs_ansatz.py", JAX_NQS_BASELINE_PRODUCT_TEST]
NQS_ANSATZ_TYPING_RATCHET = [
    NQS_ANSATZ_SOURCE,
    JAX_NQS_BASELINE_PRODUCT_SOURCE,
    *NQS_ANSATZ_COVERAGE_COHORT,
    "tools/nqs_ansatz_quality_gates.py",
    "tests/test_nqs_ansatz_quality_gate.py",
]
NQS_ANSATZ_DOCSTRING_RATCHET = [*NQS_ANSATZ_TYPING_RATCHET]
NQS_ANSATZ_COVERAGE_DATA_FILE = "/tmp/scpn-qc-nqs-ansatz-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-nqs-ansatz-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *NQS_ANSATZ_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D NQS-ansatz quality ratchet",
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
                *NQS_ANSATZ_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real NQS execution and exact source-coverage gates."""
    return [
        (
            "NQS-ansatz focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={NQS_ANSATZ_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *NQS_ANSATZ_COVERAGE_COHORT,
            ],
        ),
        (
            "NQS-ansatz exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={NQS_ANSATZ_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/nqs_ansatz.py,*/jax_nqs_baseline_product.py",
            ],
        ),
    ]


__all__ = [
    "JAX_NQS_BASELINE_PRODUCT_SOURCE",
    "JAX_NQS_BASELINE_PRODUCT_TEST",
    "NQS_ANSATZ_COVERAGE_COHORT",
    "NQS_ANSATZ_COVERAGE_DATA_FILE",
    "NQS_ANSATZ_DOCSTRING_RATCHET",
    "NQS_ANSATZ_SOURCE",
    "NQS_ANSATZ_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
