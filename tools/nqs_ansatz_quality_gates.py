# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — NQS-ansatz quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

import os
import tempfile
from os import devnull
from pathlib import Path

Gate = tuple[str, list[str]]
NQS_ANSATZ_SOURCE = "src/scpn_quantum_control/phase/nqs_ansatz.py"
JAX_NQS_BASELINE_PRODUCT_SOURCE = "src/scpn_quantum_control/jax_nqs_baseline_product.py"
JAX_NQS_BASELINE_PRODUCT_TEST = "tests/test_jax_nqs_baseline_product.py"
JAX_NQS_SOURCE = "src/scpn_quantum_control/phase/jax_nqs.py"
VARIATIONAL_QUALIFICATION_SOURCE = "src/scpn_quantum_control/variational_baseline_qualification.py"
NQS_ANSATZ_COVERAGE_COHORT = [
    "tests/test_nqs_ansatz.py",
    JAX_NQS_BASELINE_PRODUCT_TEST,
    "tests/test_jax_nqs.py",
    "tests/test_variational_baseline_qualification.py",
    "tests/test_variational_quantum_baselines.py",
    "tests/test_nqs_ansatz_quality_gate.py",
]
NQS_ANSATZ_COVERAGE_INCLUDE = (
    "*/phase/nqs_ansatz.py,*/jax_nqs_baseline_product.py,"
    "*/phase/jax_nqs.py,*/variational_baseline_qualification.py,"
    "*/tools/nqs_ansatz_quality_gates.py,*/examples/variational_baseline_qualification.py"
)
NQS_ANSATZ_TYPING_RATCHET = [
    NQS_ANSATZ_SOURCE,
    JAX_NQS_BASELINE_PRODUCT_SOURCE,
    JAX_NQS_SOURCE,
    VARIATIONAL_QUALIFICATION_SOURCE,
    "examples/variational_baseline_qualification.py",
    *NQS_ANSATZ_COVERAGE_COHORT,
    "tools/nqs_ansatz_quality_gates.py",
]
NQS_ANSATZ_DOCSTRING_RATCHET = [*NQS_ANSATZ_TYPING_RATCHET]
NQS_ANSATZ_COVERAGE_DATA_FILE = str(
    Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "scpn-qc-nqs-ansatz-quality.coverage"
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates.

    Parameters
    ----------
    python : str
        Interpreter for the existing locked project environment.

    Returns
    -------
    list
        Named executable commands preserving required real-runtime acceptance.

    """
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
    """Build real NQS execution and exact source-coverage gates.

    Parameters
    ----------
    python : str
        Interpreter for the existing locked project environment.

    Returns
    -------
    list
        Named executable commands preserving required real-runtime acceptance.

    """
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
            "NQS-ansatz complete file example",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={NQS_ANSATZ_COVERAGE_DATA_FILE}",
                "--append",
                "--branch",
                "examples/variational_baseline_qualification.py",
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
                f"--include={NQS_ANSATZ_COVERAGE_INCLUDE}",
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
