# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — learning-generalisation quality gates
"""Build strict typing, documentation and exact coverage gates for the QSNN training owner."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
LEARNING_GENERALISATION_SOURCES = [
    "src/scpn_quantum_control/qsnn/training.py",
]
LEARNING_GENERALISATION_CONFORMANCE_TEST = "tests/test_learning_rng_and_generalisation.py"
LEARNING_GENERALISATION_COVERAGE_COHORT = [
    LEARNING_GENERALISATION_CONFORMANCE_TEST,
    "tests/test_qsnn_training.py",
    "tests/test_qsnn_training_validation.py",
    "tests/test_qsnn_native_parameter_shift.py",
    "tests/test_ml_convergence_evidence.py",
    "tests/test_dense_allocation_admission.py",
]
LEARNING_GENERALISATION_COVERAGE_INCLUDE = "*/qsnn/training.py"
LEARNING_GENERALISATION_TYPING_RATCHET = [
    *LEARNING_GENERALISATION_SOURCES,
    LEARNING_GENERALISATION_CONFORMANCE_TEST,
    "tools/learning_generalisation_quality_gates.py",
    "tests/test_learning_generalisation_quality_gate.py",
]
LEARNING_GENERALISATION_DOCSTRING_RATCHET = [*LEARNING_GENERALISATION_TYPING_RATCHET]
LEARNING_GENERALISATION_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-learning-generalisation-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates.

    Parameters
    ----------
    python
        Interpreter that runs the gates.

    Returns
    -------
    list[Gate]
        Named commands.

    """
    return [
        (
            "mypy-strict-learning-generalisation-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *LEARNING_GENERALISATION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D learning-generalisation quality ratchet",
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
                "lint.explicit-preview-rules = true",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *LEARNING_GENERALISATION_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build the focused run and the exact coverage threshold.

    Parameters
    ----------
    python
        Interpreter that runs the gates.

    Returns
    -------
    list[Gate]
        Named commands.

    """
    return [
        (
            "learning-generalisation focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={LEARNING_GENERALISATION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *LEARNING_GENERALISATION_COVERAGE_COHORT,
            ],
        ),
        (
            "learning-generalisation exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={LEARNING_GENERALISATION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={LEARNING_GENERALISATION_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "LEARNING_GENERALISATION_CONFORMANCE_TEST",
    "LEARNING_GENERALISATION_COVERAGE_COHORT",
    "LEARNING_GENERALISATION_COVERAGE_DATA_FILE",
    "LEARNING_GENERALISATION_COVERAGE_INCLUDE",
    "LEARNING_GENERALISATION_DOCSTRING_RATCHET",
    "LEARNING_GENERALISATION_SOURCES",
    "LEARNING_GENERALISATION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
