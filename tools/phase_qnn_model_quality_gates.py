# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase classifier model quality gates
"""Build strict checks and exact coverage for the bounded classifier model."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

PHASE_QNN_MODEL_SOURCES = [
    "src/scpn_quantum_control/phase_qnn_model_numerics.py",
    "src/scpn_quantum_control/phase_qnn_numpy_model.py",
]
"""Numerical LEARN owners enforced at exact statement and branch coverage."""

PHASE_QNN_MODEL_QUALITY_RATCHET = [
    *PHASE_QNN_MODEL_SOURCES,
    "tests/test_phase_qnn_numpy_model.py",
    "tests/test_phase_qnn_model_quality_gate.py",
    "tools/phase_qnn_model_quality_gates.py",
]
"""Strict typing and NumPy-docstring cohort for the numerical model."""

PHASE_QNN_MODEL_COVERAGE_COHORT = [
    "tests/test_phase_qnn_numpy_model.py",
    "tests/test_phase_qnn_training_source_contracts.py",
    "tests/test_phase_qnn_model_quality_gate.py",
]
"""Real model execution and original-source compatibility tests."""

PHASE_QNN_MODEL_COVERAGE_DATA_FILE = ".coverage.phase-qnn-model"
"""Isolated coverage database for this model owner."""

PHASE_QNN_MODEL_COVERAGE_INCLUDE = (
    "*/phase_qnn_model_numerics.py,*/phase_qnn_numpy_model.py,*/phase_qnn_model_quality_gates.py"
)
"""Production model, contracts and their executable gate specification."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring commands.

    Parameters
    ----------
    python
        Python interpreter admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered commands for the numerical model and its original caller.

    """
    return [
        (
            "mypy-strict-phase-qnn-model",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                "src/scpn_quantum_control/phase/qnn_training.py",
                *PHASE_QNN_MODEL_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D phase-qnn-model",
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
                *PHASE_QNN_MODEL_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build isolated execution and an exact owner coverage report.

    Parameters
    ----------
    python
        Python interpreter admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Real model tests followed by the strict statement and branch threshold.

    """
    return [
        (
            "phase-qnn-model focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_QNN_MODEL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PHASE_QNN_MODEL_COVERAGE_COHORT,
            ],
        ),
        (
            "phase-qnn-model exact coverage",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PHASE_QNN_MODEL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                f"--include={PHASE_QNN_MODEL_COVERAGE_INCLUDE}",
            ],
        ),
    ]
