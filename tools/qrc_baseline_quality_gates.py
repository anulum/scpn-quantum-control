# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QRC-baseline quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QRC_BASELINE_SOURCE = "src/scpn_quantum_control/applications/qrc_baseline.py"
QUANTUM_RESERVOIR_SOURCE = "src/scpn_quantum_control/applications/quantum_reservoir.py"
SURROGATE_MODEL_SOURCE = "src/scpn_quantum_control/surrogates/models.py"
SURROGATE_TRAIN_SOURCE = "src/scpn_quantum_control/surrogates/train.py"
SURROGATE_FIDELITY_SOURCE = "src/scpn_quantum_control/surrogates/fidelity.py"
QRC_BASELINE_SOURCES = [
    QRC_BASELINE_SOURCE,
    QUANTUM_RESERVOIR_SOURCE,
    SURROGATE_MODEL_SOURCE,
    SURROGATE_TRAIN_SOURCE,
    SURROGATE_FIDELITY_SOURCE,
]
QRC_BASELINE_COVERAGE_COHORT = [
    "tests/test_qrc_baseline.py",
    "tests/test_quantum_reservoir.py",
    "tests/test_surrogate_models.py",
    "tests/test_surrogate_train.py",
    "tests/test_surrogate_fidelity.py",
]
QRC_BASELINE_TYPING_RATCHET = [
    *QRC_BASELINE_SOURCES,
    *QRC_BASELINE_COVERAGE_COHORT,
    "tools/qrc_baseline_quality_gates.py",
    "tests/test_qrc_baseline_quality_gate.py",
]
QRC_BASELINE_DOCSTRING_RATCHET = [*QRC_BASELINE_TYPING_RATCHET]
QRC_BASELINE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-qrc-baseline-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-qrc-baseline-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QRC_BASELINE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D QRC-baseline quality ratchet",
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
                *QRC_BASELINE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real offline QRC/surrogate execution and exact coverage gates."""
    return [
        (
            "QRC-baseline focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QRC_BASELINE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QRC_BASELINE_COVERAGE_COHORT,
            ],
        ),
        (
            "QRC-baseline exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QRC_BASELINE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/applications/qrc_baseline.py,*/applications/quantum_reservoir.py,*/surrogates/models.py,*/surrogates/train.py,*/surrogates/fidelity.py",
            ],
        ),
    ]


__all__ = [
    "QRC_BASELINE_COVERAGE_COHORT",
    "QRC_BASELINE_COVERAGE_DATA_FILE",
    "QRC_BASELINE_DOCSTRING_RATCHET",
    "QRC_BASELINE_SOURCE",
    "QRC_BASELINE_SOURCES",
    "QRC_BASELINE_TYPING_RATCHET",
    "SURROGATE_MODEL_SOURCE",
    "SURROGATE_FIDELITY_SOURCE",
    "SURROGATE_TRAIN_SOURCE",
    "QUANTUM_RESERVOIR_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
