# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-geometry quality gates
"""Build strict typing, documentation and exact coverage gates for the quantum metric owners."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QUANTUM_GEOMETRY_SOURCES = [
    "src/scpn_quantum_control/analysis/qfi.py",
    "src/scpn_quantum_control/analysis/quantum_fisher_information.py",
    "src/scpn_quantum_control/phase/natural_gradient.py",
]
QUANTUM_GEOMETRY_CONFORMANCE_TEST = "tests/test_quantum_geometry_conformance.py"
QUANTUM_GEOMETRY_COVERAGE_COHORT = [
    QUANTUM_GEOMETRY_CONFORMANCE_TEST,
    "tests/test_qfi.py",
    "tests/test_phase_natural_gradient.py",
    "tests/test_observables.py",
]
QUANTUM_GEOMETRY_COVERAGE_INCLUDE = (
    "*/analysis/qfi.py,*/analysis/quantum_fisher_information.py,*/phase/natural_gradient.py"
)
QUANTUM_GEOMETRY_TYPING_RATCHET = [
    *QUANTUM_GEOMETRY_SOURCES,
    QUANTUM_GEOMETRY_CONFORMANCE_TEST,
    "tools/quantum_geometry_quality_gates.py",
    "tests/test_quantum_geometry_quality_gate.py",
]
QUANTUM_GEOMETRY_DOCSTRING_RATCHET = [*QUANTUM_GEOMETRY_TYPING_RATCHET]
QUANTUM_GEOMETRY_COVERAGE_DATA_FILE = "/tmp/scpn-qc-quantum-geometry-quality.coverage"  # nosec B108


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
            "mypy-strict-quantum-geometry-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QUANTUM_GEOMETRY_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D quantum-geometry quality ratchet",
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
                *QUANTUM_GEOMETRY_DOCSTRING_RATCHET,
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
            "quantum-geometry focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_GEOMETRY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QUANTUM_GEOMETRY_COVERAGE_COHORT,
            ],
        ),
        (
            "quantum-geometry exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_GEOMETRY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={QUANTUM_GEOMETRY_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "QUANTUM_GEOMETRY_CONFORMANCE_TEST",
    "QUANTUM_GEOMETRY_COVERAGE_COHORT",
    "QUANTUM_GEOMETRY_COVERAGE_DATA_FILE",
    "QUANTUM_GEOMETRY_COVERAGE_INCLUDE",
    "QUANTUM_GEOMETRY_DOCSTRING_RATCHET",
    "QUANTUM_GEOMETRY_SOURCES",
    "QUANTUM_GEOMETRY_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
