# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Circuit preservation quality gates
"""Owning strict typing, documentation and native circuit conformance commands."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
SOURCES = [
    "src/scpn_quantum_control/compiler/circuit_pass_records.py",
    "src/scpn_quantum_control/compiler/circuit_source.py",
    "src/scpn_quantum_control/compiler/circuit_pass_qualification.py",
]
"""Immutable records, actual source import and semantic qualification owners."""
TESTS = [
    "tests/test_circuit_pass_records.py",
    "tests/test_circuit_source.py",
    "tests/test_circuit_pass_qualification.py",
    "tests/test_compiler_observable_preservation.py",
    "tests/test_circuit_preservation_quality_gate.py",
]
"""Dedicated owners and public compiler/quality integration tests."""
TYPING = [*SOURCES, *TESTS, "tools/circuit_preservation_quality_gates.py"]
"""Same strictly typed and NumPy-documented cohort in local and hosted gates."""
INCLUDE = "*/compiler/circuit_pass_records.py,*/compiler/circuit_source.py,*/compiler/circuit_pass_qualification.py,*/tools/circuit_preservation_quality_gates.py"
"""Exact full owner denominators, without exclusions or lowered thresholds."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build the owning strict typing and native documentation commands.

    Parameters
    ----------
    python
        Existing runner executable; no dependency installation occurs.

    Returns
    -------
    list[Gate]
        Two named commands for the actual source and dedicated test owners.

    """
    return [
        (
            "mypy-strict-circuit-preservation",
            [python, "-m", "mypy", "--strict", "--explicit-package-bases", *TYPING],
        ),
        (
            "ruff D circuit preservation",
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
                *TYPING,
            ],
        ),
    ]


def build_coverage_gates(python: str, data_file: str) -> list[Gate]:
    """Build the real native conformance and complete owner-coverage gates.

    Parameters
    ----------
    python
        Existing runner executable for both commands.
    data_file
        Exact task-owned or runner-local coverage database path.

    Returns
    -------
    list[Gate]
        Real public-source tests followed by the original 100 percent branch gate.

    """
    return [
        (
            "circuit preservation focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data_file}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                "--no-cov",
                *TESTS,
            ],
        ),
        (
            "circuit preservation exact coverage",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data_file}",
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                f"--include={INCLUDE}",
            ],
        ),
    ]
