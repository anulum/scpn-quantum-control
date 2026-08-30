# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio simulation quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
STUDIO_SIMULATION_SOURCE = "src/scpn_quantum_control/studio/executive_simulate.py"
STUDIO_EXECUTIVE_CLI_SOURCE = "src/scpn_quantum_control/studio/executive_cli.py"
STUDIO_SIMULATION_TEST = "tests/test_studio_executive_simulate.py"
STUDIO_EXECUTIVE_CLI_TEST = "tests/test_studio_executive_cli.py"
STUDIO_SIMULATION_COVERAGE_COHORT = [
    STUDIO_SIMULATION_TEST,
    STUDIO_EXECUTIVE_CLI_TEST,
]
STUDIO_SIMULATION_TYPING_RATCHET = [
    STUDIO_SIMULATION_SOURCE,
    STUDIO_EXECUTIVE_CLI_SOURCE,
    *STUDIO_SIMULATION_COVERAGE_COHORT,
    "tools/studio_simulation_quality_gates.py",
    "tests/test_studio_simulation_quality_gate.py",
]
STUDIO_SIMULATION_DOCSTRING_RATCHET = [
    STUDIO_SIMULATION_SOURCE,
    STUDIO_SIMULATION_TEST,
    "tools/studio_simulation_quality_gates.py",
    "tests/test_studio_simulation_quality_gate.py",
]
STUDIO_SIMULATION_COVERAGE_DATA_FILE = "/tmp/scpn-qc-studio-simulation-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-studio-simulation-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STUDIO_SIMULATION_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D studio-simulation quality ratchet",
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
                *STUDIO_SIMULATION_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real simulation execution and exact source-coverage gates."""
    return [
        (
            "studio-simulation focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_SIMULATION_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STUDIO_SIMULATION_COVERAGE_COHORT,
            ],
        ),
        (
            "studio-simulation exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_SIMULATION_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/studio/executive_simulate.py,*/studio/executive_cli.py",
            ],
        ),
    ]


__all__ = [
    "STUDIO_EXECUTIVE_CLI_SOURCE",
    "STUDIO_EXECUTIVE_CLI_TEST",
    "STUDIO_SIMULATION_COVERAGE_COHORT",
    "STUDIO_SIMULATION_COVERAGE_DATA_FILE",
    "STUDIO_SIMULATION_DOCSTRING_RATCHET",
    "STUDIO_SIMULATION_SOURCE",
    "STUDIO_SIMULATION_TEST",
    "STUDIO_SIMULATION_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
