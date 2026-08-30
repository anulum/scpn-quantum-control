# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio replay quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HARDWARE_RESULT_PACKS_SOURCE = "src/scpn_quantum_control/hardware_result_packs.py"
STUDIO_REPLAY_SOURCE = "src/scpn_quantum_control/studio/executive_replay.py"
STUDIO_EXECUTIVE_CLI_SOURCE = "src/scpn_quantum_control/studio/executive_cli.py"
HARDWARE_RESULT_PACKS_TEST = "tests/test_hardware_result_packs.py"
STUDIO_REPLAY_TEST = "tests/test_studio_executive_replay.py"
STUDIO_EXECUTIVE_CLI_TEST = "tests/test_studio_executive_cli.py"
STUDIO_REPLAY_COVERAGE_COHORT = [
    HARDWARE_RESULT_PACKS_TEST,
    STUDIO_REPLAY_TEST,
    STUDIO_EXECUTIVE_CLI_TEST,
]
STUDIO_REPLAY_TYPING_RATCHET = [
    HARDWARE_RESULT_PACKS_SOURCE,
    STUDIO_REPLAY_SOURCE,
    STUDIO_EXECUTIVE_CLI_SOURCE,
    *STUDIO_REPLAY_COVERAGE_COHORT,
    "tools/studio_replay_quality_gates.py",
    "tests/test_studio_replay_quality_gate.py",
]
STUDIO_REPLAY_DOCSTRING_RATCHET = [
    HARDWARE_RESULT_PACKS_SOURCE,
    STUDIO_REPLAY_SOURCE,
    HARDWARE_RESULT_PACKS_TEST,
    STUDIO_REPLAY_TEST,
    "tools/studio_replay_quality_gates.py",
    "tests/test_studio_replay_quality_gate.py",
]
STUDIO_REPLAY_COVERAGE_DATA_FILE = "/tmp/scpn-qc-studio-replay-quality.coverage"  # nosec B108
STUDIO_REPLAY_COVERAGE_INCLUDE = ",".join(
    [
        "*/hardware_result_packs.py",
        "*/studio/executive_replay.py",
        "*/studio/executive_cli.py",
    ]
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-studio-replay-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STUDIO_REPLAY_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D studio-replay quality ratchet",
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
                *STUDIO_REPLAY_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline replay execution and exact source-coverage gates."""
    return [
        (
            "studio-replay focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_REPLAY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STUDIO_REPLAY_COVERAGE_COHORT,
            ],
        ),
        (
            "studio-replay exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_REPLAY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={STUDIO_REPLAY_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "HARDWARE_RESULT_PACKS_SOURCE",
    "HARDWARE_RESULT_PACKS_TEST",
    "STUDIO_EXECUTIVE_CLI_SOURCE",
    "STUDIO_EXECUTIVE_CLI_TEST",
    "STUDIO_REPLAY_COVERAGE_COHORT",
    "STUDIO_REPLAY_COVERAGE_DATA_FILE",
    "STUDIO_REPLAY_COVERAGE_INCLUDE",
    "STUDIO_REPLAY_DOCSTRING_RATCHET",
    "STUDIO_REPLAY_SOURCE",
    "STUDIO_REPLAY_TEST",
    "STUDIO_REPLAY_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
