# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — QPU result-pack quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QPU_RESULT_PACK_SOURCE = "src/scpn_quantum_control/studio/qpu_result_pack.py"
QPU_RESULT_PACK_BRIDGE = "src/scpn_quantum_control/hardware/qpu_result_pack_bridge.py"
EXECUTE_SOURCE = "src/scpn_quantum_control/studio/executive_execute.py"
EXECUTIVE_CLI_SOURCE = "src/scpn_quantum_control/studio/executive_cli.py"
EXECUTE_TEST = "tests/test_studio_executive_execute.py"
EXECUTIVE_CLI_TEST = "tests/test_studio_executive_cli.py"
QPU_RESULT_PACK_COVERAGE_COHORT = [
    "tests/test_studio_qpu_result_pack.py",
    "tests/test_qpu_result_pack_bridge.py",
    EXECUTE_TEST,
    EXECUTIVE_CLI_TEST,
]
QPU_RESULT_PACK_TYPING_RATCHET = [
    QPU_RESULT_PACK_SOURCE,
    QPU_RESULT_PACK_BRIDGE,
    EXECUTE_SOURCE,
    EXECUTIVE_CLI_SOURCE,
    EXECUTE_TEST,
    EXECUTIVE_CLI_TEST,
    "tools/qpu_result_pack_quality_gates.py",
    "tests/test_qpu_result_pack_quality_gate.py",
]
QPU_RESULT_PACK_DOCSTRING_RATCHET = [
    QPU_RESULT_PACK_SOURCE,
    QPU_RESULT_PACK_BRIDGE,
    EXECUTE_SOURCE,
    "tests/test_studio_qpu_result_pack.py",
    "tests/test_qpu_result_pack_bridge.py",
    EXECUTE_TEST,
    "tools/qpu_result_pack_quality_gates.py",
    "tests/test_qpu_result_pack_quality_gate.py",
]
QPU_RESULT_PACK_COVERAGE_DATA_FILE = "/tmp/scpn-qc-qpu-result-pack-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-qpu-result-pack-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QPU_RESULT_PACK_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D QPU result-pack quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "D,D413,D417",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *QPU_RESULT_PACK_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline QPU result-pack execution and exact coverage gates."""
    return [
        (
            "QPU result-pack focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QPU_RESULT_PACK_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QPU_RESULT_PACK_COVERAGE_COHORT,
            ],
        ),
        (
            "QPU result-pack exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QPU_RESULT_PACK_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                (
                    "--include=*/studio/qpu_result_pack.py,"
                    "*/studio/executive_execute.py,*/studio/executive_cli.py"
                ),
            ],
        ),
    ]


__all__ = [
    "EXECUTE_SOURCE",
    "EXECUTE_TEST",
    "EXECUTIVE_CLI_SOURCE",
    "EXECUTIVE_CLI_TEST",
    "QPU_RESULT_PACK_BRIDGE",
    "QPU_RESULT_PACK_COVERAGE_COHORT",
    "QPU_RESULT_PACK_COVERAGE_DATA_FILE",
    "QPU_RESULT_PACK_DOCSTRING_RATCHET",
    "QPU_RESULT_PACK_SOURCE",
    "QPU_RESULT_PACK_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
