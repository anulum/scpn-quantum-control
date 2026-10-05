# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — XY-compile recompute-artifact quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
XY_COMPILE_RECOMPUTE_ARTIFACT_SOURCE = (
    "src/scpn_quantum_control/studio/xy_compile_recompute_artifact.py"
)
COMPILE_SOURCE = "src/scpn_quantum_control/studio/executive_compile.py"
EXECUTIVE_CLI_SOURCE = "src/scpn_quantum_control/studio/executive_cli.py"
COMPILE_TEST = "tests/test_studio_executive_compile.py"
EXECUTIVE_CLI_TEST = "tests/test_studio_executive_cli.py"
XY_COMPILE_RECOMPUTE_ARTIFACT_COVERAGE_COHORT = [
    "tests/test_studio_xy_compile_recompute_artifact.py",
    COMPILE_TEST,
    EXECUTIVE_CLI_TEST,
]
XY_COMPILE_RECOMPUTE_ARTIFACT_TYPING_RATCHET = [
    XY_COMPILE_RECOMPUTE_ARTIFACT_SOURCE,
    COMPILE_SOURCE,
    EXECUTIVE_CLI_SOURCE,
    *XY_COMPILE_RECOMPUTE_ARTIFACT_COVERAGE_COHORT,
    "tools/xy_compile_recompute_artifact_quality_gates.py",
    "tests/test_xy_compile_recompute_artifact_quality_gate.py",
]
XY_COMPILE_RECOMPUTE_ARTIFACT_DOCSTRING_RATCHET = [
    XY_COMPILE_RECOMPUTE_ARTIFACT_SOURCE,
    COMPILE_SOURCE,
    "tests/test_studio_xy_compile_recompute_artifact.py",
    COMPILE_TEST,
    "tools/xy_compile_recompute_artifact_quality_gates.py",
    "tests/test_xy_compile_recompute_artifact_quality_gate.py",
]
XY_COMPILE_RECOMPUTE_ARTIFACT_COVERAGE_DATA_FILE = (  # nosec B108
    "/tmp/scpn-qc-xy-compile-recompute-artifact-quality.coverage"
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-XY-compile-recompute-artifact-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *XY_COMPILE_RECOMPUTE_ARTIFACT_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D XY-compile-recompute-artifact quality ratchet",
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
                *XY_COMPILE_RECOMPUTE_ARTIFACT_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build recompute-artifact execution and exact source-coverage gates."""
    return [
        (
            "XY-compile-recompute-artifact focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={XY_COMPILE_RECOMPUTE_ARTIFACT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *XY_COMPILE_RECOMPUTE_ARTIFACT_COVERAGE_COHORT,
            ],
        ),
        (
            "XY-compile-recompute-artifact exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={XY_COMPILE_RECOMPUTE_ARTIFACT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                (
                    "--include=*/studio/xy_compile_recompute_artifact.py,"
                    "*/studio/executive_compile.py,*/studio/executive_cli.py"
                ),
            ],
        ),
    ]


__all__ = [
    "COMPILE_SOURCE",
    "COMPILE_TEST",
    "EXECUTIVE_CLI_SOURCE",
    "EXECUTIVE_CLI_TEST",
    "XY_COMPILE_RECOMPUTE_ARTIFACT_COVERAGE_COHORT",
    "XY_COMPILE_RECOMPUTE_ARTIFACT_COVERAGE_DATA_FILE",
    "XY_COMPILE_RECOMPUTE_ARTIFACT_DOCSTRING_RATCHET",
    "XY_COMPILE_RECOMPUTE_ARTIFACT_SOURCE",
    "XY_COMPILE_RECOMPUTE_ARTIFACT_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
