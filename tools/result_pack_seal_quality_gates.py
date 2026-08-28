# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — result-pack-seal quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
RESULT_PACK_SEAL_SOURCE = "src/scpn_quantum_control/studio/result_pack_seal.py"
RESULT_PACK_SEAL_COVERAGE_COHORT = ["tests/test_result_pack_seal.py"]
RESULT_PACK_SEAL_TYPING_RATCHET = [
    RESULT_PACK_SEAL_SOURCE,
    *RESULT_PACK_SEAL_COVERAGE_COHORT,
    "tools/result_pack_seal_quality_gates.py",
    "tests/test_result_pack_seal_quality_gate.py",
]
RESULT_PACK_SEAL_DOCSTRING_RATCHET = [*RESULT_PACK_SEAL_TYPING_RATCHET]
RESULT_PACK_SEAL_COVERAGE_DATA_FILE = (  # nosec B108
    "/tmp/scpn-qc-result-pack-seal-quality.coverage"
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-result-pack-seal-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *RESULT_PACK_SEAL_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D result-pack-seal quality ratchet",
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
                *RESULT_PACK_SEAL_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real result-pack sealing and exact source-coverage gates."""
    return [
        (
            "result-pack-seal focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={RESULT_PACK_SEAL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *RESULT_PACK_SEAL_COVERAGE_COHORT,
            ],
        ),
        (
            "result-pack-seal exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={RESULT_PACK_SEAL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/studio/result_pack_seal.py",
            ],
        ),
    ]


__all__ = [
    "RESULT_PACK_SEAL_COVERAGE_COHORT",
    "RESULT_PACK_SEAL_COVERAGE_DATA_FILE",
    "RESULT_PACK_SEAL_DOCSTRING_RATCHET",
    "RESULT_PACK_SEAL_SOURCE",
    "RESULT_PACK_SEAL_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
