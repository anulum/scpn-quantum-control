# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — KYMA v2 coupling quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
KYMA_V2_COUPLING_SOURCE = "src/scpn_quantum_control/benchmarks/kyma_v2/coupling.py"
KYMA_V2_COUPLING_COVERAGE_COHORT = [
    "tests/test_kyma_v2_coupling.py",
]
KYMA_V2_COUPLING_TYPING_RATCHET = [
    KYMA_V2_COUPLING_SOURCE,
    *KYMA_V2_COUPLING_COVERAGE_COHORT,
    "tools/kyma_v2_coupling_quality_gates.py",
    "tests/test_kyma_v2_coupling_quality_gate.py",
]
KYMA_V2_COUPLING_DOCSTRING_RATCHET = [
    *KYMA_V2_COUPLING_TYPING_RATCHET,
]
KYMA_V2_COUPLING_COVERAGE_DATA_FILE = "/tmp/scpn-qc-kyma-v2-coupling-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-kyma-v2-coupling-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KYMA_V2_COUPLING_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D KYMA v2 coupling quality ratchet",
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
                *KYMA_V2_COUPLING_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real coupling execution and exact source-coverage gates."""
    return [
        (
            "KYMA v2 coupling focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KYMA_V2_COUPLING_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *KYMA_V2_COUPLING_COVERAGE_COHORT,
            ],
        ),
        (
            "KYMA v2 coupling exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KYMA_V2_COUPLING_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/benchmarks/kyma_v2/coupling.py",
            ],
        ),
    ]


__all__ = [
    "KYMA_V2_COUPLING_COVERAGE_COHORT",
    "KYMA_V2_COUPLING_COVERAGE_DATA_FILE",
    "KYMA_V2_COUPLING_DOCSTRING_RATCHET",
    "KYMA_V2_COUPLING_SOURCE",
    "KYMA_V2_COUPLING_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
