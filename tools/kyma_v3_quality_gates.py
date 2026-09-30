# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — KYMA v3 quality gates
"""Build strict documentation, typing, and exact coverage gates for KYMA v3."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
KYMA_V3_SOURCES = [
    "src/scpn_quantum_control/benchmarks/kyma_v3/__init__.py",
    "src/scpn_quantum_control/benchmarks/kyma_v3/task.py",
    "src/scpn_quantum_control/benchmarks/kyma_v3/substrate.py",
    "src/scpn_quantum_control/benchmarks/kyma_v3/baselines.py",
    "src/scpn_quantum_control/benchmarks/kyma_v3/probe.py",
    "scripts/run_kyma_v3_probe.py",
    "scripts/run_kyma_v3_ablation.py",
]
"""Production sources of the symbolic probe and its runner."""
KYMA_V3_COVERAGE_COHORT = [
    "tests/test_kyma_v3_task.py",
    "tests/test_kyma_v3_substrate.py",
    "tests/test_kyma_v3_baselines.py",
    "tests/test_kyma_v3_probe.py",
    "tests/test_kyma_v3_runner.py",
    "tests/test_kyma_v3_ablation.py",
]
"""Direct execution suites of every KYMA v3 source."""
KYMA_V3_QUALITY_RATCHET = [
    *KYMA_V3_SOURCES,
    *KYMA_V3_COVERAGE_COHORT,
    "tools/kyma_v3_quality_gates.py",
    "tests/test_kyma_v3_quality_gate.py",
]
"""Strict-typing and NumPy-docstring cohort."""
KYMA_V3_COVERAGE_DATA_FILE = "/tmp/scpn-qc-kyma-v3-quality.coverage"  # nosec B108
"""Isolated coverage database for KYMA v3."""
KYMA_V3_COVERAGE_INCLUDE = (
    "*/benchmarks/kyma_v3/*.py,*/scripts/run_kyma_v3_probe.py,*/scripts/run_kyma_v3_ablation.py"
)
"""Exact production paths required to remain at full branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Interpreter used to run the gates.

    Returns
    -------
    list of Gate
        Named commands.

    """
    return [
        (
            "mypy-strict-kyma-v3-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KYMA_V3_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D kyma-v3 quality ratchet",
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
                *KYMA_V3_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates.

    Parameters
    ----------
    python
        Interpreter used to run the gates.

    Returns
    -------
    list of Gate
        Named commands.

    """
    return [
        (
            "kyma-v3 focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KYMA_V3_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *KYMA_V3_COVERAGE_COHORT,
            ],
        ),
        (
            "kyma-v3 exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KYMA_V3_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={KYMA_V3_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "KYMA_V3_COVERAGE_COHORT",
    "KYMA_V3_COVERAGE_DATA_FILE",
    "KYMA_V3_COVERAGE_INCLUDE",
    "KYMA_V3_QUALITY_RATCHET",
    "KYMA_V3_SOURCES",
    "build_coverage_gates",
    "build_static_quality_gates",
]
