# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — KYMA v2 dynamics quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
KYMA_V2_DYNAMICS_SOURCE = "src/scpn_quantum_control/benchmarks/kyma_v2/dynamics.py"
"""Production source owned by the per-trial Kuramoto dynamics."""
KYMA_V2_TASK_SOURCE = "src/scpn_quantum_control/benchmarks/kyma_v2/task.py"
"""Production source owning masks, encoding, and the compositional split."""
KYMA_V2_DYNAMICS_COVERAGE_COHORT = [
    "tests/test_kyma_v2_dynamics.py",
    "tests/test_kyma_v2_teacher.py",
    "tests/test_kyma_v2_task.py",
]
"""Direct dynamics and teacher-consumer tests."""
KYMA_V2_DYNAMICS_TYPING_RATCHET = [
    KYMA_V2_DYNAMICS_SOURCE,
    KYMA_V2_TASK_SOURCE,
    "tests/test_kyma_v2_task.py",
    "tools/kyma_v2_dynamics_quality_gates.py",
    "tests/test_kyma_v2_dynamics_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""
KYMA_V2_DYNAMICS_DOCSTRING_RATCHET = [
    KYMA_V2_DYNAMICS_SOURCE,
    KYMA_V2_TASK_SOURCE,
    *KYMA_V2_DYNAMICS_COVERAGE_COHORT,
    "tools/kyma_v2_dynamics_quality_gates.py",
    "tests/test_kyma_v2_dynamics_quality_gate.py",
]
"""Complete dynamics, teacher-consumer, and gate docstring cohort."""
KYMA_V2_DYNAMICS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-kyma-v2-dynamics-quality.coverage"  # nosec B108
"""Isolated coverage database for KYMA v2 dynamics."""
KYMA_V2_DYNAMICS_COVERAGE_INCLUDE = (
    "--include=*/benchmarks/kyma_v2/dynamics.py,*/benchmarks/kyma_v2/task.py"
)
"""Exact production paths required to remain at full branch coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-kyma-v2-dynamics-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KYMA_V2_DYNAMICS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D kyma-v2-dynamics quality ratchet",
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
                *KYMA_V2_DYNAMICS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "kyma-v2-dynamics focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KYMA_V2_DYNAMICS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *KYMA_V2_DYNAMICS_COVERAGE_COHORT,
            ],
        ),
        (
            "kyma-v2-dynamics exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KYMA_V2_DYNAMICS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                KYMA_V2_DYNAMICS_COVERAGE_INCLUDE,
            ],
        ),
    ]


__all__ = [
    "KYMA_V2_DYNAMICS_COVERAGE_COHORT",
    "KYMA_V2_DYNAMICS_COVERAGE_DATA_FILE",
    "KYMA_V2_DYNAMICS_COVERAGE_INCLUDE",
    "KYMA_V2_DYNAMICS_DOCSTRING_RATCHET",
    "KYMA_V2_DYNAMICS_SOURCE",
    "KYMA_V2_DYNAMICS_TYPING_RATCHET",
    "KYMA_V2_TASK_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
