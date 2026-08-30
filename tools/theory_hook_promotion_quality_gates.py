# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — theory hook promotion quality gates
"""Build strict documentation, evidence-drift, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
THEORY_HOOK_PROMOTION_SOURCE = "src/scpn_quantum_control/analysis/theory_hook_promotion.py"
MAGIC_NONSTABILIZERNESS_SOURCE = "src/scpn_quantum_control/analysis/magic_nonstabilizerness.py"
THEORY_HOOK_PROMOTION_COVERAGE_INCLUDE = (
    "*/analysis/theory_hook_promotion.py,*/analysis/magic_nonstabilizerness.py"
)
THEORY_HOOK_PROMOTION_QUALITY_RATCHET = [
    THEORY_HOOK_PROMOTION_SOURCE,
    MAGIC_NONSTABILIZERNESS_SOURCE,
    "tests/test_theory_hook_promotion.py",
    "tests/test_magic_nonstabilizerness.py",
    "scripts/run_theory_hook_promotion_evidence.py",
    "tools/theory_hook_promotion_quality_gates.py",
    "tests/test_theory_hook_promotion_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
THEORY_HOOK_PROMOTION_COVERAGE_COHORT = [
    "tests/test_theory_hook_promotion.py",
    "tests/test_magic_nonstabilizerness.py",
]
"""Tests that own exact theory-hook-promotion coverage."""
THEORY_HOOK_PROMOTION_COVERAGE_DATA_FILE = "/tmp/scpn-qc-theory-hook-promotion-quality.coverage"  # nosec B108
"""Isolated coverage database for theory-hook-promotion diagnostics."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing, NumPy-docstring, and evidence-drift gates."""
    return [
        (
            "mypy-strict-theory-hook-promotion-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *THEORY_HOOK_PROMOTION_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D theory-hook-promotion quality ratchet",
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
                *THEORY_HOOK_PROMOTION_QUALITY_RATCHET,
            ],
        ),
        (
            "theory-hook-promotion evidence drift",
            [python, "scripts/run_theory_hook_promotion_evidence.py", "--check"],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = THEORY_HOOK_PROMOTION_COVERAGE_DATA_FILE
    return [
        (
            "theory-hook-promotion focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *THEORY_HOOK_PROMOTION_COVERAGE_COHORT,
            ],
        ),
        (
            "theory-hook-promotion exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                f"--include={THEORY_HOOK_PROMOTION_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "MAGIC_NONSTABILIZERNESS_SOURCE",
    "THEORY_HOOK_PROMOTION_COVERAGE_COHORT",
    "THEORY_HOOK_PROMOTION_COVERAGE_DATA_FILE",
    "THEORY_HOOK_PROMOTION_COVERAGE_INCLUDE",
    "THEORY_HOOK_PROMOTION_QUALITY_RATCHET",
    "THEORY_HOOK_PROMOTION_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
