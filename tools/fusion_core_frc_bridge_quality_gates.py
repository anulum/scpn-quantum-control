# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — fusion-core FRC bridge quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
FUSION_CORE_FRC_BRIDGE_SOURCE = "src/scpn_quantum_control/bridge/fusion_core_frc.py"
FUSION_CORE_FRC_BRIDGE_COVERAGE_COHORT = [
    "tests/test_fusion_core_frc.py",
    "tests/test_frc_pulsed_qaoa_branches.py",
]
FUSION_CORE_FRC_BRIDGE_TYPING_RATCHET = [
    FUSION_CORE_FRC_BRIDGE_SOURCE,
    "tools/fusion_core_frc_bridge_quality_gates.py",
    "tests/test_fusion_core_frc_bridge_quality_gate.py",
]
FUSION_CORE_FRC_BRIDGE_DOCSTRING_RATCHET = [
    FUSION_CORE_FRC_BRIDGE_SOURCE,
    *FUSION_CORE_FRC_BRIDGE_COVERAGE_COHORT,
    "tools/fusion_core_frc_bridge_quality_gates.py",
    "tests/test_fusion_core_frc_bridge_quality_gate.py",
]
FUSION_CORE_FRC_BRIDGE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-fusion-core-frc-bridge-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-fusion-core-frc-bridge-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *FUSION_CORE_FRC_BRIDGE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D fusion-core-frc-bridge quality ratchet",
            [
                python,
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--select",
                "D,D413",
                "--config",
                'lint.pydocstyle.convention = "numpy"',
                *FUSION_CORE_FRC_BRIDGE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline execution and exact source coverage gates."""
    return [
        (
            "fusion-core-frc-bridge focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={FUSION_CORE_FRC_BRIDGE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *FUSION_CORE_FRC_BRIDGE_COVERAGE_COHORT,
            ],
        ),
        (
            "fusion-core-frc-bridge exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={FUSION_CORE_FRC_BRIDGE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/bridge/fusion_core_frc.py",
            ],
        ),
    ]


__all__ = [
    "FUSION_CORE_FRC_BRIDGE_COVERAGE_COHORT",
    "FUSION_CORE_FRC_BRIDGE_COVERAGE_DATA_FILE",
    "FUSION_CORE_FRC_BRIDGE_DOCSTRING_RATCHET",
    "FUSION_CORE_FRC_BRIDGE_SOURCE",
    "FUSION_CORE_FRC_BRIDGE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
