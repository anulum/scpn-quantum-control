# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — synchronisation-uncertainty quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
SYNC_UNCERTAINTY_SOURCE = "src/scpn_quantum_control/analysis/sync_uncertainty.py"
ZNE_UNCERTAINTY_SOURCE = "src/scpn_quantum_control/mitigation/zne_uncertainty.py"
STUDIO_MITIGATE_SOURCE = "src/scpn_quantum_control/studio/executive_mitigate.py"
STUDIO_EXECUTIVE_CLI_SOURCE = "src/scpn_quantum_control/studio/executive_cli.py"
SYNC_UNCERTAINTY_TEST = "tests/test_sync_uncertainty.py"
COUPLING_INVARIANT_TEST = "tests/test_studio_coupling_invariant.py"
ZNE_UNCERTAINTY_TEST = "tests/test_zne_uncertainty.py"
STUDIO_MITIGATE_TEST = "tests/test_studio_executive_mitigate.py"
STUDIO_EXECUTIVE_CLI_TEST = "tests/test_studio_executive_cli.py"
SYNC_UNCERTAINTY_COVERAGE_COHORT = [
    SYNC_UNCERTAINTY_TEST,
    COUPLING_INVARIANT_TEST,
    ZNE_UNCERTAINTY_TEST,
    STUDIO_MITIGATE_TEST,
    STUDIO_EXECUTIVE_CLI_TEST,
]
SYNC_UNCERTAINTY_TYPING_RATCHET = [
    SYNC_UNCERTAINTY_SOURCE,
    ZNE_UNCERTAINTY_SOURCE,
    STUDIO_MITIGATE_SOURCE,
    STUDIO_EXECUTIVE_CLI_SOURCE,
    *SYNC_UNCERTAINTY_COVERAGE_COHORT,
    "tools/sync_uncertainty_quality_gates.py",
    "tests/test_sync_uncertainty_quality_gate.py",
]
SYNC_UNCERTAINTY_DOCSTRING_RATCHET = [
    SYNC_UNCERTAINTY_SOURCE,
    ZNE_UNCERTAINTY_SOURCE,
    STUDIO_MITIGATE_SOURCE,
    SYNC_UNCERTAINTY_TEST,
    ZNE_UNCERTAINTY_TEST,
    STUDIO_MITIGATE_TEST,
    "tools/sync_uncertainty_quality_gates.py",
    "tests/test_sync_uncertainty_quality_gate.py",
]
SYNC_UNCERTAINTY_COVERAGE_DATA_FILE = "/tmp/scpn-qc-sync-uncertainty-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-sync-uncertainty-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *SYNC_UNCERTAINTY_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D synchronisation-uncertainty quality ratchet",
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
                *SYNC_UNCERTAINTY_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build connected uncertainty execution and exact coverage gates."""
    return [
        (
            "synchronisation-uncertainty focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={SYNC_UNCERTAINTY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *SYNC_UNCERTAINTY_COVERAGE_COHORT,
            ],
        ),
        (
            "synchronisation-uncertainty exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={SYNC_UNCERTAINTY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                (
                    "--include=*/analysis/sync_uncertainty.py,"
                    "*/mitigation/zne_uncertainty.py,*/studio/executive_mitigate.py,"
                    "*/studio/executive_cli.py"
                ),
            ],
        ),
    ]


__all__ = [
    "COUPLING_INVARIANT_TEST",
    "STUDIO_EXECUTIVE_CLI_SOURCE",
    "STUDIO_EXECUTIVE_CLI_TEST",
    "STUDIO_MITIGATE_SOURCE",
    "STUDIO_MITIGATE_TEST",
    "SYNC_UNCERTAINTY_COVERAGE_COHORT",
    "SYNC_UNCERTAINTY_COVERAGE_DATA_FILE",
    "SYNC_UNCERTAINTY_DOCSTRING_RATCHET",
    "SYNC_UNCERTAINTY_SOURCE",
    "SYNC_UNCERTAINTY_TYPING_RATCHET",
    "SYNC_UNCERTAINTY_TEST",
    "ZNE_UNCERTAINTY_SOURCE",
    "ZNE_UNCERTAINTY_TEST",
    "build_coverage_gates",
    "build_static_quality_gates",
]
