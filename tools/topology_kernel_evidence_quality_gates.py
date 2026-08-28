# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-kernel evidence quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
TOPOLOGY_KERNEL_EVIDENCE_SOURCE = "src/scpn_quantum_control/topology_kernel_product/evidence.py"
"""Production source owned by deterministic topology-kernel evidence."""
TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_COHORT = [
    "tests/test_topology_kernel_product_evidence.py",
]
"""Tests that own evidence construction, custody, and exact coverage."""
TOPOLOGY_KERNEL_EVIDENCE_TYPING_RATCHET = [
    TOPOLOGY_KERNEL_EVIDENCE_SOURCE,
    "tools/topology_kernel_evidence_quality_gates.py",
    "tests/test_topology_kernel_evidence_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""
TOPOLOGY_KERNEL_EVIDENCE_DOCSTRING_RATCHET = [
    TOPOLOGY_KERNEL_EVIDENCE_SOURCE,
    "tests/test_topology_kernel_product_evidence.py",
    "tools/topology_kernel_evidence_quality_gates.py",
    "tests/test_topology_kernel_evidence_quality_gate.py",
]
"""Complete evidence-owner and gate-contract docstring cohort."""
TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-topology-kernel-evidence-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the topology-kernel evidence owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-topology-kernel-evidence-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *TOPOLOGY_KERNEL_EVIDENCE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D topology-kernel-evidence quality ratchet",
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
                *TOPOLOGY_KERNEL_EVIDENCE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "topology-kernel-evidence focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_COHORT,
            ],
        ),
        (
            "topology-kernel-evidence exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/topology_kernel_product/evidence.py",
            ],
        ),
    ]


__all__ = [
    "TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_COHORT",
    "TOPOLOGY_KERNEL_EVIDENCE_COVERAGE_DATA_FILE",
    "TOPOLOGY_KERNEL_EVIDENCE_DOCSTRING_RATCHET",
    "TOPOLOGY_KERNEL_EVIDENCE_SOURCE",
    "TOPOLOGY_KERNEL_EVIDENCE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
