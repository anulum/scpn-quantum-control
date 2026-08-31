# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — topology-kernel schema quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

TOPOLOGY_KERNEL_SCHEMA_SOURCE = "src/scpn_quantum_control/topology_kernel_product/schema.py"
"""Production source owned by the topology-kernel schema."""
TOPOLOGY_KERNEL_SYNTHETIC_SOURCE = "src/scpn_quantum_control/topology_kernel_product/synthetic.py"
"""Production source for deterministic topology controls and teacher data."""

TOPOLOGY_KERNEL_SCHEMA_COVERAGE_COHORT = [
    "tests/test_topology_kernel_product_schema.py",
    "tests/test_topology_kernel_product_synthetic.py",
]
"""Tests that own exact topology-kernel schema and synthetic-task coverage."""

TOPOLOGY_KERNEL_SCHEMA_TYPING_RATCHET = [
    TOPOLOGY_KERNEL_SCHEMA_SOURCE,
    TOPOLOGY_KERNEL_SYNTHETIC_SOURCE,
    "tools/topology_kernel_schema_quality_gates.py",
    "tests/test_topology_kernel_schema_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""

TOPOLOGY_KERNEL_SCHEMA_DOCSTRING_RATCHET = [
    TOPOLOGY_KERNEL_SCHEMA_SOURCE,
    TOPOLOGY_KERNEL_SYNTHETIC_SOURCE,
    *TOPOLOGY_KERNEL_SCHEMA_COVERAGE_COHORT,
    "tools/topology_kernel_schema_quality_gates.py",
    "tests/test_topology_kernel_schema_quality_gate.py",
]
"""Complete production, owner-test, and gate-contract docstring cohort."""

TOPOLOGY_KERNEL_SCHEMA_COVERAGE_DATA_FILE = "/tmp/scpn-qc-topology-kernel-schema-quality.coverage"  # nosec B108
"""Isolated coverage database for the topology-kernel schema owner."""
TOPOLOGY_KERNEL_SCHEMA_COVERAGE_INCLUDE = (
    "*/topology_kernel_product/schema.py,*/topology_kernel_product/synthetic.py"
)
"""Production schema and synthetic-task sources enforced at exact coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-topology-kernel-schema-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *TOPOLOGY_KERNEL_SCHEMA_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D topology-kernel-schema quality ratchet",
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
                *TOPOLOGY_KERNEL_SCHEMA_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "topology-kernel-schema focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={TOPOLOGY_KERNEL_SCHEMA_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *TOPOLOGY_KERNEL_SCHEMA_COVERAGE_COHORT,
            ],
        ),
        (
            "topology-kernel-schema exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={TOPOLOGY_KERNEL_SCHEMA_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={TOPOLOGY_KERNEL_SCHEMA_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "TOPOLOGY_KERNEL_SCHEMA_COVERAGE_COHORT",
    "TOPOLOGY_KERNEL_SCHEMA_COVERAGE_DATA_FILE",
    "TOPOLOGY_KERNEL_SCHEMA_COVERAGE_INCLUDE",
    "TOPOLOGY_KERNEL_SCHEMA_DOCSTRING_RATCHET",
    "TOPOLOGY_KERNEL_SCHEMA_SOURCE",
    "TOPOLOGY_KERNEL_SCHEMA_TYPING_RATCHET",
    "TOPOLOGY_KERNEL_SYNTHETIC_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
