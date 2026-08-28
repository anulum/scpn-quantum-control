# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-kernel classifier quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
TOPOLOGY_KERNEL_CLASSIFIER_SOURCE = (
    "src/scpn_quantum_control/topology_kernel_product/classifier.py"
)
TOPOLOGY_KERNEL_CLASSIFIER_COVERAGE_COHORT = [
    "tests/test_topology_kernel_product_classifier.py",
]
TOPOLOGY_KERNEL_CLASSIFIER_TYPING_RATCHET = [
    TOPOLOGY_KERNEL_CLASSIFIER_SOURCE,
    *TOPOLOGY_KERNEL_CLASSIFIER_COVERAGE_COHORT,
    "tools/topology_kernel_classifier_quality_gates.py",
    "tests/test_topology_kernel_classifier_quality_gate.py",
]
TOPOLOGY_KERNEL_CLASSIFIER_DOCSTRING_RATCHET = [
    *TOPOLOGY_KERNEL_CLASSIFIER_TYPING_RATCHET,
]
TOPOLOGY_KERNEL_CLASSIFIER_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-topology-kernel-classifier-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-topology-kernel-classifier-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *TOPOLOGY_KERNEL_CLASSIFIER_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D topology-kernel classifier quality ratchet",
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
                *TOPOLOGY_KERNEL_CLASSIFIER_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build classifier execution and exact source-coverage gates."""
    return [
        (
            "topology-kernel classifier focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={TOPOLOGY_KERNEL_CLASSIFIER_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *TOPOLOGY_KERNEL_CLASSIFIER_COVERAGE_COHORT,
            ],
        ),
        (
            "topology-kernel classifier exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={TOPOLOGY_KERNEL_CLASSIFIER_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/topology_kernel_product/classifier.py",
            ],
        ),
    ]


__all__ = [
    "TOPOLOGY_KERNEL_CLASSIFIER_COVERAGE_COHORT",
    "TOPOLOGY_KERNEL_CLASSIFIER_COVERAGE_DATA_FILE",
    "TOPOLOGY_KERNEL_CLASSIFIER_DOCSTRING_RATCHET",
    "TOPOLOGY_KERNEL_CLASSIFIER_SOURCE",
    "TOPOLOGY_KERNEL_CLASSIFIER_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
