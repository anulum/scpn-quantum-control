# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hierarchical-key quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HIERARCHICAL_KEYS_SOURCE = "src/scpn_quantum_control/crypto/hierarchical_keys.py"
HIERARCHICAL_KEYS_PRIMARY_TEST = "tests/test_crypto_hierarchical_keys.py"
HIERARCHICAL_KEYS_COVERAGE_COHORT = [
    HIERARCHICAL_KEYS_PRIMARY_TEST,
    "tests/test_crypto_expanded.py",
    "tests/test_crypto_properties.py",
    "tests/test_crypto_exports.py",
]
HIERARCHICAL_KEYS_TYPING_RATCHET = [
    HIERARCHICAL_KEYS_SOURCE,
    HIERARCHICAL_KEYS_PRIMARY_TEST,
    "tools/hierarchical_keys_quality_gates.py",
    "tests/test_hierarchical_keys_quality_gate.py",
]
HIERARCHICAL_KEYS_DOCSTRING_RATCHET = [
    *HIERARCHICAL_KEYS_TYPING_RATCHET,
]
HIERARCHICAL_KEYS_COVERAGE_DATA_FILE = "/tmp/scpn-qc-hierarchical-keys-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-hierarchical-keys-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *HIERARCHICAL_KEYS_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D hierarchical-keys quality ratchet",
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
                *HIERARCHICAL_KEYS_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build public crypto execution and exact source-coverage gates."""
    return [
        (
            "hierarchical-keys focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={HIERARCHICAL_KEYS_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *HIERARCHICAL_KEYS_COVERAGE_COHORT,
            ],
        ),
        (
            "hierarchical-keys exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={HIERARCHICAL_KEYS_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/crypto/hierarchical_keys.py",
            ],
        ),
    ]


__all__ = [
    "HIERARCHICAL_KEYS_COVERAGE_COHORT",
    "HIERARCHICAL_KEYS_COVERAGE_DATA_FILE",
    "HIERARCHICAL_KEYS_DOCSTRING_RATCHET",
    "HIERARCHICAL_KEYS_PRIMARY_TEST",
    "HIERARCHICAL_KEYS_SOURCE",
    "HIERARCHICAL_KEYS_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
