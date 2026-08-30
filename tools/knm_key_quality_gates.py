# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — K_nm key-material quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
KNM_KEY_SOURCE = "src/scpn_quantum_control/crypto/knm_key.py"
KNM_KEY_PRIMARY_TEST = "tests/test_crypto_knm_key.py"
KNM_KEY_COVERAGE_COHORT = [
    KNM_KEY_PRIMARY_TEST,
    "tests/test_crypto_entanglement_qkd.py",
]
KNM_KEY_TYPING_RATCHET = [
    KNM_KEY_SOURCE,
    KNM_KEY_PRIMARY_TEST,
    "tools/knm_key_quality_gates.py",
    "tests/test_knm_key_quality_gate.py",
]
KNM_KEY_DOCSTRING_RATCHET = [*KNM_KEY_TYPING_RATCHET]
KNM_KEY_COVERAGE_DATA_FILE = "/tmp/scpn-qc-knm-key-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-knm-key-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *KNM_KEY_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D K_nm key-material quality ratchet",
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
                *KNM_KEY_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build local VQE/key execution and exact source-coverage gates."""
    return [
        (
            "K_nm key-material focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={KNM_KEY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *KNM_KEY_COVERAGE_COHORT,
            ],
        ),
        (
            "K_nm key-material exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={KNM_KEY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/crypto/knm_key.py",
            ],
        ),
    ]


__all__ = [
    "KNM_KEY_COVERAGE_COHORT",
    "KNM_KEY_COVERAGE_DATA_FILE",
    "KNM_KEY_DOCSTRING_RATCHET",
    "KNM_KEY_PRIMARY_TEST",
    "KNM_KEY_SOURCE",
    "KNM_KEY_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
