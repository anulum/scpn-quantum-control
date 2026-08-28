# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — ML-DSA honesty-seal quality gates
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
ML_DSA_SEAL_SOURCE = "src/scpn_quantum_control/crypto/ml_dsa_seal.py"
"""Production source owned by the ML-DSA signer and verifier."""
ML_DSA_SEAL_COVERAGE_COHORT = [
    "tests/test_ml_dsa_seal.py",
    "tests/test_result_pack_seal.py",
    "tests/test_qpu_result_pack_bridge.py",
    "tests/test_studio_qpu_result_pack.py",
]
"""Offline signer, seal, and result-pack integration tests."""
ML_DSA_SEAL_TYPING_RATCHET = [
    ML_DSA_SEAL_SOURCE,
    "tools/ml_dsa_seal_quality_gates.py",
    "tests/test_ml_dsa_seal_quality_gate.py",
]
"""Strict-typing cohort for production and gate contracts."""
ML_DSA_SEAL_DOCSTRING_RATCHET = [
    ML_DSA_SEAL_SOURCE,
    *ML_DSA_SEAL_COVERAGE_COHORT,
    "tools/ml_dsa_seal_quality_gates.py",
    "tests/test_ml_dsa_seal_quality_gate.py",
]
"""Complete signer, integration, and gate-contract docstring cohort."""
ML_DSA_SEAL_COVERAGE_DATA_FILE = "/tmp/scpn-qc-ml-dsa-seal-quality.coverage"  # nosec B108
"""Isolated coverage database for the ML-DSA seal owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-ml-dsa-seal-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *ML_DSA_SEAL_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D ml-dsa-seal quality ratchet",
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
                *ML_DSA_SEAL_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source coverage gates."""
    return [
        (
            "ml-dsa-seal focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={ML_DSA_SEAL_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *ML_DSA_SEAL_COVERAGE_COHORT,
            ],
        ),
        (
            "ml-dsa-seal exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={ML_DSA_SEAL_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/crypto/ml_dsa_seal.py",
            ],
        ),
    ]


__all__ = [
    "ML_DSA_SEAL_COVERAGE_COHORT",
    "ML_DSA_SEAL_COVERAGE_DATA_FILE",
    "ML_DSA_SEAL_DOCSTRING_RATCHET",
    "ML_DSA_SEAL_SOURCE",
    "ML_DSA_SEAL_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
