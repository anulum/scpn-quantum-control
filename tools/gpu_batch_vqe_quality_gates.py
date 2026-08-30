# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — GPU-batch VQE quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
GPU_BATCH_VQE_SOURCE = "src/scpn_quantum_control/phase/gpu_batch_vqe.py"
GPU_BATCH_VQE_PRIMARY_TEST = "tests/test_gpu_batch_vqe.py"
GPU_BATCH_VQE_COVERAGE_COHORT = [GPU_BATCH_VQE_PRIMARY_TEST]
GPU_BATCH_VQE_TYPING_RATCHET = [
    GPU_BATCH_VQE_SOURCE,
    GPU_BATCH_VQE_PRIMARY_TEST,
    "tools/gpu_batch_vqe_quality_gates.py",
    "tests/test_gpu_batch_vqe_quality_gate.py",
]
GPU_BATCH_VQE_DOCSTRING_RATCHET = [*GPU_BATCH_VQE_TYPING_RATCHET]
GPU_BATCH_VQE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-gpu-batch-vqe-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-GPU-batch-VQE-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *GPU_BATCH_VQE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D GPU-batch VQE quality ratchet",
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
                *GPU_BATCH_VQE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build bounded local batch-VQE execution and exact coverage gates."""
    return [
        (
            "GPU-batch VQE focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={GPU_BATCH_VQE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *GPU_BATCH_VQE_COVERAGE_COHORT,
            ],
        ),
        (
            "GPU-batch VQE exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={GPU_BATCH_VQE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/phase/gpu_batch_vqe.py",
            ],
        ),
    ]


__all__ = [
    "GPU_BATCH_VQE_COVERAGE_COHORT",
    "GPU_BATCH_VQE_COVERAGE_DATA_FILE",
    "GPU_BATCH_VQE_DOCSTRING_RATCHET",
    "GPU_BATCH_VQE_PRIMARY_TEST",
    "GPU_BATCH_VQE_SOURCE",
    "GPU_BATCH_VQE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
