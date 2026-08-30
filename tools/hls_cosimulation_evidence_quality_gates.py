# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — HLS cosimulation-evidence quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
HLS_COSIMULATION_EVIDENCE_SOURCE = (
    "src/scpn_quantum_control/benchmarks/hls_cosimulation_evidence.py"
)
ULTRASCALE_HLS_SOURCE = "src/scpn_quantum_control/codegen/ultrascale_hls.py"
HLS_COSIMULATION_EVIDENCE_SOURCES = [
    HLS_COSIMULATION_EVIDENCE_SOURCE,
    ULTRASCALE_HLS_SOURCE,
]
HLS_COSIMULATION_EVIDENCE_COVERAGE_COHORT = [
    "tests/test_hls_cosimulation_evidence.py",
    "tests/test_run_hls_cosimulation_evidence.py",
    "tests/test_ultrascale_hls.py",
    "tests/test_ultrascale_hls_branch.py",
]
HLS_COSIMULATION_EVIDENCE_TYPING_RATCHET = [
    *HLS_COSIMULATION_EVIDENCE_SOURCES,
    "tests/test_ultrascale_hls_branch.py",
    "tools/hls_cosimulation_evidence_quality_gates.py",
    "tests/test_hls_cosimulation_evidence_quality_gate.py",
]
HLS_COSIMULATION_EVIDENCE_DOCSTRING_RATCHET = [
    *HLS_COSIMULATION_EVIDENCE_SOURCES,
    *HLS_COSIMULATION_EVIDENCE_COVERAGE_COHORT,
    "tools/hls_cosimulation_evidence_quality_gates.py",
    "tests/test_hls_cosimulation_evidence_quality_gate.py",
]
HLS_COSIMULATION_EVIDENCE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-hls-cosimulation-evidence-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-hls-cosimulation-evidence-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *HLS_COSIMULATION_EVIDENCE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D HLS cosimulation-evidence quality ratchet",
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
                *HLS_COSIMULATION_EVIDENCE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build offline host-cosimulation execution and exact coverage gates."""
    return [
        (
            "HLS cosimulation-evidence focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={HLS_COSIMULATION_EVIDENCE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *HLS_COSIMULATION_EVIDENCE_COVERAGE_COHORT,
            ],
        ),
        (
            "HLS cosimulation-evidence exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={HLS_COSIMULATION_EVIDENCE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/hls_cosimulation_evidence.py,*/codegen/ultrascale_hls.py",
            ],
        ),
    ]


__all__ = [
    "HLS_COSIMULATION_EVIDENCE_COVERAGE_COHORT",
    "HLS_COSIMULATION_EVIDENCE_COVERAGE_DATA_FILE",
    "HLS_COSIMULATION_EVIDENCE_DOCSTRING_RATCHET",
    "HLS_COSIMULATION_EVIDENCE_SOURCE",
    "HLS_COSIMULATION_EVIDENCE_SOURCES",
    "HLS_COSIMULATION_EVIDENCE_TYPING_RATCHET",
    "ULTRASCALE_HLS_SOURCE",
    "build_coverage_gates",
    "build_static_quality_gates",
]
