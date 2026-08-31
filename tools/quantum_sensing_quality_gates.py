# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum-sensing quality-gate specification
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
QUANTUM_SENSING_SOURCE = "src/scpn_quantum_control/analysis/sensing.py"
"""No-submit S11 quantum-sensing readiness surface."""
NV_MAGNETOMETRY_SOURCE = "src/scpn_quantum_control/sensing/nv_magnetometry_20T.py"
"""Simulation-only NV-centre magnetometry model through 20 tesla."""
NV_MAGNETOMETRY_TEST = "tests/test_nv_magnetometry.py"
"""Primary public, calibration, and native-parity NV suite."""
NV_MAGNETOMETRY_BRANCH_TEST = "tests/test_nv_magnetometry_high_field.py"
"""Focused validation and Python-fallback branch suite."""
NV_MAGNETOMETRY_RUSTDOC_SOURCE = "scpn_quantum_engine/src/sensing.rs"
"""Documented native Lorentzian kernel owned by global Rust CI."""
QUANTUM_SENSING_EXPORTER = "scripts/export_s11_quantum_sensing_readiness.py"
"""Executable JSON and Markdown readiness exporter."""
QUANTUM_SENSING_TYPING_RATCHET = [
    QUANTUM_SENSING_SOURCE,
    NV_MAGNETOMETRY_SOURCE,
    NV_MAGNETOMETRY_TEST,
    NV_MAGNETOMETRY_BRANCH_TEST,
    "tests/test_quantum_sensing_readiness.py",
    "tests/test_sensing_branches.py",
    "tests/test_sensing_readiness_contracts.py",
    QUANTUM_SENSING_EXPORTER,
    "tests/test_export_s11_quantum_sensing_readiness.py",
    "tools/quantum_sensing_quality_gates.py",
    "tests/test_quantum_sensing_quality_gate.py",
    "tools/preflight.py",
    "tests/test_preflight_tool.py",
]
"""Production, export, tests, and gate surfaces held to strict MyPy."""
QUANTUM_SENSING_DOCSTRING_RATCHET = [
    QUANTUM_SENSING_SOURCE,
    NV_MAGNETOMETRY_SOURCE,
    NV_MAGNETOMETRY_TEST,
    NV_MAGNETOMETRY_BRANCH_TEST,
    "tests/test_quantum_sensing_readiness.py",
    "tests/test_sensing_branches.py",
    "tests/test_sensing_readiness_contracts.py",
    QUANTUM_SENSING_EXPORTER,
    "tests/test_export_s11_quantum_sensing_readiness.py",
    "tools/quantum_sensing_quality_gates.py",
    "tests/test_quantum_sensing_quality_gate.py",
    "tests/test_preflight_tool.py",
]
"""Whole owner cohort held to complete NumPy docstrings."""
QUANTUM_SENSING_COVERAGE_COHORT = [
    NV_MAGNETOMETRY_TEST,
    NV_MAGNETOMETRY_BRANCH_TEST,
    "tests/test_quantum_sensing_readiness.py",
    "tests/test_sensing_branches.py",
    "tests/test_sensing_readiness_contracts.py",
    "tests/test_export_s11_quantum_sensing_readiness.py",
]
"""Public, contract, and executable-export tests that own source coverage."""
QUANTUM_SENSING_COVERAGE_DATA_FILE = "/tmp/scpn-qc-quantum-sensing-quality.coverage"  # nosec B108
"""Isolated coverage database for the quantum-sensing owner."""
QUANTUM_SENSING_COVERAGE_INCLUDE = "*/analysis/sensing.py,*/sensing/nv_magnetometry_20T.py"
"""Exact production source include for the coverage report."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-quantum-sensing-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *QUANTUM_SENSING_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D quantum-sensing quality ratchet",
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
                *QUANTUM_SENSING_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real owner execution and exact source-coverage gates."""
    return [
        (
            "quantum-sensing focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_SENSING_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *QUANTUM_SENSING_COVERAGE_COHORT,
            ],
        ),
        (
            "quantum-sensing exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={QUANTUM_SENSING_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={QUANTUM_SENSING_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "NV_MAGNETOMETRY_BRANCH_TEST",
    "NV_MAGNETOMETRY_RUSTDOC_SOURCE",
    "NV_MAGNETOMETRY_SOURCE",
    "NV_MAGNETOMETRY_TEST",
    "QUANTUM_SENSING_COVERAGE_COHORT",
    "QUANTUM_SENSING_COVERAGE_DATA_FILE",
    "QUANTUM_SENSING_COVERAGE_INCLUDE",
    "QUANTUM_SENSING_DOCSTRING_RATCHET",
    "QUANTUM_SENSING_EXPORTER",
    "QUANTUM_SENSING_SOURCE",
    "QUANTUM_SENSING_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
