# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA protected-memory quality-gate specification
"""Build strict Python and Rust DLA protected-memory quality gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
DLA_PROTECTED_MEMORY_SOURCES = [
    "src/scpn_quantum_control/qec/dla_protected_scar.py",
    "src/scpn_quantum_control/qec/dla_protected_subspace.py",
]
"""Cohesive Python implementation owner."""
DLA_PROTECTED_MEMORY_TESTS = [
    "tests/test_dla_protected_scar.py",
    "tests/test_dla_protected_scar_branches.py",
    "tests/test_dla_protected_subspace.py",
    "tests/test_dla_protected_subspace_branches.py",
]
"""Direct and fail-closed public execution cohort."""
DLA_PROTECTED_MEMORY_TYPING_RATCHET = [
    *DLA_PROTECTED_MEMORY_SOURCES,
    *DLA_PROTECTED_MEMORY_TESTS,
    "tools/dla_protected_memory_quality_gates.py",
    "tests/test_dla_protected_memory_quality_gate.py",
    "tools/preflight.py",
    "tests/test_preflight_tool.py",
]
"""Touched Python surfaces held to strict MyPy."""
DLA_PROTECTED_MEMORY_DOCSTRING_RATCHET = [
    *DLA_PROTECTED_MEMORY_SOURCES,
    *DLA_PROTECTED_MEMORY_TESTS,
    "tools/dla_protected_memory_quality_gates.py",
    "tests/test_dla_protected_memory_quality_gate.py",
]
"""Direct owner surfaces held to complete NumPy docstrings."""
DLA_PROTECTED_MEMORY_COVERAGE_DATA_FILE = "/tmp/scpn-qc-dla-protected-memory-quality.coverage"  # nosec B108
"""Isolated coverage database for the protected-memory owner."""
DLA_PROTECTED_MEMORY_COVERAGE_INCLUDE = (
    "*/qec/dla_protected_scar.py,*/qec/dla_protected_subspace.py"
)
"""Exact production sources for the branch report."""
DLA_PROTECTED_MEMORY_RUST_SOURCE = "scpn_quantum_engine/src/dla.rs"
DLA_PROTECTED_MEMORY_RUST_FFI_TEST = "tests/test_rust_ffi_validation.py"
DLA_PROTECTED_MEMORY_POLYGLOT_EVIDENCE = [
    DLA_PROTECTED_MEMORY_RUST_SOURCE,
    DLA_PROTECTED_MEMORY_RUST_FFI_TEST,
]
"""Native kernel and Python FFI validation evidence."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-dla-protected-memory-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DLA_PROTECTED_MEMORY_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D DLA-protected-memory quality ratchet",
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
                *DLA_PROTECTED_MEMORY_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real connected execution and exact branch-coverage gates."""
    return [
        (
            "DLA-protected-memory focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DLA_PROTECTED_MEMORY_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DLA_PROTECTED_MEMORY_TESTS,
            ],
        ),
        (
            "DLA-protected-memory exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DLA_PROTECTED_MEMORY_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                f"--include={DLA_PROTECTED_MEMORY_COVERAGE_INCLUDE}",
            ],
        ),
    ]


def build_polyglot_gates(cargo: str, python: str) -> list[Gate]:
    """Build filtered native-kernel and installed-extension parity gates."""
    return [
        (
            "Rust DLA-protected-memory parity tests",
            [
                cargo,
                "test",
                "--manifest-path",
                "scpn_quantum_engine/Cargo.toml",
                "--lib",
                "test_memory_",
            ],
        ),
        (
            "DLA-protected-memory Rust FFI validation",
            [
                python,
                "-m",
                "pytest",
                "-q",
                DLA_PROTECTED_MEMORY_RUST_FFI_TEST,
                "-k",
                "dla_protected_memory_metrics",
            ],
        ),
    ]


__all__ = [
    "DLA_PROTECTED_MEMORY_COVERAGE_DATA_FILE",
    "DLA_PROTECTED_MEMORY_COVERAGE_INCLUDE",
    "DLA_PROTECTED_MEMORY_DOCSTRING_RATCHET",
    "DLA_PROTECTED_MEMORY_POLYGLOT_EVIDENCE",
    "DLA_PROTECTED_MEMORY_RUST_FFI_TEST",
    "DLA_PROTECTED_MEMORY_RUST_SOURCE",
    "DLA_PROTECTED_MEMORY_SOURCES",
    "DLA_PROTECTED_MEMORY_TESTS",
    "DLA_PROTECTED_MEMORY_TYPING_RATCHET",
    "build_coverage_gates",
    "build_polyglot_gates",
    "build_static_quality_gates",
]
