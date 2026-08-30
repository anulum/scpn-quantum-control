# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — psi-field lattice quality gates
"""Build strict Python and Rust lattice quality gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
PSI_LATTICE_SOURCE = "src/scpn_quantum_control/psi_field/lattice.py"
PSI_LATTICE_OWNER_TEST = "tests/test_psi_field.py"
PSI_LATTICE_IMPORT_TEST = "tests/test_lattice_branches.py"
PSI_LATTICE_RUST_SOURCE = "scpn_quantum_engine/src/gauge_lattice.rs"
PSI_LATTICE_RUST_FFI_TEST = "tests/test_rust_ffi_validation.py"
PSI_LATTICE_STUB = "src/scpn_quantum_engine.pyi"
PSI_LATTICE_POLYGLOT_EVIDENCE = [
    PSI_LATTICE_RUST_SOURCE,
    PSI_LATTICE_RUST_FFI_TEST,
    PSI_LATTICE_STUB,
]
PSI_LATTICE_TYPING_RATCHET = [
    PSI_LATTICE_SOURCE,
    PSI_LATTICE_OWNER_TEST,
    PSI_LATTICE_IMPORT_TEST,
    "tools/psi_lattice_quality_gates.py",
    "tests/test_psi_lattice_quality_gate.py",
]
PSI_LATTICE_DOCSTRING_RATCHET = [*PSI_LATTICE_TYPING_RATCHET]
PSI_LATTICE_COVERAGE_COHORT = [PSI_LATTICE_OWNER_TEST, PSI_LATTICE_IMPORT_TEST]
PSI_LATTICE_COVERAGE_DATA_FILE = "/tmp/scpn-qc-psi-lattice-quality.coverage"  # nosec B108


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-psi-lattice-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *PSI_LATTICE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D psi-lattice quality ratchet",
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
                *PSI_LATTICE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build real lattice execution and exact source-coverage gates."""
    return [
        (
            "psi-lattice focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={PSI_LATTICE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *PSI_LATTICE_COVERAGE_COHORT,
            ],
        ),
        (
            "psi-lattice exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={PSI_LATTICE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/psi_field/lattice.py",
            ],
        ),
    ]


def build_polyglot_gates(cargo: str) -> list[Gate]:
    """Build the filtered Rust hot-path parity gate."""
    return [
        (
            "Rust psi-lattice parity tests",
            [
                cargo,
                "test",
                "--manifest-path",
                "scpn_quantum_engine/Cargo.toml",
                "--lib",
                "gauge_lattice",
            ],
        )
    ]


__all__ = [
    "PSI_LATTICE_COVERAGE_COHORT",
    "PSI_LATTICE_COVERAGE_DATA_FILE",
    "PSI_LATTICE_DOCSTRING_RATCHET",
    "PSI_LATTICE_IMPORT_TEST",
    "PSI_LATTICE_OWNER_TEST",
    "PSI_LATTICE_POLYGLOT_EVIDENCE",
    "PSI_LATTICE_RUST_FFI_TEST",
    "PSI_LATTICE_RUST_SOURCE",
    "PSI_LATTICE_SOURCE",
    "PSI_LATTICE_STUB",
    "PSI_LATTICE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_polyglot_gates",
    "build_static_quality_gates",
]
