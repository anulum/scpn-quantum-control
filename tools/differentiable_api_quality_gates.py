# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable API quality-gate specification
"""Build exact quality gates for the unified differentiable API."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from os import devnull

Gate = tuple[str, list[str]]

DIFFERENTIABLE_API_QUALITY_RATCHET = [
    "src/scpn_quantum_control/differentiable_api.py",
    "src/scpn_quantum_control/differentiable_api_core.py",
    "src/scpn_quantum_control/differentiable_api_core_algorithms.py",
    "src/scpn_quantum_control/differentiable_api_core_bindings.py",
    "src/scpn_quantum_control/differentiable_canonical_api.py",
    "src/scpn_quantum_control/differentiable_result_contracts.py",
    "src/scpn_quantum_control/differentiable_tensor_admission.py",
    "src/scpn_quantum_control/differentiable_tensor_construction.py",
    "src/scpn_quantum_control/differentiable_tensor_materialization.py",
    "src/scpn_quantum_control/differentiable_axis_compilation.py",
    "src/scpn_quantum_control/differentiable_backend_capabilities.py",
    "tests/test_differentiable_api.py",
    "tests/test_differentiable_api_core.py",
    "tests/test_differentiable_api_core_algorithms.py",
    "tests/test_differentiable_api_core_bindings.py",
    "tests/test_differentiable_api_contracts.py",
    "tests/test_differentiable_canonical_modes.py",
    "tests/test_differentiable_package_exports.py",
    "tests/test_differentiable_result_contracts.py",
    "tests/test_differentiable_result_provenance.py",
    "tests/test_differentiable_api_quality_gate.py",
    "tests/test_differentiable_tensor_admission.py",
    "tests/test_differentiable_tensor_construction.py",
    "tests/test_differentiable_tensor_construction_native.py",
    "tests/test_differentiable_tensor_materialization.py",
    "tests/test_differentiable_tensor_materialization_native.py",
    "tests/test_differentiable_axis_compilation.py",
    "tests/test_differentiable_backend_capabilities.py",
    "tests/test_differentiable_backend_capabilities_native.py",
    "tools/differentiable_api_quality_gates.py",
    "tools/differentiable_quality_gates.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

DIFFERENTIABLE_API_COVERAGE_COHORT = [
    "tests/test_differentiable_api.py",
    "tests/test_differentiable_api_core.py",
    "tests/test_differentiable_api_core_algorithms.py",
    "tests/test_differentiable_api_core_bindings.py",
    "tests/test_differentiable_api_contracts.py",
    "tests/test_differentiable_canonical_modes.py",
    "tests/test_differentiable_package_exports.py",
    "tests/test_differentiable_result_contracts.py",
    "tests/test_differentiable_result_provenance.py",
    "tests/test_differentiable_api_quality_gate.py",
    "tests/test_differentiable_tensor_admission.py",
    "tests/test_transform_algebra.py",
]
"""Tests that own exact unified-API statement and branch coverage."""

DIFFERENTIABLE_API_COVERAGE_SELECTOR = (
    "not unified_differentiable_benchmark_report_is_non_performance_evidence"
)
"""Exclude the environment-sensitive benchmark while stubbing its route."""

DIFFERENTIABLE_API_COVERAGE_DATA_FILE = ".coverage.differentiable-api-quality"
"""Isolated coverage database for the unified API owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static quality gates for the API owner cohort.

    """
    return [
        (
            "mypy-strict-differentiable-api-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *DIFFERENTIABLE_API_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D differentiable-api quality ratchet",
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
                *DIFFERENTIABLE_API_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build isolated exact statement and branch coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Focused execution followed by the exact owner-only report.

    """
    return [
        (
            "differentiable-api focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_API_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *DIFFERENTIABLE_API_COVERAGE_COHORT,
                "-k",
                DIFFERENTIABLE_API_COVERAGE_SELECTOR,
            ],
        ),
        (
            "differentiable-api exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={DIFFERENTIABLE_API_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                "--include=*/differentiable_api.py,*/differentiable_api_core.py,*/differentiable_api_core_algorithms.py,*/differentiable_api_core_bindings.py,*/differentiable_canonical_api.py,*/differentiable_result_contracts.py,*/differentiable_tensor_admission.py",
            ],
        ),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one canonical static gate from the hosted workflow.

    Parameters
    ----------
    argv
        Static check selector, or the actual process arguments when omitted.

    Returns
    -------
    int
        Zero after the selected command succeeds. Command failures propagate.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=("typing", "docstrings"))
    args = parser.parse_args(argv)
    index = 0 if args.check == "typing" else 1
    _name, command = build_static_quality_gates(sys.executable)[index]
    subprocess.run(command, check=True)
    return 0


def build_native_tensor_coverage_gates(python: str) -> list[Gate]:
    """Build real SDK execution and exact native-construction coverage gates.

    Parameters
    ----------
    python
        Interpreter with the admitted CPU PyTorch, JAX and TensorFlow dependencies.

    Returns
    -------
    list[Gate]
        Native tests and their exact statement and branch coverage report.

    """
    data = ".coverage.differentiable-tensor-construction"
    return [
        (
            "native tensor construction focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                "--no-cov",
                "tests/test_differentiable_tensor_construction.py",
                "tests/test_differentiable_tensor_construction_native.py",
                "tests/test_differentiable_tensor_materialization.py",
                "tests/test_differentiable_tensor_materialization_native.py",
                "tests/test_differentiable_backend_capabilities.py",
                "tests/test_differentiable_backend_capabilities_native.py",
                "tests/test_differentiable_axis_compilation.py",
            ],
        ),
        (
            "native tensor construction exact coverage",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                "--include=*/differentiable_tensor_construction.py,*/differentiable_tensor_materialization.py,*/differentiable_backend_capabilities.py,*/differentiable_axis_compilation.py",
            ],
        ),
    ]


__all__ = [
    "DIFFERENTIABLE_API_COVERAGE_COHORT",
    "DIFFERENTIABLE_API_COVERAGE_DATA_FILE",
    "DIFFERENTIABLE_API_COVERAGE_SELECTOR",
    "DIFFERENTIABLE_API_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_native_tensor_coverage_gates",
    "build_static_quality_gates",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
