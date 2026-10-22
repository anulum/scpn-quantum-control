# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native transform quality gates
"""Build one required-runtime cohort with serial native execution and exact coverage."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

Gate = tuple[str, list[str]]
SOURCES = [
    "src/scpn_quantum_control/differentiable_transform_algebra.py",
    "src/scpn_quantum_control/differentiable_transform_comparison.py",
    "src/scpn_quantum_control/differentiable_batched_transform_execution.py",
    "src/scpn_quantum_control/differentiable_native_curvature.py",
    "src/scpn_quantum_control/differentiable_native_gradient.py",
    "src/scpn_quantum_control/differentiable_native_linearization.py",
    "src/scpn_quantum_control/differentiable_module_hardening_audit.py",
    "src/scpn_quantum_control/differentiable_structured_transform_execution.py",
    "src/scpn_quantum_control/differentiable_transform_support_matrix.py",
    "src/scpn_quantum_control/differentiable_jax_transform_algebra.py",
    "src/scpn_quantum_control/differentiable_torch_transform_algebra.py",
    "src/scpn_quantum_control/differentiable_tensorflow_transform_algebra.py",
]
COVERAGE_OWNERS = [
    *SOURCES,
    "tools/check_differentiable_transform_algebra.py",
    "tools/differentiable_transform_algebra_quality_gates.py",
    "examples/native_transform_algebra.py",
]
LOCAL_TESTS = [
    "tests/test_differentiable_transform_algebra.py",
    "tests/test_differentiable_transform_comparison.py",
    "tests/test_differentiable_transform_support_matrix.py",
    "tests/test_differentiable_transform_algebra_quality_gate.py",
    "tests/test_differentiable_module_hardening_audit.py",
]
RUNTIME_TESTS = {
    "jax": "tests/test_differentiable_jax_transform_algebra.py",
    "pytorch": "tests/test_differentiable_torch_transform_algebra.py",
    "tensorflow": "tests/test_differentiable_tensorflow_transform_algebra.py",
}
SHARED_RUNTIME_TESTS = [
    "tests/test_native_transform_composition.py",
    "tests/test_differentiable_batched_transform_execution.py",
    "tests/test_differentiable_native_curvature.py",
    "tests/test_differentiable_native_gradient.py",
    "tests/test_differentiable_native_linearization.py",
    "tests/test_differentiable_structured_transform_execution.py",
]
STATIC_COHORT = [
    *SOURCES,
    *LOCAL_TESTS,
    *RUNTIME_TESTS.values(),
    "src/scpn_quantum_control/differentiable_api.py",
    "tools/differentiable_quality_gates.py",
    "tests/test_differentiable_api.py",
    *SHARED_RUNTIME_TESTS,
    "tools/check_differentiable_transform_algebra.py",
    "tools/differentiable_transform_algebra_quality_gates.py",
    "examples/native_transform_algebra.py",
]


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict types and complete native NumPy docstring checks.

    Parameters
    ----------
    python:
        Interpreter used by both local and hosted execution.

    Returns
    -------
    list[Gate]
        Ordered type and documentation commands covering each native owner.

    """
    return [
        (
            "mypy-strict-native-transform-algebra",
            [python, "-m", "mypy", "--strict", "--explicit-package-bases", *STATIC_COHORT],
        ),
        (
            "numpy-docs-native-transform-algebra",
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
                *STATIC_COHORT,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build serial real-runtime tests and an exact owner-only coverage threshold.

    Parameters
    ----------
    python:
        Interpreter with all three required CPU frameworks installed.

    Returns
    -------
    list[Gate]
        Each framework runs in a separate process. Import/transform failures
        propagate; a sparse environment cannot pass this owning cohort.

    """
    data_file = str(
        Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "native-transform.coverage"
    )
    prefix = [
        python,
        "-m",
        "coverage",
        "run",
        "--rcfile=" + os.devnull,
        "--data-file=" + data_file,
        "--branch",
        "--include=" + ",".join("*/" + Path(path).name for path in COVERAGE_OWNERS),
    ]
    pytest_args = ["-m", "pytest", "-q", "--no-cov", "-p", "no:cacheprovider"]
    gates: list[Gate] = [
        (
            "native-transform local boundaries",
            [
                *prefix,
                *pytest_args,
                *LOCAL_TESTS,
                *SHARED_RUNTIME_TESTS,
                "-m",
                "not framework_imports",
            ],
        ),
    ]
    for framework, test_path in RUNTIME_TESTS.items():
        gates.append(
            (
                f"native-transform required {framework}",
                [
                    *prefix,
                    "--append",
                    *pytest_args,
                    test_path,
                    *SHARED_RUNTIME_TESTS,
                    "-m",
                    "framework_imports",
                    "-k",
                    framework if framework != "pytorch" else "pytorch or torch",
                ],
            )
        )
        gates.append(
            (
                f"native-transform CLI {framework}",
                [
                    *prefix,
                    "--append",
                    "tools/check_differentiable_transform_algebra.py",
                    "--framework",
                    framework,
                    *(
                        ["--values", "[[0.2,0.5],[-0.4,0.7]]", "--batch-axis", "1"]
                        if framework == "pytorch"
                        else []
                    ),
                ],
            )
        )
    gates.extend(
        [
            (
                "native-transform legacy CLI",
                [*prefix, "--append", "tools/check_differentiable_transform_algebra.py"],
            ),
            (
                "native-transform example jax",
                [*prefix, "--append", "examples/native_transform_algebra.py", "jax"],
            ),
        ]
    )
    gates.append(
        (
            "native-transform exact coverage",
            [
                python,
                "-m",
                "coverage",
                "report",
                "--rcfile=" + os.devnull,
                "--data-file=" + data_file,
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                "--include=" + ",".join("*/" + Path(path).name for path in COVERAGE_OWNERS),
            ],
        )
    )
    return gates


__all__ = [
    "SOURCES",
    "COVERAGE_OWNERS",
    "LOCAL_TESTS",
    "RUNTIME_TESTS",
    "SHARED_RUNTIME_TESTS",
    "STATIC_COHORT",
    "build_static_quality_gates",
    "build_coverage_gates",
]
