# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio-executive quality-gate specification
"""Build strict documentation, typing, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET = [
    "src/scpn_quantum_control/studio_executive_product.py",
    "src/scpn_quantum_control/studio/manifest.py",
    "src/scpn_quantum_control/studio/federation.py",
    "src/scpn_quantum_control/studio/verbs.py",
    "src/scpn_quantum_control/studio/executive_cli.py",
    "tests/test_studio_executive_product.py",
    "tests/test_studio_manifest.py",
    "tests/test_studio_executive_cli.py",
    "tools/studio_executive_product_quality_gates.py",
    "tests/test_studio_executive_product_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT = [
    "tests/test_studio_executive_product.py",
    "tests/test_studio_manifest.py",
    "tests/test_studio_executive_cli.py",
]
"""Tests that own exact Studio-executive product coverage."""
STUDIO_EXECUTIVE_PRODUCT_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-studio-executive-product-quality.coverage"  # nosec B108
)
"""Isolated coverage database for the Studio-executive product owner."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and NumPy-docstring gates."""
    return [
        (
            "mypy-strict-studio-executive-product-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D studio-executive-product quality ratchet",
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
                *STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    return [
        (
            "studio-executive-product focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_EXECUTIVE_PRODUCT_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT,
            ],
        ),
        (
            "studio-executive-product exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_EXECUTIVE_PRODUCT_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/studio_executive_product.py,*/studio/manifest.py,*/studio/federation.py,*/studio/verbs.py,*/studio/executive_cli.py",
            ],
        ),
    ]


def build_program_authoring_quality_gates(python: str) -> list[Gate]:
    """Build the additive native program-source owner gates for Studio CI.

    Parameters
    ----------
    python
        Existing locked interpreter selected by the caller.

    Returns
    -------
    list
        Strict typing, native documentation and exact branch-coverage commands.

    """
    production = [
        "src/scpn_quantum_control/studio/program_authoring.py",
        "src/scpn_quantum_control/studio/program_authoring_contracts.py",
        "src/scpn_quantum_control/studio/executive_compile.py",
        "src/scpn_quantum_control/studio/compiler_trace.py",
    ]
    tests = [
        "tests/test_studio_program_authoring.py",
        "tests/test_studio_executive_compile.py",
        "tests/test_studio_compiler_trace.py",
    ]
    owners = [
        *production,
        *tests,
        "tools/studio_program_authoring_browser.py",
        "tools/tests/test_studio_program_authoring_browser.py",
        "tools/studio_compiler_trace_browser.py",
        "tools/tests/test_studio_compiler_trace_browser.py",
        "tools/studio_executive_product_quality_gates.py",
    ]
    data_file = "/tmp/scpn-qc-studio-program-authoring.coverage"  # nosec B108
    return [
        (
            "studio-program-authoring-strict",
            [python, "-m", "mypy", "--strict", "--explicit-package-bases", *owners],
        ),
        (
            "studio-program-authoring-native-docs",
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
                *owners,
            ],
        ),
        (
            "studio-program-authoring-native-coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data_file}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *tests,
            ],
        ),
        (
            "studio-program-authoring-native-exact",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data_file}",
                "--precision=2",
                "--show-missing",
                "--fail-under=100",
                "--include=*/studio/program_authoring.py,*/studio/program_authoring_contracts.py,*/studio/executive_compile.py,*/studio/compiler_trace.py",
            ],
        ),
    ]


__all__ = [
    "STUDIO_EXECUTIVE_PRODUCT_COVERAGE_COHORT",
    "STUDIO_EXECUTIVE_PRODUCT_COVERAGE_DATA_FILE",
    "STUDIO_EXECUTIVE_PRODUCT_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
    "build_program_authoring_quality_gates",
]
