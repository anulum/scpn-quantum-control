# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Whole retained-population comparison quality gates
"""Build the coherent native comparison gates without weakening existing owners."""

from __future__ import annotations

from os import devnull
from pathlib import Path
from time import time_ns

from tools.studio_executive_product_quality_gates import Gate


def build_method_comparison_quality_gates(python: str, *, output_root: Path) -> list[Gate]:
    """Bind original native producers, dedicated tests and unchanged exact thresholds.

    Parameters
    ----------
    python : str
        Existing locked interpreter; no environment installation occurs here.
    output_root : Path
        Explicit caller-owned task allocation or standalone runner temp directory.
        Each invocation names fresh raw data and never overwrites an older run.

    Returns
    -------
    list
        Strict types/native NumPy documentation, complete dedicated native
        regressions and 100 percent statement/branch gates on every native owner.
        Actual built/source browser/WASM acceptance belongs to the shared runner.

    """
    production = [
        "src/scpn_quantum_control/benchmarks/reproducible_comparison.py",
        "src/scpn_quantum_control/benchmarks/method_comparison_contracts.py",
        "src/scpn_quantum_control/benchmarks/method_comparison.py",
        "src/scpn_quantum_control/studio/method_comparison_artifact.py",
        "tools/export_method_comparison.py",
        "tools/studio_method_comparison_quality_gates.py",
        "examples/method_comparison.py",
    ]
    tests = [
        "tests/test_reproducible_comparison.py",
        "tests/test_method_comparison_contracts.py",
        "tests/test_method_comparison.py",
        "tests/test_studio_method_comparison_artifact.py",
        "tools/tests/test_export_method_comparison.py",
        "tests/test_studio_method_comparison_quality_gates.py",
    ]
    owners = [
        *production,
        *tests,
        "tools/studio_method_comparison_browser.py",
        "tools/tests/test_studio_method_comparison_browser.py",
        "tools/studio_browser_journey.py",
        "tests/test_studio_browser_journey.py",
        "tools/tests/test_studio_browser_runtime_refusal.py",
        "tools/studio_native_coverage_gate.py",
        "tools/tests/test_studio_native_coverage_gate.py",
    ]
    data = str(output_root / f"studio-method-comparison-{time_ns()}.coverage")
    include = ",".join(
        "*/" + path.removeprefix("src/scpn_quantum_control/") for path in production
    )
    return [
        (
            "studio-method-comparison-strict",
            [python, "-m", "mypy", "--strict", "--explicit-package-bases", *owners],
        ),
        (
            "studio-method-comparison-native-docs",
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
            "studio-method-comparison-native-coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--branch",
                f"--include={include}",
                "-m",
                "pytest",
                "-q",
                "--basetemp",
                str(output_root / f"studio-method-comparison-pytest-{time_ns()}"),
                *tests,
            ],
        ),
        (
            "studio-method-comparison-native-exact",
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
                f"--include={include}",
            ],
        ),
    ]
