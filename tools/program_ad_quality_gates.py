# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — focused Studio Program-AD quality-gate specification
"""Build the local polyglot quality gates for the Studio Program-AD owner."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]

STUDIO_PROGRAM_AD_QUALITY_RATCHET = [
    "src/scpn_quantum_control/studio/program_ad_replay_artifact.py",
    "tests/test_studio_program_ad_replay_artifact.py",
    "tools/program_ad_quality_gates.py",
    "tests/test_studio_program_ad_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""

STUDIO_PROGRAM_AD_COVERAGE_COHORT = [
    "tests/test_studio_program_ad_replay_artifact.py",
]
"""Python tests that own exact artifact-emitter coverage."""

STUDIO_PROGRAM_AD_BROWSER_TESTS = [
    "src/panel/programAd.test.ts",
    "src/panel/ProgramADReplayCard.test.tsx",
]
"""Studio-web tests that execute the real Program-AD WASM kernel."""

STUDIO_PROGRAM_AD_COVERAGE_DATA_FILE = ".coverage.studio-program-ad"
"""Isolated coverage database for the Python artifact owner."""

STUDIO_PROGRAM_AD_REQUIRE_NATIVE_ENV = "SCPN_PROGRAM_AD_REQUIRE_NATIVE"
"""Environment switch that makes the dedicated native CI owner fail hard."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict Python typing and NumPy-docstring gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered static quality gates for the Python owner and gate policy.

    """
    return [
        (
            "mypy-strict-studio-program-ad",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STUDIO_PROGRAM_AD_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D studio-program-ad quality ratchet",
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
                *STUDIO_PROGRAM_AD_QUALITY_RATCHET,
            ],
        ),
    ]


def build_python_coverage_gates(python: str) -> list[Gate]:
    """Build isolated exact Python statement and branch coverage gates.

    Parameters
    ----------
    python
        Absolute Python interpreter path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Coverage execution followed by its exact owner-only report.

    """
    return [
        (
            "studio Program-AD focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_PROGRAM_AD_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STUDIO_PROGRAM_AD_COVERAGE_COHORT,
            ],
        ),
        (
            "studio Program-AD exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_PROGRAM_AD_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                "--include=*/program_ad_replay_artifact.py",
            ],
        ),
    ]


def build_runtime_gates(cargo: str, pnpm: str) -> list[Gate]:
    """Build native Rust, release-WASM, and strict TypeScript gates.

    Parameters
    ----------
    cargo
        Absolute Cargo executable path admitted by the preflight runner.
    pnpm
        Absolute pnpm executable path admitted by the preflight runner.

    Returns
    -------
    list[Gate]
        Ordered commands that prepare the real browser replay runtime.

    """
    manifest = "scpn_quantum_engine/studio_program_ad_wasm/Cargo.toml"
    return [
        (
            "studio Program-AD Rust kernel tests",
            [cargo, "test", "--locked", "--manifest-path", manifest],
        ),
        (
            "studio Program-AD WASM release build",
            [
                cargo,
                "build",
                "--release",
                "--locked",
                "--target",
                "wasm32-unknown-unknown",
                "--manifest-path",
                manifest,
            ],
        ),
        (
            "studio Program-AD browser strict typecheck",
            [pnpm, "--dir", "studio-web", "typecheck"],
        ),
    ]


def build_browser_test_gate(pnpm: str) -> Gate:
    """Build the focused browser test command used without coverage.

    Parameters
    ----------
    pnpm
        Absolute pnpm executable path admitted by the preflight runner.

    Returns
    -------
    Gate
        Real-WASM Vitest gate for the verifier and React card.

    """
    return (
        "studio Program-AD focused browser tests",
        [pnpm, "--dir", "studio-web", "exec", "vitest", "run", *STUDIO_PROGRAM_AD_BROWSER_TESTS],
    )


def build_browser_coverage_gate(pnpm: str) -> Gate:
    """Build exact browser owner coverage over the real WASM tests.

    Parameters
    ----------
    pnpm
        Absolute pnpm executable path admitted by the preflight runner.

    Returns
    -------
    Gate
        Vitest command with 100% statement, branch, function, and line gates.

    """
    return (
        "studio Program-AD exact browser coverage",
        [
            pnpm,
            "--dir",
            "studio-web",
            "exec",
            "vitest",
            "run",
            *STUDIO_PROGRAM_AD_BROWSER_TESTS,
            "--coverage",
            "--coverage.reporter=text",
            "--coverage.include=src/panel/programAd.ts",
            "--coverage.include=src/panel/ProgramADReplayCard.tsx",
            "--coverage.thresholds.statements=100",
            "--coverage.thresholds.branches=100",
            "--coverage.thresholds.functions=100",
            "--coverage.thresholds.lines=100",
        ],
    )


__all__ = [
    "STUDIO_PROGRAM_AD_BROWSER_TESTS",
    "STUDIO_PROGRAM_AD_COVERAGE_COHORT",
    "STUDIO_PROGRAM_AD_COVERAGE_DATA_FILE",
    "STUDIO_PROGRAM_AD_QUALITY_RATCHET",
    "STUDIO_PROGRAM_AD_REQUIRE_NATIVE_ENV",
    "build_browser_coverage_gate",
    "build_browser_test_gate",
    "build_python_coverage_gates",
    "build_runtime_gates",
    "build_static_quality_gates",
]
