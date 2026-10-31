# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Studio readout-mitigation bundle quality gates
"""Build strict documentation, typing, execution, and coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
STUDIO_READOUT_MITIGATION_BUNDLE_SOURCE = (
    "src/scpn_quantum_control/studio/readout_mitigation_bundle.py"
)
STUDIO_MITIGATION_INSTRUMENT_SOURCE = (
    "src/scpn_quantum_control/studio/mitigation_instrument_artifact.py"
)
STUDIO_COMMITTED_DATA_SOURCE = "src/scpn_quantum_control/studio/committed_data.py"
STUDIO_PULSE_INSTRUMENT_SOURCE = "src/scpn_quantum_control/studio/pulse_instrument_artifact.py"
STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_COHORT = [
    "tests/test_studio_readout_mitigation_bundle.py",
    "tests/test_studio_mitigation_instrument_artifact.py",
    "tests/test_studio_committed_data.py",
    "tests/test_studio_pulse_instrument_artifact.py",
]
STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_INCLUDE = (
    "--include=*/studio/readout_mitigation_bundle.py,"
    "*/studio/mitigation_instrument_artifact.py,*/studio/committed_data.py,"
    "*/studio/pulse_instrument_artifact.py"
)
STUDIO_READOUT_MITIGATION_BUNDLE_TYPING_RATCHET = [
    STUDIO_READOUT_MITIGATION_BUNDLE_SOURCE,
    STUDIO_MITIGATION_INSTRUMENT_SOURCE,
    STUDIO_COMMITTED_DATA_SOURCE,
    STUDIO_PULSE_INSTRUMENT_SOURCE,
    *STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_COHORT,
    "tools/studio_readout_mitigation_bundle_quality_gates.py",
    "tests/test_studio_readout_mitigation_bundle_quality_gate.py",
]
STUDIO_READOUT_MITIGATION_BUNDLE_DOCSTRING_RATCHET = [
    *STUDIO_READOUT_MITIGATION_BUNDLE_TYPING_RATCHET,
]
STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_DATA_FILE = (
    "/tmp/scpn-qc-studio-readout-mitigation-bundle-quality.coverage"  # nosec B108
)


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing and complete NumPy-docstring gates."""
    return [
        (
            "mypy-strict-studio-readout-mitigation-bundle-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *STUDIO_READOUT_MITIGATION_BUNDLE_TYPING_RATCHET,
            ],
        ),
        (
            "ruff D Studio readout-mitigation bundle quality ratchet",
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
                *STUDIO_READOUT_MITIGATION_BUNDLE_DOCSTRING_RATCHET,
            ],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build public entrypoint execution and exact source-coverage gates."""
    return [
        (
            "Studio readout-mitigation bundle focused coverage",
            [
                python,
                "-m",
                "coverage",
                "run",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_DATA_FILE}",
                "--branch",
                "-m",
                "pytest",
                "-q",
                *STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_COHORT,
            ],
        ),
        (
            "Studio readout-mitigation bundle exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_DATA_FILE}",
                "--precision=2",
                "--fail-under=100",
                STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_INCLUDE,
            ],
        ),
    ]


__all__ = [
    "STUDIO_COMMITTED_DATA_SOURCE",
    "STUDIO_MITIGATION_INSTRUMENT_SOURCE",
    "STUDIO_PULSE_INSTRUMENT_SOURCE",
    "STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_COHORT",
    "STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_DATA_FILE",
    "STUDIO_READOUT_MITIGATION_BUNDLE_COVERAGE_INCLUDE",
    "STUDIO_READOUT_MITIGATION_BUNDLE_DOCSTRING_RATCHET",
    "STUDIO_READOUT_MITIGATION_BUNDLE_SOURCE",
    "STUDIO_READOUT_MITIGATION_BUNDLE_TYPING_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
