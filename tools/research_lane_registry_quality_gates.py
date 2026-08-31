# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — research-lane registry quality gates
"""Build strict documentation, evidence-drift, and exact coverage gates."""

from __future__ import annotations

from os import devnull

Gate = tuple[str, list[str]]
RESEARCH_LANE_REGISTRY_QUALITY_RATCHET = [
    "src/scpn_quantum_control/analysis/research_lane_registry.py",
    "src/scpn_quantum_control/analysis/rl_discovery_agent.py",
    "src/scpn_quantum_control/analysis/rl_pulse_optimizer.py",
    "src/scpn_quantum_control/analysis/rl_research_governance.py",
    "src/scpn_quantum_control/analysis/witness_discovery.py",
    "tests/test_research_lane_registry.py",
    "tests/test_rl_research_governance.py",
    "tests/test_rl_discovery_agent_branches.py",
    "tests/test_witness_discovery.py",
    "tests/test_witness_discovery_engine_fallback.py",
    "scripts/run_research_lane_registry.py",
    "scripts/run_rl_research_governance_evidence.py",
    "tools/research_lane_registry_quality_gates.py",
    "tests/test_research_lane_registry_quality_gate.py",
]
"""Ordered strict-typing and NumPy-docstring cohort."""
RESEARCH_LANE_REGISTRY_COVERAGE_COHORT = [
    "tests/test_research_lane_registry.py",
    "tests/test_rl_research_governance.py",
    "tests/test_rl_discovery_agent_branches.py",
    "tests/test_witness_discovery.py",
    "tests/test_witness_discovery_engine_fallback.py",
    "tests/test_frontier_interface_guards.py",
]
"""Tests that own exact research-lane registry coverage."""
RESEARCH_LANE_REGISTRY_COVERAGE_DATA_FILE = "/tmp/scpn-qc-research-lane-registry-quality.coverage"  # nosec B108
"""Isolated coverage database for research-lane registry diagnostics."""
RESEARCH_LANE_REGISTRY_COVERAGE_INCLUDE = (
    "*/analysis/research_lane_registry.py,*/analysis/rl_discovery_agent.py,"
    "*/analysis/rl_pulse_optimizer.py,*/analysis/rl_research_governance.py,"
    "*/analysis/witness_discovery.py"
)
"""Connected registry and governed RL sources owned by exact coverage."""


def build_static_quality_gates(python: str) -> list[Gate]:
    """Build strict typing, NumPy-docstring, and evidence-drift gates."""
    return [
        (
            "mypy-strict-research-lane-registry-quality",
            [
                python,
                "-m",
                "mypy",
                "--strict",
                "--explicit-package-bases",
                *RESEARCH_LANE_REGISTRY_QUALITY_RATCHET,
            ],
        ),
        (
            "ruff D research-lane-registry quality ratchet",
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
                *RESEARCH_LANE_REGISTRY_QUALITY_RATCHET,
            ],
        ),
        (
            "research-lane-registry evidence drift",
            [python, "scripts/run_research_lane_registry.py", "--check"],
        ),
        (
            "RL research-governance evidence drift",
            [python, "scripts/run_rl_research_governance_evidence.py", "--check"],
        ),
    ]


def build_coverage_gates(python: str) -> list[Gate]:
    """Build focused execution and exact source-only coverage gates."""
    data = RESEARCH_LANE_REGISTRY_COVERAGE_DATA_FILE
    return [
        (
            "research-lane-registry focused coverage",
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
                *RESEARCH_LANE_REGISTRY_COVERAGE_COHORT,
            ],
        ),
        (
            "research-lane-registry exact coverage threshold",
            [
                python,
                "-m",
                "coverage",
                "report",
                f"--rcfile={devnull}",
                f"--data-file={data}",
                "--precision=2",
                "--fail-under=100",
                f"--include={RESEARCH_LANE_REGISTRY_COVERAGE_INCLUDE}",
            ],
        ),
    ]


__all__ = [
    "RESEARCH_LANE_REGISTRY_COVERAGE_COHORT",
    "RESEARCH_LANE_REGISTRY_COVERAGE_DATA_FILE",
    "RESEARCH_LANE_REGISTRY_COVERAGE_INCLUDE",
    "RESEARCH_LANE_REGISTRY_QUALITY_RATCHET",
    "build_coverage_gates",
    "build_static_quality_gates",
]
