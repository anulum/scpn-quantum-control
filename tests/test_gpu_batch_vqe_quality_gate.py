# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — GPU-batch VQE quality-gate tests
"""Lock the GPU-batch VQE quality gate into preflight and CI."""

from pathlib import Path

from tools import gpu_batch_vqe_quality_gates as quality_gates
from tools import preflight


def test_static_gate_is_strict_and_completely_documented() -> None:
    """Require strict typing and complete direct-owner docstrings."""
    gates = dict(quality_gates.build_static_quality_gates("/python"))
    assert (
        gates["mypy-strict-GPU-batch-VQE-quality"][5:]
        == quality_gates.GPU_BATCH_VQE_TYPING_RATCHET
    )
    ruff = gates["ruff D GPU-batch VQE quality ratchet"]
    assert (
        ruff[-len(quality_gates.GPU_BATCH_VQE_DOCSTRING_RATCHET) :]
        == quality_gates.GPU_BATCH_VQE_DOCSTRING_RATCHET
    )
    assert "--preview" in ruff and "D,D413,D417,D420" in ruff


def test_coverage_gate_is_isolated_public_and_exact() -> None:
    """Require bounded local scan execution and exact source coverage."""
    gates = dict(quality_gates.build_coverage_gates("/python"))
    run = gates["GPU-batch VQE focused coverage"]
    report = gates["GPU-batch VQE exact coverage threshold"]
    assert "--branch" in run
    assert (
        run[-len(quality_gates.GPU_BATCH_VQE_COVERAGE_COHORT) :]
        == quality_gates.GPU_BATCH_VQE_COVERAGE_COHORT
    )
    assert "--fail-under=100" in report
    assert "--include=*/phase/gpu_batch_vqe.py" in report


def test_preflight_uses_helper_defined_gates() -> None:
    """Keep helper commands verbatim in preflight."""
    assert dict(preflight.GPU_BATCH_VQE_COVERAGE_GATES) == dict(
        quality_gates.build_coverage_gates(preflight._PY)
    )
    for name, command in quality_gates.build_static_quality_gates(preflight._PY):
        assert dict(preflight.STATIC_GATES)[name] == command
    assert "gates.extend(GPU_BATCH_VQE_COVERAGE_GATES)" in Path("tools/preflight.py").read_text(
        encoding="utf-8"
    )


def test_ci_runs_and_aggregates_gpu_batch_vqe_gate() -> None:
    """Keep the focused CI job and aggregate dependency required."""
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    start = workflow.index("  gpu-batch-vqe-quality:")
    end = workflow.index("\n\n  avqds-quality:", start)
    block = workflow[start:end]
    for path in quality_gates.GPU_BATCH_VQE_DOCSTRING_RATCHET:
        assert path in block
    assert "--fail-under=100" in block
    assert quality_gates.GPU_BATCH_VQE_SOURCE in block
    assert "gpu-batch-vqe-quality" in workflow[workflow.index("  ci-gate:") :]
