# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Catalyst differentiable comparison tests.
"""Tests for dedicated Catalyst compiler-workflow comparison evidence."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import pytest

import scpn_quantum_control.benchmarks.differentiable_external_comparison as comparison
from scpn_quantum_control.benchmarks.differentiable_catalyst_comparison import (
    CATALYST_UNSUPPORTED_PROVIDER_ROUTES,
    CatalystCompilerWorkflowComparison,
    catalyst_compiler_workflow_comparison,
)
from scpn_quantum_control.benchmarks.differentiable_external_comparison import (
    ExternalComparisonRow,
    run_differentiable_external_comparison_suite,
)


def _catalyst_gap_row(
    profile: CatalystCompilerWorkflowComparison | None = None,
    *,
    backend: str = "catalyst",
) -> ExternalComparisonRow:
    """Return a Catalyst-shaped hard-gap row for validation tests."""
    return ExternalComparisonRow(
        case_id="bounded_phase_objective",
        backend=backend,
        status="hard_gap",
        failure_class="dependency_missing",
        value_error=None,
        gradient_error=None,
        runtime_seconds=None,
        memory_peak_bytes=None,
        batching_support="not_evaluated",
        transform_support="Catalyst qjit/MLIR/QIR runner",
        dtype="float64",
        device="cpu",
        source_of_truth="scpn_reference",
        setup_instructions="Install PennyLane Catalyst and configure a runner.",
        claim_boundary="Catalyst dependency hard gap only.",
        dependency_versions={"pennylane-catalyst": "not_installed"},
        catalyst_comparison=profile,
    )


def test_catalyst_workflow_profile_serializes_required_scope() -> None:
    """Catalyst workflow profiles should name every open comparison axis."""
    profile = catalyst_compiler_workflow_comparison(runner_status="dependency_gap")
    payload = profile.to_dict()

    assert payload["runner_status"] == "dependency_gap"
    assert "qjit/MLIR/QIR" in cast(str, payload["compiled_quantum_classical_workflows"])
    assert "compiled differentiation" in cast(str, payload["compiled_differentiation"])
    assert "control flow" in cast(str, payload["control_flow"])
    assert "finite-shot" in cast(str, payload["finite_shot_limitations"])
    assert "broadcast/vmap trainability" in cast(str, payload["finite_shot_limitations"])
    assert "provider" in cast(str, payload["provider_route_support"])
    assert payload["unsupported_provider_routes"] == list(CATALYST_UNSUPPORTED_PROVIDER_ROUTES)
    assert payload["promotion_ready"] is False


def test_catalyst_workflow_profile_rejects_incomplete_provider_boundaries() -> None:
    """Catalyst profiles should not mark promotion-ready rows with open routes."""
    with pytest.raises(ValueError, match="unsupported_provider_routes"):
        CatalystCompilerWorkflowComparison(
            runner_status="dependency_gap",
            compiled_quantum_classical_workflows="not evaluated",
            compiled_differentiation="not evaluated",
            control_flow="not evaluated",
            finite_shot_limitations="finite-shot route not evaluated",
            provider_route_support="provider routes unsupported",
            unsupported_provider_routes=("hardware_qpu_execution", ""),
            claim_boundary="Catalyst hard gap.",
        )

    with pytest.raises(ValueError, match="promotion_ready"):
        CatalystCompilerWorkflowComparison(
            runner_status="success",
            compiled_quantum_classical_workflows="bounded qjit runner success",
            compiled_differentiation="bounded first-order gradient only",
            control_flow="control flow not evaluated",
            finite_shot_limitations="finite-shot route not evaluated",
            provider_route_support="provider routes unsupported",
            unsupported_provider_routes=("hardware_qpu_execution",),
            claim_boundary="Bounded comparison only.",
            promotion_ready=True,
        )

    promoted = CatalystCompilerWorkflowComparison(
        runner_status="success",
        compiled_quantum_classical_workflows="validated Catalyst workflow parity",
        compiled_differentiation="validated compiled differentiation",
        control_flow="validated control flow",
        finite_shot_limitations="finite-shot route validated",
        provider_route_support="provider routes validated",
        unsupported_provider_routes=(),
        claim_boundary="Promotion-ready Catalyst profile.",
        promotion_ready=True,
    )

    assert promoted.to_dict()["unsupported_provider_routes"] == []
    assert promoted.to_dict()["promotion_ready"] is True


def test_catalyst_external_rows_require_dedicated_workflow_profile() -> None:
    """Catalyst external rows should carry the dedicated workflow profile."""
    with pytest.raises(ValueError, match="catalyst_comparison"):
        _catalyst_gap_row()

    profile = catalyst_compiler_workflow_comparison(runner_status="dependency_gap")
    with pytest.raises(ValueError, match="catalyst_comparison"):
        _catalyst_gap_row(profile, backend="jax")


def test_external_suite_records_catalyst_workflow_hard_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconfigured Catalyst should emit a dedicated workflow hard-gap profile."""
    monkeypatch.setattr(comparison, "is_phase_jax_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_torch_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_tensorflow_available", lambda: False)
    monkeypatch.setattr(comparison, "is_phase_pennylane_available", lambda: False)
    monkeypatch.setattr(comparison, "_enzyme_runner_configured", lambda: False)
    monkeypatch.setattr(comparison, "_catalyst_runner_configured", lambda: False)

    rows = run_differentiable_external_comparison_suite()
    catalyst = {row.backend: row for row in rows if row.case_id == "bounded_phase_objective"}[
        "catalyst"
    ]
    catalyst_payload = cast(dict[str, Any], catalyst.to_dict()["catalyst_comparison"])

    assert catalyst.status == "hard_gap"
    assert catalyst.failure_class == "dependency_missing"
    assert catalyst.artifact_fields_ready
    assert catalyst_payload["runner_status"] == "dependency_gap"
    assert catalyst_payload["promotion_ready"] is False
    assert catalyst_payload["unsupported_provider_routes"] == list(
        CATALYST_UNSUPPORTED_PROVIDER_ROUTES
    )
    trainability = {row.case_id: row for row in rows if row.backend == "catalyst"}[
        "trainability_adaptive_shot_dry_run"
    ]
    trainability_payload = cast(dict[str, Any], trainability.to_dict()["catalyst_comparison"])
    assert trainability.status == "hard_gap"
    assert trainability.failure_class == "unsupported_batching"
    assert trainability.closure_status == "permanent_boundary"
    assert trainability.batching_support == "no_broadcast_no_vmap"
    assert "adaptive finite-shot trainability" in str(trainability.setup_instructions)
    assert "broadcast/vmap trainability" in cast(
        str, trainability_payload["finite_shot_limitations"]
    )


def test_configured_catalyst_runner_keeps_provider_routes_non_promotional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful bounded runner should still record open Catalyst provider routes."""
    runner = tmp_path / "catalyst_runner.py"
    runner.write_text(
        "\n".join(
            (
                "#!/usr/bin/env python3",
                "import json, math, sys",
                "payload = json.load(sys.stdin)",
                "values = payload['values']",
                "print(json.dumps({",
                "    'value': math.cos(values[0]) + 0.25 * math.sin(values[1]),",
                "    'gradient': [-math.sin(values[0]), 0.25 * math.cos(values[1])],",
                "    'toolchain': {'catalyst': 'test-runner', 'mlir': 'test-mlir'},",
                "}))",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    monkeypatch.setenv("SCPN_CATALYST_RUNNER", str(runner))
    monkeypatch.setattr(comparison, "_catalyst_tooling_available", lambda: True)

    row = comparison._catalyst_row()
    payload = row.to_dict()
    catalyst_payload = cast(dict[str, Any], payload["catalyst_comparison"])

    assert row.status == "success"
    assert row.value_error is not None and math.isclose(row.value_error, 0.0, abs_tol=1e-12)
    assert catalyst_payload["runner_status"] == "success"
    assert catalyst_payload["promotion_ready"] is False
    assert "finite-shot" in cast(str, catalyst_payload["finite_shot_limitations"])
    assert "hardware_qpu_execution" in catalyst_payload["unsupported_provider_routes"]
