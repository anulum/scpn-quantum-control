# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Torch Maturity Integration Tests
"""Integration tests for Torch maturity, lowering, and cloud planning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from _phase_torch_bridge_test_helpers import _FakeTorch

import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    PhaseTorchCloudValidationRunSpec,
    PhaseTorchEcosystemMaturityAuditResult,
    PhaseTorchMaturityAuditResult,
    PhaseTorchPhaseQNodeLoweringMatrixResult,
    plan_torch_cloud_validation_batch,
    run_torch_ecosystem_maturity_audit,
    run_torch_maturity_audit,
    run_torch_phase_qnode_lowering_matrix,
)


def test_torch_maturity_audit_records_bounded_passes_and_provider_gaps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify that PyTorch maturity audit records bounded passes and provider gaps."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.45], dtype=float)
    params_batch = np.array([[0.25], [0.45], [0.65]], dtype=float)
    live_overlay_artifact = tmp_path / "diff-qnode-external-comparison.json"
    live_overlay_artifact.write_text(
        json.dumps(
            {
                "artifact_id": "diff-qnode-external-comparison-local",
                "classification": "functional_non_isolated",
                "promotion_ready": False,
                "rows": [
                    {
                        "backend": "pytorch",
                        "status": "success",
                        "value_error": 0.0,
                        "gradient_error": 0.0,
                        "runtime_seconds": 0.1,
                        "memory_peak_bytes": 4096,
                        "batching_support": "torch.func.vmap",
                        "transform_support": "torch.func.grad/jacrev",
                        "dependency_versions": {"torch": "2.11.0+cpu"},
                        "claim_boundary": "bounded CPU comparison only",
                    }
                ],
            },
        ),
        encoding="utf-8",
    )

    result = run_torch_maturity_audit(
        features=features,
        labels=labels,
        params=params,
        params_batch=params_batch,
        tolerance=1e-12,
        live_overlay_artifact_path=live_overlay_artifact,
    )

    assert isinstance(result, PhaseTorchMaturityAuditResult)
    assert result.bounded_model_ready
    assert not result.ready_for_provider_exceedance
    evidence = cast(dict[str, Any], result.evidence)
    assert evidence["analytic_tensor"].passed
    assert evidence["custom_autograd"].passed
    assert evidence["torch_func"].passed
    assert evidence["torch_compile"].passed
    assert evidence["module_layer_wrapper"].passed
    assert evidence["training_loop"].passed
    assert evidence["ecosystem_maturity"].route_status("torch_compile_callable") == "passed"
    assert evidence["cloud_validation_batch"].ready_for_cloud_dispatch
    assert evidence["live_overlay"].passed
    assert evidence["live_overlay"].artifact_id == "diff-qnode-external-comparison-local"
    assert "finite_shot_provider_hardware_torch_phase_qnode_lowering" in result.open_gaps
    assert "torch_ecosystem_maturity" in result.open_gaps
    assert "promotion_grade_isolated_benchmarks" in result.open_gaps
    assert "live_overlay_execution" not in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    json.dumps(payload)
    required_capabilities = cast(dict[str, str], payload["required_capabilities"])
    assert required_capabilities["torch_compile"] == "passed"
    assert required_capabilities["training_loop"] == "passed"
    assert required_capabilities["torch_ecosystem_maturity"] == "blocked"
    assert required_capabilities["cloud_validation_batch"] == "scheduled"
    assert required_capabilities["live_overlay_execution"] == "passed"
    assert (
        cast(dict[str, Any], payload["evidence"])["live_overlay"]["torch_version"] == "2.11.0+cpu"
    )
    assert payload["claim_boundary"] == "bounded_torch_provider_maturity_audit"


def test_torch_maturity_audit_rejects_invalid_lowering_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Torch maturity aggregation should fail closed on invalid lowering evidence."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    monkeypatch.setattr(
        torch_bridge,
        "plan_torch_cloud_validation_batch",
        lambda: PhaseTorchCloudValidationRunSpec(
            runner="local",
            local_execution_status="skipped",
            local_skip_reason="unit test",
            torch_version="fake",
            cuda_available=False,
            cuda_device_count=0,
            cuda_device_names=(),
            blocked_local_routes=(),
            required_artifacts=(),
            required_environment={},
            commands=(),
            ready_for_cloud_dispatch=False,
        ),
    )
    monkeypatch.setattr(torch_bridge, "run_torch_phase_qnode_lowering_matrix", lambda: object())

    with pytest.raises(RuntimeError, match="phase-QNode lowering matrix"):
        run_torch_maturity_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.45], dtype=float),
            params_batch=np.array([[0.25], [0.45], [0.65]], dtype=float),
        )


def test_torch_ecosystem_maturity_audit_records_broad_module_func_compile_device_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch ecosystem maturity audit records broad module func compile
    device gaps.
    """
    fake_torch = _FakeTorch()
    fake_torch.__version__ = "2.11.0+cpu"
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    result = run_torch_ecosystem_maturity_audit()

    assert isinstance(result, PhaseTorchEcosystemMaturityAuditResult)
    assert not result.ready_for_provider_exceedance
    assert result.route_status("nn_module_parameter_surface") == "passed"
    assert result.route_status("torch_func_grad_vmap_jacrev") == "passed"
    assert result.route_status("torch_func_jacfwd_hessian") == "blocked"
    assert result.route_status("torch_compile_callable") == "passed"
    assert result.route_status("registered_phase_qnode_torch_compile_lowering") == "passed"
    assert (
        result.route_status("registered_phase_qnode_torch_compile_fullgraph_lowering") == "blocked"
    )
    assert result.route_status("cuda_accelerator_device") == "blocked"
    assert result.torch_version == "2.11.0+cpu"
    assert not result.cuda_available
    assert "cuda_accelerator_device" in result.open_gaps
    payload = cast(dict[str, Any], result.to_dict())
    assert (
        cast(dict[str, Any], payload["routes"])["cuda_accelerator_device"]["status"] == "blocked"
    )
    assert payload["claim_boundary"] == "torch_ecosystem_device_maturity_audit"


def test_torch_maturity_audit_rejects_incomplete_live_overlay_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify that PyTorch maturity audit rejects incomplete live overlay artifact."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)
    bad_artifact = tmp_path / "diff-qnode-external-comparison.json"
    bad_artifact.write_text(
        json.dumps(
            {
                "artifact_id": "diff-qnode-external-comparison-local",
                "classification": "functional_non_isolated",
                "rows": [{"backend": "pytorch", "status": "hard_gap"}],
            },
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="successful PyTorch row"):
        run_torch_maturity_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.45], dtype=float),
            params_batch=np.array([[0.25], [0.45], [0.65]], dtype=float),
            live_overlay_artifact_path=bad_artifact,
        )


def test_torch_phase_qnode_lowering_matrix_fails_closed_for_arbitrary_qnodes() -> None:
    """Verify that PyTorch phase QNode lowering matrix fails closed for arbitrary
    qnodes.
    """
    result = run_torch_phase_qnode_lowering_matrix()

    assert isinstance(result, PhaseTorchPhaseQNodeLoweringMatrixResult)
    assert result.bounded_qnn_routes_ready
    assert not result.arbitrary_phase_qnode_lowering_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("bounded_qnn_custom_autograd") == "passed"
    assert result.route_status("registered_phase_qnode_statevector_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_torch_func_transform_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_torch_compile_lowering") == "passed"
    assert (
        result.route_status("registered_phase_qnode_torch_compile_fullgraph_lowering") == "blocked"
    )
    assert result.route_status("registered_phase_qnode_cuda_device_lowering") == "blocked"
    assert result.route_status("registered_phase_qnode_provider_lowering") == "blocked"
    assert result.route_status("registered_phase_qnode_hardware_lowering") == "blocked"
    assert "registered_phase_qnode_statevector_lowering" not in result.open_gaps
    assert "registered_phase_qnode_torch_func_transform_lowering" not in result.open_gaps
    assert "registered_phase_qnode_torch_compile_lowering" not in result.open_gaps
    assert "registered_phase_qnode_torch_compile_fullgraph_lowering" in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    assert result.claim_boundary == "bounded_torch_phase_qnode_lowering_matrix"

    payload = cast(dict[str, Any], result.to_dict())
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert routes["bounded_qnn_torch_compile"]["status"] == "passed"
    assert routes["registered_phase_qnode_hardware_lowering"]["status"] == "blocked"
    assert routes["registered_phase_qnode_hardware_lowering"]["requires"] == [
        "live_ticket",
        "provider_allowlist",
        "shot_budget",
        "hardware_evidence_id",
    ]


def test_torch_cloud_validation_batch_schedules_incompatible_local_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch cloud validation batch schedules incompatible local device."""
    fake_torch = _FakeTorch()
    fake_torch.__version__ = "2.11.0+cpu"
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    result = plan_torch_cloud_validation_batch(runner="jarvislabs")

    assert isinstance(result, PhaseTorchCloudValidationRunSpec)
    assert result.runner == "jarvislabs"
    assert result.ready_for_cloud_dispatch
    assert result.local_execution_status == "skipped_incompatible_local_hardware"
    assert "CUDA" in result.local_skip_reason
    assert "registered_phase_qnode_torch_compile_fullgraph_lowering" in result.blocked_local_routes
    assert "registered_phase_qnode_cuda_device_lowering" in result.blocked_local_routes
    assert "registered_phase_qnode_fullgraph_compile_artifact" in result.required_artifacts
    assert "cuda_device_phase_qnode_gradient_artifact" in result.required_artifacts
    assert "isolated_benchmark_artifact" in result.required_artifacts
    assert any("test_phase_framework_bridges.py" in command for command in result.commands)
    assert result.required_environment["accelerator_backend"] == "cuda"
    assert result.claim_boundary == "torch_cloud_validation_batch_plan"
    payload = result.to_dict()
    assert payload["local_execution_status"] == "skipped_incompatible_local_hardware"
    json.dumps(payload)


def test_torch_maturity_audit_fails_closed_on_bad_batch_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that PyTorch maturity audit fails closed on bad batch shape."""
    fake_torch = _FakeTorch()
    monkeypatch.setattr(torch_bridge, "_load_torch", lambda: fake_torch)

    with pytest.raises(ValueError, match="params_batch"):
        run_torch_maturity_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.45], dtype=float),
            params_batch=np.array([0.25], dtype=float),
        )
