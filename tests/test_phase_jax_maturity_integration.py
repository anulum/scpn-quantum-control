# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Maturity Integration Tests
"""Integration tests for JAX maturity, lowering, and cloud planning."""

from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
import pytest
from _phase_jax_bridge_test_helpers import (
    _FakeJAX,
)

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import (
    PhaseJAXCloudValidationRunSpec,
    PhaseJAXMaturityAuditResult,
    PhaseJAXPhaseQNodeLoweringMatrixResult,
    plan_jax_cloud_validation_batch,
    run_jax_maturity_audit,
    run_jax_phase_qnode_lowering_matrix,
)


def test_phase_jax_maturity_audit_records_bounded_passes_and_provider_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = 2
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0, 0.2], [np.pi, np.pi + 0.2]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params = np.array([0.25, 0.45], dtype=float)
    params_batch = np.array([[0.25, 0.45], [0.35, 0.55]], dtype=float)
    params_tree = {"phase": params}

    result = run_jax_maturity_audit(
        features=features,
        labels=labels,
        params=params,
        params_batch=params_batch,
        params_pytree=params_tree,
        tolerance=1e-10,
    )
    payload = result.to_dict()

    assert isinstance(result, PhaseJAXMaturityAuditResult)
    assert result.bounded_model_ready
    assert not result.ready_for_provider_exceedance
    assert cast(Any, result.evidence["custom_vjp"]).passed
    assert cast(Any, result.evidence["jit"]).passed
    assert cast(Any, result.evidence["vmap"]).passed
    assert cast(Any, result.evidence["pmap_sharding"]).passed
    assert cast(Any, result.evidence["pytree"]).passed
    assert cast(Any, result.evidence["cloud_validation_batch"]).ready_for_cloud_dispatch
    assert "arbitrary_quantum_kernel_jax_lowering" in result.open_gaps
    assert "hardware_or_provider_callback_transform_safety" in result.open_gaps
    assert cast(Any, payload["required_capabilities"])["jit"] == "passed"
    assert cast(Any, payload["required_capabilities"])["cloud_validation_batch"] == "scheduled"
    assert (
        cast(Any, payload["required_capabilities"])["arbitrary_quantum_kernel_jax_lowering"]
        == "blocked"
    )
    assert payload["claim_boundary"] == "bounded_jax_provider_maturity_audit"


def test_phase_jax_cloud_validation_batch_schedules_gtx1060_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = 1
    cast(Any, fake_jax).local_devices = lambda: ("NVIDIA GeForce GTX 1060",)
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = plan_jax_cloud_validation_batch(runner="jarvislabs")

    assert isinstance(result, PhaseJAXCloudValidationRunSpec)
    assert result.runner == "jarvislabs"
    assert result.ready_for_cloud_dispatch
    assert result.local_execution_status == "skipped_incompatible_local_hardware"
    assert "GTX 1060" in result.local_skip_reason
    assert "jax_cuda_accelerator_device" in result.blocked_local_routes
    assert "registered_phase_qnode_pmap_multi_device_lowering" in result.blocked_local_routes
    assert "jax_cuda_device_metadata_artifact" in result.required_artifacts
    assert "registered_phase_qnode_jax_pmap_sharding_artifact" in result.required_artifacts
    assert "isolated_benchmark_artifact" in result.required_artifacts
    assert any("test_phase_jax_bridge.py" in command for command in result.commands)
    assert result.required_environment["accelerator_backend"] == "cuda"
    assert result.claim_boundary == "jax_cloud_validation_batch_plan"
    payload = result.to_dict()
    assert payload["local_execution_status"] == "skipped_incompatible_local_hardware"
    json.dumps(payload)


def test_phase_jax_phase_qnode_lowering_matrix_fails_closed_for_arbitrary_qnodes() -> None:
    result = run_jax_phase_qnode_lowering_matrix()

    assert isinstance(result, PhaseJAXPhaseQNodeLoweringMatrixResult)
    assert result.bounded_no_host_callback_routes_ready
    assert result.arbitrary_phase_qnode_lowering_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("bounded_qnn_native_value_and_grad") == "passed"
    assert result.route_status("bounded_qnn_jit_value_and_grad") == "passed"
    assert result.route_status("bounded_qnn_vmap_value_and_grad") == "passed"
    assert result.route_status("registered_phase_qnode_statevector_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_native_transform_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_pytree_transform_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_pmap_sharding_lowering") == "passed"
    assert result.route_status("registered_phase_qnode_provider_lowering") == "blocked"
    assert "registered_phase_qnode_statevector_lowering" not in result.open_gaps
    assert "registered_phase_qnode_native_transform_lowering" not in result.open_gaps
    assert "registered_phase_qnode_pytree_transform_lowering" not in result.open_gaps
    assert "registered_phase_qnode_pmap_sharding_lowering" not in result.open_gaps
    assert "isolated_benchmark_artifact" in result.open_gaps
    assert result.claim_boundary == "bounded_jax_phase_qnode_lowering_matrix"

    payload = result.to_dict()
    assert cast(Any, payload["routes"])["bounded_qnn_jit_value_and_grad"]["host_callback"] is False
    assert (
        cast(Any, payload["routes"])["registered_phase_qnode_statevector_lowering"][
            "host_callback"
        ]
        is False
    )
    assert (
        cast(Any, payload["routes"])["registered_phase_qnode_native_transform_lowering"][
            "host_callback"
        ]
        is False
    )
    assert (
        cast(Any, payload["routes"])["registered_phase_qnode_pytree_transform_lowering"][
            "host_callback"
        ]
        is False
    )
    assert (
        cast(Any, payload["routes"])["registered_phase_qnode_pmap_sharding_lowering"][
            "host_callback"
        ]
        is False
    )


def test_phase_jax_maturity_audit_fails_closed_on_bad_batch_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    with pytest.raises(ValueError, match="params_batch"):
        run_jax_maturity_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.25], dtype=float),
            params_batch=np.array([0.25], dtype=float),
            params_pytree={"phase": np.array([0.25], dtype=float)},
        )
