# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for QNN Framework Bridge Matrix
"""Tests for bounded phase-QNN framework bridge capability declarations."""

from __future__ import annotations

import pytest

from scpn_quantum_control.phase import (
    BoundedQNNFrameworkBridgeMatrixResult,
    assert_bounded_qnn_framework_bridge_supported,
    run_bounded_qnn_framework_bridge_matrix,
)


def test_bounded_qnn_framework_bridge_matrix_declares_supported_routes() -> None:
    result = run_bounded_qnn_framework_bridge_matrix()

    assert isinstance(result, BoundedQNNFrameworkBridgeMatrixResult)
    assert result.passed
    assert result.framework_count == 5
    assert result.supported_count == 3
    assert result.fail_closed_count == 2
    assert result.native_framework_autodiff_count == 2
    assert result.tensor_output_count == 2
    assert result.host_boundary_count == 0

    jax = result.capability_by_framework("jax")
    assert jax.supported
    assert jax.native_framework_autodiff
    assert jax.public_api == "jax_native_qnn_value_and_grad,jax_custom_vjp_qnn_value_and_grad"
    assert "jax_custom_vjp_bounded_phase_qnn_value_and_grad" in jax.gradient_route
    assert not jax.host_boundary

    pytorch = result.capability_by_framework("pytorch")
    assert pytorch.supported
    assert pytorch.tensor_output
    assert pytorch.native_framework_autodiff
    assert pytorch.analytic_framework_gradient
    assert (
        pytorch.public_api == "torch_bounded_qnn_value_and_grad,torch_autograd_qnn_value_and_grad"
        ",run_torch_func_compatibility_audit,run_torch_compile_compatibility_audit"
        ",torch_bounded_qnn_module,torch_bounded_qnn_layer,run_torch_module_wrapper_audit"
    )
    assert "torch_bounded_phase_qnn_custom_autograd_function" in pytorch.gradient_route
    assert "bounded_torch_func_grad_vmap_jacrev" in pytorch.gradient_route
    assert "bounded_torch_compile_gradient" in pytorch.gradient_route
    assert "bounded_torch_module_layer_wrapper_gradient" in pytorch.gradient_route

    tensorflow = result.capability_by_framework("tensorflow")
    assert tensorflow.supported
    assert tensorflow.tensor_output
    assert tensorflow.analytic_framework_gradient
    assert tensorflow.public_api == "tensorflow_bounded_qnn_value_and_grad"


def test_bounded_qnn_framework_bridge_matrix_records_fail_closed_gaps() -> None:
    result = run_bounded_qnn_framework_bridge_matrix(
        frameworks=("generic_simulator_autodiff", "provider_hardware_gradient"),
    )

    assert result.framework_count == 2
    assert result.supported_count == 0
    assert result.fail_closed_count == 2
    assert result.passed
    simulator = result.capability_by_framework("generic_simulator_autodiff")
    hardware = result.capability_by_framework("provider_hardware_gradient")
    assert not simulator.supported
    assert "arbitrary simulator kernels" in str(simulator.fail_closed_reason)
    assert not hardware.supported
    assert "hardware QNN gradients" in str(hardware.fail_closed_reason)


def test_bounded_qnn_framework_bridge_assertion_fails_closed() -> None:
    capability = assert_bounded_qnn_framework_bridge_supported("jax")
    assert capability.supported

    with pytest.raises(RuntimeError, match="unsupported"):
        assert_bounded_qnn_framework_bridge_supported("provider_hardware_gradient")

    with pytest.raises(ValueError, match="unknown bounded QNN framework bridge"):
        run_bounded_qnn_framework_bridge_matrix(frameworks=("missing",))

    with pytest.raises(ValueError, match="framework name must be non-empty"):
        run_bounded_qnn_framework_bridge_matrix(frameworks=("",))


def test_bounded_qnn_framework_bridge_matrix_to_dict_is_json_ready() -> None:
    result = run_bounded_qnn_framework_bridge_matrix(frameworks=("jax", "tensorflow"))
    payload = result.to_dict()

    assert payload["passed"] is True
    assert payload["framework_count"] == 2
    assert payload["supported_count"] == 2
    assert payload["native_framework_autodiff_count"] == 1
    assert payload["tensor_output_count"] == 1
    assert "not arbitrary framework autodiff" in str(payload["claim_boundary"])
    assert [item["framework"] for item in payload["capabilities"]] == [
        "jax",
        "tensorflow",
    ]
