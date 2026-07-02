# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- PyTorch device-state audit tests
"""Device-transfer state replay tests for bounded PyTorch phase-QNN modules."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.phase import (
    PhaseTorchDeviceStateAuditResult,
    run_torch_module_device_state_audit,
)

pytest.importorskip("torch")


def _features() -> NDArray[np.float64]:
    """Return a deterministic two-parameter bounded phase-QNN fixture."""
    return np.array(
        [
            [0.0, 1.0],
            [np.pi / 2.0, -0.4],
            [np.pi, 0.25],
            [3.0 * np.pi / 2.0, 0.75],
        ],
        dtype=np.float64,
    )


def _labels() -> NDArray[np.float64]:
    """Return deterministic labels for the device-state fixture."""
    return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)


def _params() -> NDArray[np.float64]:
    """Return deterministic initial parameters for the device-state fixture."""
    return np.array([0.25, -0.35], dtype=np.float64)


def test_torch_module_device_state_audit_replays_cpu_and_classifies_cuda() -> None:
    """The audit should replay CPU state and classify CUDA with real runtime metadata."""
    result = run_torch_module_device_state_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        tolerance=1.0e-8,
    )

    assert isinstance(result, PhaseTorchDeviceStateAuditResult)
    assert result.passed
    assert result.route_status("cpu_module_state_transfer") == "passed"
    assert result.route_status("cuda_module_state_transfer") in {"passed", "blocked"}
    assert set(result.state_dict_keys) == {"features", "labels", "params"}
    assert result.cpu_loss_error <= result.tolerance
    assert result.cpu_gradient_error <= result.tolerance
    assert result.cpu_state_devices == {
        "features": "cpu",
        "labels": "cpu",
        "params": "cpu",
    }
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False
    assert result.open_gaps in {(), ("cuda_module_state_transfer",)}

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert routes["cpu_module_state_transfer"]["status"] == "passed"
    assert routes["cuda_module_state_transfer"]["status"] in {"passed", "blocked"}
    assert "no provider" in str(payload["claim_boundary"])
    if result.cuda_smoke_passed:
        assert routes["cuda_module_state_transfer"]["status"] == "passed"
        assert result.cuda_loss_error is not None
        assert result.cuda_gradient_error is not None
        assert result.cuda_loss_error <= result.tolerance
        assert result.cuda_gradient_error <= result.tolerance
    else:
        assert routes["cuda_module_state_transfer"]["status"] == "blocked"
        assert "compatible_cuda_device" in routes["cuda_module_state_transfer"]["requires"]


def test_torch_module_device_state_audit_rejects_invalid_target_devices() -> None:
    """Device-state audit should fail closed for unsupported target-device names."""
    with pytest.raises(ValueError, match="target_devices"):
        run_torch_module_device_state_audit(
            features=_features(),
            labels=_labels(),
            initial_params=_params(),
            target_devices=("cpu", "tpu"),
        )


def test_torch_module_device_state_audit_keeps_cpu_baseline_for_cuda_request() -> None:
    """CUDA requests should retain CPU replay as the mandatory baseline route."""
    result = run_torch_module_device_state_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        target_devices=("cuda",),
    )

    assert result.route_status("cpu_module_state_transfer") == "passed"
    assert result.route_status("cuda_module_state_transfer") in {"passed", "blocked"}
    assert result.passed


def test_torch_module_device_state_audit_rejects_unknown_route() -> None:
    """Route lookups should fail closed for unknown device-state rows."""
    result = run_torch_module_device_state_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        target_devices=("cpu",),
    )

    with pytest.raises(KeyError, match="unknown PyTorch device-state route"):
        result.route_status("missing")
