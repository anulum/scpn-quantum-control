# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — PyTorch module-state audit tests
"""Module-state round-trip tests for the bounded PyTorch phase-QNN route."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.phase.torch_module_state as torch_module_state
from scpn_quantum_control.phase import (
    PhaseTorchModuleStateAuditResult,
    PhaseTorchModuleStateValidationResult,
    run_torch_module_state_audit,
    torch_bounded_qnn_module,
    validate_torch_bounded_qnn_state_dict,
)

pytest.importorskip("torch")


class _StateDictMissing:
    """Object without a ``state_dict`` method."""


class _StateDictReturnsList:
    """Object returning a non-mapping state dictionary."""

    @staticmethod
    def state_dict() -> list[str]:
        """Return a malformed state dictionary."""
        return ["params"]


class _StateDictNonStringKey:
    """Object returning a state dictionary with a non-string key."""

    @staticmethod
    def state_dict() -> dict[int, float]:
        """Return a state dictionary with an invalid key type."""
        return {1: 0.0}


class _LoadStateMissing:
    """Object without ``load_state_dict``."""


class _LoadResult:
    """Strict-load result carrying arbitrary key containers."""

    missing_keys = ("params", 2)
    unexpected_keys = "not-a-sequence"


class _LoadStateReturnsKeys:
    """Object exposing a strict load method with key diagnostics."""

    @staticmethod
    def load_state_dict(
        state_dict: dict[str, object],
        *,
        strict: bool,
    ) -> _LoadResult:
        """Return fixed load diagnostics after checking strict mode."""
        assert strict is True
        assert "params" in state_dict
        return _LoadResult()


class _LossWithoutBackward:
    """Scalar-like object that cannot backpropagate."""


class _CallableLossWithoutBackward:
    """Module-like object returning a non-backpropagating loss."""

    @staticmethod
    def __call__() -> _LossWithoutBackward:
        """Return a malformed loss object."""
        return _LossWithoutBackward()


class _LossWithBackward:
    """Scalar-like loss object exposing backward."""

    @staticmethod
    def backward() -> None:
        """Pretend to run backward without populating gradients."""


class _ParamsWithoutGradient:
    """Parameter holder with no gradient."""

    grad = None


class _CallableLossWithoutGradient:
    """Module-like object whose parameters receive no gradient."""

    params = _ParamsWithoutGradient()

    @staticmethod
    def __call__() -> _LossWithBackward:
        """Return a loss with backward but no parameter gradient."""
        return _LossWithBackward()


class _TorchWithoutAdam:
    """PyTorch-like module without an Adam optimizer."""

    class optim:
        """Optim namespace without Adam."""


class _ModuleWithoutParameters:
    """Module-like object without a ``parameters`` method."""


class _ModuleWithNoParameters:
    """Module-like object exposing an empty parameter sequence."""

    @staticmethod
    def parameters() -> tuple[()]:
        """Return no trainable parameters."""
        return ()


class _OptimizerStateMissing:
    """Optimizer-like object without ``state_dict``."""


class _OptimizerStateReturnsList:
    """Optimizer-like object returning malformed state."""

    @staticmethod
    def state_dict() -> list[str]:
        """Return a malformed optimizer state dictionary."""
        return ["state"]


class _OptimizerLoadMissing:
    """Optimizer-like object without ``load_state_dict``."""


class _OptimizerWithoutStep:
    """Optimizer exposing ``zero_grad`` but not ``step``."""

    @staticmethod
    def zero_grad() -> None:
        """Pretend to clear gradients."""


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
    """Return deterministic labels for the module-state fixture."""
    return np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float64)


def _params() -> NDArray[np.float64]:
    """Return deterministic initial parameters for the module-state fixture."""
    return np.array([0.25, -0.35], dtype=np.float64)


def test_torch_module_state_audit_round_trips_module_and_optimizer() -> None:
    """The audit should replay strict module and Adam optimizer state exactly."""
    result = run_torch_module_state_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
        learning_rate=0.05,
        tolerance=1.0e-8,
    )

    assert isinstance(result, PhaseTorchModuleStateAuditResult)
    assert result.passed
    assert result.route_status("module_state_dict_round_trip") == "passed"
    assert result.route_status("optimizer_state_dict_round_trip") == "passed"
    assert result.route_status("device_state_transfer") == "blocked"
    assert set(result.state_dict_keys) == {"features", "labels", "params"}
    assert result.strict_load_missing_keys == ()
    assert result.strict_load_unexpected_keys == ()
    assert result.optimizer_state_entry_count == 1
    assert result.optimizer_param_group_count == 1
    assert result.module_loss_error <= result.tolerance
    assert result.module_gradient_error <= result.tolerance
    assert result.optimizer_replay_parameter_error <= result.tolerance
    assert result.optimizer_replay_loss_error <= result.tolerance
    assert result.provider_claim is False
    assert result.hardware_claim is False
    assert result.performance_claim is False

    payload = result.to_dict()
    routes = cast(dict[str, dict[str, Any]], payload["routes"])
    assert routes["module_state_dict_round_trip"]["status"] == "passed"
    assert routes["optimizer_state_dict_round_trip"]["status"] == "passed"
    assert routes["device_state_transfer"]["status"] == "blocked"
    assert "strict=True" in routes["module_state_dict_round_trip"]["reason"]
    assert "same module parameters" in routes["optimizer_state_dict_round_trip"]["reason"]
    assert "no provider" in str(payload["claim_boundary"])


def test_torch_module_state_validation_rejects_malformed_state_dict() -> None:
    """Validation should fail closed for missing keys and tensor shape drift."""
    module = torch_bounded_qnn_module(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
    )
    state_dict = dict(module.state_dict())
    state_dict_without_params = dict(state_dict)
    state_dict_without_params.pop("params")

    missing_result = validate_torch_bounded_qnn_state_dict(
        module,
        state_dict_without_params,
    )

    assert isinstance(missing_result, PhaseTorchModuleStateValidationResult)
    assert not missing_result.passed
    assert missing_result.missing_keys == ("params",)
    assert missing_result.unexpected_keys == ()
    assert missing_result.mismatched_tensors == ()
    assert "strict PyTorch module state_dict validation" in missing_result.claim_boundary

    mismatched_state = dict(state_dict)
    mismatched_state["params"] = state_dict["params"].reshape(1, 2)
    mismatched_result = validate_torch_bounded_qnn_state_dict(module, mismatched_state)

    assert not mismatched_result.passed
    assert mismatched_result.missing_keys == ()
    assert mismatched_result.unexpected_keys == ()
    assert len(mismatched_result.mismatched_tensors) == 1
    mismatch = mismatched_result.mismatched_tensors[0]
    assert mismatch.key == "params"
    assert mismatch.expected_shape == (2,)
    assert mismatch.observed_shape == (1, 2)
    assert "params" in str(mismatched_result.to_dict())


def test_torch_module_state_audit_rejects_unknown_route() -> None:
    """Route lookups should fail closed for unknown module-state rows."""
    result = run_torch_module_state_audit(
        features=_features(),
        labels=_labels(),
        initial_params=_params(),
    )

    with pytest.raises(KeyError, match="unknown PyTorch module-state route"):
        result.route_status("missing")


def test_torch_module_state_tensor_descriptor_and_clone_edges() -> None:
    """State helper descriptors should handle array and container leaves."""
    shape, dtype = torch_module_state._tensor_descriptor([1.0, 2.0])
    assert shape == (2,)
    assert dtype == "float64"

    nested = {"left": [np.array([1.0], dtype=np.float64)], "right": (2.0,)}
    cloned = torch_module_state._clone_state_mapping(nested)

    assert cloned is not nested
    assert isinstance(cloned["left"], list)
    assert isinstance(cloned["right"], tuple)


def test_torch_module_state_runtime_helper_fail_closed_edges() -> None:
    """Runtime helper boundaries should reject malformed module and optimizer surfaces."""
    with pytest.raises(RuntimeError, match="state_dict"):
        torch_module_state._module_state_dict(_StateDictMissing())
    with pytest.raises(RuntimeError, match="must return a mapping"):
        torch_module_state._module_state_dict(_StateDictReturnsList())
    with pytest.raises(RuntimeError, match="keys must be strings"):
        torch_module_state._module_state_dict(_StateDictNonStringKey())

    with pytest.raises(RuntimeError, match="load_state_dict"):
        torch_module_state._strict_load_module_state(_LoadStateMissing(), {"params": 0.0})
    missing, unexpected = torch_module_state._strict_load_module_state(
        _LoadStateReturnsKeys(),
        {"params": 0.0},
    )
    assert missing == ("params", "2")
    assert unexpected == ()
    assert torch_module_state._string_tuple_from_sequence("params") == ()

    with pytest.raises(RuntimeError, match="backward"):
        torch_module_state._module_loss_and_gradient(_CallableLossWithoutBackward(), 1)
    with pytest.raises(RuntimeError, match="did not receive a gradient"):
        torch_module_state._module_loss_and_gradient(_CallableLossWithoutGradient(), 1)

    with pytest.raises(RuntimeError, match="Adam"):
        torch_module_state._adam_optimizer(_TorchWithoutAdam(), object(), 0.1)
    with pytest.raises(RuntimeError, match="callable parameters"):
        torch_module_state._adam_optimizer(
            cast(Any, __import__("torch")),
            _ModuleWithoutParameters(),
            0.1,
        )
    with pytest.raises(RuntimeError, match="at least one trainable parameter"):
        torch_module_state._adam_optimizer(
            cast(Any, __import__("torch")),
            _ModuleWithNoParameters(),
            0.1,
        )

    with pytest.raises(RuntimeError, match="state_dict"):
        torch_module_state._optimizer_state_dict(_OptimizerStateMissing())
    with pytest.raises(RuntimeError, match="must return a mapping"):
        torch_module_state._optimizer_state_dict(_OptimizerStateReturnsList())
    with pytest.raises(RuntimeError, match=r"\['state'\]"):
        torch_module_state._optimizer_state_counts({"state": [], "param_groups": []})
    with pytest.raises(RuntimeError, match=r"\['param_groups'\]"):
        torch_module_state._optimizer_state_counts({"state": {}, "param_groups": "bad"})
    with pytest.raises(RuntimeError, match="load_state_dict"):
        torch_module_state._load_optimizer_state(_OptimizerLoadMissing(), {})
    with pytest.raises(RuntimeError, match="backward"):
        torch_module_state._optimizer_step(
            _CallableLossWithoutBackward(),
            _OptimizerWithoutStep(),
        )
    with pytest.raises(RuntimeError, match="step"):
        torch_module_state._optimizer_step(
            _CallableLossWithoutGradient(),
            _OptimizerWithoutStep(),
        )
