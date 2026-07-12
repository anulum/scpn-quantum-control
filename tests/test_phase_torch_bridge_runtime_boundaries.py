# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — phase torch bridge runtime boundaries tests
# scpn-quantum-control -- PyTorch bridge runtime-boundary tests
"""Runtime-boundary edge tests for the phase PyTorch bridge."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import scpn_quantum_control.phase.torch_bridge as torch_bridge
from scpn_quantum_control.phase import (
    run_torch_compile_compatibility_audit,
    run_torch_ecosystem_maturity_audit,
    run_torch_func_compatibility_audit,
    torch_autograd_qnn_value_and_grad,
    torch_bounded_qnn_module,
)

pytest.importorskip("torch")  # the torch bridge is an optional extra; skip when torch is absent


class _ArrayCarrier:
    """Tiny tensor-like carrier exposing detach/cpu/numpy conversion hooks."""

    def __init__(self, value: object) -> None:
        self._value = value

    def detach(self) -> _ArrayCarrier:
        """Return the detached carrier."""

        return self

    def cpu(self) -> _ArrayCarrier:
        """Return the host carrier."""

        return self

    def numpy(self) -> object:
        """Return the stored NumPy-compatible value."""

        return self._value


class _NoGradTensor:
    """Tensor-like object that intentionally omits ``requires_grad_``."""

    def detach(self) -> _NoGradTensor:
        """Return a detached tensor-like object."""

        return self

    def clone(self) -> _NoGradTensor:
        """Return a cloned tensor-like object."""

        return self


class _TorchReturningNoGradTensor:
    """Minimal PyTorch-like module returning tensors without grad mutators."""

    float64 = "float64"

    def as_tensor(self, values: object, *, dtype: object | None = None) -> _NoGradTensor:
        """Return a tensor-like object without ``requires_grad_``."""

        _ = values, dtype
        return _NoGradTensor()


class _TorchAsTensorNoDtype:
    """Minimal module with ``as_tensor`` and no dtype constant."""

    def as_tensor(self, values: object) -> tuple[str, object]:
        """Return a marker proving the no-dtype branch was used."""

        return ("as_tensor", values)


class _TorchTensorWithDtype:
    """Minimal module with only ``tensor`` and a dtype constant."""

    float64 = "float64"

    def tensor(self, values: object, *, dtype: object | None = None) -> tuple[str, object, object]:
        """Return a marker proving the tensor fallback branch was used."""

        return ("tensor", values, dtype)


class _TorchTensorNoDtype:
    """Minimal module with only ``tensor`` and no dtype constant."""

    def tensor(self, values: object) -> tuple[str, object]:
        """Return a marker proving the tensor no-dtype branch was used."""

        return ("tensor", values)


class _SmokeTensor:
    """Tensor-like object used for CUDA metadata smoke probes."""

    def __add__(self, other: object) -> _SmokeTensor:
        """Return a tensor-like sum marker."""

        _ = other
        return self

    def detach(self) -> _SmokeTensor:
        """Return the detached smoke tensor."""

        return self

    def cpu(self) -> _SmokeTensor:
        """Return the host smoke tensor."""

        return self

    def numpy(self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Return a successful smoke result."""

        return np.array([2.0], dtype=np.float64)


class _CudaAvailableNoNames:
    """CUDA facade with devices but no device-name accessor."""

    def is_available(self) -> bool:
        """Return CUDA availability."""

        return True

    def device_count(self) -> int:
        """Return a visible device count."""

        return 2


class _CudaUnavailableWithRuntime:
    """CUDA facade with metadata hooks but no available device."""

    def is_available(self) -> bool:
        """Return CUDA unavailability."""

        return False

    def device_count(self) -> int:
        """Return a count that should be ignored while unavailable."""

        return 3


class _CudaAvailableSmokeFailure:
    """CUDA facade whose smoke execution fails after metadata succeeds."""

    def is_available(self) -> bool:
        """Return CUDA availability."""

        return True

    def device_count(self) -> int:
        """Return a visible device count."""

        return 1

    def get_device_name(self, index: int) -> str:
        """Return a deterministic device name."""

        return f"cuda-test-{index}"


class _TorchCudaSmokePass:
    """PyTorch-like module whose CUDA metadata smoke test passes."""

    float64 = "float64"
    cuda = _CudaAvailableNoNames()

    def device(self, kind: str, index: int) -> str:
        """Return a deterministic device token."""

        return f"{kind}:{index}"

    def ones(
        self,
        shape: tuple[int, ...],
        *,
        dtype: object,
        device: object,
    ) -> _SmokeTensor:
        """Return a tensor-like smoke value."""

        _ = shape, dtype, device
        return _SmokeTensor()


class _TorchCudaUnavailable:
    """PyTorch-like module exposing unavailable CUDA runtime hooks."""

    cuda = _CudaUnavailableWithRuntime()


class _TorchCudaSmokeFailure:
    """PyTorch-like module whose CUDA smoke operation raises."""

    float64 = "float64"
    cuda = _CudaAvailableSmokeFailure()

    def device(self, kind: str, index: int) -> str:
        """Return a deterministic device token."""

        return f"{kind}:{index}"

    def ones(self, shape: tuple[int, ...], *, dtype: object, device: object) -> _SmokeTensor:
        """Raise during smoke allocation."""

        _ = shape, dtype, device
        raise RuntimeError("smoke blocked")


class _ParameterlessModule:
    """Object without a callable ``parameters`` method."""


class _CountedParameterModule:
    """Object exposing a counted parameter iterator."""

    def parameters(self) -> Iterator[object]:
        """Yield deterministic parameter markers."""

        yield object()
        yield object()


def test_torch_conversion_helpers_validate_tensor_like_boundaries() -> None:
    """Tensor conversion helpers should validate hooks, scalars, and fallbacks."""

    with pytest.raises(ValueError, match="must not be empty"):
        torch_bridge._as_parameter_matrix("matrix", np.empty((0, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="width"):
        torch_bridge._as_parameter_matrix(
            "matrix",
            np.array([[1.0, 2.0]], dtype=np.float64),
            width=3,
        )

    matrix = torch_bridge._torch_matrix_to_numpy(
        "matrix",
        _ArrayCarrier(np.array([[1.0, 2.0]], dtype=np.float64)),
    )
    scalar = torch_bridge._torch_scalar_to_float(_ArrayCarrier(np.array([3.0])))

    assert matrix.shape == (1, 2)
    assert scalar == 3.0
    with pytest.raises(ValueError, match="scalar-like"):
        torch_bridge._torch_scalar_to_float(_ArrayCarrier(np.array([1.0, 2.0])))
    with pytest.raises(ValueError, match="finite"):
        torch_bridge._torch_scalar_to_float(_ArrayCarrier(np.array([np.nan])))

    assert torch_bridge._torch_tensor(_TorchAsTensorNoDtype(), [1.0]) == ("as_tensor", [1.0])
    assert torch_bridge._torch_tensor(_TorchTensorWithDtype(), [1.0]) == (
        "tensor",
        [1.0],
        "float64",
    )
    assert torch_bridge._torch_tensor(_TorchTensorNoDtype(), [1.0]) == ("tensor", [1.0])
    with pytest.raises(RuntimeError, match="as_tensor or tensor"):
        torch_bridge._torch_tensor(SimpleNamespace(), [1.0])


def test_torch_optional_runtime_helpers_fail_closed() -> None:
    """Optional PyTorch runtime helpers should fail closed when hooks are absent."""

    with pytest.raises(RuntimeError, match="autograd.grad"):
        torch_bridge._torch_autograd_grad(SimpleNamespace(autograd=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="torch.func"):
        torch_bridge._torch_func_transforms(SimpleNamespace(func=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="torch.compile"):
        torch_bridge._torch_compile(SimpleNamespace())
    with pytest.raises(RuntimeError, match="torch.nn.Module"):
        torch_bridge._torch_nn_module_and_parameter(SimpleNamespace(nn=SimpleNamespace()))
    with pytest.raises(RuntimeError, match="requires_grad"):
        torch_bridge._torch_trainable_tensor(
            _TorchReturningNoGradTensor(),
            np.array([0.1], dtype=np.float64),
        )

    assert torch_bridge._torch_parameter_count(_ParameterlessModule()) == 0
    assert torch_bridge._torch_parameter_count(_CountedParameterModule()) == 2


def test_torch_cuda_metadata_branches_are_deterministic() -> None:
    """CUDA metadata probing should report missing, passing, and failing smoke states."""

    assert torch_bridge._torch_cuda_metadata(SimpleNamespace()) == (
        False,
        0,
        (),
        False,
        "PyTorch CUDA runtime metadata is unavailable",
    )
    assert torch_bridge._torch_cuda_metadata(_TorchCudaUnavailable()) == (
        False,
        0,
        (),
        False,
        "no visible CUDA devices",
    )
    assert torch_bridge._torch_cuda_metadata(_TorchCudaSmokePass()) == (
        True,
        2,
        ("cuda:0", "cuda:1"),
        True,
        "CUDA smoke execution passed",
    )
    available, count, names, smoke_passed, reason = torch_bridge._torch_cuda_metadata(
        _TorchCudaSmokeFailure(),
    )
    assert (available, count, names, smoke_passed) == (True, 1, ("cuda-test-0",), False)
    assert "CUDA smoke execution failed" in reason


def test_torch_private_qnode_guards_reject_unknown_internal_labels() -> None:
    """Private QNode matrix helpers should retain explicit unsupported-branch errors."""

    torch_module = torch_bridge._load_torch()
    state = torch_bridge._torch_complex_tensor(torch_module, [1.0, 0.0])

    with pytest.raises(ValueError, match="unsupported PyTorch Phase-QNode gate"):
        torch_bridge._torch_gate_matrix(torch_module, "unknown", None)
    with pytest.raises(ValueError, match="unsupported PyTorch Phase-QNode observable"):
        torch_bridge._torch_expectation_value(torch_module, state, 1, object())
    with pytest.raises(ValueError, match="unsupported PyTorch Pauli label"):
        torch_bridge._torch_pauli_matrix(torch_module, "q")


def test_torch_public_qnn_routes_reject_width_mismatches() -> None:
    """Public QNN routes should reject parameter widths before framework dispatch."""

    features = np.array([[0.0, 0.2], [0.4, 0.6]], dtype=np.float64)
    labels = np.array([0.0, 1.0], dtype=np.float64)
    short_params = np.array([0.1], dtype=np.float64)

    with pytest.raises(ValueError, match="params width"):
        torch_autograd_qnn_value_and_grad(features, labels, short_params)
    with pytest.raises(ValueError, match="params width"):
        run_torch_func_compatibility_audit(
            features=features,
            labels=labels,
            params=short_params,
            params_batch=np.array([[0.1]], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="params width"):
        run_torch_compile_compatibility_audit(
            features=features, labels=labels, params=short_params
        )
    with pytest.raises(ValueError, match="initial_params width"):
        torch_bounded_qnn_module(features=features, labels=labels, initial_params=short_params)


def test_torch_ecosystem_maturity_reports_blocked_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ecosystem maturity audit should serialize blocked optional-runtime routes."""

    monkeypatch.setattr(
        torch_bridge,
        "_load_torch",
        lambda: SimpleNamespace(__version__="fake", func=SimpleNamespace(), nn=SimpleNamespace()),
    )

    audit = run_torch_ecosystem_maturity_audit()
    statuses = {route.name: route.status for route in audit.routes}

    assert statuses["nn_module_parameter_surface"] == "blocked"
    assert statuses["torch_func_grad_vmap_jacrev"] == "blocked"
    assert statuses["torch_func_jacfwd_hessian"] == "blocked"
    assert statuses["torch_compile_callable"] == "blocked"


def test_torch_ecosystem_maturity_reports_forward_hessian_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ecosystem maturity audit should detect jacfwd/hessian when present."""

    monkeypatch.setattr(
        torch_bridge,
        "_load_torch",
        lambda: SimpleNamespace(
            __version__="fake",
            func=SimpleNamespace(
                jacfwd=lambda function: function,
                hessian=lambda function: function,
            ),
            nn=SimpleNamespace(),
        ),
    )

    audit = run_torch_ecosystem_maturity_audit()

    assert audit.route_status("torch_func_jacfwd_hessian") == "passed"


def test_torch_live_overlay_required_field_helpers_reject_bad_types() -> None:
    """Live-overlay scalar field validators should reject malformed field types."""

    with pytest.raises(ValueError, match="name"):
        torch_bridge._required_str({"name": ""}, "name")
    with pytest.raises(ValueError, match="value"):
        torch_bridge._required_float({"value": "1.0"}, "value")
    with pytest.raises(ValueError, match="count"):
        torch_bridge._required_int({"count": 1.5}, "count")
