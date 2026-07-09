# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Optional tensor input/output contract tests
"""Tests for Torch/JAX input and output on dense Kuramoto facade hot paths."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import oscillatools as kuramoto
import oscillatools.accel.tensor_io as tensor_io


class _TorchLikeTensor:
    __module__ = "torch.fake"

    dtype = "float32"
    device = "cuda:0"

    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def detach(self) -> _TorchLikeTensor:
        return self

    def cpu(self) -> _TorchLikeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _JaxLikeTensor:
    __module__ = "jaxlib.fake"


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.array([0.0, 0.4, 1.1, 2.0], dtype=np.float64)
    omega = np.array([0.1, -0.2, 0.05, 0.15], dtype=np.float64)
    coupling = np.array(
        [
            [0.0, 0.5, 0.1, 0.0],
            [0.5, 0.0, 0.2, 0.3],
            [0.1, 0.2, 0.0, 0.4],
            [0.0, 0.3, 0.4, 0.0],
        ],
        dtype=np.float64,
    )
    return theta, omega, coupling


def test_importing_oscillatools_does_not_import_optional_tensor_backends() -> None:
    script = (
        "import sys; import oscillatools; "
        "print('torch' in sys.modules, 'jax' in sys.modules, 'jax.numpy' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False False False"


class TestTorchTensorIO:
    def test_networked_force_and_jacobian_return_torch_tensors(self) -> None:
        torch = pytest.importorskip("torch")
        theta, _, coupling = _problem()
        theta_t = torch.tensor(theta, dtype=torch.float64)
        coupling_t = torch.tensor(coupling, dtype=torch.float64)

        force = kuramoto.networked_kuramoto_force(theta_t, coupling_t)
        jacobian = kuramoto.networked_kuramoto_jacobian(theta_t, coupling_t)

        assert isinstance(force, torch.Tensor)
        assert isinstance(jacobian, torch.Tensor)
        assert force.dtype is torch.float64
        assert jacobian.dtype is torch.float64
        np.testing.assert_allclose(
            force.detach().cpu().numpy(),
            kuramoto.networked_kuramoto_force(theta, coupling),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            jacobian.detach().cpu().numpy(),
            kuramoto.networked_kuramoto_jacobian(theta, coupling),
            atol=1e-12,
        )

    def test_trajectories_and_vjp_return_torch_tensors(self) -> None:
        torch = pytest.importorskip("torch")
        theta, omega, coupling = _problem()
        theta_t = torch.tensor(theta, dtype=torch.float64)
        omega_t = torch.tensor(omega, dtype=torch.float64)
        coupling_t = torch.tensor(coupling, dtype=torch.float64)

        trajectory = kuramoto.kuramoto_rk4_trajectory(theta_t, omega_t, coupling_t, 0.02, 5)
        cotangent = torch.ones(theta.shape[0], dtype=torch.float64)
        gradients = kuramoto.kuramoto_rk4_vjp(trajectory, omega_t, coupling_t, 0.02, cotangent)

        assert isinstance(trajectory, torch.Tensor)
        assert trajectory.shape == (6, 4)
        assert all(isinstance(channel, torch.Tensor) for channel in gradients)
        assert gradients[0].shape == (4,)
        assert gradients[1].shape == (4,)
        assert gradients[2].shape == (4, 4)

    def test_energy_derivatives_return_torch_tensors_but_scalar_stays_float(self) -> None:
        torch = pytest.importorskip("torch")
        theta, _, coupling = _problem()
        theta_t = torch.tensor(theta, dtype=torch.float64)
        coupling_t = torch.tensor(coupling, dtype=torch.float64)

        energy = kuramoto.kuramoto_interaction_energy(theta_t, coupling_t)
        gradient = kuramoto.kuramoto_interaction_energy_gradient(theta_t, coupling_t)
        hessian = kuramoto.kuramoto_interaction_energy_hessian(theta_t, coupling_t)

        assert isinstance(energy, float)
        assert isinstance(gradient, torch.Tensor)
        assert isinstance(hessian, torch.Tensor)
        assert gradient.shape == (4,)
        assert hessian.shape == (4, 4)


class TestJaxTensorIO:
    def test_networked_force_and_jacobian_return_jax_arrays(self) -> None:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = pytest.importorskip("jax.numpy")
        theta, _, coupling = _problem()
        theta_j = jnp.asarray(theta, dtype=jnp.float64)
        coupling_j = jnp.asarray(coupling, dtype=jnp.float64)

        force = kuramoto.networked_kuramoto_force(theta_j, coupling_j)
        jacobian = kuramoto.networked_kuramoto_jacobian(theta_j, coupling_j)

        assert type(force).__module__.startswith(("jax.", "jaxlib."))
        assert type(jacobian).__module__.startswith(("jax.", "jaxlib."))
        np.testing.assert_allclose(
            np.asarray(force), kuramoto.networked_kuramoto_force(theta, coupling)
        )
        np.testing.assert_allclose(
            np.asarray(jacobian),
            kuramoto.networked_kuramoto_jacobian(theta, coupling),
        )

    def test_euler_trajectory_and_vjp_return_jax_arrays(self) -> None:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = pytest.importorskip("jax.numpy")
        theta, omega, coupling = _problem()
        theta_j = jnp.asarray(theta, dtype=jnp.float64)
        omega_j = jnp.asarray(omega, dtype=jnp.float64)
        coupling_j = jnp.asarray(coupling, dtype=jnp.float64)

        trajectory = kuramoto.kuramoto_euler_trajectory(theta_j, omega_j, coupling_j, 0.02, 5)
        cotangent = jnp.ones(theta.shape[0], dtype=jnp.float64)
        gradients = kuramoto.kuramoto_euler_vjp(trajectory, coupling_j, 0.02, cotangent)

        assert type(trajectory).__module__.startswith(("jax.", "jaxlib."))
        assert trajectory.shape == (6, 4)
        assert all(
            type(channel).__module__.startswith(("jax.", "jaxlib.")) for channel in gradients
        )
        assert gradients[0].shape == (4,)
        assert gradients[1].shape == (4,)
        assert gradients[2].shape == (4, 4)

    def test_energy_derivatives_return_jax_arrays_but_scalar_stays_float(self) -> None:
        jax = pytest.importorskip("jax")
        jax.config.update("jax_enable_x64", True)
        jnp = pytest.importorskip("jax.numpy")
        theta, _, coupling = _problem()
        theta_j = jnp.asarray(theta, dtype=jnp.float64)
        coupling_j = jnp.asarray(coupling, dtype=jnp.float64)

        energy = kuramoto.kuramoto_interaction_energy(theta_j, coupling_j)
        gradient = kuramoto.kuramoto_interaction_energy_gradient(theta_j, coupling_j)
        hessian = kuramoto.kuramoto_interaction_energy_hessian(theta_j, coupling_j)

        assert isinstance(energy, float)
        assert type(gradient).__module__.startswith(("jax.", "jaxlib."))
        assert type(hessian).__module__.startswith(("jax.", "jaxlib."))
        assert gradient.shape == (4,)
        assert hessian.shape == (4, 4)


class TestTensorAdapterBranches:
    def test_tensor_template_detects_torch_and_jax_like_inputs(self) -> None:
        torch_like = _TorchLikeTensor(np.array([1.0, 2.0]))
        jax_like = _JaxLikeTensor()

        torch_template = tensor_io.tensor_template(torch_like, jax_like)
        jax_template = tensor_io.tensor_template(jax_like)

        assert torch_template == tensor_io.TensorTemplate(kind="torch", source=torch_like)
        assert jax_template == tensor_io.TensorTemplate(kind="jax", source=jax_like)

    def test_as_float64_array_converts_torch_like_numpy_chain(self) -> None:
        torch_like = _TorchLikeTensor(np.array([1, 2], dtype=np.int64))

        converted = tensor_io.as_float64_array(torch_like)

        assert converted.dtype == np.float64
        np.testing.assert_array_equal(converted, np.array([1.0, 2.0]))

    def test_torch_like_value_without_numpy_returns_none(self) -> None:
        torch_like = type("TorchLike", (), {"__module__": "torch.fake"})()

        assert tensor_io._torch_tensor_to_numpy(torch_like) is None

    def test_restore_array_uses_torch_template_device(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []

        def as_tensor(value: object, **kwargs: object) -> tuple[object, dict[str, object]]:
            calls.append(kwargs)
            return value, kwargs

        monkeypatch.setattr(
            tensor_io.importlib,
            "import_module",
            lambda name: SimpleNamespace(as_tensor=as_tensor, float64="float64"),
        )
        source = _TorchLikeTensor(np.array([1.0, 2.0]))
        template = tensor_io.TensorTemplate(kind="torch", source=source)
        array = np.array([3.0, 4.0])

        restored = tensor_io.restore_array(array, template)

        assert restored == (array, {"dtype": "float32", "device": "cuda:0"})
        assert calls == [{"dtype": "float32", "device": "cuda:0"}]

    def test_restore_torch_without_device_uses_dtype_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []

        def as_tensor(value: object, **kwargs: object) -> tuple[object, dict[str, object]]:
            calls.append(kwargs)
            return value, kwargs

        monkeypatch.setattr(
            tensor_io.importlib,
            "import_module",
            lambda name: SimpleNamespace(as_tensor=as_tensor, float64="float64"),
        )
        source = SimpleNamespace(dtype="float32")
        array = np.array([1.0, 2.0])

        restored = tensor_io._restore_torch(array, source)

        assert restored == (array, {"dtype": "float32"})
        assert calls == [{"dtype": "float32"}]

    def test_restore_torch_requires_as_tensor(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            tensor_io.importlib,
            "import_module",
            lambda name: SimpleNamespace(float64="float64"),
        )

        with pytest.raises(RuntimeError, match="torch.as_tensor is unavailable"):
            tensor_io._restore_torch(np.zeros(2), object())

    def test_restore_jax_requires_asarray(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            tensor_io.importlib,
            "import_module",
            lambda name: SimpleNamespace(float64="float64"),
        )

        with pytest.raises(RuntimeError, match="jax.numpy.asarray is unavailable"):
            tensor_io._restore_jax(np.zeros(2))

    def test_restore_array_uses_jax_template(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[dict[str, object]] = []

        def asarray(value: object, **kwargs: object) -> tuple[object, dict[str, object]]:
            calls.append(kwargs)
            return value, kwargs

        monkeypatch.setattr(
            tensor_io.importlib,
            "import_module",
            lambda name: SimpleNamespace(asarray=asarray, float64="float64"),
        )
        array = np.array([1.0, 2.0])
        template = tensor_io.TensorTemplate(kind="jax", source=_JaxLikeTensor())

        restored = tensor_io.restore_array(array, template)

        assert restored == (array, {"dtype": "float64"})
        assert calls == [{"dtype": "float64"}]

    def test_call_method_returns_value_when_method_is_absent(self) -> None:
        value = object()

        assert tensor_io._call_method(value, "missing") is value
