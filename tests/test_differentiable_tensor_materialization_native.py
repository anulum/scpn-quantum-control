# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native host materialization regressions
"""Exercise real CPU SDK host views, tape lifetimes and public gradient routes."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from scpn_quantum_control.differentiable_tensor_materialization import (
    TensorMaterializationPrimitives,
    materialize_tensorflow_host_values,
    materialize_torch_host_values,
)
from scpn_quantum_control.phase import (
    tensorflow_gradients,
    torch_gradients,
    torch_qnode_transforms,
)

pytestmark = pytest.mark.framework_imports


def _primitives() -> TensorMaterializationPrimitives:
    return TensorMaterializationPrimitives(attribute=lambda: getattr, predicate=lambda: callable)


def test_real_torch_host_view_shares_storage_without_detaching_original_graph() -> None:
    """Host exposure shares storage while the original graph remains usable."""
    torch = importlib.import_module("torch")
    source = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    nonleaf = source * 2
    actual = materialize_torch_host_values(nonleaf, primitives=_primitives())
    assert isinstance(actual, np.ndarray)
    assert np.shares_memory(actual, nonleaf.detach().numpy())
    np.testing.assert_array_equal(actual, [0.4, -0.8])
    copy = torch_gradients._torch_values_to_numpy(nonleaf)
    assert not np.shares_memory(copy, actual)
    nonleaf.square().sum().backward()
    np.testing.assert_allclose(source.grad.numpy(), [1.6, -3.2])
    actual[0] = 0.7
    assert nonleaf.detach().numpy()[0] == 0.7
    np.testing.assert_array_equal(source.detach().numpy(), [0.2, -0.4])


def test_real_tensorflow_materialization_keeps_variable_and_tape_live() -> None:
    """Eager materialization leaves native variables and their tape connected."""
    tensorflow = importlib.import_module("tensorflow")
    source = tensorflow.Variable([0.2, -0.4], dtype=tensorflow.float64)
    with tensorflow.GradientTape() as tape:
        host = materialize_tensorflow_host_values(source, primitives=_primitives())
        loss = tensorflow.reduce_sum(tensorflow.square(source))
    assert isinstance(host, np.ndarray)
    np.testing.assert_array_equal(host, [0.2, -0.4])
    np.testing.assert_allclose(tape.gradient(loss, source).numpy(), [0.4, -0.8])
    host[0] = 0.7
    np.testing.assert_array_equal(source.numpy(), [0.2, -0.4])
    copy = tensorflow_gradients._tensorflow_values_to_numpy(source)
    np.testing.assert_array_equal(copy, source.numpy())


@pytest.mark.parametrize(
    ("module_name", "name"),
    [
        ("torch_gradients", "_torch_values_to_numpy"),
        ("torch_qnode_transforms", "_torch_to_numpy_compatible"),
    ],
)
def test_real_torch_conversion_keeps_live_caller_lookup_and_predicate(
    monkeypatch: pytest.MonkeyPatch, module_name: str, name: str
) -> None:
    """Original aliases observe rebinding between native conversion operations.

    Parameters
    ----------
    monkeypatch
        Restore the original caller globals after rebinding probes.
    module_name
        Original module that owns the native conversion helper.
    name
        Natural original helper name.

    """
    torch = importlib.import_module("torch")
    owner = importlib.import_module("scpn_quantum_control.phase." + module_name)
    reads: list[str] = []

    def later(value: object, attr: str, default: object) -> object:
        reads.append("later:" + attr)
        return getattr(value, attr, default)

    def predicate(value: object) -> bool:
        reads.append("predicate")
        return callable(value)

    def first(value: object, attr: str, default: object) -> object:
        reads.append("first:" + attr)
        monkeypatch.setattr(owner, "getattr", later, raising=False)
        monkeypatch.setattr(owner, "callable", predicate, raising=False)
        return getattr(value, attr, default)

    monkeypatch.setattr(owner, "getattr", first, raising=False)
    monkeypatch.setattr(owner, "callable", callable, raising=False)
    actual = getattr(owner, name)(torch.tensor([0.2], requires_grad=True))
    np.testing.assert_allclose(actual, [0.2])
    assert reads == [
        "first:detach",
        "predicate",
        "later:cpu",
        "predicate",
        "later:numpy",
        "predicate",
    ]


def test_public_gradient_routes_accept_real_trainable_parameters() -> None:
    """Original public host adapters consume real SDK tensors and return native gradients."""
    torch = importlib.import_module("torch")
    tensorflow = importlib.import_module("tensorflow")
    torch_result = torch_gradients.torch_parameter_shift_value_and_grad(
        lambda values: float(np.cos(values[0])),
        torch.tensor([0.2], dtype=torch.float64, requires_grad=True),
    )
    tf_result = tensorflow_gradients.tensorflow_parameter_shift_value_and_grad(
        lambda values: float(np.cos(values[0])),
        tensorflow.Variable([0.2], dtype=tensorflow.float64),
        _tensorflow_loader=lambda: tensorflow,
    )
    for result in (torch_result, tf_result):
        np.testing.assert_allclose(result.gradient, [-np.sin(0.2)])
    assert torch.is_tensor(torch_result.torch_gradient)
    assert tf_result.tensorflow_gradient.dtype == tensorflow.float64


def test_real_torch_singleton_scalar_and_matrix_host_routes() -> None:
    """Original QNode scalar and matrix consumers keep their native input contract."""
    torch = importlib.import_module("torch")
    for shape in ((), (1,)):
        tensor = torch.full(shape, 0.2, dtype=torch.float64, requires_grad=True)
        assert torch_qnode_transforms._torch_scalar_to_float(tensor) == 0.2
    tensor = torch.tensor([[0.2, -0.4]], dtype=torch.float64, requires_grad=True)
    actual = torch_qnode_transforms._torch_matrix_to_numpy("matrix", tensor)
    np.testing.assert_array_equal(actual, [[0.2, -0.4]])
    assert not np.shares_memory(actual, tensor.detach().numpy())


def test_real_tensorflow_host_lookup_observes_rebound_caller_predicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host conversion reads the caller predicate after the native attribute lookup.

    Parameters
    ----------
    monkeypatch
        Restore the original TensorFlow helper globals after the probe.

    """
    tensorflow = importlib.import_module("tensorflow")
    reads: list[str] = []

    def predicate(value: object) -> bool:
        reads.append("predicate")
        return callable(value)

    def lookup(value: object, name: str, default: object) -> object:
        reads.append(name)
        monkeypatch.setattr(tensorflow_gradients, "callable", predicate, raising=False)
        return getattr(value, name, default)

    monkeypatch.setattr(tensorflow_gradients, "getattr", lookup, raising=False)
    actual = tensorflow_gradients._tensorflow_values_to_numpy(
        tensorflow.Variable([0.2], dtype=tensorflow.float64)
    )
    np.testing.assert_array_equal(actual, [0.2])
    assert reads == ["numpy", "predicate"]


def test_real_torch_native_host_conversion_refusal_propagates() -> None:
    """An unresolved native conjugate view raises the original NumPy refusal."""
    torch = importlib.import_module("torch")
    source = torch.tensor([1 + 2j], dtype=torch.complex128).conj()
    for convert in (
        torch_gradients._torch_values_to_numpy,
        torch_qnode_transforms._torch_to_numpy_compatible,
    ):
        with pytest.raises(RuntimeError, match="conjugate bit set"):
            convert(source)
