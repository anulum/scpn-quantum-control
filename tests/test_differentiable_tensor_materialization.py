# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native host materialization regressions
"""Preserve host fallbacks, descriptor errors and original helper identities."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.differentiable_tensor_materialization import (
    TensorMaterializationPrimitives,
    materialize_tensorflow_host_values,
    materialize_torch_host_values,
)
from scpn_quantum_control.phase import (
    tensorflow_bridge,
    tensorflow_compatibility,
    tensorflow_gradients,
    torch_bridge,
    torch_compatibility,
    torch_gradients,
    torch_qnode_transforms,
)


def _primitives() -> TensorMaterializationPrimitives:
    return TensorMaterializationPrimitives(attribute=lambda: getattr, predicate=lambda: callable)


def test_host_values_keep_identity_without_conversion_capabilities() -> None:
    """Host fallback does not copy or coerce arrays, sequences or absent methods."""
    for values in (object(), [0.2], np.array([0.2])):
        assert materialize_torch_host_values(values, primitives=_primitives()) is values
        assert materialize_tensorflow_host_values(values, primitives=_primitives()) is values


def test_conversion_descriptor_exception_propagates_unchanged() -> None:
    """Capability lookup exposes the original exception instance."""
    failure = RuntimeError("host view unavailable")

    class Unavailable:
        @property
        def numpy(self) -> object:
            raise failure

    for convert in (materialize_torch_host_values, materialize_tensorflow_host_values):
        with pytest.raises(RuntimeError) as caught:
            convert(Unavailable(), primitives=_primitives())
        assert caught.value is failure


def test_original_private_aliases_keep_function_and_global_identity() -> None:
    """Consumers retain natural original functions and their live dictionaries."""
    for owner, consumers, name in (
        (torch_gradients, (torch_bridge, torch_compatibility), "_torch_values_to_numpy"),
        (
            tensorflow_gradients,
            (tensorflow_bridge, tensorflow_compatibility),
            "_tensorflow_values_to_numpy",
        ),
        (torch_qnode_transforms, (torch_bridge,), "_torch_to_numpy_compatible"),
    ):
        original = getattr(owner, name)
        assert original.__globals__ is vars(owner)
        assert original.__module__ == owner.__name__
        assert original.__qualname__ == name
        for consumer in consumers:
            assert getattr(consumer, name) is original


@pytest.mark.parametrize("shape", [(), (1,)])
def test_original_torch_scalar_shape_contract_survives(shape: tuple[int, ...]) -> None:
    """Both original scalar-like shapes remain admitted.

    Parameters
    ----------
    shape
        Scalar or singleton-vector shape admitted by the original helper.

    """
    assert torch_qnode_transforms._torch_scalar_to_float(np.full(shape, 0.2)) == 0.2


def test_original_vector_copy_and_error_payloads_survive() -> None:
    """Old helpers retain fresh storage and their vector/scalar refusals."""
    source = np.array([0.2])
    for convert in (
        torch_gradients._torch_values_to_numpy,
        tensorflow_gradients._tensorflow_values_to_numpy,
    ):
        actual = convert(source)
        np.testing.assert_array_equal(actual, source)
        assert not np.shares_memory(actual, source)
        with pytest.raises(ValueError, match="values must be a one-dimensional array"):
            convert([[0.2]])
    with pytest.raises(ValueError, match="PyTorch scalar value must be scalar-like"):
        torch_qnode_transforms._torch_scalar_to_float(np.array([[0.2]]))
    with pytest.raises(ValueError, match="PyTorch scalar value must be finite"):
        torch_qnode_transforms._torch_scalar_to_float(np.array([np.inf]))
