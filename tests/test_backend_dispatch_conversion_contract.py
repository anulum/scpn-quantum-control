# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Backend Conversion Contract Tests
"""``to_numpy`` converts what it is given, not what the backend happens to be.

Two defects met here. ``np.array(arr, copy=False)`` means "never copy, raise
instead" under NumPy 2, so every ordinary sequence raised ``ValueError`` from a
method documented as converting any backend array. Separately, the conversion
was chosen from the module-global backend selection rather than from the object,
so an array that outlived a :func:`set_backend` call, or a caller passing an
object from a backend that is not currently selected, took the wrong path.

Protocol doubles here prove call ordering and interface dispatch only, not real
gradient, device or storage behaviour. Optional real CPU Torch and JAX contracts
live in test_backend_dispatch.py; their results require those actual frameworks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.backend_dispatch as dispatch_module
from scpn_quantum_control.backend_dispatch import to_numpy

AMBIENT_BACKENDS = ("numpy", "jax", "torch", "unknown")
"""Every backend label the module can hold, including one it never sets."""


class _ProtocolTensor:
    """Minimal object implementing the torch tensor conversion protocol.

    Records each step so a test can assert that detaching and the host move
    actually happened, rather than inferring it from the returned values.
    """

    def __init__(self, values: NDArray[np.float64], *, fail_with: type | None = None) -> None:
        """Store the payload and the optional conversion failure.

        Parameters
        ----------
        values
            Array the protocol finally yields.
        fail_with
            Exception type ``numpy()`` should raise instead of returning.

        """
        self.values = values
        self.fail_with = fail_with
        self.detached = False
        self.moved_to_host = False

    def detach(self) -> _ProtocolTensor:
        """Record the detach and return self.

        Returns
        -------
        _ProtocolTensor
            The same object, as a real tensor's ``detach`` returns a view.

        """
        self.detached = True
        return self

    def cpu(self) -> _ProtocolTensor:
        """Record the host move and return self.

        Returns
        -------
        _ProtocolTensor
            The same object, as a host tensor's ``cpu`` returns itself.

        """
        self.moved_to_host = True
        return self

    def numpy(self) -> NDArray[np.float64]:
        """Return the payload, or raise the configured failure.

        Returns
        -------
        numpy.ndarray
            The stored values.

        Raises
        ------
        Exception
            The configured ``fail_with`` type, if one was given.

        """
        if self.fail_with is not None:
            raise self.fail_with("dtype cannot be represented in NumPy")
        return self.values


class _ArrayLike:
    """Object exposing only ``__array__``, as a JAX array does."""

    def __init__(self, values: NDArray[np.float64]) -> None:
        """Store the payload.

        Parameters
        ----------
        values
            Array this object converts to.

        """
        self.values = values

    def __array__(self, dtype: Any = None, copy: Any = None) -> NDArray[np.float64]:
        """Return the payload for NumPy conversion.

        Parameters
        ----------
        dtype
            Requested dtype, honoured by delegating to ``numpy.asarray``.
        copy
            Copy request forwarded by NumPy; the payload is returned either way.

        Returns
        -------
        numpy.ndarray
            The stored values.

        """
        if dtype is None:
            return self.values
        return np.asarray(self.values, dtype=dtype)


class TestOrdinarySequences:
    """The recorded defect: ordinary sequences were refused."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([1.0, 2.0], [1.0, 2.0]),
            ((1.0, 2.0), [1.0, 2.0]),
            (range(3), [0, 1, 2]),
            ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),
            (3.0, 3.0),
        ],
    )
    def test_sequences_and_scalars_convert(self, value: Any, expected: Any) -> None:
        """Each of these raised ValueError before the copy policy changed.

        Parameters
        ----------
        value
            Input handed to the converter.
        expected
            Array the conversion must produce.

        """
        result = to_numpy(value)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.asarray(expected))

    def test_buffer_protocol_objects_convert(self) -> None:
        """A memoryview is converted without a copy of its buffer."""
        buffer = np.array([7.0, 8.0])

        result = to_numpy(memoryview(buffer.data))

        np.testing.assert_array_equal(result, buffer)


class TestDispatchIsIndependentOfAmbientState:
    """The second defect: the path was chosen from the module-global backend."""

    @pytest.mark.parametrize("backend", AMBIENT_BACKENDS)
    def test_a_list_converts_under_every_backend(
        self, backend: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Under ``torch`` this used to call ``detach`` on a list.

        Parameters
        ----------
        backend
            Ambient backend label to install.
        monkeypatch
            Used to set the module-global selection.

        """
        monkeypatch.setattr(dispatch_module._STATE, "backend", backend)

        np.testing.assert_array_equal(to_numpy([1.0, 2.0]), np.array([1.0, 2.0]))

    @pytest.mark.parametrize("backend", AMBIENT_BACKENDS)
    def test_a_tensor_converts_under_every_backend(
        self, backend: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The array-outlives-a-backend-switch case the card names.

        Parameters
        ----------
        backend
            Ambient backend label to install.
        monkeypatch
            Used to set the module-global selection.

        """
        monkeypatch.setattr(dispatch_module._STATE, "backend", backend)
        tensor = _ProtocolTensor(np.array([5.0]))

        np.testing.assert_array_equal(to_numpy(tensor), np.array([5.0]))
        assert tensor.detached

    def test_every_backend_gives_the_same_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One input, four ambient selections, one result.

        Parameters
        ----------
        monkeypatch
            Used to set the module-global selection.

        """
        results = []
        for backend in AMBIENT_BACKENDS:
            monkeypatch.setattr(dispatch_module._STATE, "backend", backend)
            results.append(to_numpy((1.5, -2.5)))

        for result in results[1:]:
            np.testing.assert_array_equal(result, results[0])


class TestNumpyInputsAreUnchanged:
    """A NumPy array is returned as it was handed in."""

    def test_an_array_is_returned_by_identity(self) -> None:
        """No copy, so the documented mutation visibility holds."""
        array = np.array([1.0, 2.0])

        assert to_numpy(array) is array

    def test_a_view_stays_a_view(self) -> None:
        """A strided view is not silently densified."""
        view = np.arange(6.0)[::2]

        result = to_numpy(view)

        assert result is view
        assert not result.flags["C_CONTIGUOUS"]

    def test_a_subclass_is_returned_unchanged(self) -> None:
        """``isinstance`` admits subclasses, and the identity holds for them."""

        class TaggedArray(np.ndarray[Any, np.dtype[np.float64]]):
            """Actual ndarray subclass used to verify identity-preserving dispatch."""

        matrix = np.asarray([[1.0, 2.0]]).view(TaggedArray)

        assert to_numpy(matrix) is matrix


class TestTensorProtocol:
    """Interface dispatch, and what it documents about gradients and devices."""

    def test_detaching_and_the_host_move_both_happen(self) -> None:
        """The docstring says the gradient history is dropped; this pins it."""
        tensor = _ProtocolTensor(np.array([4.0]))

        to_numpy(tensor)

        assert tensor.detached
        assert tensor.moved_to_host

    def test_conversion_errors_propagate_unchanged(self) -> None:
        """A dtype NumPy cannot represent raises from the source, not here."""
        tensor = _ProtocolTensor(np.array([1.0]), fail_with=TypeError)

        with pytest.raises(TypeError, match="dtype cannot be represented"):
            to_numpy(tensor)

    def test_an_array_like_object_does_not_take_the_tensor_path(self) -> None:
        """A JAX-shaped object has no ``detach`` and goes through asarray."""
        values = np.array([2.0, 4.0])

        result = to_numpy(_ArrayLike(values))

        np.testing.assert_array_equal(result, values)

    @pytest.mark.parametrize(
        "candidate",
        [
            np.array([1.0]),
            [1.0],
            (1.0,),
            3.0,
            "text",
        ],
    )
    def test_the_protocol_check_rejects_non_tensors(self, candidate: Any) -> None:
        """Only an object with all three callables takes the tensor path.

        Parameters
        ----------
        candidate
            Object that must not be treated as a tensor.

        """
        assert not dispatch_module._has_torch_tensor_protocol(candidate)

    def test_the_protocol_check_requires_every_attribute(self) -> None:
        """Two of three is not the protocol."""

        class _Partial:
            def detach(self) -> None:
                """Do nothing."""

            def cpu(self) -> None:
                """Do nothing."""

        assert not dispatch_module._has_torch_tensor_protocol(_Partial())

    def test_the_protocol_check_requires_callables(self) -> None:
        """Attributes that merely exist are not the protocol."""

        class _NotCallable:
            detach = 1
            cpu = 2
            numpy = 3

        assert not dispatch_module._has_torch_tensor_protocol(_NotCallable())

    def test_the_protocol_check_accepts_the_full_interface(self) -> None:
        """The positive case the other assertions are measured against."""
        assert dispatch_module._has_torch_tensor_protocol(_ProtocolTensor(np.array([1.0])))
