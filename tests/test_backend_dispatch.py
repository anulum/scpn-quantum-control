# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Backend Dispatch
"""Tests for multi-backend tensor dispatch module.

Covers:
    - Default backend state
    - set_backend / get_backend / get_array_module for numpy
    - set_backend for jax and torch (import error handling)
    - to_numpy / from_numpy round-trips
    - available_backends detection
    - Invalid backend rejection
    - State isolation (reset after test)
"""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from scpn_quantum_control.backend_dispatch import (
    available_backends,
    from_numpy,
    get_array_module,
    get_backend,
    set_backend,
    to_numpy,
)


class _JaxModule(ModuleType):
    """Import-boundary double with an explicit dynamically mocked numpy member."""

    numpy: Any


@pytest.fixture(autouse=True)
def _reset_backend() -> Iterator[None]:
    """Ensure numpy backend after each test."""
    yield
    set_backend("numpy")


# ── Default state ─────────────────────────────────────────────────────


class TestDefaultState:
    def test_default_backend_is_numpy(self) -> None:
        set_backend("numpy")
        assert get_backend() == "numpy"

    def test_default_array_module_is_numpy(self) -> None:
        set_backend("numpy")
        assert get_array_module() is np


# ── Numpy backend ─────────────────────────────────────────────────────


class TestNumpyBackend:
    def test_set_numpy(self) -> None:
        set_backend("numpy")
        assert get_backend() == "numpy"

    def test_to_numpy_passthrough(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert result is arr

    def test_from_numpy_passthrough(self) -> None:
        arr = np.array([1.0, 2.0])
        result = from_numpy(arr)
        assert result is arr

    def test_round_trip(self) -> None:
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = to_numpy(from_numpy(arr))
        np.testing.assert_array_equal(result, arr)


# ── JAX backend ───────────────────────────────────────────────────────


class TestJAXBackend:
    def test_jax_import_error(self) -> None:
        with (
            patch.dict("sys.modules", {"jax": None, "jax.numpy": None}),
            pytest.raises(ImportError, match="JAX unavailable"),
        ):
            set_backend("jax")

    def test_jax_if_available(self) -> None:
        """A real JAX array retained across a backend switch exports its values."""
        try:
            import jax.numpy as jnp
        except ImportError as exc:
            pytest.skip(f"JAX unavailable in this environment: {exc}")
        set_backend("jax")
        assert get_array_module() is jnp
        arr = np.array([1.0, 2.0], dtype=np.float32)
        jnp_arr = from_numpy(arr)
        set_backend("numpy")
        back = to_numpy(jnp_arr)
        np.testing.assert_array_equal(back, arr)


# ── Torch backend ─────────────────────────────────────────────────────


class TestTorchBackend:
    """Optional real CPU tensor conversion and unavailable-provider errors."""

    def test_torch_import_error(self) -> None:
        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(ImportError, match="PyTorch not installed"),
        ):
            set_backend("torch")

    def test_pytorch_alias(self) -> None:
        """'pytorch' alias also works for set_backend."""
        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(ImportError, match="PyTorch not installed"),
        ):
            set_backend("pytorch")

    @pytest.mark.parametrize("view", ["plain", "conjugate", "negative", "transpose"])
    def test_torch_if_available(self, view: str) -> None:
        """Real CPU views export values, preserve autograd and honour copy boundaries."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        set_backend("torch")
        arr = np.array([[1 + 2j, 3 - 4j]], dtype=np.complex128)
        tensor = from_numpy(arr.copy()).requires_grad_()
        assert isinstance(tensor, torch.Tensor)
        if view == "conjugate":
            tensor = tensor.conj()
            expected = arr.conj()
        elif view == "negative":
            tensor = torch._neg_view(tensor)
            expected = -arr
        elif view == "transpose":
            tensor = tensor.T
            expected = arr.T
        else:
            expected = arr
        set_backend("numpy")
        back = to_numpy(tensor)
        np.testing.assert_array_equal(back, expected)
        assert tensor.requires_grad
        gradient = torch.autograd.grad(tensor.real.sum(), tensor)[0]
        torch.testing.assert_close(gradient, torch.ones_like(tensor))
        if view in ("plain", "transpose"):
            assert np.shares_memory(back, tensor.detach().numpy())
        else:
            back.flat[0] = 99
            np.testing.assert_array_equal(
                tensor.detach().resolve_conj().resolve_neg().numpy(), expected
            )


# ── Invalid backend ───────────────────────────────────────────────────


class TestInvalidBackend:
    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("tensorflow")

    def test_case_insensitive(self) -> None:
        set_backend("NUMPY")
        assert get_backend() == "numpy"


# ── available_backends ────────────────────────────────────────────────


class TestAvailableBackends:
    def test_numpy_always_available(self) -> None:
        backends = available_backends()
        assert "numpy" in backends

    def test_returns_list(self) -> None:
        backends = available_backends()
        assert isinstance(backends, list)


# ── to_numpy edge cases ──────────────────────────────────────────────


class TestToNumpyEdgeCases:
    def test_non_array_input(self) -> None:
        """to_numpy should handle plain lists.

        This test claimed list handling but passed an ndarray, so it could not
        under NumPy 2 the old ``copy=False`` policy raised
        ValueError for every ordinary sequence. It now passes what it says.
        """
        set_backend("numpy")
        result = to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array([1, 2, 3]))

    def test_from_numpy_unknown_backend_passthrough(self) -> None:
        """When backend is unknown in module dict, from_numpy returns input."""
        arr = np.array([5.0])
        result = from_numpy(arr)
        assert result is arr


# ── JAX and torch dispatch fallback paths ──────────────────────────────


class TestMockedJaxPath:
    """``to_numpy`` no longer has per-backend branches.

    These three exercised branches keyed on the module-global backend.
    That was replaced with dispatch on the object handed in, so they
    now assert what actually decides the path. They are kept rather than deleted
    because each still covers a real input shape; the full contract has a
    dedicated owner in ``tests/test_backend_dispatch_conversion_contract.py``.
    """

    def test_to_numpy_converts_a_buffer_under_the_jax_selection(self) -> None:
        """A memoryview converts, and the jax selection does not change that."""
        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._STATE.backend
        try:
            mod._STATE.backend = "jax"
            buf = np.array([1.0, 2.0])
            result = to_numpy(buf.data)
            np.testing.assert_array_equal(result, [1.0, 2.0])
        finally:
            mod._STATE.backend = old_backend

    def test_to_numpy_uses_the_tensor_protocol_not_the_selection(self) -> None:
        """A mock exposing detach/cpu/numpy takes the tensor path."""
        from unittest.mock import MagicMock

        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._STATE.backend
        try:
            # Deliberately the numpy selection: the object decides, not this.
            mod._STATE.backend = "numpy"
            mock_tensor = MagicMock(spec=["detach", "cpu", "numpy"])
            mock_tensor.detach.return_value = mock_tensor
            mock_tensor.cpu.return_value = mock_tensor
            mock_tensor.numpy.return_value = np.array([3.0])
            result = to_numpy(mock_tensor)
            np.testing.assert_array_equal(result, [3.0])
            mock_tensor.detach.assert_called_once()
        finally:
            mod._STATE.backend = old_backend

    def test_to_numpy_converts_under_an_unknown_selection(self) -> None:
        """An unrecognised selection is simply irrelevant to the conversion."""
        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._STATE.backend
        try:
            mod._STATE.backend = "unknown"
            buf = np.array([4.0, 5.0])
            result = to_numpy(buf.data)
            np.testing.assert_array_equal(result, [4.0, 5.0])
        finally:
            mod._STATE.backend = old_backend

    def test_from_numpy_jax_branch(self) -> None:
        """Exercise from_numpy jax branch with mock jnp."""
        from unittest.mock import MagicMock

        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._STATE.backend
        try:
            mod._STATE.backend = "jax"
            mock_jnp = MagicMock()
            mock_jnp.array.return_value = "jax_array"
            fake_jax = _JaxModule("jax")
            fake_jax.numpy = mock_jnp
            with patch.dict("sys.modules", {"jax": fake_jax, "jax.numpy": mock_jnp}):
                arr = np.array([1.0])
                result = from_numpy(arr)
                assert result == "jax_array"
        finally:
            mod._STATE.backend = old_backend

    def test_from_numpy_torch_branch(self) -> None:
        """Exercise from_numpy torch branch with mock torch."""
        from unittest.mock import MagicMock

        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._STATE.backend
        try:
            mod._STATE.backend = "torch"
            mock_torch = MagicMock()
            mock_torch.from_numpy.return_value = "torch_tensor"
            with patch.dict("sys.modules", {"torch": mock_torch}):
                arr = np.array([1.0])
                result = from_numpy(arr)
                assert result == "torch_tensor"
        finally:
            mod._STATE.backend = old_backend

    def test_from_numpy_unknown_returns_arr(self) -> None:
        """from_numpy with unknown backend falls through to return arr."""
        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._STATE.backend
        try:
            mod._STATE.backend = "unknown"
            arr = np.array([9.0])
            assert from_numpy(arr) is arr
        finally:
            mod._STATE.backend = old_backend


class TestAvailableBackendsMocked:
    def test_jax_detected_when_importable(self) -> None:
        """available_backends includes jax when import succeeds."""
        from unittest.mock import MagicMock

        fake_jnp = MagicMock()
        fake_jax = _JaxModule("jax")
        fake_jax.numpy = fake_jnp
        with patch.dict("sys.modules", {"jax": fake_jax, "jax.numpy": fake_jnp}):
            backends = available_backends()
            assert "jax" in backends

    def test_torch_detected_when_importable(self) -> None:
        """available_backends includes torch when import succeeds."""
        from types import ModuleType

        fake_torch = ModuleType("torch")
        with patch.dict("sys.modules", {"torch": fake_torch}):
            backends = available_backends()
            assert "torch" in backends

    def test_torch_absence_is_ignored(self) -> None:
        """available_backends ignores an unavailable torch import."""
        with patch.dict("sys.modules", {"torch": None}):
            backends = available_backends()
            assert "numpy" in backends
            assert "torch" not in backends


class TestSetBackendMockedSuccess:
    def test_set_jax_success(self) -> None:
        """set_backend('jax') success path with a mock jax.numpy module."""
        from unittest.mock import MagicMock

        import scpn_quantum_control.backend_dispatch as mod

        fake_jnp = MagicMock()
        fake_jax = _JaxModule("jax")
        fake_jax.numpy = fake_jnp
        old_backend = mod._STATE.backend
        try:
            with patch.dict("sys.modules", {"jax": fake_jax, "jax.numpy": fake_jnp}):
                set_backend("jax")
                assert get_backend() == "jax"
                assert mod._BACKEND_MODULES["jax"] is fake_jnp
        finally:
            mod._STATE.backend = old_backend
            mod._BACKEND_MODULES.pop("jax", None)

    def test_set_torch_success(self) -> None:
        """set_backend('torch') success path with mock torch module."""
        from types import ModuleType

        import scpn_quantum_control.backend_dispatch as mod

        fake_torch = ModuleType("torch")
        old_backend = mod._STATE.backend
        try:
            with patch.dict("sys.modules", {"torch": fake_torch}):
                set_backend("torch")
                assert get_backend() == "torch"
                assert mod._BACKEND_MODULES["torch"] is fake_torch
        finally:
            mod._STATE.backend = old_backend
            mod._BACKEND_MODULES.pop("torch", None)
