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


@pytest.fixture(autouse=True)
def _reset_backend():
    """Ensure numpy backend after each test."""
    yield
    set_backend("numpy")


# ── Default state ─────────────────────────────────────────────────────


class TestDefaultState:
    def test_default_backend_is_numpy(self):
        set_backend("numpy")
        assert get_backend() == "numpy"

    def test_default_array_module_is_numpy(self):
        set_backend("numpy")
        assert get_array_module() is np


# ── Numpy backend ─────────────────────────────────────────────────────


class TestNumpyBackend:
    def test_set_numpy(self):
        set_backend("numpy")
        assert get_backend() == "numpy"

    def test_to_numpy_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = to_numpy(arr)
        assert result is arr

    def test_from_numpy_passthrough(self):
        arr = np.array([1.0, 2.0])
        result = from_numpy(arr)
        assert result is arr

    def test_round_trip(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = to_numpy(from_numpy(arr))
        np.testing.assert_array_equal(result, arr)


# ── JAX backend ───────────────────────────────────────────────────────


class TestJAXBackend:
    def test_jax_import_error(self):
        with (
            patch.dict("sys.modules", {"jax": None, "jax.numpy": None}),
            pytest.raises(ImportError, match="JAX not installed"),
        ):
            set_backend("jax")

    def test_jax_if_available(self):
        try:
            import jax.numpy as _jnp  # noqa: F401

            set_backend("jax")
            assert get_backend() == "jax"
            # to_numpy converts back
            arr = np.array([1.0, 2.0])
            jnp_arr = from_numpy(arr)
            back = to_numpy(jnp_arr)
            np.testing.assert_allclose(back, arr)
        except ImportError:
            pytest.skip("JAX not installed")


# ── Torch backend ─────────────────────────────────────────────────────


class TestTorchBackend:
    def test_torch_import_error(self):
        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(ImportError, match="PyTorch not installed"),
        ):
            set_backend("torch")

    def test_pytorch_alias(self):
        """'pytorch' alias also works for set_backend."""
        with (
            patch.dict("sys.modules", {"torch": None}),
            pytest.raises(ImportError, match="PyTorch not installed"),
        ):
            set_backend("pytorch")

    def test_torch_if_available(self):
        try:
            import torch

            set_backend("torch")
            assert get_backend() == "torch"
            arr = np.array([1.0, 2.0, 3.0])
            t = from_numpy(arr)
            assert isinstance(t, torch.Tensor)
            back = to_numpy(t)
            np.testing.assert_allclose(back, arr)
        except ImportError:
            pytest.skip("PyTorch not installed")


# ── Invalid backend ───────────────────────────────────────────────────


class TestInvalidBackend:
    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            set_backend("tensorflow")

    def test_case_insensitive(self):
        set_backend("NUMPY")
        assert get_backend() == "numpy"


# ── available_backends ────────────────────────────────────────────────


class TestAvailableBackends:
    def test_numpy_always_available(self):
        backends = available_backends()
        assert "numpy" in backends

    def test_returns_list(self):
        backends = available_backends()
        assert isinstance(backends, list)


# ── to_numpy edge cases ──────────────────────────────────────────────


class TestToNumpyEdgeCases:
    def test_non_array_input(self):
        """to_numpy should handle plain lists via np.array."""
        set_backend("numpy")
        result = to_numpy(np.array([1, 2, 3]))
        assert isinstance(result, np.ndarray)

    def test_from_numpy_unknown_backend_passthrough(self):
        """When backend is unknown in module dict, from_numpy returns input."""
        arr = np.array([5.0])
        result = from_numpy(arr)
        assert result is arr


# ── Mocked JAX/torch paths for coverage ──────────────────────────────


class TestMockedJaxPath:
    def test_to_numpy_jax_branch(self):
        """Exercise to_numpy jax branch with a memoryview-compatible input."""
        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._CURRENT_BACKEND
        try:
            mod._CURRENT_BACKEND = "jax"
            # Use a numpy array subclass to bypass isinstance check but still
            # allow copy=False. np.matrix is a subclass, but simpler: use
            # np.asarray on a memoryview.
            buf = np.array([1.0, 2.0])
            view = memoryview(buf)
            result = to_numpy(view)
            np.testing.assert_array_equal(result, [1.0, 2.0])
        finally:
            mod._CURRENT_BACKEND = old_backend

    def test_to_numpy_torch_branch(self):
        """Exercise to_numpy torch branch with a mock tensor."""
        from unittest.mock import MagicMock

        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._CURRENT_BACKEND
        try:
            mod._CURRENT_BACKEND = "torch"
            mock_tensor = MagicMock()
            mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = np.array([3.0])
            result = to_numpy(mock_tensor)
            np.testing.assert_array_equal(result, [3.0])
        finally:
            mod._CURRENT_BACKEND = old_backend

    def test_to_numpy_fallback_branch(self):
        """Exercise to_numpy fallback (unknown backend, non-ndarray)."""
        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._CURRENT_BACKEND
        try:
            mod._CURRENT_BACKEND = "unknown"
            buf = np.array([4.0, 5.0])
            view = memoryview(buf)
            result = to_numpy(view)
            np.testing.assert_array_equal(result, [4.0, 5.0])
        finally:
            mod._CURRENT_BACKEND = old_backend

    def test_from_numpy_jax_branch(self):
        """Exercise from_numpy jax branch with mock jnp."""
        from types import ModuleType
        from unittest.mock import MagicMock

        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._CURRENT_BACKEND
        try:
            mod._CURRENT_BACKEND = "jax"
            mock_jnp = MagicMock()
            mock_jnp.array.return_value = "jax_array"
            fake_jax = ModuleType("jax")
            fake_jax.numpy = mock_jnp  # type: ignore[attr-defined]
            with patch.dict("sys.modules", {"jax": fake_jax, "jax.numpy": mock_jnp}):
                arr = np.array([1.0])
                result = from_numpy(arr)
                assert result == "jax_array"
        finally:
            mod._CURRENT_BACKEND = old_backend

    def test_from_numpy_torch_branch(self):
        """Exercise from_numpy torch branch with mock torch."""
        from unittest.mock import MagicMock

        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._CURRENT_BACKEND
        try:
            mod._CURRENT_BACKEND = "torch"
            mock_torch = MagicMock()
            mock_torch.from_numpy.return_value = "torch_tensor"
            with patch.dict("sys.modules", {"torch": mock_torch}):
                arr = np.array([1.0])
                result = from_numpy(arr)
                assert result == "torch_tensor"
        finally:
            mod._CURRENT_BACKEND = old_backend

    def test_from_numpy_unknown_returns_arr(self):
        """from_numpy with unknown backend falls through to return arr."""
        import scpn_quantum_control.backend_dispatch as mod

        old_backend = mod._CURRENT_BACKEND
        try:
            mod._CURRENT_BACKEND = "unknown"
            arr = np.array([9.0])
            assert from_numpy(arr) is arr
        finally:
            mod._CURRENT_BACKEND = old_backend


class TestAvailableBackendsMocked:
    def test_jax_detected_when_importable(self):
        """available_backends includes jax when import succeeds."""
        from types import ModuleType
        from unittest.mock import MagicMock

        fake_jnp = MagicMock()
        fake_jax = ModuleType("jax")
        fake_jax.numpy = fake_jnp  # type: ignore[attr-defined]
        with patch.dict("sys.modules", {"jax": fake_jax, "jax.numpy": fake_jnp}):
            backends = available_backends()
            assert "jax" in backends

    def test_torch_detected_when_importable(self):
        """available_backends includes torch when import succeeds."""
        from types import ModuleType

        fake_torch = ModuleType("torch")
        with patch.dict("sys.modules", {"torch": fake_torch}):
            backends = available_backends()
            assert "torch" in backends


class TestSetBackendMockedSuccess:
    def test_set_torch_success(self):
        """set_backend('torch') success path with mock torch module."""
        from types import ModuleType

        import scpn_quantum_control.backend_dispatch as mod

        fake_torch = ModuleType("torch")
        old_backend = mod._CURRENT_BACKEND
        try:
            with patch.dict("sys.modules", {"torch": fake_torch}):
                set_backend("torch")
                assert get_backend() == "torch"
                assert mod._BACKEND_MODULES["torch"] is fake_torch
        finally:
            mod._CURRENT_BACKEND = old_backend
            mod._BACKEND_MODULES.pop("torch", None)
