# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Multi-Backend Tensor Dispatch
"""Runtime backend selection for array operations.

Allows switching between numpy, JAX, and PyTorch backends at runtime:

    from scpn_quantum_control.backend_dispatch import set_backend, get_backend
    set_backend("jax")   # use JAX for all array ops
    set_backend("torch") # use PyTorch
    set_backend("numpy") # default

Inspired by TensorCircuit (Tencent Quantum Lab, arXiv:2205.10091).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray


class _DispatchState:
    """One-slot mutable holder for the active backend selection."""

    backend: str = "numpy"


_STATE = _DispatchState()
_BACKEND_MODULES: dict[str, Any] = {"numpy": np}


def set_backend(name: str) -> None:
    """Set the active array backend.

    Supported: "numpy" (default), "jax", "torch".
    """
    name = name.lower()

    if name == "numpy":
        _STATE.backend = "numpy"
        _BACKEND_MODULES["numpy"] = np
    elif name == "jax":
        try:
            import jax.numpy as jnp

            _BACKEND_MODULES["jax"] = jnp
            _STATE.backend = "jax"
        except Exception as e:
            raise ImportError("JAX unavailable: install compatible jax and jaxlib") from e
    elif name in ("torch", "pytorch"):
        try:
            import torch

            _BACKEND_MODULES["torch"] = torch
            _STATE.backend = "torch"
        except ImportError as e:
            raise ImportError("PyTorch not installed: pip install torch") from e
    else:
        raise ValueError(f"Unknown backend: {name}. Use 'numpy', 'jax', or 'torch'.")


def get_backend() -> str:
    """Return the name of the current backend."""
    return _STATE.backend


def get_array_module() -> Any:
    """Return the current array module (np, jnp, or torch)."""
    return _BACKEND_MODULES.get(_STATE.backend, np)


_TORCH_TENSOR_PROTOCOL: Final = ("detach", "cpu", "numpy")
"""Attributes a tensor must expose to be converted through the torch path."""


def _has_torch_tensor_protocol(arr: Any) -> bool:
    """Return whether ``arr`` exposes the torch tensor conversion protocol.

    Dispatch is by interface rather than by the ambient backend selection,
    because the two disagree whenever an array outlives a backend switch or a
    caller passes an object from a backend that is not currently selected.
    NumPy arrays are handled before this check and JAX arrays do not carry
    ``detach``, so the three attributes together identify a torch-like tensor
    without importing torch.

    Parameters
    ----------
    arr
        Candidate object.

    Returns
    -------
    bool
        True when every protocol attribute is present and callable.

    """
    return all(callable(getattr(arr, name, None)) for name in _TORCH_TENSOR_PROTOCOL)


def to_numpy(arr: Any) -> NDArray[Any]:
    """Convert a backend array, tensor or ordinary sequence to NumPy.

    The conversion is chosen from the object handed in, not from the backend
    that happens to be selected. An array created under one backend and
    converted after :func:`set_backend` therefore converts correctly, and a
    plain list does not depend on the backend at all.

    Copies are made where a copy is necessary. Under NumPy 2 the previous
    ``np.array(arr, copy=False)`` meant "never copy, raise instead", so every
    ordinary sequence raised ``ValueError`` from a method documented as
    converting any backend array.

    What each input costs:

    - A ``numpy.ndarray`` is returned unchanged, including a view. No copy is
      made and later mutations of the result are visible through the input.
    - A torch-like tensor is detached from the autograd graph, moved to host
      memory and viewed as an array. **The gradient history is dropped**; keep
      the original tensor if it is needed. ``.cpu()`` copies a device tensor
      and is a no-op for one already on the host, in which case the returned
      array shares memory with the tensor.
      Lazy conjugation or negation is resolved before export and requires a
      copy when present. The original tensor and its autograd graph are unchanged.
    - Anything else, including JAX arrays, lists, tuples, ranges and scalars,
      goes through :func:`numpy.asarray`, which copies only when it must.

    Parameters
    ----------
    arr
        Backend array, tensor, or any object NumPy can interpret as an array.

    Returns
    -------
    numpy.ndarray
        NumPy view or copy of ``arr``.

    Raises
    ------
    Exception
        Conversion errors from the source framework propagate unchanged; a
        torch dtype NumPy cannot represent raises from torch, not from here.

    """
    if isinstance(arr, np.ndarray):
        return arr
    if _has_torch_tensor_protocol(arr):
        host = arr.detach().cpu()
        for name in ("resolve_conj", "resolve_neg"):
            resolve = getattr(host, name, None)
            if callable(resolve):
                host = resolve()
        detached: NDArray[Any] = host.numpy()
        return detached
    return np.asarray(arr)


def from_numpy(arr: NDArray[Any]) -> Any:
    """Convert numpy array to current backend."""
    if _STATE.backend == "numpy":
        return arr
    if _STATE.backend == "jax":
        import jax.numpy as jnp

        return jnp.array(arr)
    if _STATE.backend == "torch":
        import torch

        return torch.from_numpy(arr)
    return arr


def available_backends() -> list[str]:
    """List all available backends on this system."""
    backends = ["numpy"]
    try:
        import jax.numpy as _jnp

        del _jnp
        backends.append("jax")
    except Exception as exc:
        del exc
    try:
        import torch

        del torch  # used only for availability check
        backends.append("torch")
    except Exception as exc:
        del exc
    return backends
