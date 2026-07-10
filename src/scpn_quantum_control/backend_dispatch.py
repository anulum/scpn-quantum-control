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

from typing import Any

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


def to_numpy(arr: Any) -> NDArray[Any]:
    """Convert any backend array to numpy."""
    if isinstance(arr, np.ndarray):
        return arr
    if _STATE.backend == "jax":
        return np.array(arr, copy=False)
    if _STATE.backend == "torch":
        result: NDArray[Any] = arr.detach().cpu().numpy()
        return result
    return np.array(arr, copy=False)


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
