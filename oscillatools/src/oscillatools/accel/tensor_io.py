# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Optional tensor input/output adapters
"""Optional tensor input/output adapters for public Kuramoto hot paths.

The numerical kernels in :mod:`oscillatools.accel` run on a NumPy float64 floor
and then dispatch to optional Rust or Julia implementations. This module keeps
that floor intact while accepting Torch and JAX tensors at facade boundaries:
inputs are normalised to contiguous NumPy arrays for the existing kernels, and
array outputs are restored to the first tensor namespace seen in the call.

No optional backend is imported at package import time. Torch/JAX modules are
looked up only when an object from that namespace is already present.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

TensorKind = Literal["jax", "torch"]


@dataclass(frozen=True)
class TensorTemplate:
    """Tensor namespace and source object used to restore array outputs.

    Parameters
    ----------
    kind:
        Optional tensor backend detected from a call input.
    source:
        The tensor object whose device and dtype should be mirrored where the
        backend exposes that metadata.
    """

    kind: TensorKind
    source: object


def as_float64_array(value: object) -> NDArray[np.float64]:
    """Return ``value`` as a contiguous NumPy float64 array.

    Parameters
    ----------
    value:
        A NumPy-like input, Python sequence, JAX array, or Torch tensor.

    Returns
    -------
    numpy.ndarray
        Contiguous float64 array consumed by the existing accelerator kernels.
    """
    raw_value: object = _torch_tensor_to_numpy(value)
    if raw_value is None:
        raw_value = value
    return np.ascontiguousarray(raw_value, dtype=np.float64)


def tensor_template(*values: object) -> TensorTemplate | None:
    """Return the first Torch or JAX tensor template among ``values``.

    Parameters
    ----------
    *values:
        Public-call inputs in precedence order. The first detected tensor
        controls the output namespace.

    Returns
    -------
    TensorTemplate | None
        Template for restoring array outputs, or ``None`` when every input is a
        plain NumPy/Python value.
    """
    for value in values:
        kind = _tensor_kind(value)
        if kind is not None:
            return TensorTemplate(kind=kind, source=value)
    return None


def restore_array(
    array: NDArray[np.float64],
    template: TensorTemplate | None,
) -> NDArray[np.float64] | Any:
    """Return ``array`` in the namespace described by ``template``.

    Parameters
    ----------
    array:
        NumPy float64 result from the accelerator dispatch chain.
    template:
        Optional tensor template captured from the public call inputs.

    Returns
    -------
    numpy.ndarray | Any
        The original NumPy array when ``template`` is absent, otherwise a Torch
        or JAX tensor created lazily from the NumPy result.
    """
    result = np.ascontiguousarray(array, dtype=np.float64)
    if template is None:
        return result
    if template.kind == "torch":
        return _restore_torch(result, template.source)
    return _restore_jax(result)


def restore_array_tuple(
    arrays: tuple[NDArray[np.float64], ...],
    template: TensorTemplate | None,
) -> tuple[NDArray[np.float64] | Any, ...]:
    """Return a tuple of arrays restored with :func:`restore_array`.

    Parameters
    ----------
    arrays:
        NumPy float64 result channels.
    template:
        Optional tensor template captured from the public call inputs.

    Returns
    -------
    tuple
        Tuple whose array channels follow the same output namespace.
    """
    return tuple(restore_array(array, template) for array in arrays)


def _tensor_kind(value: object) -> TensorKind | None:
    module = type(value).__module__
    if module == "torch" or module.startswith("torch."):
        return "torch"
    if module.startswith("jax.") or module.startswith("jaxlib."):
        return "jax"
    return None


def _torch_tensor_to_numpy(value: object) -> NDArray[Any] | None:
    if _tensor_kind(value) != "torch":
        return None
    detached = _call_method(value, "detach")
    cpu_value = _call_method(detached, "cpu")
    numpy_method = getattr(cpu_value, "numpy", None)
    if not callable(numpy_method):
        return None
    return np.asarray(numpy_method())


def _restore_torch(array: NDArray[np.float64], source: object) -> Any:
    torch_module = importlib.import_module("torch")
    dtype = getattr(source, "dtype", getattr(torch_module, "float64", None))
    device = getattr(source, "device", None)
    as_tensor = getattr(torch_module, "as_tensor", None)
    if not callable(as_tensor):
        raise RuntimeError("torch.as_tensor is unavailable")
    if device is None:
        return as_tensor(array, dtype=dtype)
    return as_tensor(array, dtype=dtype, device=device)


def _restore_jax(array: NDArray[np.float64]) -> Any:
    jax_numpy = importlib.import_module("jax.numpy")
    asarray = getattr(jax_numpy, "asarray", None)
    if not callable(asarray):
        raise RuntimeError("jax.numpy.asarray is unavailable")
    return asarray(array, dtype=getattr(jax_numpy, "float64", None))


def _call_method(value: object, name: str) -> object:
    method = getattr(value, name, None)
    if callable(method):
        return method()
    return value


__all__ = [
    "TensorTemplate",
    "as_float64_array",
    "restore_array",
    "restore_array_tuple",
    "tensor_template",
]
