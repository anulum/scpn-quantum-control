# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""GPU acceleration via cupy for matrix-heavy quantum operations.

Offloads eigendecomposition, matrix exponentials, and dense linear
algebra to the local CUDA GPU (GTX 1060 6GB) when cupy is available.
Falls back to numpy/scipy transparently.

Accelerated operations:
    - eigvalsh: eigenvalues of Hermitian matrix (used in exact_diag, BKT, QFI)
    - eigh: eigenvalues + eigenvectors (used in ground_state, robustness)
    - expm: matrix exponential (used in OTOC, AVQDS, Lindblad)
    - matmul: dense matrix multiply (used in Lindblad, correlators)

Usage: import from here instead of numpy/scipy for GPU-eligible ops.
"""

from __future__ import annotations

import os as _os

import numpy as np

_CUPY_AVAILABLE = False
_cp = None

# GPU disabled by default unless SCPN_GPU_ENABLE=1.
# cupy import can hang on misconfigured CUDA — opt-in only.
if _os.environ.get("SCPN_GPU_ENABLE", "0") == "1":
    try:
        import cupy as _cp_module  # type: ignore[import-untyped,import-not-found]

        if _cp_module.cuda.runtime.getDeviceCount() > 0:
            _cp = _cp_module
            _CUPY_AVAILABLE = True
    except (ImportError, Exception):
        pass


def is_gpu_available() -> bool:
    """Check if CUDA GPU acceleration is available."""
    return _CUPY_AVAILABLE


def gpu_device_name() -> str:
    """Get GPU device name or 'cpu'."""
    if _CUPY_AVAILABLE and _cp is not None:
        props = _cp.cuda.runtime.getDeviceProperties(0)
        return props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    return "cpu"


def eigvalsh(matrix: np.ndarray) -> np.ndarray:
    """Eigenvalues of Hermitian matrix, GPU-accelerated if available."""
    if _CUPY_AVAILABLE and _cp is not None and matrix.shape[0] >= 64:
        m_gpu = _cp.asarray(matrix)
        eigs_gpu = _cp.linalg.eigvalsh(m_gpu)
        result: np.ndarray = _cp.asnumpy(eigs_gpu)
        return result
    out: np.ndarray = np.linalg.eigvalsh(matrix)
    return out


def eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues + eigenvectors of Hermitian matrix, GPU-accelerated."""
    if _CUPY_AVAILABLE and _cp is not None and matrix.shape[0] >= 64:
        m_gpu = _cp.asarray(matrix)
        eigs_gpu, vecs_gpu = _cp.linalg.eigh(m_gpu)
        return _cp.asnumpy(eigs_gpu), _cp.asnumpy(vecs_gpu)
    return np.linalg.eigh(matrix)


def expm(matrix: np.ndarray) -> np.ndarray:
    """Matrix exponential, GPU-accelerated if available.

    cupy doesn't have expm natively; uses eigendecomposition:
    exp(A) = V diag(exp(lambda)) V^{-1} for Hermitian A.
    Falls back to scipy for non-Hermitian.
    """
    if (
        _CUPY_AVAILABLE
        and _cp is not None
        and matrix.shape[0] >= 32
        and np.allclose(matrix, matrix.conj().T, atol=1e-10)
    ):
        m_gpu = _cp.asarray(matrix)
        eigs, vecs = _cp.linalg.eigh(m_gpu)
        exp_diag = _cp.diag(_cp.exp(eigs))
        result_gpu = vecs @ exp_diag @ vecs.conj().T
        result: np.ndarray = _cp.asnumpy(result_gpu)
        return result

    from scipy.linalg import expm as scipy_expm

    out: np.ndarray = scipy_expm(matrix)
    return out


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiplication, GPU-accelerated if available."""
    if _CUPY_AVAILABLE and _cp is not None and a.shape[0] >= 64:
        a_gpu = _cp.asarray(a)
        b_gpu = _cp.asarray(b)
        result: np.ndarray = _cp.asnumpy(a_gpu @ b_gpu)
        return result
    out: np.ndarray = a @ b
    return out


def gpu_memory_free_mb() -> float:
    """Free GPU memory in MB."""
    if _CUPY_AVAILABLE and _cp is not None:
        free, _total = _cp.cuda.runtime.memGetInfo()
        return float(free / 1e6)
    return 0.0
