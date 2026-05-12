# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — GPU Batch VQE
"""Parallel VQE evaluation on GPU using PyTorch or JAX.

Evaluates multiple parameter sets simultaneously by batching the
forward pass (wavefunction → energy) across GPU threads.

For a VQE with P parameters and B batch size:
  CPU sequential: B × T_single
  GPU batched: ~T_single + overhead (amortised across B)

Inspired by TorchQuantum (MIT HAN Lab).

Requires: pip install torch (or jax)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def batch_energy_numpy(
    H: np.ndarray,
    param_sets: np.ndarray,
    ansatz_fn: Any,
) -> np.ndarray:
    """Evaluate energy for a batch of parameter sets (CPU baseline).

    Parameters
    ----------
    H : array (dim, dim)
        Hamiltonian matrix.
    param_sets : array (batch, n_params)
        B parameter vectors to evaluate.
    ansatz_fn : callable
        Maps params → statevector (dim,) complex array.

    Returns
    -------
    energies : array (batch,)
    """
    batch = param_sets.shape[0]
    energies = np.zeros(batch)
    for i in range(batch):
        psi = ansatz_fn(param_sets[i])
        psi = psi / np.linalg.norm(psi)
        energies[i] = float(np.real(psi.conj() @ H @ psi))
    return energies


def batch_energy_torch(
    H: np.ndarray,
    param_sets: np.ndarray,
    ansatz_fn: Any,
    device: str = "cuda",
) -> np.ndarray:
    """Evaluate energy for a batch of parameter sets on GPU via PyTorch.

    Parameters
    ----------
    H : array (dim, dim)
        Hamiltonian.
    param_sets : array (batch, n_params)
        Parameter vectors.
    ansatz_fn : callable
        Maps torch.Tensor params → torch.Tensor statevector.
    device : str
        "cuda" for GPU, "cpu" for CPU.

    Returns
    -------
    energies : array (batch,)
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("PyTorch not installed: pip install torch") from e

    H_t = torch.tensor(H, dtype=torch.complex64, device=device)
    params_t = torch.tensor(param_sets, dtype=torch.float32, device=device)

    energies = torch.zeros(params_t.shape[0], device=device)

    for i in range(params_t.shape[0]):
        psi = ansatz_fn(params_t[i])
        psi = psi / torch.linalg.norm(psi)
        energies[i] = torch.real(psi.conj() @ H_t @ psi)

    result: np.ndarray = energies.detach().cpu().numpy()
    return result


def batch_vqe_scan(
    K: np.ndarray,
    omega: np.ndarray,
    n_samples: int = 100,
    n_params: int | None = None,
    seed: int = 42,
    use_gpu: bool = False,
) -> dict:
    """Scan VQE landscape by evaluating random parameter sets in batch.

    Uses a product Ry-layer diagnostic ansatz and random parameter scan.
    This is a landscape scanner, not a gradient-optimised production VQE.
    Passing ``use_gpu=True`` requires PyTorch and a CUDA device; it is not
    silently downgraded to the NumPy path.

    Returns dict with energies, parameters, best point, and contract metadata.
    """
    from ..bridge.knm_hamiltonian import knm_to_dense_matrix

    if n_samples < 1:
        raise ValueError("n_samples must be >= 1")

    n = K.shape[0]
    dim = 2**n
    H = knm_to_dense_matrix(K, omega)

    if n_params is None:
        n_params = n * 2  # 2 layers of Ry
    if n_params < 1:
        raise ValueError("n_params must be >= 1")

    rng = np.random.default_rng(seed)
    param_sets = rng.normal(0, 1.0, (n_samples, n_params)).astype(np.float32)

    def numpy_ansatz(params: np.ndarray) -> np.ndarray:
        """Product Ry-layer diagnostic ansatz producing a statevector."""
        psi = np.zeros(dim, dtype=np.complex128)
        psi[0] = 1.0
        # Apply Ry rotations layer by layer
        for layer in range(n_params // n):
            for i in range(n):
                idx = layer * n + i
                if idx < len(params):
                    angle = float(params[idx])
                    c, s = np.cos(angle / 2), np.sin(angle / 2)
                    # Apply Ry on qubit i
                    new_psi = np.zeros_like(psi)
                    for k in range(dim):
                        bi = (k >> i) & 1
                        k_flip = k ^ (1 << i)
                        if bi == 0:
                            new_psi[k] += c * psi[k]
                            new_psi[k_flip] += s * psi[k]
                        else:
                            new_psi[k] += c * psi[k]
                            new_psi[k_flip] -= s * psi[k]
                    psi[:] = new_psi
        return psi

    backend = "numpy"
    if use_gpu:
        try:
            import torch
        except ImportError as e:
            raise ImportError("PyTorch not installed: pip install torch") from e

        if not torch.cuda.is_available():
            raise RuntimeError("use_gpu=True requires an available CUDA device")
        try:
            torch.zeros(1, device="cuda").cpu()
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError("use_gpu=True requires a usable CUDA device") from e

        def torch_ansatz(params: torch.Tensor) -> torch.Tensor:
            """Product Ry-layer diagnostic ansatz producing a torch statevector."""
            psi = torch.zeros(dim, dtype=torch.complex64, device=params.device)
            psi[0] = torch.tensor(1.0 + 0.0j, dtype=torch.complex64, device=params.device)
            for layer in range(n_params // n):
                for i in range(n):
                    idx = layer * n + i
                    if idx < params.numel():
                        angle = params[idx]
                        c = torch.cos(angle / 2).to(torch.complex64)
                        s = torch.sin(angle / 2).to(torch.complex64)
                        new_psi = torch.zeros_like(psi)
                        for k in range(dim):
                            bi = (k >> i) & 1
                            k_flip = k ^ (1 << i)
                            if bi == 0:
                                new_psi[k] += c * psi[k]
                                new_psi[k_flip] += s * psi[k]
                            else:
                                new_psi[k] += c * psi[k]
                                new_psi[k_flip] -= s * psi[k]
                        psi = new_psi
            return psi

        energies = batch_energy_torch(H, param_sets, torch_ansatz, device="cuda")
        backend = "torch_cuda"
    else:
        energies = batch_energy_numpy(H, param_sets, numpy_ansatz)

    best_idx = int(np.argmin(energies))
    return {
        "energies": energies,
        "params": param_sets,
        "best_energy": float(energies[best_idx]),
        "best_params": param_sets[best_idx],
        "n_samples": n_samples,
        "backend": backend,
        "ansatz_family": "product_ry_layers",
        "optimizer": "random_parameter_scan",
        "hardware_claim": "none_statevector_expectation_scan",
    }
