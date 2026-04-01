# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Dynamical Lie Algebra
"""Dynamical Lie algebra (DLA) computation for Hamiltonian simulability analysis.

The DLA of a set of Hamiltonian generators {H_1, ..., H_k} is the smallest
Lie algebra containing all generators, closed under commutation. If the DLA
dimension grows polynomially with system size N, the system is efficiently
classically simulable (Goh et al., Phys. Rev. Research, 2025).

For the pure XY model, DLA dimension = O(N²) → classically simulable.
Adding SSGF geometry feedback, TCBO coupling, or PGBO tensor terms may
expand the DLA beyond polynomial, indicating quantum advantage.

Reference: Goh, Larocca, Cincio, Cerezo & Sauvage, arXiv:2308.01432.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass
class DLAResult:
    """Result of dynamical Lie algebra computation."""

    dimension: int
    n_qubits: int
    n_generators: int
    n_iterations: int
    basis_labels: list[str]
    is_polynomial: bool
    polynomial_degree: float
    max_hilbert_dim: int

    @property
    def classical_simulable(self) -> bool:
        """DLA dimension polynomial → classically simulable (g-sim)."""
        return self.is_polynomial


def _commutator(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix commutator [A, B] = AB - BA."""
    result: np.ndarray = a @ b - b @ a
    return result


def _is_independent(new_op: np.ndarray, basis: list[np.ndarray], tol: float = 1e-10) -> bool:
    """Check if new_op is linearly independent of existing basis vectors."""
    if not basis:
        return bool(np.linalg.norm(new_op) > tol)
    mat = np.column_stack([b.ravel() for b in basis])
    new_vec = new_op.ravel()
    residual = new_vec - mat @ np.linalg.lstsq(mat, new_vec, rcond=None)[0]
    return bool(np.linalg.norm(residual) > tol)


def compute_dla(
    generators: list[SparsePauliOp],
    max_iterations: int = 50,
    max_dimension: int = 500,
    tol: float = 1e-10,
) -> DLAResult:
    """Compute the dynamical Lie algebra from Hamiltonian generators.

    Takes generators as SparsePauliOp and computes the closure under
    commutation. Stops when no new linearly independent operators appear
    or when max_dimension is reached.

    The DLA dimension determines classical simulability:
    - dim = O(N²): classically simulable (g-sim framework)
    - dim = O(2^N): quantum advantage regime

    Returns DLAResult with dimension, polynomial assessment, and basis.
    """
    n = generators[0].num_qubits
    dim = 2**n
    max_hilbert = dim * dim  # maximum DLA dimension = dim(su(2^n))

    # Convert generators to dense matrices (iH for Hermitian generators)
    gen_mats = [1j * g.to_matrix() for g in generators]

    # Initialize basis with generators
    basis: list[np.ndarray] = []
    labels: list[str] = []
    for i, g in enumerate(gen_mats):
        if _is_independent(g, basis, tol):
            basis.append(g)
            labels.append(f"H_{i}")

    n_iters = 0
    for _step in range(1, max_iterations + 1):
        n_iters = _step
        new_ops: list[tuple[np.ndarray, str]] = []
        n_basis = len(basis)

        for i in range(n_basis):
            for j in range(i + 1, n_basis):
                comm = _commutator(basis[i], basis[j])
                if np.linalg.norm(comm) < tol:
                    continue
                if _is_independent(comm, basis + [op for op, _ in new_ops], tol):
                    new_ops.append((comm, f"[{labels[i]},{labels[j]}]"))

        if not new_ops:
            break
        for op, label in new_ops:
            basis.append(op)
            labels.append(label)

        if len(basis) >= max_dimension:
            break

    dimension = len(basis)

    # Assess polynomial scaling: compare dim vs N² and N³
    n_sq = n * n
    n_cube = n * n * n
    if dimension <= 2 * n_sq:
        is_poly = True
        degree = 2.0
    elif dimension <= 2 * n_cube:
        is_poly = True
        degree = 3.0
    elif dimension >= max_dimension:
        # Hit cap — likely exponential, but can't be sure
        is_poly = False
        degree = float("inf")
    else:
        # Between N³ and cap — estimate degree from log
        degree = np.log(dimension) / np.log(max(n, 2))
        is_poly = degree < n / 2

    return DLAResult(
        dimension=dimension,
        n_qubits=n,
        n_generators=len(generators),
        n_iterations=n_iters,
        basis_labels=labels[:20],  # truncate for readability
        is_polynomial=is_poly,
        polynomial_degree=degree,
        max_hilbert_dim=max_hilbert,
    )


def compute_dla_rust(
    generators: list[SparsePauliOp],
    max_iterations: int = 50,
    max_dimension: int = 500,
    tol: float = 1e-10,
) -> DLAResult:
    """Rust-accelerated DLA computation (50-100x faster than Python).

    Falls back to Python compute_dla if Rust engine unavailable.
    """
    try:
        from scpn_quantum_engine import dla_dimension  # type: ignore[import-not-found]
    except ImportError:
        return compute_dla(generators, max_iterations, max_dimension, tol)

    n = generators[0].num_qubits
    dim = 2**n
    max_hilbert = dim * dim

    # Convert generators to flat real array (iH for Hermitian generators)
    gen_mats = []
    for g in generators:
        mat = 1j * g.to_matrix()
        if hasattr(mat, "toarray"):
            mat = mat.toarray()
        gen_mats.append(np.array(mat, dtype=complex))

    # Rust DLA works with real matrices — use real part of iH (which is anti-Hermitian)
    # For real Hamiltonians, iH is purely imaginary → use imaginary part
    flat = np.concatenate([m.imag.ravel() for m in gen_mats])

    dimension = dla_dimension(flat, dim, len(generators), max_iterations, max_dimension, tol)

    n_sq = n * n
    n_cube = n * n * n
    if dimension <= 2 * n_sq:
        is_poly = True
        degree = 2.0
    elif dimension <= 2 * n_cube:
        is_poly = True
        degree = 3.0
    elif dimension >= max_dimension:
        is_poly = False
        degree = float("inf")
    else:
        degree = np.log(dimension) / np.log(max(n, 2))
        is_poly = degree < n / 2

    return DLAResult(
        dimension=dimension,
        n_qubits=n,
        n_generators=len(generators),
        n_iterations=0,
        basis_labels=[f"rust_basis_{i}" for i in range(min(dimension, 20))],
        is_polynomial=is_poly,
        polynomial_degree=degree,
        max_hilbert_dim=max_hilbert,
    )


def build_xy_generators(K: np.ndarray, omega: np.ndarray) -> list[SparsePauliOp]:
    """Build the standard XY Hamiltonian generators: {Z_i, X_iX_j, Y_iY_j}."""
    from ..bridge.knm_hamiltonian import KNM_SPARSITY_EPS

    n = len(omega)
    generators = []

    # Single-qubit Z terms
    for i in range(n):
        if abs(omega[i]) > KNM_SPARSITY_EPS:
            z_str = ["I"] * n
            z_str[i] = "Z"
            generators.append(SparsePauliOp("".join(reversed(z_str)), omega[i]))

    # Two-qubit XX and YY terms
    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) < KNM_SPARSITY_EPS:
                continue
            xx = ["I"] * n
            xx[i] = "X"
            xx[j] = "X"
            generators.append(SparsePauliOp("".join(reversed(xx)), K[i, j]))

            yy = ["I"] * n
            yy[i] = "Y"
            yy[j] = "Y"
            generators.append(SparsePauliOp("".join(reversed(yy)), K[i, j]))

    return generators


def build_ssgf_generators(
    K: np.ndarray,
    omega: np.ndarray,
    W: np.ndarray,
    sigma_g: float = 0.3,
) -> list[SparsePauliOp]:
    """Build generators including SSGF geometry feedback term.

    Adds σ_g × W_nm × (X_nX_m + Y_nY_m) coupling on top of base XY.
    """
    base = build_xy_generators(K, omega)
    from ..bridge.knm_hamiltonian import KNM_SPARSITY_EPS

    n = len(omega)
    for i in range(n):
        for j in range(i + 1, n):
            coupling = sigma_g * W[i, j]
            if abs(coupling) < KNM_SPARSITY_EPS:
                continue
            xx = ["I"] * n
            xx[i] = "X"
            xx[j] = "X"
            base.append(SparsePauliOp("".join(reversed(xx)), coupling))

            yy = ["I"] * n
            yy[i] = "Y"
            yy[j] = "Y"
            base.append(SparsePauliOp("".join(reversed(yy)), coupling))

    return base


def build_pgbo_generators(
    K: np.ndarray,
    omega: np.ndarray,
    h_munu: np.ndarray,
    pgbo_weight: float = 0.1,
) -> list[SparsePauliOp]:
    """Build generators including PGBO tensor field h_munu coupling.

    Adds pgbo_weight × h_nm × (X_nX_m + Y_nY_m) for the phase-geometry
    bridge. h_munu is a symmetric PSD matrix (gauge field strength).
    """
    base = build_xy_generators(K, omega)
    from ..bridge.knm_hamiltonian import KNM_SPARSITY_EPS

    n = len(omega)
    for i in range(n):
        for j in range(i + 1, n):
            coupling = pgbo_weight * h_munu[i, j]
            if abs(coupling) < KNM_SPARSITY_EPS:
                continue
            xx = ["I"] * n
            xx[i] = "X"
            xx[j] = "X"
            base.append(SparsePauliOp("".join(reversed(xx)), coupling))

            yy = ["I"] * n
            yy[i] = "Y"
            yy[j] = "Y"
            base.append(SparsePauliOp("".join(reversed(yy)), coupling))

    return base


def build_tcbo_generators(
    K: np.ndarray,
    omega: np.ndarray,
    kappa: float = 1.0,
    connectivity: str = "nearest",
) -> list[SparsePauliOp]:
    """Build generators including TCBO gap-junction diffusion coupling.

    Adds kappa × Laplacian diffusion: sum_neighbors Z_n Z_m terms.
    This is qualitatively different from XX+YY — it's ZZ coupling,
    which expands the Lie algebra beyond the XY sector.
    """
    base = build_xy_generators(K, omega)
    n = len(omega)

    # Build Laplacian connectivity
    neighbors: list[list[int]] = [[] for _ in range(n)]
    if connectivity == "nearest":
        for i in range(n - 1):
            neighbors[i].append(i + 1)
            neighbors[i + 1].append(i)
    elif connectivity == "full":
        for i in range(n):
            for j in range(i + 1, n):
                neighbors[i].append(j)

    # Add ZZ terms (gap-junction diffusion)
    for i in range(n):
        for j in neighbors[i]:
            if j <= i:
                continue
            zz = ["I"] * n
            zz[i] = "Z"
            zz[j] = "Z"
            base.append(SparsePauliOp("".join(reversed(zz)), kappa))

    return base


def build_full_scpn_generators(
    K: np.ndarray,
    omega: np.ndarray,
    W: np.ndarray | None = None,
    h_munu: np.ndarray | None = None,
    sigma_g: float = 0.3,
    pgbo_weight: float = 0.1,
    kappa: float = 1.0,
) -> list[SparsePauliOp]:
    """Build generators for the full SCPN Hamiltonian: XY + SSGF + PGBO + TCBO.

    This is the decisive test: if the DLA of the full system is polynomial,
    quantum simulation provides no advantage. If exponential, it does.
    """
    base = build_xy_generators(K, omega)
    n = len(omega)
    from ..bridge.knm_hamiltonian import KNM_SPARSITY_EPS

    # SSGF geometry feedback
    if W is not None:
        for i in range(n):
            for j in range(i + 1, n):
                coupling = sigma_g * W[i, j]
                if abs(coupling) < KNM_SPARSITY_EPS:
                    continue
                xx = ["I"] * n
                xx[i] = "X"
                xx[j] = "X"
                base.append(SparsePauliOp("".join(reversed(xx)), coupling))
                yy = ["I"] * n
                yy[i] = "Y"
                yy[j] = "Y"
                base.append(SparsePauliOp("".join(reversed(yy)), coupling))

    # PGBO tensor field
    if h_munu is not None:
        for i in range(n):
            for j in range(i + 1, n):
                coupling = pgbo_weight * h_munu[i, j]
                if abs(coupling) < KNM_SPARSITY_EPS:
                    continue
                xx = ["I"] * n
                xx[i] = "X"
                xx[j] = "X"
                base.append(SparsePauliOp("".join(reversed(xx)), coupling))
                yy = ["I"] * n
                yy[i] = "Y"
                yy[j] = "Y"
                base.append(SparsePauliOp("".join(reversed(yy)), coupling))

    # TCBO gap-junction ZZ coupling (nearest-neighbor Laplacian)
    for i in range(n - 1):
        zz = ["I"] * n
        zz[i] = "Z"
        zz[i + 1] = "Z"
        base.append(SparsePauliOp("".join(reversed(zz)), kappa))

    return base
