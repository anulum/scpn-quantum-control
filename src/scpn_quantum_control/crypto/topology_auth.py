"""Spectral fingerprint authentication for K_nm topology.

The Laplacian spectrum of K_nm provides a public authentication token.
Co-spectral graphs (different K_nm with same spectrum) exist, so
publishing the spectrum doesn't reveal K_nm — but any party with the
true K_nm can verify consistency.

Refs:
- Van Dam & Haemers (2003), "Which graphs are determined by their spectrum?"
- AAAI Symposium 2025, "Quantum Network Science: Graph Structure to Entanglement Performance"
"""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy as scipy_entropy


def spectral_fingerprint(K: np.ndarray) -> dict:
    """Compute public spectral fingerprint of coupling matrix.

    Returns dict with:
        fiedler: Second-smallest eigenvalue of graph Laplacian (algebraic connectivity).
        gap_ratio: lambda_1 / lambda_2 (spectral gap quality).
        spectral_entropy: Shannon entropy of normalized eigenvalue distribution.
        n_components: Number of connected components (eigenvalues ≈ 0).
    """
    n = K.shape[0]
    D = np.diag(K.sum(axis=1))
    L = D - K

    eigvals = np.sort(np.linalg.eigvalsh(L))

    # Fiedler value (second smallest)
    fiedler = float(eigvals[1]) if n > 1 else 0.0

    # Spectral gap ratio
    gap_ratio = float(eigvals[1] / eigvals[2]) if n > 2 and eigvals[2] > 1e-12 else 0.0

    # Spectral entropy of positive eigenvalues
    pos_eigvals = eigvals[eigvals > 1e-12]
    if len(pos_eigvals) > 0:
        p = pos_eigvals / pos_eigvals.sum()
        s_entropy = float(scipy_entropy(p, base=2))
    else:
        s_entropy = 0.0

    # Connected components
    n_components = int(np.sum(eigvals < 1e-8))

    return {
        "fiedler": fiedler,
        "gap_ratio": gap_ratio,
        "spectral_entropy": s_entropy,
        "n_components": n_components,
        "eigenvalues": eigvals.tolist(),
    }


def verify_fingerprint(K: np.ndarray, fingerprint: dict, tol: float = 1e-6) -> bool:
    """Check K against a claimed spectral fingerprint."""
    computed = spectral_fingerprint(K)
    return (
        abs(computed["fiedler"] - fingerprint["fiedler"]) < tol
        and abs(computed["spectral_entropy"] - fingerprint["spectral_entropy"]) < tol
        and computed["n_components"] == fingerprint["n_components"]
    )


def topology_distance(fp1: dict, fp2: dict) -> float:
    """L2 distance between two spectral fingerprints.

    Useful for detecting calibration drift or K_nm tampering.
    """
    e1 = np.array(fp1["eigenvalues"])
    e2 = np.array(fp2["eigenvalues"])
    if len(e1) != len(e2):
        return float("inf")
    return float(np.linalg.norm(e1 - e2))
