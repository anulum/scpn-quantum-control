# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Symmetry Verification
"""Z₂ parity symmetry verification for XY Hamiltonian error mitigation.

The XY Hamiltonian H = -Σ K_ij (X_iX_j + Y_iY_j) - Σ (ω_i/2) Z_i
conserves the total Z₂ parity P = ⊗_i Z_i. This is proven by the DLA
decomposition: DLA(N) = su(2^(N-1)) ⊕ su(2^(N-1)), where the two
sectors are the even- and odd-parity subspaces, and the only missing
generators are the identity and the parity operator itself.

Consequence: any measurement outcome whose parity differs from the
initial state's parity is guaranteed to be a hardware error. Discarding
these outcomes (post-selection) or expanding the density matrix into
the correct symmetry sector (symmetry expansion) provides error
mitigation at zero circuit overhead.

Reference:
    Bonet-Monroig et al., "Low-cost error mitigation by symmetry
    verification", Phys. Rev. A 98, 062339 (2018). arXiv:2101.03151.

    Sotek, "DLA(N) = 2^(2N-1) - 2 for heterogeneous XY on complete
    weighted graphs", scpn-quantum-control (2026).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

MeasurementCount = int | np.integer[Any]
"""Accepted count scalars; the NumPy bit precision is unrestricted."""

try:
    import scpn_quantum_engine as _engine  # pragma: no cover

    _HAS_RUST = True  # pragma: no cover
except ImportError:
    _HAS_RUST = False


@dataclass
class SymmetryVerificationResult:
    """Result of Z₂ parity post-selection."""

    raw_counts: dict[str, int]
    verified_counts: dict[str, int]
    rejected_counts: dict[str, int]
    raw_shots: int
    verified_shots: int
    rejected_shots: int
    rejection_rate: float
    expected_parity: int


def _clean_bitstring(bitstring: str) -> str:
    clean = bitstring.replace(" ", "")
    if not clean or any(bit not in {"0", "1"} for bit in clean):
        raise ValueError(f"invalid computational-basis bitstring: {bitstring!r}")
    return clean


def _validate_expected_parity(expected_parity: int) -> None:
    if expected_parity not in {0, 1}:
        raise ValueError(f"expected_parity must be 0 or 1, got {expected_parity!r}")


def _validate_count(count: MeasurementCount) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise ValueError("counts must be non-negative integers, excluding booleans")
    count_value = int(count)
    if count_value < 0:
        raise ValueError("counts must be non-negative")
    return count_value


def _validated_counts(counts: Mapping[str, MeasurementCount]) -> dict[str, int]:
    """Validate labels and return exact Python integers without changing the caller."""
    clean_counts: dict[str, int] = {}
    for bitstring, count in counts.items():
        _clean_bitstring(bitstring)
        clean_counts[bitstring] = _validate_count(count)
    return clean_counts


def bitstring_parity(bitstring: str) -> int:
    """Compute Z₂ parity of a measurement bitstring.

    Returns 0 (even) or 1 (odd) = number of '1' bits mod 2.
    """
    return _clean_bitstring(bitstring).count("1") % 2


def initial_state_parity(omega: NDArray[np.float64]) -> int:
    """Compute the dominant parity sector of the initial state |ψ_0⟩.

    The initial state is ⊗_i Ry(ω_i)|0⟩. Each qubit has probability
    sin²(ω_i/2) of being in |1⟩. The dominant parity sector is
    determined by the most probable bitstring class.

    For small angles (ω_i ≪ π), the state is near |00...0⟩ → even parity.
    """
    omega_arr = np.asarray(omega, dtype=float)
    if omega_arr.ndim != 1 or omega_arr.size == 0 or not np.all(np.isfinite(omega_arr)):
        raise ValueError("omega must be a non-empty finite one-dimensional vector")
    p_ones = np.sin(omega_arr / 2.0) ** 2
    expected_ones: float = float(np.sum(p_ones))
    return int(round(expected_ones)) % 2


def parity_postselect(
    counts: Mapping[str, MeasurementCount],
    expected_parity: int,
) -> SymmetryVerificationResult:
    """Filter measurement outcomes by Z₂ parity conservation.

    The XY Hamiltonian conserves parity: [H_XY, P] = 0 where P = ⊗Z_i.
    Any bitstring with the wrong parity is a hardware error.

    Parameters
    ----------
    counts
        Raw measurement counts ``{bitstring: count}``. Multiplicities must
        be non-negative integral scalars (including NumPy integers), not
        booleans, floats or text. The input mapping is not modified.
    expected_parity
        Expected sector: zero for even or one for odd parity.

    Returns
    -------
        SymmetryVerificationResult with verified and rejected counts.

    Raises
    ------
    ValueError
        If a bitstring, parity sector or count violates its input contract.

    """
    _validate_expected_parity(expected_parity)
    clean_counts = _validated_counts(counts)
    verified: dict[str, int] = {}
    rejected: dict[str, int] = {}

    if _HAS_RUST and clean_counts:  # pragma: no cover
        keys = list(clean_counts.keys())
        vals = [clean_counts[k] for k in keys]
        bs_ints = np.array([int(k.replace(" ", ""), 2) for k in keys], dtype=np.uint64)
        mask = np.array(_engine.parity_filter_mask(bs_ints, expected_parity))
        for i, k in enumerate(keys):
            if mask[i]:
                verified[k] = vals[i]
            else:
                rejected[k] = vals[i]
    else:
        for bitstring, count in clean_counts.items():
            if bitstring_parity(bitstring) == expected_parity:
                verified[bitstring] = count
            else:
                rejected[bitstring] = count

    raw_shots = sum(clean_counts.values())
    verified_shots = sum(verified.values())
    rejected_shots = sum(rejected.values())
    rejection_rate = rejected_shots / raw_shots if raw_shots > 0 else 0.0

    return SymmetryVerificationResult(
        raw_counts=clean_counts,
        verified_counts=verified,
        rejected_counts=rejected,
        raw_shots=raw_shots,
        verified_shots=verified_shots,
        rejected_shots=rejected_shots,
        rejection_rate=rejection_rate,
        expected_parity=expected_parity,
    )


def symmetry_expand(
    counts: Mapping[str, MeasurementCount],
    expected_parity: int,
) -> dict[str, int]:
    """Symmetry expansion: project counts into the correct parity sector.

    Unlike post-selection (which discards wrong-parity outcomes and loses
    shots), symmetry expansion redistributes wrong-parity counts by
    flipping the least-significant bit and adding them to the correct
    sector. This preserves total shot count.

    Trade-off: post-selection is cleaner (no bias); expansion preserves
    statistics (no shot loss). Use post-selection for observables that
    are sensitive to parity (energy, correlators). Use expansion for
    observables that tolerate small bias (order parameter R).

    Counts must be non-negative integral scalars, excluding booleans.
    Floats and text are refused rather than converted. Invalid count,
    bitstring or parity input raises `ValueError`; empty input returns an
    empty mapping. The caller's mapping is not modified.
    """
    _validate_expected_parity(expected_parity)
    expanded: dict[str, int] = {}

    for bitstring, count in counts.items():
        clean = _clean_bitstring(bitstring)
        validated = _validate_count(count)
        if bitstring_parity(clean) == expected_parity:
            expanded[clean] = expanded.get(clean, 0) + validated
        else:
            # Flip least-significant bit to correct parity
            flipped = clean[:-1] + ("0" if clean[-1] == "1" else "1")
            expanded[flipped] = expanded.get(flipped, 0) + validated

    return expanded


def parity_verified_expectation(
    counts: Mapping[str, MeasurementCount],
    n_qubits: int,
    expected_parity: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Compute per-qubit ⟨Z⟩ from parity-verified counts only.

    Counts follow `parity_postselect`'s integral-scalar contract; invalid
    input raises `ValueError` before expectation reduction.

    Returns
    -------
        (exp_vals, std_vals, rejection_rate)

    """
    result = parity_postselect(counts, expected_parity)

    if result.verified_shots == 0:
        return (
            np.zeros(n_qubits),
            np.ones(n_qubits),
            result.rejection_rate,
        )

    total = result.verified_shots
    exp_vals = np.zeros(n_qubits)
    for bitstring, count in result.verified_counts.items():
        bits = bitstring.replace(" ", "")
        for q in range(min(n_qubits, len(bits))):
            bit = int(bits[-(q + 1)])
            exp_vals[q] += (1 - 2 * bit) * count
    exp_vals /= total
    std_vals: NDArray[np.float64] = np.sqrt(np.maximum(1.0 - exp_vals**2, 0.0) / total)
    return exp_vals, std_vals, result.rejection_rate


def parity_verified_R(
    z_counts: Mapping[str, MeasurementCount],
    x_counts: Mapping[str, MeasurementCount],
    y_counts: Mapping[str, MeasurementCount],
    n_qubits: int,
    expected_parity: int,
) -> dict[str, float]:
    """Compute order parameter R from parity-verified XYZ measurements.

    Drop-in replacement for experiments._R_from_xyz with symmetry
    verification. Returns dict with both raw and verified R.

    All three count maps must satisfy the integral-scalar contract of
    `parity_postselect`. Invalid input raises `ValueError` before either
    raw or verified reduction; caller-owned maps remain unchanged.
    """
    from scpn_quantum_control.hardware.experiments import _R_from_xyz

    # Normalise integral scalar representations before any observable reduction.
    z_clean = _validated_counts(z_counts)
    x_clean = _validated_counts(x_counts)
    y_clean = _validated_counts(y_counts)

    # Z-basis: parity applies directly
    z_ver = parity_postselect(z_clean, expected_parity)

    # X-basis: after H rotation, parity of measurement outcomes doesn't
    # directly correspond to Z-parity. But the physical constraint still
    # holds: the state before basis rotation has definite parity.
    # For X and Y bases, we use symmetry expansion (less aggressive).
    x_expanded = symmetry_expand(x_clean, expected_parity)
    y_expanded = symmetry_expand(y_clean, expected_parity)

    # Validate all axes before either raw or verified observable reduction.
    raw = _R_from_xyz(z_clean, x_clean, y_clean, n_qubits)
    R_raw, R_raw_std = raw[0], raw[1]

    # Recompute R from verified Z, expanded X/Y
    R_ver, R_ver_std = _R_from_xyz(z_ver.verified_counts, x_expanded, y_expanded, n_qubits)[:2]

    return {
        "R_raw": R_raw,
        "R_raw_std": R_raw_std,
        "R_verified": R_ver,
        "R_verified_std": R_ver_std,
        "z_rejection_rate": z_ver.rejection_rate,
        "improvement": R_ver - R_raw,
        "improvement_pct": (R_ver - R_raw) / max(abs(R_raw), 1e-12) * 100,
    }
