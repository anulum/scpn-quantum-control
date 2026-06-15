# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — NV-centre high-field magnetometry (simulation only)
"""Nitrogen-vacancy (NV) centre magnetometry response model into the 20 T regime.

Simulation-only model: the NV ground-state spin-1 Hamiltonian is diagonalised
exactly, so the optically-detected magnetic-resonance (ODMR) frequencies, the
shot-noise-limited DC sensitivity, and a noisy-spectrum field-calibration loop
remain valid past the ground-state level anti-crossing (GSLAC, ~102 mT axial)
and into the high-field regime where the electron Zeeman term dominates the
zero-field splitting. Hardware calibration against a NIST-traceable reference is
a separate, hardware-gated workstream.

References:

- M. W. Doherty et al., *The nitrogen-vacancy colour centre in diamond*, Physics
  Reports 528, 1 (2013) — ground-state spin Hamiltonian.
- J. F. Barry et al., *Sensitivity optimization for NV-diamond magnetometry*,
  Reviews of Modern Physics 92, 015004 (2020) — CW-ODMR DC sensitivity.
- E. Bauch et al., *Ultra-long dephasing times in solid-state spin ensembles via
  quantum control*, Physical Review X 8, 031025 (2018) — coherence regime.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

# Ground-state zero-field splitting D and the electron gyromagnetic ratio
# gamma_e = g_e mu_B / h (g_e = 2.0028 for the NV centre).
NV_ZERO_FIELD_SPLITTING_HZ = 2.870e9
ELECTRON_GYROMAGNETIC_HZ_PER_T = 28.024951e9
# CW-ODMR DC-sensitivity prefactor for a Lorentzian line (Barry RMP 2020).
_LORENTZIAN_SENSITIVITY_PREFACTOR = 4.0 / (3.0 * np.sqrt(3.0))

# Spin-1 operators (hbar = 1), basis ordered |+1>, |0>, |-1>.
_SQRT2 = np.sqrt(2.0)
_SX = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.complex128) / _SQRT2
_SY = (
    np.array([[0.0, -1.0j, 0.0], [1.0j, 0.0, -1.0j], [0.0, 1.0j, 0.0]], dtype=np.complex128)
    / _SQRT2
)
_SZ = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.complex128)
_SZ2 = _SZ @ _SZ
_MS0_INDEX = 1  # |0> is the second basis vector


@dataclass(frozen=True)
class NVCenter:
    """NV-centre parameters for the ground-state magnetometry model."""

    zero_field_splitting_hz: float = NV_ZERO_FIELD_SPLITTING_HZ
    gyromagnetic_hz_per_t: float = ELECTRON_GYROMAGNETIC_HZ_PER_T
    transverse_strain_hz: float = 0.0  # E term [Hz]
    linewidth_hz: float = 1.0e6  # ODMR FWHM
    contrast: float = 0.03  # fractional PL contrast
    photon_rate_per_s: float = 1.0e12  # detected photons / s

    def __post_init__(self) -> None:
        if not np.isfinite(self.zero_field_splitting_hz) or self.zero_field_splitting_hz <= 0.0:
            raise ValueError("zero_field_splitting_hz must be finite and positive")
        if not np.isfinite(self.gyromagnetic_hz_per_t) or self.gyromagnetic_hz_per_t <= 0.0:
            raise ValueError("gyromagnetic_hz_per_t must be finite and positive")
        if not np.isfinite(self.transverse_strain_hz) or self.transverse_strain_hz < 0.0:
            raise ValueError("transverse_strain_hz must be finite and non-negative")
        if not np.isfinite(self.linewidth_hz) or self.linewidth_hz <= 0.0:
            raise ValueError("linewidth_hz must be finite and positive")
        if not (0.0 < self.contrast <= 1.0):
            raise ValueError("contrast must lie in (0, 1]")
        if not np.isfinite(self.photon_rate_per_s) or self.photon_rate_per_s <= 0.0:
            raise ValueError("photon_rate_per_s must be finite and positive")


def nv_ground_state_hamiltonian(
    nv: NVCenter, field_tesla: float, theta_rad: float = 0.0, phi_rad: float = 0.0
) -> NDArray[np.complex128]:
    """NV ground-state spin-1 Hamiltonian in Hz (D, strain E, electron Zeeman)."""
    if field_tesla < 0.0:
        raise ValueError("field_tesla must be non-negative")
    d = nv.zero_field_splitting_hz
    e = nv.transverse_strain_hz
    gamma_b = nv.gyromagnetic_hz_per_t * field_tesla
    bx = np.sin(theta_rad) * np.cos(phi_rad)
    by = np.sin(theta_rad) * np.sin(phi_rad)
    bz = np.cos(theta_rad)
    hamiltonian = (
        d * (_SZ2 - (2.0 / 3.0) * np.eye(3))
        + e * (_SX @ _SX - _SY @ _SY)
        + gamma_b * (bx * _SX + by * _SY + bz * _SZ)
    )
    return np.asarray(hamiltonian, dtype=np.complex128)


def nv_energy_levels_hz(
    nv: NVCenter, field_tesla: float, theta_rad: float = 0.0, phi_rad: float = 0.0
) -> NDArray[np.float64]:
    """Sorted ground-state spin energy levels [Hz]."""
    hamiltonian = nv_ground_state_hamiltonian(nv, field_tesla, theta_rad, phi_rad)
    return np.asarray(np.linalg.eigvalsh(hamiltonian), dtype=np.float64)


def odmr_resonances_hz(
    nv: NVCenter, field_tesla: float, theta_rad: float = 0.0, phi_rad: float = 0.0
) -> tuple[float, float]:
    """The two ODMR transition frequencies from the ``ms=0``-like state [Hz].

    Returns ``(f_lower, f_upper)`` sorted ascending. At zero field both equal D;
    for an axial field they are ``D -/+ gamma_e B`` (the lower branch reflects
    about the GSLAC).
    """
    hamiltonian = nv_ground_state_hamiltonian(nv, field_tesla, theta_rad, phi_rad)
    values, vectors = np.linalg.eigh(hamiltonian)
    overlaps = np.abs(vectors[_MS0_INDEX, :]) ** 2
    ms0 = int(np.argmax(overlaps))
    transitions = sorted(abs(values[i] - values[ms0]) for i in range(3) if i != ms0)
    return float(transitions[0]), float(transitions[1])


def cw_odmr_dc_sensitivity_t_per_sqrt_hz(nv: NVCenter) -> float:
    """Shot-noise-limited CW-ODMR DC magnetic sensitivity [T/sqrt(Hz)].

    ``eta = P_F * delta_nu / (gamma_e * C * sqrt(R))`` (Barry RMP 2020), where the
    field response ``d nu / dB = gamma_e`` for an axial field.
    """
    return float(
        _LORENTZIAN_SENSITIVITY_PREFACTOR
        * nv.linewidth_hz
        / (nv.gyromagnetic_hz_per_t * nv.contrast * np.sqrt(nv.photon_rate_per_s))
    )


def _lorentzian_dip(
    freqs: NDArray[np.float64], centers: NDArray[np.float64], fwhm: float, contrast: float
) -> NDArray[np.float64]:
    half = fwhm / 2.0
    half_sq = half * half  # multiplication (not **2) to stay bit-true with the Rust kernel
    spectrum = np.ones_like(freqs)
    for center in centers:
        delta = freqs - center
        spectrum -= contrast * half_sq / (delta * delta + half_sq)
    return spectrum


def odmr_spectrum(
    freqs: NDArray[np.floating],
    nv: NVCenter,
    field_tesla: float,
    theta_rad: float = 0.0,
    phi_rad: float = 0.0,
) -> NDArray[np.float64]:
    """Normalised CW-ODMR photoluminescence spectrum (a Lorentzian dip per line)."""
    grid = np.ascontiguousarray(freqs, dtype=np.float64)
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError("freqs must be a non-empty one-dimensional array")
    lower, upper = odmr_resonances_hz(nv, field_tesla, theta_rad, phi_rad)
    centers = np.array([lower, upper], dtype=np.float64)
    return _odmr_spectrum_dispatch(grid, centers, nv.linewidth_hz, nv.contrast)


def _odmr_spectrum_dispatch(
    freqs: NDArray[np.float64], centers: NDArray[np.float64], fwhm: float, contrast: float
) -> NDArray[np.float64]:
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "nv_odmr_spectrum"):
            return np.asarray(
                _engine.nv_odmr_spectrum(freqs, np.ascontiguousarray(centers), fwhm, contrast),
                dtype=np.float64,
            )
    except (ImportError, AttributeError, ValueError):
        pass
    return _lorentzian_dip(freqs, centers, fwhm, contrast)


def simulate_odmr_measurement(
    freqs: NDArray[np.floating],
    nv: NVCenter,
    field_tesla: float,
    *,
    theta_rad: float = 0.0,
    phi_rad: float = 0.0,
    noise_std: float = 0.0,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """ODMR spectrum with additive Gaussian readout noise (synthetic measurement)."""
    spectrum = odmr_spectrum(freqs, nv, field_tesla, theta_rad, phi_rad)
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if noise_std > 0.0:
        rng = np.random.default_rng(seed)
        spectrum = spectrum + rng.normal(0.0, noise_std, size=spectrum.shape)
    return spectrum


@dataclass(frozen=True)
class NVFieldCalibration:
    """Field recovered from a noisy ODMR spectrum."""

    field_tesla: float
    true_field_tesla: float
    residual: float
    abs_error_tesla: float
    n_grid: int
    details: Mapping[str, float] = field(default_factory=dict)


def _parabolic_peak(grid: NDArray[np.float64], signal: NDArray[np.float64], idx: int) -> float:
    """Sub-grid peak position by parabolic interpolation around index ``idx``."""
    if 0 < idx < grid.size - 1:
        y0, y1, y2 = signal[idx - 1], signal[idx], signal[idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            offset = 0.5 * (y0 - y2) / denom
            return float(grid[idx] + offset * (grid[1] - grid[0]))
    return float(grid[idx])


def calibrate_field_from_odmr(
    freqs: NDArray[np.floating],
    measured_spectrum: NDArray[np.floating],
    nv: NVCenter,
    *,
    theta_rad: float = 0.0,
    true_field_tesla: float | None = None,
) -> NVFieldCalibration:
    """Recover the axial field from a noisy ODMR spectrum.

    The upper ODMR resonance is ``D + gamma_e B`` for every field magnitude (it is
    monotonic and does not reflect about the GSLAC), so the highest-frequency dip
    in the spectrum determines ``B`` unambiguously. Dips are located by prominence
    and refined to sub-grid resolution; the field follows from the upper dip.

    ``freqs`` must be a fine scan around the upper resonance (a few samples per
    FWHM), as a real ODMR field-tracking loop does. The deepest dip in the window
    is the upper resonance; an under-sampled grid cannot locate a sub-linewidth
    dip reliably.
    """
    grid = np.ascontiguousarray(freqs, dtype=np.float64)
    measured = np.ascontiguousarray(measured_spectrum, dtype=np.float64)
    if grid.shape != measured.shape:
        raise ValueError("freqs and measured_spectrum must have the same shape")
    if grid.size < 3:
        raise ValueError("at least three frequency samples are required")

    dip_signal = 1.0 - measured
    deepest = int(np.argmax(dip_signal))
    if dip_signal[deepest] < 0.5 * nv.contrast:
        raise ValueError("no ODMR dip resolved in the scan window")
    f_upper = _parabolic_peak(grid, dip_signal, deepest)
    b_recovered = max((f_upper - nv.zero_field_splitting_hz) / nv.gyromagnetic_hz_per_t, 0.0)

    model = odmr_spectrum(grid, nv, b_recovered, theta_rad)
    residual = float(np.sum((model - measured) ** 2))
    abs_error = (
        abs(b_recovered - true_field_tesla) if true_field_tesla is not None else float("nan")
    )
    return NVFieldCalibration(
        field_tesla=float(b_recovered),
        true_field_tesla=float(true_field_tesla) if true_field_tesla is not None else float("nan"),
        residual=residual,
        abs_error_tesla=float(abs_error),
        n_grid=int(grid.size),
        details={"f_upper_hz": float(f_upper)},
    )


__all__ = [
    "ELECTRON_GYROMAGNETIC_HZ_PER_T",
    "NV_ZERO_FIELD_SPLITTING_HZ",
    "NVCenter",
    "NVFieldCalibration",
    "calibrate_field_from_odmr",
    "cw_odmr_dc_sensitivity_t_per_sqrt_hz",
    "nv_energy_levels_hz",
    "nv_ground_state_hamiltonian",
    "odmr_resonances_hz",
    "odmr_spectrum",
    "simulate_odmr_measurement",
]
