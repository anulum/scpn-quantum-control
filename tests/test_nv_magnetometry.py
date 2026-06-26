# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for NV-centre 20 T magnetometry (QUA-C.5)
"""Tests for sensing/nv_magnetometry_20T.py (simulation-only + hardware-gated)."""

import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.sensing.nv_magnetometry_20T import (
    ELECTRON_GYROMAGNETIC_HZ_PER_T as GAMMA,
)
from scpn_quantum_control.sensing.nv_magnetometry_20T import (
    NV_ZERO_FIELD_SPLITTING_HZ as D,
)
from scpn_quantum_control.sensing.nv_magnetometry_20T import (
    NVCenter,
    _lorentzian_dip,
    calibrate_field_from_odmr,
    cw_odmr_dc_sensitivity_t_per_sqrt_hz,
    nv_ground_state_hamiltonian,
    odmr_resonances_hz,
    odmr_spectrum,
    simulate_odmr_measurement,
)

try:
    import scpn_quantum_engine as _engine

    _HAS_RUST = hasattr(_engine, "nv_odmr_spectrum")
except ImportError:  # pragma: no cover - engine optional
    _engine = None
    _HAS_RUST = False


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kw",
    [
        {"zero_field_splitting_hz": 0.0},
        {"gyromagnetic_hz_per_t": -1.0},
        {"contrast": 0.0},
        {"contrast": 1.5},
        {"linewidth_hz": 0.0},
        {"photon_rate_per_s": 0.0},
    ],
)
def test_nv_center_rejects_bad_params(kw):
    with pytest.raises(ValueError):
        NVCenter(**kw)


def test_hamiltonian_hermitian():
    h = nv_ground_state_hamiltonian(NVCenter(), 5.0, theta_rad=0.7, phi_rad=0.3)
    assert np.allclose(h, h.conj().T)


def test_negative_field_rejected():
    with pytest.raises(ValueError):
        nv_ground_state_hamiltonian(NVCenter(), -1.0)


# --------------------------------------------------------------------------- #
# Spin-Hamiltonian physics
# --------------------------------------------------------------------------- #
def test_zero_field_resonances_equal_d():
    lo, hi = odmr_resonances_hz(NVCenter(), 0.0)
    assert lo == pytest.approx(D, rel=1e-9)
    assert hi == pytest.approx(D, rel=1e-9)


@pytest.mark.parametrize("b", [0.01, 0.05, 0.08])
def test_axial_resonances_linear_below_gslac(b):
    lo, hi = odmr_resonances_hz(NVCenter(), b, theta_rad=0.0)
    assert lo == pytest.approx(D - GAMMA * b, rel=1e-9)
    assert hi == pytest.approx(D + GAMMA * b, rel=1e-9)


def test_gslac_lower_resonance_vanishes():
    b_gslac = D / GAMMA
    lo, _ = odmr_resonances_hz(NVCenter(), b_gslac, theta_rad=0.0)
    assert lo == pytest.approx(0.0, abs=1.0)  # within 1 Hz of the crossing


@pytest.mark.parametrize("b", [1.0, 5.0, 20.0])
def test_high_field_regime(b):
    lo, hi = odmr_resonances_hz(NVCenter(), b, theta_rad=0.0)
    assert lo == pytest.approx(GAMMA * b - D, rel=1e-9)  # |D - gamma B| past GSLAC
    assert hi == pytest.approx(D + GAMMA * b, rel=1e-9)


def test_upper_resonance_monotonic_in_field():
    nv = NVCenter()
    uppers = [odmr_resonances_hz(nv, b)[1] for b in np.linspace(0.0, 20.0, 50)]
    assert np.all(np.diff(uppers) > 0)


# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #
def test_sensitivity_positive_and_scales():
    base = cw_odmr_dc_sensitivity_t_per_sqrt_hz(NVCenter())
    assert base > 0.0
    # Narrower line, higher contrast, more photons all improve (lower) sensitivity.
    better = cw_odmr_dc_sensitivity_t_per_sqrt_hz(
        NVCenter(linewidth_hz=5.0e5, contrast=0.06, photon_rate_per_s=4.0e12)
    )
    assert better < base


# --------------------------------------------------------------------------- #
# ODMR spectrum
# --------------------------------------------------------------------------- #
def test_spectrum_dips_at_resonances():
    nv = NVCenter()
    lo, hi = odmr_resonances_hz(nv, 0.05)
    freqs = np.linspace(lo - 5e6, hi + 5e6, 4000)
    spectrum = odmr_spectrum(freqs, nv, 0.05)
    assert spectrum.max() == pytest.approx(1.0, abs=1e-3)
    # The two deepest points sit at the two resonances.
    assert spectrum.min() == pytest.approx(1.0 - nv.contrast, abs=1e-3)


def test_spectrum_rejects_empty_grid():
    with pytest.raises(ValueError):
        odmr_spectrum(np.array([]), NVCenter(), 1.0)


# --------------------------------------------------------------------------- #
# Field calibration (simulation only)
# --------------------------------------------------------------------------- #
def _resolved_window(nv: NVCenter, b_true: float, half_width_hz: float = 5.0e7, n: int = 4000):
    """A fine ODMR scan window around the upper resonance (resolves the linewidth)."""
    f_upper = odmr_resonances_hz(nv, b_true)[1]
    return np.linspace(f_upper - half_width_hz, f_upper + half_width_hz, n)


@pytest.mark.parametrize("b_true", [0.0734, 0.12, 0.5, 1.5, 5.0, 20.0])
def test_calibration_recovers_field_to_microtesla(b_true):
    nv = NVCenter()
    freqs = _resolved_window(nv, b_true)
    measured = simulate_odmr_measurement(
        nv=nv, freqs=freqs, field_tesla=b_true, noise_std=0.004, seed=5
    )
    cal = calibrate_field_from_odmr(freqs, measured, nv, true_field_tesla=b_true)
    assert cal.abs_error_tesla < 5.0e-5  # < 50 microtesla across 0.07-20 T


def test_calibration_shape_mismatch_raises():
    nv = NVCenter()
    with pytest.raises(ValueError):
        calibrate_field_from_odmr(np.linspace(0, 1, 10), np.ones(9), nv)


@settings(max_examples=20, deadline=None)
@given(b_true=st.floats(min_value=0.2, max_value=20.0))
def test_calibration_property(b_true):
    nv = NVCenter()
    freqs = _resolved_window(nv, b_true)
    measured = simulate_odmr_measurement(
        nv=nv, freqs=freqs, field_tesla=b_true, noise_std=0.003, seed=1
    )
    cal = calibrate_field_from_odmr(freqs, measured, nv, true_field_tesla=b_true)
    assert cal.abs_error_tesla < 1.0e-4


# --------------------------------------------------------------------------- #
# Rust ↔ NumPy parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine nv_odmr_spectrum not built")
@settings(max_examples=40, deadline=None)
@given(
    n_freqs=st.integers(min_value=1, max_value=2000),
    fwhm_mhz=st.floats(min_value=0.1, max_value=10.0),
    contrast=st.floats(min_value=0.001, max_value=0.3),
)
def test_odmr_spectrum_rust_parity(n_freqs, fwhm_mhz, contrast):
    freqs = np.linspace(2.0e9, 4.0e9, n_freqs)
    centers = np.array([2.5e9, 3.3e9], dtype=np.float64)
    rust = np.asarray(
        _engine.nv_odmr_spectrum(
            np.ascontiguousarray(freqs), np.ascontiguousarray(centers), fwhm_mhz * 1e6, contrast
        )
    )
    python = _lorentzian_dip(freqs, centers, fwhm_mhz * 1e6, contrast)
    assert np.array_equal(rust, python)  # bit-true


# --------------------------------------------------------------------------- #
# Hardware-gated (skips unless MIF_NV_HARDWARE_CI=1)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    os.environ.get("MIF_NV_HARDWARE_CI") != "1",
    reason="hardware NV magnetometry gated by MIF_NV_HARDWARE_CI=1",
)
def test_hardware_nv_calibration():  # pragma: no cover - hardware only
    assert os.environ.get("MIF_NV_HARDWARE_CI") == "1"
    raise AssertionError("NV hardware calibration requires a magnet and is not run in CI")
