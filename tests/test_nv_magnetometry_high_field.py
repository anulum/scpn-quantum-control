# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for high-field NV magnetometry
"""Branch tests for the high-field NV-centre magnetometry model.

Covers the strain validation guard, the energy-level helper, the Python
Lorentzian fallback of the ODMR spectrum dispatch, the noise guard, the
parabolic-peak boundary case, and the two field-calibration guards.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.sensing.nv_magnetometry_20T import (
    NVCenter,
    _lorentzian_dip,
    _odmr_spectrum_dispatch,
    _parabolic_peak,
    calibrate_field_from_odmr,
    nv_energy_levels_hz,
    simulate_odmr_measurement,
)


def test_rejects_negative_transverse_strain() -> None:
    """A negative transverse strain term is rejected at construction."""
    with pytest.raises(ValueError, match="transverse_strain_hz must be finite and non-negative"):
        NVCenter(transverse_strain_hz=-1.0)


def test_energy_levels_are_three_sorted_reals() -> None:
    """The spin-1 ground state yields three ascending real energy levels."""
    levels = nv_energy_levels_hz(NVCenter(), 0.1)
    assert levels.shape == (3,)
    assert levels.dtype == np.float64
    assert np.all(np.diff(levels) >= 0.0)


def test_odmr_dispatch_falls_back_to_python_lorentzian(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the native kernel is absent or errors, the Python Lorentzian is used."""
    freqs = np.linspace(2.80e9, 2.94e9, 64, dtype=np.float64)
    centers = np.array([2.85e9, 2.89e9], dtype=np.float64)
    fwhm, contrast = 1.0e6, 0.03

    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "nv_odmr_spectrum"):

            def _raise(*_args: object, **_kwargs: object) -> NDArray[np.float64]:
                raise ValueError("forced fallback to the Python kernel")

            monkeypatch.setattr(engine, "nv_odmr_spectrum", _raise)
    except ImportError:
        pass

    result = _odmr_spectrum_dispatch(freqs, centers, fwhm, contrast)
    expected = _lorentzian_dip(freqs, centers, fwhm, contrast)
    np.testing.assert_allclose(result, expected)


def test_rejects_negative_noise_std() -> None:
    """A negative readout-noise standard deviation is rejected."""
    freqs = np.linspace(2.80e9, 2.94e9, 16, dtype=np.float64)
    with pytest.raises(ValueError, match="noise_std must be non-negative"):
        simulate_odmr_measurement(freqs, NVCenter(), 0.05, noise_std=-1.0)


def test_parabolic_peak_returns_grid_point_at_boundary() -> None:
    """At a boundary index the sub-grid refinement falls back to the grid point."""
    grid = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    signal = np.array([1.0, 0.5, 0.2], dtype=np.float64)
    assert _parabolic_peak(grid, signal, 0) == 0.0


def test_calibration_requires_three_samples() -> None:
    """Field calibration needs at least three frequency samples."""
    freqs = np.array([2.88e9, 2.89e9], dtype=np.float64)
    spectrum = np.array([1.0, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="at least three frequency samples"):
        calibrate_field_from_odmr(freqs, spectrum, NVCenter())


def test_calibration_rejects_dip_free_window() -> None:
    """A flat (dip-free) spectrum yields no resolvable resonance."""
    freqs = np.linspace(2.80e9, 2.94e9, 32, dtype=np.float64)
    spectrum = np.ones_like(freqs)
    with pytest.raises(ValueError, match="no ODMR dip resolved"):
        calibrate_field_from_odmr(freqs, spectrum, NVCenter())
