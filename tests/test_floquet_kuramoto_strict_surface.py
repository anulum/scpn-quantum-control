# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Floquet Kuramoto Strict Surface Tests
"""Strict-surface tests for typed Floquet-Kuramoto behavior."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.phase import floquet_kuramoto as module


def test_build_h_matrix_normalizes_dense_complex_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dense Hamiltonian wrapper preserves budget forwarding and complex dtype."""
    calls: list[tuple[tuple[int, int], tuple[int], float | None]] = []

    def fake_dense_matrix(
        K: module.FloatArray,
        omega: module.FloatArray,
        *,
        max_dense_gib: float | None = None,
    ) -> module.FloatArray:
        calls.append((K.shape, omega.shape, max_dense_gib))
        return np.eye(2, dtype=np.float64)

    monkeypatch.setattr(module, "knm_to_dense_matrix", fake_dense_matrix)

    matrix = module._build_H_matrix(
        np.zeros((2, 2), dtype=np.float64),
        np.zeros(2, dtype=np.float64),
        max_dense_gib=0.25,
    )

    assert calls == [((2, 2), (2,), 0.25)]
    assert matrix.dtype == np.complex128
    assert matrix.shape == (2, 2)


def test_phase_facade_exports_dtc_threshold() -> None:
    """The package-level phase facade exposes the documented DTC threshold."""
    from scpn_quantum_control import phase

    assert phase.DTC_SUBHARMONIC_THRESHOLD == module.DTC_SUBHARMONIC_THRESHOLD


def test_order_parameter_uses_rust_fast_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional Rust engine path consumes contiguous real and imaginary buffers."""

    class Engine:
        def all_xy_expectations(
            self,
            psi_re: module.FloatArray,
            psi_im: module.FloatArray,
            n: int,
        ) -> tuple[module.FloatArray, module.FloatArray]:
            assert psi_re.flags.c_contiguous
            assert psi_im.flags.c_contiguous
            assert n == 2
            return (
                np.array([1.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0], dtype=np.float64),
            )

    monkeypatch.setattr(module, "optional_rust_engine", Engine)

    psi = np.ones(4, dtype=np.complex128) / 2.0
    assert module._order_parameter(psi, 2) == pytest.approx(np.sqrt(0.5))


def test_order_parameter_falls_back_without_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    """The NumPy fallback remains available when the Rust extension is absent."""
    monkeypatch.setattr(module, "optional_rust_engine", lambda: None)

    psi = np.ones(4, dtype=np.complex128) / 2.0
    assert module._order_parameter(psi, 2) == pytest.approx(1.0)


def test_subharmonic_ratio_boundaries_and_positive_response() -> None:
    """Subharmonic detection handles short, zero-power, and positive spectra."""
    assert module._subharmonic_ratio(np.array([0.2, 0.1], dtype=np.float64), 1.0, 0.1) == 0.0
    assert module._subharmonic_ratio(np.zeros(64, dtype=np.float64), 10.0, 0.01) == 0.0

    drive_frequency = 8.0 * np.pi
    dt = 1.0 / 128.0
    times = np.arange(256, dtype=np.float64) * dt
    signal = np.cos(0.5 * drive_frequency * times) + 0.25 * np.cos(drive_frequency * times)

    assert module._subharmonic_ratio(signal.astype(np.float64), drive_frequency, dt) > 1.0


def test_floquet_evolve_and_scan_with_typed_dense_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A small driven system returns typed arrays and scan summaries."""
    monkeypatch.setattr(module, "optional_rust_engine", lambda: None)

    topology = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
    omega = np.array([0.5, 1.0], dtype=np.float64)
    result = module.floquet_evolve(
        topology,
        omega,
        K_base=0.8,
        drive_amplitude=0.25,
        drive_frequency=2.0,
        n_periods=1,
        steps_per_period=4,
        max_dense_gib=1.0,
    )

    assert result.times.dtype == np.float64
    assert result.R_values.dtype == np.float64
    assert result.drive_signal.dtype == np.float64
    assert result.times.shape == (5,)
    assert result.subharmonic_ratio >= 0.0
    assert result.is_dtc_candidate is (result.subharmonic_ratio > module.DTC_SUBHARMONIC_THRESHOLD)

    scan = module.scan_drive_amplitude(
        topology,
        omega,
        K_base=0.8,
        drive_frequency=2.0,
        amplitudes=np.array([0.1, 0.3], dtype=np.float64),
        n_periods=1,
        steps_per_period=4,
        max_dense_gib=1.0,
    )

    assert scan["amplitude"] == [0.1, 0.3]
    assert len(scan["subharmonic_ratio"]) == 2
    assert len(scan["mean_R"]) == 2
    assert all(value in {0.0, 1.0} for value in scan["is_dtc"])


def test_scan_drive_amplitude_default_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default amplitude grid remains a ten-point public API."""
    calls: list[float] = []

    def fake_evolve(
        K_topology: module.FloatArray,
        omega: module.FloatArray,
        K_base: float,
        drive_amplitude: float,
        drive_frequency: float,
        n_periods: int = 10,
        steps_per_period: int = 20,
        *,
        max_dense_gib: float | None = None,
    ) -> module.FloquetResult:
        del K_topology, omega, K_base, drive_frequency, n_periods, steps_per_period, max_dense_gib
        calls.append(drive_amplitude)
        return module.FloquetResult(
            times=np.array([0.0], dtype=np.float64),
            R_values=np.array([0.25], dtype=np.float64),
            drive_signal=np.array([drive_amplitude], dtype=np.float64),
            subharmonic_ratio=drive_amplitude,
            is_dtc_candidate=drive_amplitude > module.DTC_SUBHARMONIC_THRESHOLD,
        )

    monkeypatch.setattr(module, "floquet_evolve", fake_evolve)

    scan = module.scan_drive_amplitude(
        np.zeros((1, 1), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        K_base=1.0,
        drive_frequency=2.0,
    )

    assert len(calls) == 10
    assert scan["amplitude"] == pytest.approx(np.linspace(0.1, 2.0, 10).tolist())
    assert scan["mean_R"] == [0.25] * 10
