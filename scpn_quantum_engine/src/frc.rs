// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — FRC pulsed-shot physics kernels

//! Magneto-Rayleigh-Taylor (MRTI) growth integral for the FRC pulsed-shot cost.
//!
//! Replicates the NumPy reference in
//! `scpn_quantum_control.control.qaoa_pulsed_cost.FRCPlasmaSurrogate.mrti_amplitude`:
//! second-order central-difference gradient of the magnetic pressure (matching
//! `numpy.gradient`), magnetic-tension-stabilised growth rate
//! `gamma = sqrt(max(A k g - k^2 B^2 / (mu0 (rho_h + rho_l)), 0))`, integrated to
//! the clipped e-folding count `sum(gamma) dt`.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const MU0: f64 = 4.0e-7 * std::f64::consts::PI;

/// `numpy.gradient(values, dt)` — second-order interior, first-order edges.
fn gradient(values: &[f64], dt: f64) -> Vec<f64> {
    let n = values.len();
    let mut out = vec![0.0_f64; n];
    if n == 1 {
        return out;
    }
    out[0] = (values[1] - values[0]) / dt;
    out[n - 1] = (values[n - 1] - values[n - 2]) / dt;
    for i in 1..n - 1 {
        out[i] = (values[i + 1] - values[i - 1]) / (2.0 * dt);
    }
    out
}

/// Integrated MRTI e-foldings over a field profile (clipped to `max_growth`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn frc_mrti_growth(
    field: PyReadonlyArray1<'_, f64>,
    dt_s: f64,
    wavenumber: f64,
    atwood: f64,
    areal_mass: f64,
    density: f64,
    max_growth: f64,
) -> PyResult<f64> {
    let b = field.as_slice()?;
    if b.len() < 2 {
        return Ok(0.0);
    }
    if dt_s <= 0.0 || areal_mass <= 0.0 || density <= 0.0 {
        return Err(PyValueError::new_err(
            "dt_s, areal_mass, density must be positive",
        ));
    }
    let magnetic_pressure: Vec<f64> = b.iter().map(|&x| x * x / (2.0 * MU0)).collect();
    let g = gradient(&magnetic_pressure, dt_s);
    let mut total = 0.0_f64;
    for i in 0..b.len() {
        let tension = wavenumber * wavenumber * b[i] * b[i] / (MU0 * 2.0 * density);
        let gamma_sq = atwood * wavenumber * (g[i] / areal_mass) - tension;
        if gamma_sq > 0.0 {
            total += gamma_sq.sqrt();
        }
    }
    let growth = (total * dt_s).clamp(0.0, max_growth);
    Ok(growth)
}
