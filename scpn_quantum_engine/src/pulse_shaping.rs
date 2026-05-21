// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — (α,β)-Hypergeometric Pulse Shaping

//! Rust-accelerated (α,β)-hypergeometric pulse envelope computation.
//!
//! Evaluates Ω(t)/Ω₀ = sech(γt) × ₂F₁(α, β; (α+β+1)/2; (1+tanh(γt))/2)
//! in parallel across time points via rayon.
//!
//! The Gauss hypergeometric series ₂F₁(a,b;c;z) = Σ (a)_n(b)_n/(c)_n × z^n/n!
//! converges for |z| < 1. Since z = (1+tanh(γt))/2 ∈ [0,1), convergence is
//! guaranteed for all finite t.
//!
//! Ref: Ventura Meinersen et al., arXiv:2504.08031 (2025)

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::validation::{
    validate_contiguous_slice, validate_finite, validate_n, validate_positive,
};

fn validate_finite_scalar(value: f64, name: &str) -> PyResult<()> {
    if !value.is_finite() {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite, got {value}"
        )));
    }
    Ok(())
}

fn validate_non_negative_scalar(value: f64, name: &str) -> PyResult<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and non-negative, got {value}"
        )));
    }
    Ok(())
}

fn validate_unit_interval(value: f64, name: &str) -> PyResult<()> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(PyValueError::new_err(format!(
            "{name} must be finite and in [0, 1], got {value}"
        )));
    }
    Ok(())
}

#[inline]
fn rho_idx(row: usize, col: usize) -> usize {
    row * 3 + col
}

/// Gauss hypergeometric function ₂F₁(a, b; c; z) via series expansion.
///
/// Convergent for |z| < 1. Uses Pochhammer rising factorial.
/// Terminates when |term| < tol or after max_terms iterations.
pub fn hyp2f1(a: f64, b: f64, c: f64, z: f64) -> f64 {
    // Special case: z = 0
    if z.abs() < 1e-300 {
        return 1.0;
    }
    // Special case: a = 0 or b = 0
    if a.abs() < 1e-300 || b.abs() < 1e-300 {
        return 1.0;
    }

    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    let max_terms = 500;
    let tol = 1e-15;

    for n in 0..max_terms {
        let nf = n as f64;
        term *= (a + nf) * (b + nf) / ((c + nf) * (nf + 1.0)) * z;
        sum += term;
        if term.abs() < tol * sum.abs() {
            break;
        }
    }

    sum
}

/// Compute hypergeometric pulse envelope for an array of time points.
///
/// envelope[i] = sech(γ × t[i]) × ₂F₁(α, β; (α+β+1)/2; (1+tanh(γ×t[i]))/2)
///
/// Parallelised across time points via rayon.
#[pyfunction]
pub fn hypergeometric_envelope_batch<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'_, f64>,
    alpha: f64,
    beta: f64,
    gamma_width: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_positive(gamma_width, "gamma_width")?;
    validate_finite_scalar(alpha, "alpha")?;
    validate_finite_scalar(beta, "beta")?;
    validate_n(times.len()?, "times")?;

    let t = validate_contiguous_slice(&times, "times")?;
    validate_finite(t, "times")?;
    let c = (alpha + beta + 1.0) / 2.0;

    let envelope: Vec<f64> = t
        .par_iter()
        .map(|&ti| {
            let gt = gamma_width * ti;
            let sech = 1.0 / gt.cosh();
            let z = 0.5 * (1.0 + gt.tanh());
            // Clamp z away from 1.0 for series convergence
            let z_safe = z.min(1.0 - 1e-15);
            sech * hyp2f1(alpha, beta, c, z_safe)
        })
        .collect();

    Ok(PyArray1::from_vec(py, envelope))
}

/// Build complete ICI mixing angle array.
///
/// Three-segment PMP-optimal trajectory:
/// Seg 1 (0..t1): linear ramp 0 → θ_jump
/// Seg 2 (t1..t2): smooth sweep θ_jump → π/2−θ_jump
/// Seg 3 (t2..T): linear ramp → π/2
#[pyfunction]
pub fn ici_mixing_angle_batch<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'_, f64>,
    t_total: f64,
    theta_jump: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_positive(t_total, "t_total")?;
    validate_positive(theta_jump, "theta_jump")?;

    let t = validate_contiguous_slice(&times, "times")?;
    validate_finite(t, "times")?;
    let t1 = 0.05 * t_total;
    let t2 = 0.95 * t_total;
    let half_pi = std::f64::consts::FRAC_PI_2;

    let theta: Vec<f64> = t
        .par_iter()
        .map(|&ti| {
            if ti < t1 {
                theta_jump * (ti / t1)
            } else if ti < t2 {
                let s = (ti - t1) / (t2 - t1);
                theta_jump + (half_pi - 2.0 * theta_jump) * s
            } else {
                let s = (ti - t2) / (t_total - t2);
                (half_pi - theta_jump) + theta_jump * s
            }
        })
        .collect();

    Ok(PyArray1::from_vec(py, theta))
}

/// Simulate 3-level Lambda system under an ICI pulse sequence with
/// Lindblad decay from the excited state.
///
/// Hamiltonian in the rotating frame:
///   H(t) = Omega_P(t) |e><g| + Omega_S(t) |e><s| + h.c.
///
/// Lindblad operator: L = sqrt(gamma) |g><e| (decay from |e> to |g>)
///
/// Returns populations [P_g, P_e, P_s] at each time point as a
/// flattened (n_t, 3) array stored row-major.
#[pyfunction]
pub fn ici_three_level_evolution_batch<'py>(
    py: Python<'py>,
    times: PyReadonlyArray1<'_, f64>,
    omega_p: PyReadonlyArray1<'_, f64>,
    omega_s: PyReadonlyArray1<'_, f64>,
    gamma: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_non_negative_scalar(gamma, "gamma")?;
    let t = validate_contiguous_slice(&times, "times")?;
    let op = validate_contiguous_slice(&omega_p, "omega_p")?;
    let os = validate_contiguous_slice(&omega_s, "omega_s")?;
    validate_finite(t, "times")?;
    validate_finite(op, "omega_p")?;
    validate_finite(os, "omega_s")?;
    let n_t = t.len();
    if n_t < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "times must have at least 2 points",
        ));
    }
    if op.len() != n_t || os.len() != n_t {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "omega_p and omega_s must match times length",
        ));
    }

    // Density matrix rho[3][3] stored as (re, im) pairs.
    // 9 complex entries = 18 f64 slots.
    // rho[i][j] = (re[i*3+j], im[i*3+j])
    let mut rho_re = [0.0_f64; 9];
    let mut rho_im = [0.0_f64; 9];
    rho_re[0] = 1.0; // start in ground state |g><g|

    let mut populations = vec![0.0_f64; n_t * 3];
    populations[0] = 1.0; // P_g(0) = 1
                          // populations[1..3] already zero

    for i in 1..n_t {
        let dt = t[i] - t[i - 1];
        let p = op[i];
        let s = os[i];

        // H is real symmetric 3x3: rows are
        //   row 0: [0, p, 0]
        //   row 1: [p, 0, s]
        //   row 2: [0, s, 0]
        // H @ rho:  (3x3 real) x (3x3 complex) = 3x3 complex
        // drho = -i [H, rho] = -i (H rho - rho H)

        let mut comm_re = [0.0_f64; 9];
        let mut comm_im = [0.0_f64; 9];

        // Compute H*rho
        // (H*rho)[i][j] = sum_k H[i][k] * rho[k][j]
        // H[0] = [0, p, 0]   so (H*rho)[0][j] = p * rho[1][j]
        // H[1] = [p, 0, s]   so (H*rho)[1][j] = p * rho[0][j] + s * rho[2][j]
        // H[2] = [0, s, 0]   so (H*rho)[2][j] = s * rho[1][j]
        for j in 0..3 {
            let hr0 = p * rho_re[rho_idx(1, j)];
            let hi0 = p * rho_im[rho_idx(1, j)];
            let hr1 = p * rho_re[rho_idx(0, j)] + s * rho_re[rho_idx(2, j)];
            let hi1 = p * rho_im[rho_idx(0, j)] + s * rho_im[rho_idx(2, j)];
            let hr2 = s * rho_re[rho_idx(1, j)];
            let hi2 = s * rho_im[rho_idx(1, j)];

            // rho*H: (rho*H)[i][j] = sum_k rho[i][k] * H[k][j]
            // H[k][0]: H[0][0]=0, H[1][0]=p, H[2][0]=0  => column 0 = [0, p, 0]
            // H[k][1]: H[0][1]=p, H[1][1]=0, H[2][1]=s  => column 1 = [p, 0, s]
            // H[k][2]: H[0][2]=0, H[1][2]=s, H[2][2]=0  => column 2 = [0, s, 0]
            // So (rho*H)[i][0] = p * rho[i][1]
            //    (rho*H)[i][1] = p * rho[i][0] + s * rho[i][2]
            //    (rho*H)[i][2] = s * rho[i][1]
            // But we need [i][j] here, not just j. Redo this loop differently.

            // For each i (row of comm):
            // comm[i][j] = (H*rho)[i][j] - (rho*H)[i][j]
            // (H*rho)[i][j] depends on i via H row
            // (rho*H)[i][j] depends on j via H column
            //
            // I computed (H*rho)[0][j], [1][j], [2][j] above. Good.
            // Now need (rho*H)[0][j], [1][j], [2][j] for same j.
            let rh0 = match j {
                0 => p * rho_re[rho_idx(0, 1)],
                1 => p * rho_re[rho_idx(0, 0)] + s * rho_re[rho_idx(0, 2)],
                _ => s * rho_re[rho_idx(0, 1)],
            };
            let ri0 = match j {
                0 => p * rho_im[rho_idx(0, 1)],
                1 => p * rho_im[rho_idx(0, 0)] + s * rho_im[rho_idx(0, 2)],
                _ => s * rho_im[rho_idx(0, 1)],
            };
            let rh1 = match j {
                0 => p * rho_re[rho_idx(1, 1)],
                1 => p * rho_re[rho_idx(1, 0)] + s * rho_re[rho_idx(1, 2)],
                _ => s * rho_re[rho_idx(1, 1)],
            };
            let ri1 = match j {
                0 => p * rho_im[rho_idx(1, 1)],
                1 => p * rho_im[rho_idx(1, 0)] + s * rho_im[rho_idx(1, 2)],
                _ => s * rho_im[rho_idx(1, 1)],
            };
            let rh2 = match j {
                0 => p * rho_re[rho_idx(2, 1)],
                1 => p * rho_re[rho_idx(2, 0)] + s * rho_re[rho_idx(2, 2)],
                _ => s * rho_re[rho_idx(2, 1)],
            };
            let ri2 = match j {
                0 => p * rho_im[rho_idx(2, 1)],
                1 => p * rho_im[rho_idx(2, 0)] + s * rho_im[rho_idx(2, 2)],
                _ => s * rho_im[rho_idx(2, 1)],
            };

            comm_re[rho_idx(0, j)] = hr0 - rh0;
            comm_im[rho_idx(0, j)] = hi0 - ri0;
            comm_re[rho_idx(1, j)] = hr1 - rh1;
            comm_im[rho_idx(1, j)] = hi1 - ri1;
            comm_re[rho_idx(2, j)] = hr2 - rh2;
            comm_im[rho_idx(2, j)] = hi2 - ri2;
        }

        // drho = -i * commutator:  multiplying by -i flips real/imag and
        // negates the new real part.
        //   (-i) * (a + ib) = b - ia
        let mut drho_re = [0.0_f64; 9];
        let mut drho_im = [0.0_f64; 9];
        for k in 0..9 {
            drho_re[k] = comm_im[k];
            drho_im[k] = -comm_re[k];
        }

        // Lindblad decay from |e>: rho[1][1] population transfers to rho[0][0]
        // plus off-diagonal dephasing at rate gamma/2.
        if gamma > 0.0 {
            let pop_e = rho_re[rho_idx(1, 1)]; // real, diagonal
            drho_re[rho_idx(0, 0)] += gamma * pop_e;
            drho_re[rho_idx(1, 1)] -= gamma * pop_e;

            // Off-diagonal dephasing: rows/cols involving |e>
            drho_re[rho_idx(0, 1)] -= 0.5 * gamma * rho_re[rho_idx(0, 1)];
            drho_im[rho_idx(0, 1)] -= 0.5 * gamma * rho_im[rho_idx(0, 1)];
            drho_re[rho_idx(1, 0)] -= 0.5 * gamma * rho_re[rho_idx(1, 0)];
            drho_im[rho_idx(1, 0)] -= 0.5 * gamma * rho_im[rho_idx(1, 0)];
            drho_re[rho_idx(1, 2)] -= 0.5 * gamma * rho_re[rho_idx(1, 2)];
            drho_im[rho_idx(1, 2)] -= 0.5 * gamma * rho_im[rho_idx(1, 2)];
            drho_re[rho_idx(2, 1)] -= 0.5 * gamma * rho_re[rho_idx(2, 1)];
            drho_im[rho_idx(2, 1)] -= 0.5 * gamma * rho_im[rho_idx(2, 1)];
        }

        // Forward Euler step
        for k in 0..9 {
            rho_re[k] += drho_re[k] * dt;
            rho_im[k] += drho_im[k] * dt;
        }

        // Trace preservation (same normalisation as Python implementation)
        let trace = rho_re[0] + rho_re[4] + rho_re[8];
        if trace > 1e-15 {
            for k in 0..9 {
                rho_re[k] /= trace;
                rho_im[k] /= trace;
            }
        }

        populations[i * 3] = rho_re[0];
        populations[i * 3 + 1] = rho_re[4];
        populations[i * 3 + 2] = rho_re[8];
    }

    Ok(PyArray1::from_vec(py, populations))
}

/// Estimate a pi-pulse amplitude from a sampled Rabi sweep.
///
/// Uses the maximum excited-state population index with optional quadratic
/// interpolation against adjacent samples for sub-grid refinement.
#[pyfunction]
pub fn rabi_pi_amplitude_fit(
    amplitudes: PyReadonlyArray1<'_, f64>,
    excited_population: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64)> {
    let amps = validate_contiguous_slice(&amplitudes, "amplitudes")?;
    let pops = validate_contiguous_slice(&excited_population, "excited_population")?;
    validate_finite(amps, "amplitudes")?;
    validate_finite(pops, "excited_population")?;
    if amps.len() != pops.len() {
        return Err(PyValueError::new_err(format!(
            "amplitudes length {} != excited_population length {}",
            amps.len(),
            pops.len()
        )));
    }
    if amps.len() < 3 {
        return Err(PyValueError::new_err(
            "amplitudes/excited_population must contain at least 3 points",
        ));
    }

    for index in 1..amps.len() {
        if amps[index] <= amps[index - 1] {
            return Err(PyValueError::new_err(
                "amplitudes must be strictly increasing for Rabi fit",
            ));
        }
    }
    for &pop in pops {
        validate_unit_interval(pop, "excited_population entry")?;
    }

    let mut peak_index = 0usize;
    let mut peak_population = pops[0];
    for (index, &pop) in pops.iter().enumerate().skip(1) {
        if pop > peak_population {
            peak_population = pop;
            peak_index = index;
        }
    }

    let mut pi_amplitude = amps[peak_index];
    if peak_index > 0 && peak_index + 1 < amps.len() {
        let x0 = amps[peak_index - 1];
        let x1 = amps[peak_index];
        let x2 = amps[peak_index + 1];
        let y0 = pops[peak_index - 1];
        let y1 = pops[peak_index];
        let y2 = pops[peak_index + 1];

        let h0 = x0 - x1;
        let h2 = x2 - x1;
        let denom = (h0 * h2) * (h0 - h2);
        if denom.abs() > 1e-15 {
            let a = (h2 * (y0 - y1) - h0 * (y2 - y1)) / denom;
            let b = (h2.powi(2) * (y1 - y0) + h0.powi(2) * (y2 - y1)) / denom;
            if a.abs() > 1e-15 {
                let vertex = x1 - b / (2.0 * a);
                if vertex >= x0 && vertex <= x2 {
                    pi_amplitude = vertex;
                }
            }
        }
    }

    // Contrast score: how strongly the peak dominates edges (0..1+).
    let edge_mean = 0.5 * (pops[0] + pops[pops.len() - 1]);
    let contrast = (peak_population - edge_mean).max(0.0);
    let confidence = if peak_population > 1e-12 {
        (contrast / peak_population).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok((pi_amplitude, peak_population, confidence))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyp2f1_trivial() {
        // ₂F₁(a, b; c; 0) = 1 for any a, b, c
        assert!((hyp2f1(0.5, 0.5, 0.75, 0.0) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_hyp2f1_allen_eberly() {
        // ₂F₁(0, 0; 0.5; z) = 1 for any z (both a and b zero)
        assert!((hyp2f1(0.0, 0.0, 0.5, 0.3) - 1.0).abs() < 1e-15);
        assert!((hyp2f1(0.0, 0.0, 0.5, 0.9) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_hyp2f1_known_value() {
        // ₂F₁(1, 1; 2; z) = -ln(1-z)/z for z > 0
        let z: f64 = 0.5;
        let expected = -(1.0_f64 - z).ln() / z;
        let computed = hyp2f1(1.0, 1.0, 2.0, z);
        assert!(
            (computed - expected).abs() < 1e-10,
            "₂F₁(1,1;2;0.5) = {computed}, expected {expected}"
        );
    }

    #[test]
    fn test_hyp2f1_stirap() {
        // ₂F₁(0.5, 0.5; 0.75; z) should be > 1 for z > 0
        let val = hyp2f1(0.5, 0.5, 0.75, 0.5);
        assert!(val > 1.0, "STIRAP case: ₂F₁(0.5,0.5;0.75;0.5) = {val}");
    }

    #[test]
    fn test_envelope_sech_at_zero() {
        // At t=0: sech(0)=1, z=0.5, ₂F₁(0,0;0.5;0.5)=1 → envelope = 1
        let gt: f64 = 0.0;
        let sech = 1.0 / gt.cosh();
        let z = 0.5 * (1.0 + gt.tanh());
        let val = sech * hyp2f1(0.0, 0.0, 0.5, z);
        assert!((val - 1.0).abs() < 1e-15, "Allen-Eberly at t=0 = {val}");
    }

    #[test]
    fn test_ici_angle_boundaries() {
        let t_total: f64 = 1.0;
        let theta_jump: f64 = 0.3;
        let t1: f64 = 0.05;
        let t2: f64 = 0.95;

        // At t=0: θ = 0
        let theta_0 = theta_jump * (0.0_f64 / t1);
        assert!(theta_0.abs() < 1e-15);

        // At t=t_total: θ ≈ π/2
        let s = (t_total - t2) / (t_total - t2);
        let theta_end = (std::f64::consts::FRAC_PI_2 - theta_jump) + theta_jump * s;
        assert!(
            (theta_end - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "θ(T) = {theta_end}, expected π/2"
        );
    }
}
