// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Monte Carlo XY Model Simulation

//! Metropolis Monte Carlo for the classical XY model on arbitrary coupling graphs.
//!
//! Simulates the XY Hamiltonian H = −Σ_{i<j} K_ij cos(θ_i − θ_j) via single-site
//! Metropolis updates. Computes energy, order parameter, and helicity modulus
//! (superfluid stiffness) averaged over measurement sweeps.

use pyo3::prelude::*;
use rand::prelude::*;

/// Monte Carlo XY model simulation on arbitrary coupling graph.
///
/// Returns (energy, order_parameter, helicity_modulus) averaged over n_measure sweeps.
#[pyfunction]
pub fn mc_xy_simulate(
    k_flat: numpy::PyReadonlyArray1<'_, f64>,
    n: usize,
    temperature: f64,
    n_thermalize: usize,
    n_measure: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let k_data = k_flat.as_slice().unwrap();
    let beta = if temperature > 1e-15 {
        1.0 / temperature
    } else {
        1e15
    };
    let mut rng = StdRng::seed_from_u64(seed);

    let mut theta: Vec<f64> = (0..n)
        .map(|_| rng.random::<f64>() * 2.0 * std::f64::consts::PI)
        .collect();

    for _ in 0..n_thermalize {
        mc_sweep(&mut theta, k_data, n, beta, &mut rng);
    }

    let mut e_sum = 0.0_f64;
    let mut r_sum = 0.0_f64;
    let mut cos_sum_acc = 0.0_f64;
    let mut sin2_sum_acc = 0.0_f64;

    for _ in 0..n_measure {
        mc_sweep(&mut theta, k_data, n, beta, &mut rng);

        let (e, cos_s, sin_s) = xy_observables(&theta, k_data, n);
        e_sum += e;
        cos_sum_acc += cos_s;
        sin2_sum_acc += sin_s * sin_s;

        let (re, im) = theta
            .iter()
            .fold((0.0, 0.0), |(r, i), &t| (r + t.cos(), i + t.sin()));
        r_sum += (re * re + im * im).sqrt() / n as f64;
    }

    let nm = n_measure as f64;
    let energy = e_sum / nm;
    let order = r_sum / nm;
    let rho_s = (cos_sum_acc / nm - beta * sin2_sum_acc / nm) / n as f64;

    (energy, order, rho_s)
}

/// Single Metropolis sweep over all sites.
pub fn mc_sweep(theta: &mut [f64], k: &[f64], n: usize, beta: f64, rng: &mut StdRng) {
    let pi = std::f64::consts::PI;
    for i in 0..n {
        let old = theta[i];
        let proposal = old + (rng.random::<f64>() - 0.5) * 2.0 * pi;

        let mut delta_e = 0.0;
        for j in 0..n {
            if i != j {
                let kij = k[i * n + j];
                if kij.abs() > 1e-15 {
                    delta_e -= kij * ((proposal - theta[j]).cos() - (old - theta[j]).cos());
                }
            }
        }

        if delta_e < 0.0 || rng.random::<f64>() < (-beta * delta_e).exp() {
            theta[i] = proposal;
        }
    }
}

/// Compute XY observables: (energy, cos_sum, sin_sum).
pub fn xy_observables(theta: &[f64], k: &[f64], n: usize) -> (f64, f64, f64) {
    let mut energy = 0.0;
    let mut cos_sum = 0.0;
    let mut sin_sum = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            let kij = k[i * n + j];
            if kij.abs() > 1e-15 {
                let d = theta[j] - theta[i];
                energy -= kij * d.cos();
                cos_sum += kij * d.cos();
                sin_sum += kij * d.sin();
            }
        }
    }
    (energy, cos_sum, sin_sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_mc_sweep_preserves_length() {
        let n = 4;
        let k = vec![0.0; n * n];
        let mut theta = vec![0.0, 1.0, 2.0, 3.0];
        let mut rng = StdRng::seed_from_u64(42);
        mc_sweep(&mut theta, &k, n, 1.0, &mut rng);
        assert_eq!(theta.len(), n);
    }

    #[test]
    fn test_xy_observables_zero_coupling() {
        let n = 3;
        let k = vec![0.0; n * n];
        let theta = vec![0.1, 0.5, 1.2];
        let (e, cos_s, sin_s) = xy_observables(&theta, &k, n);
        assert!(e.abs() < 1e-12, "zero coupling → zero energy");
        assert!(cos_s.abs() < 1e-12);
        assert!(sin_s.abs() < 1e-12);
    }

    #[test]
    fn test_xy_observables_aligned() {
        let n = 3;
        let mut k = vec![0.0; n * n];
        k[0 * n + 1] = 1.0;
        k[1 * n + 0] = 1.0;
        k[0 * n + 2] = 1.0;
        k[2 * n + 0] = 1.0;
        k[1 * n + 2] = 1.0;
        k[2 * n + 1] = 1.0;
        let theta = vec![0.0, 0.0, 0.0];
        let (e, cos_s, _) = xy_observables(&theta, &k, n);
        assert!((e - (-3.0)).abs() < 1e-12);
        assert!((cos_s - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_xy_observables_antiparallel() {
        let n = 2;
        let mut k = vec![0.0; n * n];
        k[0 * n + 1] = 1.0;
        k[1 * n + 0] = 1.0;
        let theta = vec![0.0, std::f64::consts::PI];
        let (e, cos_s, _) = xy_observables(&theta, &k, n);
        // cos(π) = −1, E = −K×cos(π) = +1
        assert!((e - 1.0).abs() < 1e-12);
        assert!((cos_s - (-1.0)).abs() < 1e-12);
    }
}
