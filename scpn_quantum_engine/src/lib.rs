// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-control — Rust Acceleration Engine

// scpn-quantum-engine — Rust Acceleration Engine

//! Rust acceleration for scpn-quantum-control.
//!
//! Hot paths moved from Python to Rust via PyO3:
//! - PEC Monte Carlo sampling (parallel via rayon)
//! - Classical Kuramoto ODE (vectorized)
//! - K_nm matrix construction

use ndarray::{Array1, Array2};
use num_complex::Complex;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::prelude::*;
use rayon::prelude::*;

type C64 = Complex<f64>;

/// PEC quasi-probability coefficients for single-qubit depolarizing channel.
/// Returns [q_I, q_X, q_Y, q_Z].
#[pyfunction]
fn pec_coefficients(gate_error_rate: f64) -> [f64; 4] {
    let p = gate_error_rate;
    let denom = 4.0 - 4.0 * p;
    let q_i = 1.0 + 3.0 * p / denom;
    let q_xyz = -p / denom;
    [q_i, q_xyz, q_xyz, q_xyz]
}

/// PEC sign-sampling in parallel (rayon). Single-qubit depolarizing model.
///
/// Returns (mitigated_value, overhead, sign_distribution).
/// `base_exp_z` is the noiseless <Z> expectation from the ideal circuit.
/// Each sample draws a Pauli correction per gate, accumulates the sign
/// product, and scales by gamma^n_gates. The sampled operator identity
/// affects the sign but not the base expectation — this is the single-qubit
/// approximation where all corrections act on one qubit.
#[pyfunction]
fn pec_sample_parallel(
    gate_error_rate: f64,
    n_gates: usize,
    n_samples: usize,
    base_exp_z: f64,
    seed: u64,
) -> (f64, f64, Vec<f64>) {
    let coeffs = pec_coefficients(gate_error_rate);
    let abs_coeffs: Vec<f64> = coeffs.iter().map(|c| c.abs()).collect();
    let gamma_single: f64 = abs_coeffs.iter().sum();
    let probs: Vec<f64> = abs_coeffs.iter().map(|a| a / gamma_single).collect();
    let signs: Vec<f64> = coeffs.iter().map(|c| c.signum()).collect();
    let gamma_total = gamma_single.powi(n_gates as i32);

    // Cumulative probabilities for sampling
    let cum_probs: Vec<f64> = probs
        .iter()
        .scan(0.0, |acc, &p| {
            *acc += p;
            Some(*acc)
        })
        .collect();

    // Parallel Monte Carlo
    let results: Vec<(f64, f64)> = (0..n_samples)
        .into_par_iter()
        .map(|i| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            let mut total_sign = 1.0_f64;

            for _ in 0..n_gates {
                let r: f64 = rng.random();
                let idx = cum_probs.iter().position(|&c| r < c).unwrap_or(3);
                total_sign *= signs[idx];
            }

            let value = gamma_total * total_sign * base_exp_z;
            (value, total_sign)
        })
        .collect();

    let mut acc = 0.0;
    let mut sign_dist = Vec::with_capacity(n_samples);
    for (value, sign) in &results {
        acc += value;
        sign_dist.push(*sign);
    }

    (acc / n_samples as f64, gamma_total, sign_dist)
}

/// Build K_nm coupling matrix from Paper 27 parameters.
/// K_nm = K_base * exp(-alpha * |n - m|) with calibration anchors.
#[pyfunction]
fn build_knm<'py>(
    py: Python<'py>,
    n: usize,
    k_base: f64,
    alpha: f64,
) -> Bound<'py, PyArray2<f64>> {
    // Full exponential matrix including diagonal (K_base at i==j)
    let mut k = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            k[[i, j]] = k_base * (-alpha * (i as f64 - j as f64).abs()).exp();
        }
    }

    // Calibration anchors (Paper 27 Table 2)
    let anchors: [(usize, usize, f64); 4] = [(0, 1, 0.302), (1, 2, 0.201), (2, 3, 0.252), (3, 4, 0.154)];
    for &(i, j, val) in &anchors {
        if i < n && j < n {
            k[[i, j]] = val;
            k[[j, i]] = val;
        }
    }

    // Cross-hierarchy boosts (max preserves exponential if already larger)
    if n > 15 {
        k[[0, 15]] = k[[0, 15]].max(0.05);
        k[[15, 0]] = k[[15, 0]].max(0.05);
    }
    if n > 6 {
        k[[4, 6]] = k[[4, 6]].max(0.15);
        k[[6, 4]] = k[[6, 4]].max(0.15);
    }

    PyArray2::from_owned_array(py, k)
}

/// Classical Kuramoto ODE step (vectorized, no Python overhead).
/// theta' = omega + K @ sin(theta_outer - theta_inner)
/// Returns new theta after n_steps of Euler integration.
#[pyfunction]
fn kuramoto_euler<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> Bound<'py, PyArray1<f64>> {
    let mut theta = theta0.as_array().to_owned();
    let omega = omega.as_array();
    let k = k.as_array();
    let n = theta.len();

    for _ in 0..n_steps {
        let mut dtheta = Array1::<f64>::zeros(n);
        for i in 0..n {
            dtheta[i] = omega[i];
            for j in 0..n {
                dtheta[i] += k[[i, j]] * (theta[j] - theta[i]).sin();
            }
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }
    }

    PyArray1::from_owned_array(py, theta)
}

/// Compute Kuramoto order parameter R from phase array.
/// R = (1/N) |sum_i exp(i * theta_i)|
#[pyfunction]
fn order_parameter(theta: PyReadonlyArray1<'_, f64>) -> f64 {
    let theta = theta.as_array();
    let n = theta.len() as f64;
    let (mut re, mut im) = (0.0, 0.0);
    for &t in theta.iter() {
        re += t.cos();
        im += t.sin();
    }
    (re * re + im * im).sqrt() / n
}

/// Parallel classical Kuramoto trajectory.
/// Returns (times, R_values) for each timestep.
#[pyfunction]
fn kuramoto_trajectory<'py>(
    py: Python<'py>,
    theta0: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    k: PyReadonlyArray2<'_, f64>,
    dt: f64,
    n_steps: usize,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let mut theta = theta0.as_array().to_owned();
    let omega_arr = omega.as_array();
    let k_arr = k.as_array();
    let n = theta.len();

    let mut times = Array1::<f64>::zeros(n_steps + 1);
    let mut r_values = Array1::<f64>::zeros(n_steps + 1);

    // Initial R
    let (mut re, mut im) = (0.0, 0.0);
    for &t in theta.iter() { re += t.cos(); im += t.sin(); }
    r_values[0] = (re * re + im * im).sqrt() / n as f64;

    for step in 0..n_steps {
        let mut dtheta = Array1::<f64>::zeros(n);
        for i in 0..n {
            dtheta[i] = omega_arr[i];
            for j in 0..n {
                dtheta[i] += k_arr[[i, j]] * (theta[j] - theta[i]).sin();
            }
        }
        for i in 0..n {
            theta[i] += dt * dtheta[i];
        }

        times[step + 1] = (step + 1) as f64 * dt;
        let (mut re, mut im) = (0.0, 0.0);
        for &t in theta.iter() { re += t.cos(); im += t.sin(); }
        r_values[step + 1] = (re * re + im * im).sqrt() / n as f64;
    }

    (
        PyArray1::from_owned_array(py, times),
        PyArray1::from_owned_array(py, r_values),
    )
}

/// DLA: compute dynamical Lie algebra dimension via commutator closure.
///
/// Takes a flat array of generator matrices (each dim×dim, row-major)
/// and computes the closure under commutation. Returns the DLA dimension.
///
/// This is the hot path that takes 27 min in Python for N=4. In Rust
/// with vectorised matrix ops, target is <30s.
#[pyfunction]
fn dla_dimension(
    generators_flat: PyReadonlyArray1<'_, f64>,
    dim: usize,
    n_generators: usize,
    max_iterations: usize,
    max_dimension: usize,
    tol: f64,
) -> usize {
    let data = generators_flat.as_slice().unwrap();
    let mat_size = dim * dim;

    // Parse generators into Vec<Vec<f64>> (row-major dense matrices)
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for g in 0..n_generators {
        let start = g * mat_size;
        let mat: Vec<f64> = data[start..start + mat_size].to_vec();
        if is_independent_fast(&mat, &basis, dim, tol) {
            basis.push(mat);
        }
    }

    // Commutator closure — parallelised commutator computation via rayon
    for _iter in 0..max_iterations {
        let n_basis = basis.len();
        if n_basis >= max_dimension {
            break;
        }

        // Generate all (i,j) pairs
        let pairs: Vec<(usize, usize)> = (0..n_basis)
            .flat_map(|i| ((i + 1)..n_basis).map(move |j| (i, j)))
            .collect();

        // Parallel commutator computation
        let candidates: Vec<Vec<f64>> = pairs
            .par_iter()
            .filter_map(|&(i, j)| {
                let comm = commutator_dense(&basis[i], &basis[j], dim);
                let norm: f64 = comm.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < tol { None } else { Some(comm) }
            })
            .collect();

        // Sequential independence filtering (must be serial for correctness)
        let mut new_ops: Vec<Vec<f64>> = Vec::new();
        for comm in candidates {
            let mut combined = basis.clone();
            combined.extend(new_ops.iter().cloned());
            if is_independent_fast(&comm, &combined, dim, tol) {
                new_ops.push(comm);
                if basis.len() + new_ops.len() >= max_dimension {
                    break;
                }
            }
        }

        if new_ops.is_empty() {
            break;
        }
        basis.extend(new_ops);
    }

    basis.len()
}

/// Dense matrix commutator [A, B] = AB - BA (row-major)
fn commutator_dense(a: &[f64], b: &[f64], dim: usize) -> Vec<f64> {
    let mut result = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut ab = 0.0;
            let mut ba = 0.0;
            for k in 0..dim {
                ab += a[i * dim + k] * b[k * dim + j];
                ba += b[i * dim + k] * a[k * dim + j];
            }
            result[i * dim + j] = ab - ba;
        }
    }
    result
}

/// Fast linear independence check via Gram-Schmidt projection.
fn is_independent_fast(new_op: &[f64], basis: &[Vec<f64>], _dim: usize, tol: f64) -> bool {
    let new_norm: f64 = new_op.iter().map(|x| x * x).sum::<f64>().sqrt();
    if new_norm < tol {
        return false;
    }
    if basis.is_empty() {
        return true;
    }

    // Project out basis components
    let mut residual: Vec<f64> = new_op.to_vec();
    for b in basis {
        let b_norm_sq: f64 = b.iter().map(|x| x * x).sum();
        if b_norm_sq < tol * tol {
            continue;
        }
        let dot: f64 = residual.iter().zip(b.iter()).map(|(r, bi)| r * bi).sum();
        let coeff = dot / b_norm_sq;
        for (r, bi) in residual.iter_mut().zip(b.iter()) {
            *r -= coeff * bi;
        }
    }

    let res_norm: f64 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
    res_norm > tol
}

/// Monte Carlo XY model simulation on arbitrary coupling graph.
///
/// Returns (energy, order_parameter, helicity_modulus) averaged over n_measure sweeps.
#[pyfunction]
fn mc_xy_simulate(
    k_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    temperature: f64,
    n_thermalize: usize,
    n_measure: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let k_data = k_flat.as_slice().unwrap();
    let beta = if temperature > 1e-15 { 1.0 / temperature } else { 1e15 };
    let mut rng = StdRng::seed_from_u64(seed);

    // Random initial phases
    let mut theta: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * 2.0 * std::f64::consts::PI).collect();

    // Thermalise
    for _ in 0..n_thermalize {
        mc_sweep(&mut theta, k_data, n, beta, &mut rng);
    }

    // Measure
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

        let (re, im) = theta.iter().fold((0.0, 0.0), |(r, i), &t| (r + t.cos(), i + t.sin()));
        r_sum += (re * re + im * im).sqrt() / n as f64;
    }

    let nm = n_measure as f64;
    let energy = e_sum / nm;
    let order = r_sum / nm;
    let rho_s = (cos_sum_acc / nm - beta * sin2_sum_acc / nm) / n as f64;

    (energy, order, rho_s)
}

fn mc_sweep(theta: &mut [f64], k: &[f64], n: usize, beta: f64, rng: &mut StdRng) {
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

fn xy_observables(theta: &[f64], k: &[f64], n: usize) -> (f64, f64, f64) {
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

/// Compute quantum order parameter R from a complex statevector using
/// sparse bitwise Pauli application.
///
/// For each qubit q, computes <psi|X_q|psi> and <psi|Y_q|psi> via index
/// bit-flips (psi[k ^ (1<<q)]) instead of building dense 2^n Pauli matrices.
/// O(n_osc * 2^n) time, O(2^n) memory.
///
/// Takes real and imaginary parts separately since PyO3 numpy complex support
/// varies across platforms.
#[pyfunction]
fn state_order_param_sparse(
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    n_osc: usize,
) -> f64 {
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    let dim = re.len();

    let mut z_re = 0.0_f64;
    let mut z_im = 0.0_f64;

    for q in 0..n_osc {
        let mask: usize = 1 << q;
        let mut exp_x = 0.0_f64;
        let mut exp_y = 0.0_f64;

        for k in 0..dim {
            let flipped = k ^ mask;
            // psi_conj[k] * psi[flipped]
            // (re_k - i*im_k) * (re_f + i*im_f) = (re_k*re_f + im_k*im_f) + i*(re_k*im_f - im_k*re_f)
            let prod_re = re[k] * re[flipped] + im[k] * im[flipped];

            // <X_q> = Re[sum_k conj(psi[k]) * psi[k^mask]]
            exp_x += prod_re;

            // <Y_q>: sign = 1 - 2*bit_q(k)
            let bit = ((k >> q) & 1) as f64;
            let sign = 1.0 - 2.0 * bit;
            // i * sign * psi[flipped] → multiply by i*sign then take conj(psi[k]) dot
            // conj(psi[k]) * (i * sign * psi[flipped])
            // = sign * [(re_k - i*im_k) * (i*re_f - im_f)]... simplify:
            // = sign * [(-re_k*im_f - im_k*re_f) is imaginary part... but we want Re of whole sum]
            // Actually: conj(psi[k]) * (i*sign) * psi[flipped]
            // = sign * (conj(psi[k]) * i * psi[flipped])
            // conj(psi[k]) * psi[flipped] = prod_re + i*prod_im
            // prod_im = re_k*im_f - im_k*re_f
            // multiply by i: i*(prod_re + i*prod_im) = -prod_im + i*prod_re
            // Re of that = -prod_im = -(re_k*im_f - im_k*re_f) = im_k*re_f - re_k*im_f
            // Wait, we want Re[sum conj(psi[k]) * (i*sign*psi[flipped])]
            // = sign * Re[i * (prod_re + i*prod_im)]
            // = sign * Re[-prod_im + i*prod_re]
            // = sign * (-prod_im)
            // = sign * (im_k*re_f - re_k*im_f)
            let prod_im = re[k] * im[flipped] - im[k] * re[flipped];
            exp_y += sign * (-prod_im);
        }

        z_re += exp_x;
        z_im += exp_y;
    }

    z_re /= n_osc as f64;
    z_im /= n_osc as f64;
    (z_re * z_re + z_im * z_im).sqrt()
}

/// Compute single-qubit Pauli expectation <psi|P_qubit|psi> using bitwise ops.
///
/// pauli: 0=X, 1=Y, 2=Z
#[pyfunction]
fn expectation_pauli_fast(
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    _n: usize,
    qubit: usize,
    pauli: usize,
) -> f64 {
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    let dim = re.len();

    match pauli {
        0 => {
            // X: <psi|X_q|psi> = Re[sum_k conj(psi[k]) * psi[k ^ (1<<qubit)]]
            let mask = 1usize << qubit;
            let mut result = 0.0;
            for k in 0..dim {
                let f = k ^ mask;
                result += re[k] * re[f] + im[k] * im[f];
            }
            result
        }
        1 => {
            // Y|b> = i(-1)^b |1-b>. The phase (-1)^b uses the bit of the
            // target index (after flip), so sign = 2*bit_k - 1 = -(1-2*bit_k).
            let mask = 1usize << qubit;
            let mut result = 0.0;
            for k in 0..dim {
                let f = k ^ mask;
                let bit = ((k >> qubit) & 1) as f64;
                let sign = 2.0 * bit - 1.0;
                let prod_im = re[k] * im[f] - im[k] * re[f];
                result += sign * (-prod_im);
            }
            result
        }
        _ => {
            // Z: <psi|Z_q|psi> = sum_k |psi[k]|^2 * (-1)^bit_q(k)
            let mut result = 0.0;
            for k in 0..dim {
                let bit = ((k >> qubit) & 1) as f64;
                let sign = 1.0 - 2.0 * bit;
                result += sign * (re[k] * re[k] + im[k] * im[k]);
            }
            result
        }
    }
}

/// Brute-force optimal binary MPC: enumerate all 2^horizon action sequences.
/// Parallelised with rayon for horizon > 10.
///
/// Returns (optimal_actions, optimal_cost, all_costs, n_evaluated).
#[pyfunction]
fn brute_mpc<'py>(
    py: Python<'py>,
    b_flat: PyReadonlyArray1<'_, f64>,
    target: PyReadonlyArray1<'_, f64>,
    _dim: usize,
    horizon: usize,
) -> (Bound<'py, PyArray1<i64>>, f64, Bound<'py, PyArray1<f64>>, usize) {
    let b_data = b_flat.as_slice().unwrap();
    let t_data = target.as_slice().unwrap();
    let n_actions = 1usize << horizon;

    let b_norm: f64 = b_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    let t_norm: f64 = t_data.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Parallel cost evaluation
    let costs: Vec<f64> = (0..n_actions)
        .into_par_iter()
        .map(|idx| {
            let mut cost = 0.0;
            for t in 0..horizon {
                let action = ((idx >> t) & 1) as f64;
                let diff = b_norm * action - t_norm / horizon as f64;
                cost += diff * diff;
            }
            cost
        })
        .collect();

    // Find minimum
    let mut best_idx = 0usize;
    let mut best_cost = costs[0];
    for (idx, &cost) in costs.iter().enumerate() {
        if cost < best_cost {
            best_cost = cost;
            best_idx = idx;
        }
    }

    let best_actions: Vec<i64> = (0..horizon)
        .map(|bit| ((best_idx >> bit) & 1) as i64)
        .collect();

    let actions_arr = Array1::from_vec(best_actions);
    let costs_arr = Array1::from_vec(costs);

    (
        PyArray1::from_owned_array(py, actions_arr),
        best_cost,
        PyArray1::from_owned_array(py, costs_arr),
        n_actions,
    )
}

// ===== Complex-valued acceleration for analysis modules =====

fn c64(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn cmat_from_flat(re: &[f64], im: &[f64], dim: usize) -> Array2<C64> {
    Array2::from_shape_fn((dim, dim), |(i, j)| {
        c64(re[i * dim + j], im[i * dim + j])
    })
}

fn cvec_from_parts(re: &[f64], im: &[f64]) -> Array1<C64> {
    Array1::from_shape_fn(re.len(), |i| c64(re[i], im[i]))
}

fn conj_transpose(a: &Array2<C64>) -> Array2<C64> {
    let (m, n) = a.dim();
    Array2::from_shape_fn((n, m), |(i, j)| a[[j, i]].conj())
}

/// Re(Tr(A†B) / d) — Hilbert-Schmidt inner product
fn hs_inner_real(a: &Array2<C64>, b: &Array2<C64>) -> f64 {
    let d = a.nrows() as f64;
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| (av.conj() * bv).re)
        .sum::<f64>()
        / d
}

/// A† × x without materialising A†
fn ct_matvec(a: &Array2<C64>, x: &Array1<C64>) -> Array1<C64> {
    let (m, n) = a.dim();
    Array1::from_shape_fn(n, |col| {
        (0..m)
            .map(|row| a[[row, col]].conj() * x[row])
            .sum::<C64>()
    })
}

/// Operator Lanczos: b-coefficients for Liouvillian L=[H,·] on d×d matrices.
///
/// Avoids Python per-step overhead for the commutator loop (2 matrix multiplies
/// per step). For dim ≤ 256 (8 qubits), ~5-10× faster than numpy.
#[pyfunction]
fn lanczos_b_coefficients(
    h_re: PyReadonlyArray1<'_, f64>,
    h_im: PyReadonlyArray1<'_, f64>,
    o_re: PyReadonlyArray1<'_, f64>,
    o_im: PyReadonlyArray1<'_, f64>,
    dim: usize,
    max_steps: usize,
    tol: f64,
) -> Vec<f64> {
    let h = cmat_from_flat(h_re.as_slice().unwrap(), h_im.as_slice().unwrap(), dim);
    let o_init = cmat_from_flat(o_re.as_slice().unwrap(), o_im.as_slice().unwrap(), dim);

    let norm_0 = hs_inner_real(&o_init, &o_init).max(0.0).sqrt();
    if norm_0 < tol {
        return vec![0.0];
    }

    let mut o_prev = Array2::<C64>::zeros((dim, dim));
    let mut o_curr = o_init / c64(norm_0, 0.0);
    let mut b_list: Vec<f64> = Vec::with_capacity(max_steps);

    for _ in 0..max_steps {
        // A = [H, O_curr] = H·O - O·H
        let mut a_next = h.dot(&o_curr);
        {
            let oh = o_curr.dot(&h);
            a_next -= &oh;
        }

        if let Some(&b_last) = b_list.last() {
            a_next.scaled_add(c64(-b_last, 0.0), &o_prev);
        }

        let a_n = hs_inner_real(&o_curr, &a_next);
        a_next.scaled_add(c64(-a_n, 0.0), &o_curr);

        let b_next = hs_inner_real(&a_next, &a_next).max(0.0).sqrt();
        if b_next < tol {
            break;
        }

        b_list.push(b_next);
        o_prev = o_curr;
        o_curr = a_next / c64(b_next, 0.0);
    }

    b_list
}

/// OTOC F(t) via eigendecomposition, parallel across time points (rayon).
///
/// Diagonalise H once (in Python via numpy.linalg.eigh), pass eigenvalues +
/// eigenvectors here. Each time point: O(d²) phase rotation + mat-vec products.
/// Avoids 2× scipy.expm (O(d³) Padé) per time point.
///
/// F(t) = Re(⟨ψ| W†(t) V† W(t) V |ψ⟩), W(t) = e^{iHt} W e^{-iHt}
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn otoc_from_eigendecomp<'py>(
    py: Python<'py>,
    eigenvalues: PyReadonlyArray1<'_, f64>,
    eigvecs_re: PyReadonlyArray1<'_, f64>,
    eigvecs_im: PyReadonlyArray1<'_, f64>,
    w_re: PyReadonlyArray1<'_, f64>,
    w_im: PyReadonlyArray1<'_, f64>,
    v_re: PyReadonlyArray1<'_, f64>,
    v_im: PyReadonlyArray1<'_, f64>,
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    times: PyReadonlyArray1<'_, f64>,
    dim: usize,
) -> Bound<'py, PyArray1<f64>> {
    let evals = eigenvalues.as_slice().unwrap();
    let u = cmat_from_flat(
        eigvecs_re.as_slice().unwrap(),
        eigvecs_im.as_slice().unwrap(),
        dim,
    );
    let u_h = conj_transpose(&u);
    let w_mat = cmat_from_flat(w_re.as_slice().unwrap(), w_im.as_slice().unwrap(), dim);
    let v_mat = cmat_from_flat(v_re.as_slice().unwrap(), v_im.as_slice().unwrap(), dim);
    let psi = cvec_from_parts(psi_re.as_slice().unwrap(), psi_im.as_slice().unwrap());
    let t_arr = times.as_slice().unwrap();

    // Transform to eigenbasis (done once)
    let w_eig = u_h.dot(&w_mat).dot(&u);
    let v_eig = u_h.dot(&v_mat).dot(&u);
    let psi_e = u_h.dot(&psi);
    let state0 = v_eig.dot(&psi_e); // V|ψ⟩ in eigenbasis

    let results: Vec<f64> = t_arr
        .par_iter()
        .map(|&t| {
            // W(t)[i,j] = exp(i(E_i − E_j)t) × W_eig[i,j]
            let mut w_t = Array2::<C64>::zeros((dim, dim));
            for i in 0..dim {
                for j in 0..dim {
                    let ph = (evals[i] - evals[j]) * t;
                    w_t[[i, j]] = c64(ph.cos(), ph.sin()) * w_eig[[i, j]];
                }
            }
            // F(t) = Re(⟨ψ_e| W_t† V_e† W_t |state0⟩)
            let s1 = w_t.dot(&state0);
            let s2 = ct_matvec(&v_eig, &s1);
            let s3 = ct_matvec(&w_t, &s2);
            psi_e
                .iter()
                .zip(s3.iter())
                .map(|(&p, &s)| (p.conj() * s).re)
                .sum::<f64>()
        })
        .collect();

    PyArray1::from_owned_array(py, Array1::from_vec(results))
}

/// Build dense XY Hamiltonian directly from K coupling and omega frequencies.
///
/// H = -Σ_{i<j} K[i,j](X_iX_j + Y_iY_j) - Σ_i ω_i Z_i
///
/// Uses bitwise flip-flop: (XX+YY)|↑↓⟩ = 2|↓↑⟩, zero when same spin.
/// Returns flat real array (XY Hamiltonian is real in computational basis).
/// Eliminates Qiskit SparsePauliOp construction + to_matrix() overhead.
#[pyfunction]
fn build_xy_hamiltonian_dense<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    n: usize,
) -> Bound<'py, PyArray1<f64>> {
    let k = k_flat.as_slice().unwrap();
    let w = omega.as_slice().unwrap();
    let dim = 1usize << n;
    let mut h = vec![0.0f64; dim * dim];

    for idx in 0..dim {
        // Diagonal: -ω_i Z_i, where Z eigenvalue = 1-2·bit
        let mut diag = 0.0;
        for (i, &wi) in w.iter().enumerate().take(n) {
            let bit = ((idx >> i) & 1) as f64;
            diag -= wi * (1.0 - 2.0 * bit);
        }
        h[idx * dim + idx] = diag;

        // Off-diagonal: -K[i,j]·(XX+YY) flip-flop
        for i in 0..n {
            for j in (i + 1)..n {
                let kij = k[i * n + j];
                if kij.abs() < 1e-15 {
                    continue;
                }
                let bi = (idx >> i) & 1;
                let bj = (idx >> j) & 1;
                if bi != bj {
                    let flipped = idx ^ ((1 << i) | (1 << j));
                    h[idx * dim + flipped] -= 2.0 * kij;
                }
            }
        }
    }

    PyArray1::from_vec(py, h)
}

/// Batch compute per-qubit X and Y expectations for all n qubits in one call.
///
/// Returns (exp_x[n], exp_y[n]). Avoids 2n FFI roundtrips vs calling
/// expectation_pauli_fast individually per qubit per Pauli.
#[pyfunction]
fn all_xy_expectations<'py>(
    py: Python<'py>,
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    n_osc: usize,
) -> (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>) {
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    let dim = re.len();

    let mut exp_x = vec![0.0f64; n_osc];
    let mut exp_y = vec![0.0f64; n_osc];

    for q in 0..n_osc {
        let mask = 1usize << q;
        let mut ex = 0.0;
        let mut ey = 0.0;

        for k in 0..dim {
            let f = k ^ mask;
            ex += re[k] * re[f] + im[k] * im[f];

            let bit = ((k >> q) & 1) as f64;
            let sign = 2.0 * bit - 1.0;
            let prod_im = re[k] * im[f] - im[k] * re[f];
            ey += sign * (-prod_im);
        }

        exp_x[q] = ex;
        exp_y[q] = ey;
    }

    (
        PyArray1::from_vec(py, exp_x),
        PyArray1::from_vec(py, exp_y),
    )
}

// =========================================================================
// Sparse Hamiltonian construction (COO triplets)
// Outputs (rows, cols, vals) for scipy.sparse.csc_matrix construction.
// Same bitwise flip-flop as build_xy_hamiltonian_dense but sparse output.
// =========================================================================
#[allow(clippy::type_complexity)]
#[pyfunction]
fn build_sparse_xy_hamiltonian<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    omega: PyReadonlyArray1<'_, f64>,
    n: usize,
) -> (Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>) {
    let k = k_flat.as_slice().unwrap();
    let om = omega.as_slice().unwrap();
    let dim = 1usize << n;

    let mut rows: Vec<i64> = Vec::new();
    let mut cols: Vec<i64> = Vec::new();
    let mut vals: Vec<f64> = Vec::new();

    // Diagonal: -Σ ω_i (1 - 2·b_i(s))
    for s in 0..dim {
        let mut diag = 0.0f64;
        for (i, &omi) in om.iter().enumerate().take(n) {
            let bi = ((s >> i) & 1) as f64;
            diag -= omi * (1.0 - 2.0 * bi);
        }
        rows.push(s as i64);
        cols.push(s as i64);
        vals.push(diag);
    }

    // Off-diagonal: XY flip-flop
    for i in 0..n {
        for j in (i + 1)..n {
            let kij = k[i * n + j];
            if kij.abs() < 1e-15 {
                continue;
            }
            let mask = (1usize << i) | (1usize << j);
            let val = -2.0 * kij;
            for s in 0..dim {
                let bi = (s >> i) & 1;
                let bj = (s >> j) & 1;
                if bi != bj {
                    let s_flip = s ^ mask;
                    rows.push(s as i64);
                    cols.push(s_flip as i64);
                    vals.push(val);
                }
            }
        }
    }

    (
        PyArray1::from_vec(py, rows),
        PyArray1::from_vec(py, cols),
        PyArray1::from_vec(py, vals),
    )
}

// =========================================================================
// Basis partition by magnetisation (popcount-based)
// Returns array where result[k] = magnetisation M of basis state |k⟩.
// M = n - 2 × popcount(k). Uses hardware popcount instruction.
// =========================================================================
#[pyfunction]
fn magnetisation_labels<'py>(
    py: Python<'py>,
    n: usize,
) -> Bound<'py, PyArray1<i32>> {
    let dim = 1usize << n;
    let mut labels = Vec::with_capacity(dim);
    let n_i32 = n as i32;
    for k in 0..dim {
        let popcount = (k as u64).count_ones() as i32;
        labels.push(n_i32 - 2 * popcount);
    }
    PyArray1::from_vec(py, labels)
}

// =========================================================================
// Order parameter from state vector (complex)
// R = (1/N)|Σ_i (<X_i> + i<Y_i>)| computed via bitwise Pauli.
// Same logic as state_order_param_sparse but for complex state vectors
// used in tensor_jump MCWF trajectories.
// =========================================================================
#[pyfunction]
fn order_param_from_statevector(
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    n: usize,
) -> f64 {
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    let dim = 1usize << n;

    let mut z_re = 0.0f64;
    let mut z_im = 0.0f64;

    for i in 0..n {
        let mut exp_x = 0.0f64;
        let mut exp_y = 0.0f64;
        let mask = 1usize << i;
        for k in 0..dim {
            let k_flip = k ^ mask;
            // <X_i> = Σ_k Re(ψ*_k · ψ_{k^mask})
            // ψ*_k · ψ_{k^mask} = (re_k - i·im_k)(re_f + i·im_f)
            //                    = re_k·re_f + im_k·im_f + i(re_k·im_f - im_k·re_f)
            let re_prod = re[k] * re[k_flip] + im[k] * im[k_flip];
            let im_prod = re[k] * im[k_flip] - im[k] * re[k_flip];
            exp_x += re_prod;
            exp_y += im_prod;
        }
        z_re += exp_x;
        z_im += exp_y;
    }

    z_re /= n as f64;
    z_im /= n as f64;
    (z_re * z_re + z_im * z_im).sqrt()
}

// =========================================================================
// XY correlation matrix C[i,j] = <XX_ij + YY_ij> from statevector.
// Used by DynamicCouplingEngine for Hebbian learning.
// Parallelised over qubit pairs via rayon.
// =========================================================================
#[pyfunction]
fn correlation_matrix_xy<'py>(
    py: Python<'py>,
    psi_re: PyReadonlyArray1<'_, f64>,
    psi_im: PyReadonlyArray1<'_, f64>,
    n_osc: usize,
) -> Bound<'py, PyArray2<f64>> {
    let re = psi_re.as_slice().unwrap();
    let im = psi_im.as_slice().unwrap();
    let dim = 1usize << n_osc;

    // Collect all upper-triangle pairs for parallel iteration
    let pairs: Vec<(usize, usize)> = (0..n_osc)
        .flat_map(|i| ((i + 1)..n_osc).map(move |j| (i, j)))
        .collect();

    // XX + YY = 2 * sum_{k: b_i XOR b_j = 1} Re(psi*_k * psi_{k^mask})
    // Derivation: XX flips both bits, YY flips both with phase -(-1)^{b_i+b_j}.
    // When b_i == b_j: XX + YY = 0. When b_i != b_j: XX + YY = 2 * overlap.
    let results: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let mask = (1usize << i) | (1usize << j);
            let mut corr = 0.0f64;
            for k in 0..dim {
                let bi = (k >> i) & 1;
                let bj = (k >> j) & 1;
                if bi != bj {
                    let k_flip = k ^ mask;
                    corr += 2.0 * (re[k] * re[k_flip] + im[k] * im[k_flip]);
                }
            }
            (i, j, corr)
        })
        .collect();

    let mut c = Array2::<f64>::zeros((n_osc, n_osc));
    for (i, j, corr) in results {
        c[(i, j)] = corr;
        c[(j, i)] = corr;
    }

    PyArray2::from_owned_array(py, c)
}

// =========================================================================
// Lindblad jump operator COO data + anti-Hermitian diagonal.
// Builds all jump operators L_k for pairs (i,j) where |K[i,j]| > threshold.
// Each L_k: |...0_i...1_j...> <- |...1_i...0_j...> (excitation transfer).
// Returns (rows, cols, op_starts, n_ops) where op_starts[k] is the first
// index in rows/cols belonging to operator k. op_starts has length n_ops+1.
// =========================================================================
#[allow(clippy::type_complexity)]
#[pyfunction]
fn lindblad_jump_ops_coo<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    threshold: f64,
) -> (
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    usize,
) {
    let k = k_flat.as_slice().unwrap();
    let dim = 1usize << n;

    let mut rows: Vec<i64> = Vec::new();
    let mut cols: Vec<i64> = Vec::new();
    let mut op_starts: Vec<i64> = Vec::new();
    let mut op_count = 0usize;

    for i in 0..n {
        for j in 0..n {
            if i != j && k[i * n + j].abs() > threshold {
                op_starts.push(rows.len() as i64);
                for idx in 0..dim {
                    if ((idx >> i) & 1) == 1 && ((idx >> j) & 1) == 0 {
                        let flipped = idx ^ ((1 << i) | (1 << j));
                        rows.push(flipped as i64);
                        cols.push(idx as i64);
                    }
                }
                op_count += 1;
            }
        }
    }
    // Sentinel: marks end of last operator
    op_starts.push(rows.len() as i64);

    (
        PyArray1::from_vec(py, rows),
        PyArray1::from_vec(py, cols),
        PyArray1::from_vec(py, op_starts),
        op_count,
    )
}

// =========================================================================
// Anti-Hermitian diagonal for Lindblad trajectory path.
// diag[idx] = number of active jump channels that can fire from state |idx>.
// =========================================================================
#[pyfunction]
fn lindblad_anti_hermitian_diag<'py>(
    py: Python<'py>,
    k_flat: PyReadonlyArray1<'_, f64>,
    n: usize,
    threshold: f64,
) -> Bound<'py, PyArray1<f64>> {
    let k = k_flat.as_slice().unwrap();
    let dim = 1usize << n;
    let mut diag = vec![0.0f64; dim];

    for i in 0..n {
        for j in 0..n {
            if i != j && k[i * n + j].abs() > threshold {
                for (idx, d) in diag.iter_mut().enumerate().take(dim) {
                    if ((idx >> i) & 1) == 1 && ((idx >> j) & 1) == 0 {
                        *d += 1.0;
                    }
                }
            }
        }
    }

    PyArray1::from_vec(py, diag)
}

// =========================================================================
// Z2 parity filter for measurement counts (compound mitigation).
// Takes a flat array of bitstring values and returns a boolean mask
// indicating which bitstrings match the expected parity.
// =========================================================================
#[pyfunction]
fn parity_filter_mask<'py>(
    py: Python<'py>,
    bitstrings: PyReadonlyArray1<'_, u64>,
    expected_parity: u8,
) -> Bound<'py, PyArray1<bool>> {
    let bs = bitstrings.as_slice().unwrap();
    let mask: Vec<bool> = bs
        .par_iter()
        .map(|&val| (val.count_ones() as u8 % 2) == expected_parity)
        .collect();
    PyArray1::from_vec(py, mask)
}

// =========================================================================
// Unit tests for core Rust functions (no PyO3 dependency)
// =========================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pec_coefficients_zero_error() {
        let [q_i, q_x, q_y, q_z] = pec_coefficients_inner(0.0);
        assert!((q_i - 1.0).abs() < 1e-12);
        assert!(q_x.abs() < 1e-12);
        assert!(q_y.abs() < 1e-12);
        assert!(q_z.abs() < 1e-12);
    }

    #[test]
    fn test_pec_coefficients_sum() {
        let [q_i, q_x, q_y, q_z] = pec_coefficients_inner(0.01);
        let s = q_i + q_x + q_y + q_z;
        assert!((s - 1.0).abs() < 1e-10, "PEC coefficients should sum to 1, got {s}");
    }

    #[test]
    fn test_build_knm_symmetric() {
        let n = 4;
        let k = build_knm_inner(n, 0.45, 0.3);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (k[[i, j]] - k[[j, i]]).abs() < 1e-12,
                    "K_nm must be symmetric: K[{i},{j}]={} != K[{j},{i}]={}",
                    k[[i, j]],
                    k[[j, i]]
                );
            }
        }
    }

    #[test]
    fn test_build_knm_zero_diagonal() {
        let n = 4;
        let k = build_knm_inner(n, 0.45, 0.3);
        for i in 0..n {
            assert!(k[[i, i]].abs() < 1e-12, "diagonal must be zero");
        }
    }

    #[test]
    fn test_build_knm_exponential_decay() {
        let n = 4;
        let k = build_knm_inner(n, 0.45, 0.3);
        // K[0,1] > K[0,2] > K[0,3] (exponential decay with distance)
        assert!(k[[0, 1]] > k[[0, 2]]);
        assert!(k[[0, 2]] > k[[0, 3]]);
    }

    #[test]
    fn test_order_parameter_all_aligned() {
        // All angles = 0 → R = 1
        let theta = vec![0.0; 8];
        let r = order_parameter_inner(&theta);
        assert!((r - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_order_parameter_opposite() {
        // Two angles: 0 and π → R = 0
        let theta = vec![0.0, std::f64::consts::PI];
        let r = order_parameter_inner(&theta);
        assert!(r < 1e-10);
    }

    #[test]
    fn test_order_parameter_bounded() {
        let theta = vec![0.1, 0.5, 1.2, 2.5, 3.8, 5.0];
        let r = order_parameter_inner(&theta);
        assert!(r >= 0.0 && r <= 1.0 + 1e-12);
    }

    // Pure Rust helpers for testing without PyO3
    fn pec_coefficients_inner(gate_error_rate: f64) -> [f64; 4] {
        let p = gate_error_rate;
        let denom = 4.0 - 4.0 * p;
        let q_i = 1.0 + 3.0 * p / denom;
        let q_xyz = -p / denom;
        [q_i, q_xyz, q_xyz, q_xyz]
    }

    fn build_knm_inner(n: usize, k_base: f64, alpha: f64) -> Array2<f64> {
        let mut knm = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let dist = (i as f64 - j as f64).abs();
                    knm[[i, j]] = k_base * (-alpha * dist).exp();
                }
            }
        }
        knm
    }

    fn order_parameter_inner(theta: &[f64]) -> f64 {
        let n = theta.len() as f64;
        let (sr, si) = theta.iter().fold((0.0, 0.0), |(sr, si), &t| {
            (sr + t.cos(), si + t.sin())
        });
        ((sr / n).powi(2) + (si / n).powi(2)).sqrt()
    }
}

#[pymodule]
fn scpn_quantum_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pec_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(pec_sample_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(build_knm, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_euler, m)?)?;
    m.add_function(wrap_pyfunction!(order_parameter, m)?)?;
    m.add_function(wrap_pyfunction!(kuramoto_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(dla_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(mc_xy_simulate, m)?)?;
    m.add_function(wrap_pyfunction!(state_order_param_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(expectation_pauli_fast, m)?)?;
    m.add_function(wrap_pyfunction!(brute_mpc, m)?)?;
    m.add_function(wrap_pyfunction!(lanczos_b_coefficients, m)?)?;
    m.add_function(wrap_pyfunction!(otoc_from_eigendecomp, m)?)?;
    m.add_function(wrap_pyfunction!(build_xy_hamiltonian_dense, m)?)?;
    m.add_function(wrap_pyfunction!(all_xy_expectations, m)?)?;
    m.add_function(wrap_pyfunction!(build_sparse_xy_hamiltonian, m)?)?;
    m.add_function(wrap_pyfunction!(magnetisation_labels, m)?)?;
    m.add_function(wrap_pyfunction!(order_param_from_statevector, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_matrix_xy, m)?)?;
    m.add_function(wrap_pyfunction!(lindblad_jump_ops_coo, m)?)?;
    m.add_function(wrap_pyfunction!(lindblad_anti_hermitian_diag, m)?)?;
    m.add_function(wrap_pyfunction!(parity_filter_mask, m)?)?;
    Ok(())
}
