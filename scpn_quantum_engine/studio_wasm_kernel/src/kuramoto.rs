// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Studio WASM Kuramoto live simulator

//! Bounded Kuramoto phase-synchronisation simulator for the Studio Play panel.
//!
//! Two kernels share one fixed-step RK4 integrator:
//!
//! * **mean-field** — the all-to-all approximation
//!   `dθ_i/dt = ω_i + K · (1/N) Σ_j sin(θ_j − θ_i)`, evaluated in `O(N)` per
//!   step from the global order parameter;
//! * **networked** — the explicit coupling matrix
//!   `dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j − θ_i)`, `O(N²)` per step.
//!
//! Both are bounded: `N` and the step count are refused past declared limits so
//! the browser panel can show the fail-closed boundary rather than silently
//! diverging. The output is the order-parameter trajectory `R(t)` plus the final
//! phase snapshot; it is a live visualisation, not a bit-exact evidence claim.

/// Largest oscillator count the kernel will integrate (the fail-closed N
/// boundary the Play panel displays).
pub const MAX_OSCILLATORS: usize = 128;

/// Largest step count the kernel will integrate in one call.
pub const MAX_STEPS: usize = 4096;

/// Version stamped into the canonical little-endian simulator input.
pub const KURAMOTO_INPUT_VERSION: u32 = 1;

const HEADER_LEN: usize = 32;

/// Fail-closed status codes returned across the FFI boundary.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[repr(i32)]
pub enum KuramotoStatus {
    Ok = 0,
    NullPointer = -1,
    InvalidLength = -2,
    InvalidVersion = -3,
    InvalidOscillatorCount = -4,
    InvalidFloat = -5,
    InvalidSteps = -6,
    InvalidMode = -7,
    OutputMismatch = -8,
}

impl From<KuramotoStatus> for i32 {
    fn from(value: KuramotoStatus) -> Self {
        value as i32
    }
}

/// The two coupling kernels the simulator exposes.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum KuramotoMode {
    MeanField,
    Networked,
}

impl KuramotoMode {
    fn from_code(code: u32) -> Result<Self, KuramotoStatus> {
        match code {
            0 => Ok(Self::MeanField),
            1 => Ok(Self::Networked),
            _ => Err(KuramotoStatus::InvalidMode),
        }
    }
}

/// A validated Kuramoto simulation request.
#[derive(Debug, Clone)]
pub struct KuramotoInput {
    pub mode: KuramotoMode,
    pub n: usize,
    pub steps: usize,
    pub dt: f64,
    pub coupling: f64,
    pub omega: Vec<f64>,
    pub theta0: Vec<f64>,
    /// Row-major `n × n` coupling matrix; empty for the mean-field kernel.
    pub k_nm: Vec<f64>,
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    let mut raw = [0_u8; 4];
    raw.copy_from_slice(&bytes[offset..offset + 4]);
    u32::from_le_bytes(raw)
}

fn read_f64(bytes: &[u8], offset: usize) -> f64 {
    let mut raw = [0_u8; 8];
    raw.copy_from_slice(&bytes[offset..offset + 8]);
    f64::from_le_bytes(raw)
}

fn read_finite_vec(
    bytes: &[u8],
    offset: &mut usize,
    count: usize,
) -> Result<Vec<f64>, KuramotoStatus> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let value = read_f64(bytes, *offset);
        if !value.is_finite() {
            return Err(KuramotoStatus::InvalidFloat);
        }
        out.push(value);
        *offset += 8;
    }
    Ok(out)
}

/// Parse the canonical little-endian simulator input, failing closed.
///
/// Layout: `u32 version | u32 mode | u32 n | u32 steps | f64 dt | f64 coupling`
/// followed by `omega[n]`, `theta0[n]`, and — for the networked kernel only —
/// the row-major `k_nm[n*n]`.
pub fn parse_kuramoto_input(bytes: &[u8]) -> Result<KuramotoInput, KuramotoStatus> {
    if bytes.len() < HEADER_LEN {
        return Err(KuramotoStatus::InvalidLength);
    }
    let version = read_u32(bytes, 0);
    if version != KURAMOTO_INPUT_VERSION {
        return Err(KuramotoStatus::InvalidVersion);
    }
    let mode = KuramotoMode::from_code(read_u32(bytes, 4))?;
    let n = read_u32(bytes, 8) as usize;
    if n == 0 || n > MAX_OSCILLATORS {
        return Err(KuramotoStatus::InvalidOscillatorCount);
    }
    let steps = read_u32(bytes, 12) as usize;
    if steps == 0 || steps > MAX_STEPS {
        return Err(KuramotoStatus::InvalidSteps);
    }
    let dt = read_f64(bytes, 16);
    if !dt.is_finite() || dt <= 0.0 {
        return Err(KuramotoStatus::InvalidFloat);
    }
    let coupling = read_f64(bytes, 24);
    if !coupling.is_finite() {
        return Err(KuramotoStatus::InvalidFloat);
    }

    let k_len = match mode {
        KuramotoMode::MeanField => 0,
        KuramotoMode::Networked => n.checked_mul(n).ok_or(KuramotoStatus::InvalidLength)?,
    };
    let values_len = n
        .checked_mul(2)
        .and_then(|value| value.checked_add(k_len))
        .ok_or(KuramotoStatus::InvalidLength)?;
    let expected_len = values_len
        .checked_mul(8)
        .and_then(|value| value.checked_add(HEADER_LEN))
        .ok_or(KuramotoStatus::InvalidLength)?;
    if bytes.len() != expected_len {
        return Err(KuramotoStatus::InvalidLength);
    }

    let mut offset = HEADER_LEN;
    let omega = read_finite_vec(bytes, &mut offset, n)?;
    let theta0 = read_finite_vec(bytes, &mut offset, n)?;
    let k_nm = read_finite_vec(bytes, &mut offset, k_len)?;

    Ok(KuramotoInput {
        mode,
        n,
        steps,
        dt,
        coupling,
        omega,
        theta0,
        k_nm,
    })
}

/// Kuramoto order parameter `R = |(1/N) Σ_j exp(i θ_j)|`.
pub fn order_parameter(theta: &[f64]) -> f64 {
    if theta.is_empty() {
        return 0.0;
    }
    let n = theta.len() as f64;
    let mut cos_sum = 0.0;
    let mut sin_sum = 0.0;
    for &angle in theta {
        cos_sum += angle.cos();
        sin_sum += angle.sin();
    }
    ((cos_sum / n).powi(2) + (sin_sum / n).powi(2)).sqrt()
}

/// Write `dθ/dt` for the request's kernel into `out`.
fn derivative(input: &KuramotoInput, theta: &[f64], out: &mut [f64]) {
    let n = input.n;
    match input.mode {
        KuramotoMode::MeanField => {
            // mean_j sin(θ_j − θ_i) = cos θ_i · ⟨sin⟩ − sin θ_i · ⟨cos⟩
            let mut cos_mean = 0.0;
            let mut sin_mean = 0.0;
            for &angle in theta {
                cos_mean += angle.cos();
                sin_mean += angle.sin();
            }
            cos_mean /= n as f64;
            sin_mean /= n as f64;
            for (i, slot) in out.iter_mut().enumerate() {
                let (sin_i, cos_i) = theta[i].sin_cos();
                *slot = input.omega[i] + input.coupling * (sin_mean * cos_i - cos_mean * sin_i);
            }
        }
        KuramotoMode::Networked => {
            for (i, slot) in out.iter_mut().enumerate() {
                let theta_i = theta[i];
                let row = &input.k_nm[i * n..(i + 1) * n];
                let mut acc = 0.0;
                for (j, &k_ij) in row.iter().enumerate() {
                    acc += k_ij * (theta[j] - theta_i).sin();
                }
                *slot = input.omega[i] + acc;
            }
        }
    }
}

/// The element count [`simulate`] returns: `steps + 1` order-parameter samples
/// followed by the `n` final phases.
pub fn output_len(input: &KuramotoInput) -> usize {
    input.steps + 1 + input.n
}

/// Integrate the request with fixed-step RK4.
///
/// Returns `steps + 1` order-parameter samples (index 0 is the initial state)
/// followed by the `n` final phases — a `[R(t) ; θ_final]` layout.
pub fn simulate(input: &KuramotoInput) -> Vec<f64> {
    let n = input.n;
    let dt = input.dt;
    let mut theta = input.theta0.clone();
    let mut result = Vec::with_capacity(output_len(input));
    result.push(order_parameter(&theta));

    let mut k1 = vec![0.0_f64; n];
    let mut k2 = vec![0.0_f64; n];
    let mut k3 = vec![0.0_f64; n];
    let mut k4 = vec![0.0_f64; n];
    let mut stage = vec![0.0_f64; n];

    for _ in 0..input.steps {
        derivative(input, &theta, &mut k1);
        for (s, (&t, &d)) in stage.iter_mut().zip(theta.iter().zip(k1.iter())) {
            *s = t + 0.5 * dt * d;
        }
        derivative(input, &stage, &mut k2);
        for (s, (&t, &d)) in stage.iter_mut().zip(theta.iter().zip(k2.iter())) {
            *s = t + 0.5 * dt * d;
        }
        derivative(input, &stage, &mut k3);
        for (s, (&t, &d)) in stage.iter_mut().zip(theta.iter().zip(k3.iter())) {
            *s = t + dt * d;
        }
        derivative(input, &stage, &mut k4);
        for (i, angle) in theta.iter_mut().enumerate() {
            *angle += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        result.push(order_parameter(&theta));
    }

    result.extend_from_slice(&theta);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    fn encode(input: &KuramotoInput) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend(KURAMOTO_INPUT_VERSION.to_le_bytes());
        let mode = match input.mode {
            KuramotoMode::MeanField => 0_u32,
            KuramotoMode::Networked => 1_u32,
        };
        bytes.extend(mode.to_le_bytes());
        bytes.extend((input.n as u32).to_le_bytes());
        bytes.extend((input.steps as u32).to_le_bytes());
        bytes.extend(input.dt.to_le_bytes());
        bytes.extend(input.coupling.to_le_bytes());
        for value in input.omega.iter().chain(&input.theta0).chain(&input.k_nm) {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    fn mean_field(n: usize, coupling: f64) -> KuramotoInput {
        let omega: Vec<f64> = (0..n).map(|i| 0.1 * i as f64).collect();
        let theta0: Vec<f64> = (0..n).map(|i| 0.3 * i as f64).collect();
        KuramotoInput {
            mode: KuramotoMode::MeanField,
            n,
            steps: 200,
            dt: 0.01,
            coupling,
            omega,
            theta0,
            k_nm: Vec::new(),
        }
    }

    fn uniform_network(base: &KuramotoInput) -> KuramotoInput {
        // K_ij = coupling / N off-diagonal, 0 on the diagonal — the explicit
        // matrix whose dynamics equal the mean-field kernel.
        let n = base.n;
        let mut k_nm = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    k_nm[i * n + j] = base.coupling / n as f64;
                }
            }
        }
        KuramotoInput {
            mode: KuramotoMode::Networked,
            k_nm,
            ..base.clone()
        }
    }

    #[test]
    fn parse_round_trips_both_modes() {
        for input in [mean_field(4, 0.8), uniform_network(&mean_field(4, 0.8))] {
            let parsed = parse_kuramoto_input(&encode(&input)).expect("valid input");
            assert_eq!(parsed.mode, input.mode);
            assert_eq!(parsed.n, input.n);
            assert_eq!(parsed.steps, input.steps);
            assert_eq!(parsed.dt, input.dt);
            assert_eq!(parsed.omega, input.omega);
            assert_eq!(parsed.theta0, input.theta0);
            assert_eq!(parsed.k_nm, input.k_nm);
        }
    }

    #[test]
    fn parse_fails_closed_on_malformed_headers() {
        let good = encode(&mean_field(3, 0.5));

        let mut short = good.clone();
        short.truncate(HEADER_LEN - 1);
        assert_eq!(
            parse_kuramoto_input(&short).expect_err("short"),
            KuramotoStatus::InvalidLength
        );

        let mut bad_version = good.clone();
        bad_version[0..4].copy_from_slice(&99_u32.to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&bad_version).expect_err("version"),
            KuramotoStatus::InvalidVersion
        );

        let mut bad_mode = good.clone();
        bad_mode[4..8].copy_from_slice(&7_u32.to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&bad_mode).expect_err("mode"),
            KuramotoStatus::InvalidMode
        );

        let mut zero_n = good.clone();
        zero_n[8..12].copy_from_slice(&0_u32.to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&zero_n).expect_err("n=0"),
            KuramotoStatus::InvalidOscillatorCount
        );

        let mut huge_n = good.clone();
        huge_n[8..12].copy_from_slice(&(MAX_OSCILLATORS as u32 + 1).to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&huge_n).expect_err("n>max"),
            KuramotoStatus::InvalidOscillatorCount
        );

        let mut zero_steps = good.clone();
        zero_steps[12..16].copy_from_slice(&0_u32.to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&zero_steps).expect_err("steps=0"),
            KuramotoStatus::InvalidSteps
        );

        let mut huge_steps = good.clone();
        huge_steps[12..16].copy_from_slice(&(MAX_STEPS as u32 + 1).to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&huge_steps).expect_err("steps>max"),
            KuramotoStatus::InvalidSteps
        );
    }

    #[test]
    fn parse_fails_closed_on_bad_floats_and_length() {
        let good = encode(&mean_field(3, 0.5));

        let mut bad_dt = good.clone();
        bad_dt[16..24].copy_from_slice(&(-0.01_f64).to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&bad_dt).expect_err("dt<=0"),
            KuramotoStatus::InvalidFloat
        );

        let mut nan_dt = good.clone();
        nan_dt[16..24].copy_from_slice(&f64::NAN.to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&nan_dt).expect_err("dt NaN"),
            KuramotoStatus::InvalidFloat
        );

        let mut inf_coupling = good.clone();
        inf_coupling[24..32].copy_from_slice(&f64::INFINITY.to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&inf_coupling).expect_err("coupling inf"),
            KuramotoStatus::InvalidFloat
        );

        let mut nan_omega = good.clone();
        nan_omega[HEADER_LEN..HEADER_LEN + 8].copy_from_slice(&f64::NAN.to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&nan_omega).expect_err("omega NaN"),
            KuramotoStatus::InvalidFloat
        );

        let mut wrong_len = good.clone();
        wrong_len.extend_from_slice(&0.0_f64.to_le_bytes());
        assert_eq!(
            parse_kuramoto_input(&wrong_len).expect_err("trailing bytes"),
            KuramotoStatus::InvalidLength
        );
    }

    #[test]
    fn order_parameter_spans_incoherent_to_locked() {
        assert!(order_parameter(&[]) == 0.0);
        let locked = order_parameter(&[0.7, 0.7, 0.7, 0.7]);
        assert!((locked - 1.0).abs() < 1e-12);
        let splayed = order_parameter(&[0.0, PI / 2.0, PI, 3.0 * PI / 2.0]);
        assert!(splayed < 1e-12);
    }

    #[test]
    fn simulate_is_deterministic_and_shaped() {
        let input = mean_field(6, 1.2);
        let first = simulate(&input);
        let second = simulate(&input);
        assert_eq!(first, second);
        assert_eq!(first.len(), output_len(&input));
        assert_eq!(first.len(), input.steps + 1 + input.n);
        // every order-parameter sample is a valid magnitude
        for &r in &first[..=input.steps] {
            assert!((0.0..=1.0 + 1e-9).contains(&r));
        }
    }

    #[test]
    fn strong_coupling_drives_synchronisation() {
        let mut input = mean_field(12, 4.0);
        input.steps = 1500;
        input.dt = 0.01;
        let series = simulate(&input);
        let r_initial = series[0];
        let r_final = series[input.steps];
        assert!(r_final > r_initial);
        assert!(r_final > 0.9, "expected near-lock, got R={r_final}");
    }

    #[test]
    fn mean_field_matches_the_uniform_network() {
        // The mean-field kernel is the all-to-all network with K_ij = K/N.
        let base = mean_field(8, 1.5);
        let network = uniform_network(&base);
        let mean_series = simulate(&base);
        let network_series = simulate(&network);
        assert_eq!(mean_series.len(), network_series.len());
        for (a, b) in mean_series.iter().zip(&network_series) {
            assert!(
                (a - b).abs() < 1e-9,
                "mean-field vs network drift: {a} vs {b}"
            );
        }
    }
}
