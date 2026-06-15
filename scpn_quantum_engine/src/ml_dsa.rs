// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — FIPS 204 ML-DSA number-theoretic transform

//! Negacyclic NTT over Z_q (q = 8380417) for the ML-DSA polynomial ring.
//!
//! Bit-true (exact integer) with the NumPy/Python reference in
//! `scpn_quantum_control.crypto.ml_dsa`: the same in-place butterflies and the
//! same zeta table `zetas[k] = 1753^bitrev8(k) mod q`. Inputs are reduced mod q
//! first, so signed secret-polynomial coefficients transform identically.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const Q: i64 = 8380417;
const N: usize = 256;
const ZETA: i64 = 1753;
const F_256_INV: i64 = 8347681; // 256^-1 mod q

fn bit_reverse_8(value: usize) -> usize {
    let mut v = value;
    let mut result = 0;
    for _ in 0..8 {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

fn pow_mod(mut base: i64, mut exp: u32) -> i64 {
    let mut result = 1i64;
    base %= Q;
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base % Q;
        }
        base = base * base % Q;
        exp >>= 1;
    }
    result
}

fn zetas() -> [i64; N] {
    let mut z = [0i64; N];
    for (k, slot) in z.iter_mut().enumerate() {
        *slot = pow_mod(ZETA, bit_reverse_8(k) as u32);
    }
    z
}

#[inline]
fn rem_q(x: i64) -> i64 {
    x.rem_euclid(Q)
}

fn ntt_inplace(w: &mut [i64; N]) {
    let z = zetas();
    let mut k = 0usize;
    let mut len = 128usize;
    while len >= 1 {
        let mut start = 0usize;
        while start < N {
            k += 1;
            let zeta = z[k];
            for j in start..start + len {
                let t = rem_q(zeta * w[j + len]);
                w[j + len] = rem_q(w[j] - t);
                w[j] = rem_q(w[j] + t);
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

fn intt_inplace(w: &mut [i64; N]) {
    let z = zetas();
    let mut k = N;
    let mut len = 1usize;
    while len < N {
        let mut start = 0usize;
        while start < N {
            k -= 1;
            let zeta = rem_q(-z[k]);
            for j in start..start + len {
                let t = w[j];
                w[j] = rem_q(t + w[j + len]);
                w[j + len] = rem_q(t - w[j + len]);
                w[j + len] = rem_q(zeta * w[j + len]);
            }
            start += 2 * len;
        }
        len *= 2;
    }
    for slot in w.iter_mut() {
        *slot = rem_q(F_256_INV * *slot);
    }
}

fn load(poly: Vec<i64>) -> PyResult<[i64; N]> {
    if poly.len() != N {
        return Err(PyValueError::new_err(format!(
            "polynomial must have {N} coefficients, got {}",
            poly.len()
        )));
    }
    let mut w = [0i64; N];
    for (i, &c) in poly.iter().enumerate() {
        w[i] = rem_q(c);
    }
    Ok(w)
}

/// Forward NTT of a 256-coefficient polynomial over Z_q.
#[pyfunction]
pub fn ml_dsa_ntt(poly: Vec<i64>) -> PyResult<Vec<i64>> {
    let mut w = load(poly)?;
    ntt_inplace(&mut w);
    Ok(w.to_vec())
}

/// Inverse NTT of a 256-coefficient polynomial over Z_q.
#[pyfunction]
pub fn ml_dsa_intt(poly: Vec<i64>) -> PyResult<Vec<i64>> {
    let mut w = load(poly)?;
    intt_inplace(&mut w);
    Ok(w.to_vec())
}
