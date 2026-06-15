// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — NIST SP 800-22 hot-path kernels

//! Acceleration for the loop-heavy NIST SP 800-22 statistics.
//!
//! These kernels return exact integer/float statistics (LFSR length, monobit
//! sum, run-transition count, per-block longest run) and are therefore bit-true
//! identical to the NumPy reference in
//! `scpn_quantum_control.entropy.nist_sp800_22`. The final P-value special
//! functions (`erfc`, incomplete gamma) remain in SciPy on the Python side.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn validate_bits(bits: &[i8], name: &str) -> PyResult<()> {
    for &b in bits {
        if b != 0 && b != 1 {
            return Err(PyValueError::new_err(format!(
                "{name} must contain only 0 and 1"
            )));
        }
    }
    Ok(())
}

/// Length of the shortest LFSR generating `bits` (Berlekamp-Massey over GF(2)).
///
/// Bit-true with `scpn_quantum_control.entropy.nist_sp800_22.berlekamp_massey`.
#[pyfunction]
pub fn nist_berlekamp_massey(bits: PyReadonlyArray1<'_, i8>) -> PyResult<i64> {
    let s = bits.as_slice()?;
    validate_bits(s, "bits")?;
    let n = s.len();
    if n == 0 {
        return Ok(0);
    }
    let mut c = vec![0u8; n];
    let mut b = vec![0u8; n];
    c[0] = 1;
    b[0] = 1;
    let mut length: i64 = 0;
    let mut m: i64 = -1;
    for i in 0..n {
        let mut discrepancy = s[i] as u8;
        for j in 1..=(length as usize) {
            discrepancy ^= c[j] & s[i - j] as u8;
        }
        if discrepancy & 1 == 1 {
            let t = c.clone();
            let shift = (i as i64 - m) as usize;
            if shift < n {
                for k in 0..(n - shift) {
                    c[k + shift] ^= b[k];
                }
            }
            if 2 * length <= i as i64 {
                length = i as i64 + 1 - length;
                m = i as i64;
                b = t;
            }
        }
    }
    Ok(length)
}

/// Monobit sum `S_n = sum(2*bit - 1)` for the §2.1 Frequency test.
#[pyfunction]
pub fn nist_monobit_sum(bits: PyReadonlyArray1<'_, i8>) -> PyResult<i64> {
    let s = bits.as_slice()?;
    validate_bits(s, "bits")?;
    let mut acc: i64 = 0;
    for &bit in s {
        acc += 2 * bit as i64 - 1;
    }
    Ok(acc)
}

/// Return `(ones, v_obs)` for the §2.3 Runs test: the population count and the
/// number of runs `V = sum(adjacent differ) + 1`.
#[pyfunction]
pub fn nist_runs_counts(bits: PyReadonlyArray1<'_, i8>) -> PyResult<(i64, i64)> {
    let s = bits.as_slice()?;
    validate_bits(s, "bits")?;
    let n = s.len();
    let mut ones: i64 = 0;
    for &bit in s {
        ones += bit as i64;
    }
    if n == 0 {
        return Ok((0, 0));
    }
    let mut transitions: i64 = 0;
    for i in 1..n {
        if s[i] != s[i - 1] {
            transitions += 1;
        }
    }
    Ok((ones, transitions + 1))
}

/// Longest run of ones within each `block_size`-bit block (§2.4), as a flat list.
#[pyfunction]
pub fn nist_block_longest_runs(
    bits: PyReadonlyArray1<'_, i8>,
    block_size: usize,
) -> PyResult<Vec<i64>> {
    let s = bits.as_slice()?;
    validate_bits(s, "bits")?;
    if block_size == 0 {
        return Err(PyValueError::new_err("block_size must be positive"));
    }
    let n_blocks = s.len() / block_size;
    let mut out = Vec::with_capacity(n_blocks);
    for b in 0..n_blocks {
        let block = &s[b * block_size..(b + 1) * block_size];
        let mut longest = 0i64;
        let mut current = 0i64;
        for &value in block {
            if value == 1 {
                current += 1;
                if current > longest {
                    longest = current;
                }
            } else {
                current = 0;
            }
        }
        out.push(longest);
    }
    Ok(out)
}
