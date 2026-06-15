// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Q-format quantisation for UltraScale+ HLS codegen

//! Signed fixed-point (Q-format) quantisation of a pulse waveform.
//!
//! Bit-true with the pure-Python reference in
//! `scpn_quantum_control.codegen.ultrascale_hls`: both evaluate
//! `floor(x * 2^frac_bits + 0.5)` in IEEE-754 binary64 (round half toward
//! +∞) and saturate to the signed `total_bits`-wide two's-complement range
//! `[-2^(total_bits-1), 2^(total_bits-1) - 1]`. Non-finite inputs are rejected
//! by the Python façade before dispatch, so this kernel never sees NaN/Inf.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Quantise `values` to signed Q(total_bits-frac_bits).frac_bits fixed point.
///
/// `frac_bits` is the number of fractional bits; `total_bits` is the full word
/// width including the sign bit. Returns the saturated integer code-words.
#[pyfunction]
pub fn quantise_q_format(values: Vec<f64>, frac_bits: u32, total_bits: u32) -> PyResult<Vec<i64>> {
    if !(2..=63).contains(&total_bits) {
        return Err(PyValueError::new_err(
            "total_bits must be in the range 2..=63",
        ));
    }
    if frac_bits >= total_bits {
        return Err(PyValueError::new_err(
            "frac_bits must be strictly less than total_bits (one sign bit)",
        ));
    }
    let scale = (1i64 << frac_bits) as f64;
    let max = (1i64 << (total_bits - 1)) - 1;
    let min = -(1i64 << (total_bits - 1));
    let out = values
        .iter()
        .map(|&x| {
            // `as i64` saturates float overflow to i64::MIN/MAX in Rust, matching
            // the clamp that follows; the Python reference clamps an unbounded int.
            let code = (x * scale + 0.5).floor() as i64;
            code.clamp(min, max)
        })
        .collect();
    Ok(out)
}
