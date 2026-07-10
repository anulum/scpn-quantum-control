// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — K_nm validator fuzz target

#![no_main]

//! Coverage-guided fuzz target for the shared Rust input validators.
//!
//! `validation::check_*` are the Rust mirrors of the Python-side
//! `_as_coupling_matrix`-family guards — every PyO3 kernel funnels
//! caller-supplied sizes and floats through them. The harness feeds arbitrary
//! bit patterns (including NaN/±inf floats and extreme sizes) and asserts the
//! validators fail CLOSED — an error `Result`, never a panic or arithmetic
//! overflow — and that their verdicts match independent predicates. The
//! bounded `build_knm_inner` replay checks the accepted-path constructor
//! stays shape-correct for admissible sizes (THREAT_MODEL B8).

use libfuzzer_sys::fuzz_target;
use scpn_quantum_engine::knm::build_knm_inner;
use scpn_quantum_engine::validation::{
    check_domain_range, check_finite, check_flat_square, check_n, check_positive, check_range,
    check_statevec_len,
};

/// Constructor replay bound: `build_knm_inner` is O(n²), so cap the accepted
/// path well above the physical 16-oscillator system but far below anything
/// that would slow the fuzzer.
const MAX_BUILD_N: usize = 64;

fn read_usize(data: &[u8], offset: usize) -> usize {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&data[offset..offset + 8]);
    usize::from_le_bytes(raw)
}

fn read_f64(data: &[u8], offset: usize) -> f64 {
    let mut raw = [0u8; 8];
    raw.copy_from_slice(&data[offset..offset + 8]);
    f64::from_le_bytes(raw)
}

fuzz_target!(|data: &[u8]| {
    // Header: 4 usizes (n, len, start, end) + 3 f64 (val, lo, hi), then the
    // remaining bytes become the float slice under validation.
    const HEADER: usize = 4 * 8 + 3 * 8;
    if data.len() < HEADER || data.len() > 16_384 {
        return;
    }
    let n = read_usize(data, 0);
    let len = read_usize(data, 8);
    let start = read_usize(data, 16);
    let end = read_usize(data, 24);
    let val = read_f64(data, 32);
    let lo = read_f64(data, 40);
    let hi = read_f64(data, 48);
    let floats: Vec<f64> = data[HEADER..]
        .chunks_exact(8)
        .map(|chunk| {
            let mut raw = [0u8; 8];
            raw.copy_from_slice(chunk);
            f64::from_le_bytes(raw)
        })
        .collect();

    // Every validator must return a verdict — never panic, never overflow.
    assert_eq!(
        check_finite(&floats, "arr").is_ok(),
        floats.iter().all(|v| v.is_finite())
    );
    assert_eq!(
        check_positive(val, "val").is_ok(),
        val.is_finite() && val > 0.0
    );
    assert_eq!(
        check_range(val, lo, hi, "val").is_ok(),
        val >= lo && val <= hi
    );
    assert_eq!(check_n(n, "n").is_ok(), n > 0);
    assert_eq!(
        check_flat_square(&floats, n, "arr").is_ok(),
        n.checked_mul(n) == Some(floats.len())
    );
    let statevec_ok = n < usize::BITS as usize && len == 1usize << n;
    assert_eq!(check_statevec_len(len, n, "psi").is_ok(), statevec_ok);
    assert_eq!(
        check_domain_range(start, end, n, "domain").is_ok(),
        start < n && end < n && start <= end
    );

    // Accepted-path constructor: shape-correct, anchor-symmetric.
    if (1..=MAX_BUILD_N).contains(&n) {
        let k = build_knm_inner(n, val, lo);
        assert_eq!(k.shape(), [n, n]);
        if n > 1 {
            assert_eq!(k[[0, 1]], k[[1, 0]]);
        }
    }
});
