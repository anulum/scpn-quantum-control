// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — ML-DSA NTT/INTT domain fuzz target

#![no_main]

//! Coverage-guided fuzz target for the ML-DSA number-theoretic transform.
//!
//! The NTT is the arithmetic core the signature scheme's security reduces to
//! implementing EXACTLY: any silent wrap-around or reduction error would
//! corrupt signatures without failing loudly. The harness feeds arbitrary
//! `i64` coefficient vectors (the full input domain of the pure-Rust cores —
//! the Python bridge normalises identically) and asserts the transform pair
//! is a bijection on `[0, q)`: forward-then-inverse reproduces the reduced
//! input bit-exactly, in both compositions, and every output stays in range
//! (THREAT_MODEL B8).

use libfuzzer_sys::fuzz_target;
use scpn_quantum_engine::ml_dsa::{ml_dsa_intt_core, ml_dsa_ntt_core, N, Q};

fuzz_target!(|data: &[u8]| {
    if data.len() < N * 8 {
        return;
    }

    let mut poly = [0i64; N];
    for (i, slot) in poly.iter_mut().enumerate() {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(&data[i * 8..i * 8 + 8]);
        *slot = i64::from_le_bytes(raw);
    }
    let reduced = poly.map(|c| c.rem_euclid(Q));

    let forward = ml_dsa_ntt_core(poly);
    assert!(forward.iter().all(|&c| (0..Q).contains(&c)));
    let roundtrip = ml_dsa_intt_core(forward);
    assert_eq!(roundtrip, reduced, "intt(ntt(x)) must equal x mod q");

    let inverse = ml_dsa_intt_core(poly);
    assert!(inverse.iter().all(|&c| (0..Q).contains(&c)));
    let counter_trip = ml_dsa_ntt_core(inverse);
    assert_eq!(counter_trip, reduced, "ntt(intt(x)) must equal x mod q");
});
