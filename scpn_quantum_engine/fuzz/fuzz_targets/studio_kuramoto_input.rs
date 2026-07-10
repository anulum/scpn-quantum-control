// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Studio WASM kernel input-boundary fuzz target

#![no_main]

//! Coverage-guided fuzz target for the browser-facing studio kernel parsers.
//!
//! These byte parsers are the ONLY input boundary of the shipped WASM kernels
//! (`parse_kuramoto_input` for the simulator ABI, `parse_compile_input` +
//! `xy_compile_digest` for the recompute ABI) — every byte a hostile page
//! feeds the kernel flows through them. The harness asserts the parsers fail
//! closed (no panic, no allocation blow-up) and that accepted inputs honour
//! the documented bounds. Simulation replay is capped so corpus growth cannot
//! turn parser checks into unbounded RK4 workloads (THREAT_MODEL B8).

use libfuzzer_sys::fuzz_target;
use scpn_quantum_studio_wasm_kernel::kuramoto::{
    output_len, parse_kuramoto_input, simulate, MAX_OSCILLATORS, MAX_STEPS,
};
use scpn_quantum_studio_wasm_kernel::{parse_compile_input, xy_compile_digest};

/// Largest wire payload either parser can accept (networked kuramoto,
/// n = 128: 24-byte header + (2n + n²)·8 bytes), rounded up.
const MAX_INPUT_BYTES: usize = 134_400;
/// Replay budget: only simulate when the output stays this small, so the
/// fuzzer spends its time in the parser, not in long RK4 integrations.
const MAX_REPLAY_OUTPUT: usize = 4_096;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    if let Ok(input) = parse_kuramoto_input(data) {
        assert!(input.n >= 1 && input.n <= MAX_OSCILLATORS);
        assert!(input.steps >= 1 && input.steps <= MAX_STEPS);
        assert!(input.dt.is_finite() && input.dt > 0.0);
        assert!(input.coupling.is_finite());
        assert_eq!(input.omega.len(), input.n);
        assert_eq!(input.theta0.len(), input.n);
        let expected = output_len(&input);
        if expected <= MAX_REPLAY_OUTPUT {
            let out = simulate(&input);
            assert_eq!(out.len(), expected);
        }
    }

    if let Ok(compile) = parse_compile_input(data) {
        let n = compile.n_qubits as usize;
        assert_eq!(compile.k_nm.len(), n * n);
        assert_eq!(compile.omega.len(), n);
        // An accepted compile input must always digest cleanly.
        let digest = xy_compile_digest(&compile).expect("accepted input must digest");
        assert_eq!(digest.len(), 32);
    }
});
