// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD IR fuzz target

#![no_main]

//! Coverage-guided fuzz target for public Program AD IR Rust APIs.
//!
//! The harness exercises parsing, bounded forward replay, and bounded
//! value-plus-gradient replay through the same public functions exposed to the
//! Python bridge. Inputs are capped so corpus growth cannot turn local fuzz
//! checks into unbounded JSON parsing workloads.

use libfuzzer_sys::fuzz_target;
use scpn_quantum_engine::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
    parse_program_ad_effect_ir,
};

const MAX_PROGRAM_AD_IR_BYTES: usize = 16_384;
const EMPTY_INPUTS: [f64; 0] = [];
const FINITE_INPUTS: [f64; 16] = [
    0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 0.25, -0.25, 4.0, -4.0, 5.0, -5.0, 6.0,
];

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_PROGRAM_AD_IR_BYTES {
        return;
    }
    let Ok(serialization) = std::str::from_utf8(data) else {
        return;
    };

    exercise_public_program_ad_apis(serialization);
});

fn exercise_public_program_ad_apis(serialization: &str) {
    let _ = parse_program_ad_effect_ir(serialization);
    for inputs in [EMPTY_INPUTS.as_slice(), FINITE_INPUTS.as_slice()] {
        let _ = interpret_program_ad_effect_ir_forward(serialization, inputs);
        let _ = interpret_program_ad_effect_ir_value_and_gradient(serialization, inputs);
    }
}
