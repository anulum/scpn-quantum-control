// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD IR parity tests

use scpn_quantum_engine::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
    mirror_program_ad_registry_metadata, parse_program_ad_effect_ir,
};

include!("program_ad_ir/fixtures.rs");
include!("program_ad_ir/parser_registry.rs");
include!("program_ad_ir/scalar_forward.rs");
include!("program_ad_ir/scalar_reverse.rs");
include!("program_ad_ir/structural.rs");
include!("program_ad_ir/reductions.rs");
include!("program_ad_ir/linalg.rs");
