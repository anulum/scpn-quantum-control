// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — Public shaped replay workspace admission tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{with_replay_memory_admission, ReplayMemoryRequest};
use std::{cell::Cell, rc::Rc};

#[test]
fn public_shaped_multiply_declares_workspace_and_preserves_inclusive_budget() {
    let source = serde_json::json!({
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":[2],"dtype":"float64","effect":0},
            {"name":"%1","producer":1,"version":0,"shape":[2],"dtype":"float64","effect":1},
            {"name":"%2","producer":2,"version":0,"shape":[],"dtype":"float64","effect":2}
        ],
        "effects":[
            {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
            {"index":1,"kind":"primitive","target":"%1","inputs":["%0","%0"],"version":0,"ordering":1,"operation":"mul"},
            {"index":2,"kind":"primitive","target":"%2","inputs":["%1"],"version":0,"ordering":2,"operation":"sum"}
        ],
        "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
    }).to_string();
    let observed = Rc::new(Cell::new(ReplayMemoryRequest::default()));
    let recorded = Rc::clone(&observed);
    let actual = with_replay_memory_admission(
        move |request| { recorded.set(request); Ok(()) },
        || interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0, 3.0]),
    ).unwrap();
    assert!(actual.supported);
    assert_eq!(actual.value, Some(13.0));
    assert_eq!(actual.gradient, [4.0, 6.0]);
    assert_eq!(observed.get().forward_bytes, 5 * 8);
    assert_eq!(observed.get().adjoint_bytes, 7 * 8);
    // A cloned cotangent and two broadcast operands alone occupy six f64s.
    assert!(observed.get().intermediate_bytes >= 6 * 8);
    let required = observed.get().total_bytes().unwrap();
    for limit in [required - 1, required, required + 1, 12 * 8] {
        let result = with_replay_memory_admission(
            move |request| {
                if request.total_bytes()? > limit { Err("shaped workspace refused".to_owned()) }
                else { Ok(()) }
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0, 3.0]),
        ).unwrap();
        if limit >= required { assert_eq!(result, actual); }
        else {
            assert!(!result.supported);
            assert_eq!(result.value, None);
            assert!(result.gradient.is_empty());
            assert!(result.blocked_reasons.iter().any(|reason| reason == "shaped workspace refused"));
        }
        assert_eq!(interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0, 3.0]).unwrap(), actual);
    }
}
