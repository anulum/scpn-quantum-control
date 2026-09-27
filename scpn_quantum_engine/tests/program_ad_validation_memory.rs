// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public input and metadata validation lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
    parse_program_ad_effect_ir,
};
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
    replay_checkpoint, with_replay_checkpoint,
};
use std::cell::Cell;
use std::rc::Rc;

fn branch_payload() -> serde_json::Value {
    let aliases: Vec<_> = (0..257).map(|index| serde_json::json!({"source":"%0","target":format!("view:{index}"),"kind":"view_alias","version":0})).collect();
    let mut incoming: Vec<_> = (0..257).map(|index| format!("unused:{index}")).collect();
    incoming.extend(["executed_true".to_owned(), "executed_false".to_owned()]);
    serde_json::json!({
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":[],"dtype":"float64","effect":0},
            {"name":"%1","producer":1,"version":0,"shape":[],"dtype":"float64","effect":1},
            {"name":"%2","producer":2,"version":0,"shape":[],"dtype":"float64","effect":2},
            {"name":"%3","producer":3,"version":0,"shape":[],"dtype":"float64","effect":3}
        ],
        "effects":[
            {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
            {"index":1,"kind":"control_branch","target":"%1","inputs":[],"version":0,"ordering":1,"operation":"branch:x:True"},
            {"index":2,"kind":"control_branch","target":"%2","inputs":[],"version":0,"ordering":2,"operation":"branch:x:False"},
            {"index":3,"kind":"pure","target":"%3","inputs":["%0","%0"],"version":0,"ordering":3,"operation":"mul"}
        ],
        "alias_edges":aliases,
        "control_regions":[
            {"index":0,"kind":"runtime_branch","predicate":"branch:x:True","entered":true,"source_line":null},
            {"index":1,"kind":"runtime_branch","predicate":"branch:x:False","entered":false,"source_line":null},
            {"index":2,"kind":"source_control_flow","predicate":"if_expression","entered":true,"source_line":3}
        ],
        "phi_nodes":[
            {"index":0,"target":"phi:0","incoming":incoming,"control_region":0,"selected":"executed_true","source_line":null},
            {"index":1,"target":"phi:1","incoming":["executed_true","executed_false"],"control_region":1,"selected":"executed_false","source_line":null},
            {"index":2,"target":"phi:2","incoming":["left","right"],"control_region":2,"selected":null,"source_line":3}
        ],
        "bytecode_offsets":[0,2,4]
    })
}

fn validate_surface(source: &str, surface: usize, inputs: &[f64]) -> Result<(), String> {
    match surface {
        0 => {
            let ir = parse_program_ad_effect_ir(source)?;
            assert_eq!(ir.effects.len(), 4);
            assert_eq!(ir.alias_edges.len(), 257);
            assert_eq!(ir.control_regions.len(), 3);
            assert_eq!(ir.phi_nodes.len(), 3);
        }
        1 => {
            let result = interpret_program_ad_effect_ir_forward(source, inputs)?;
            if !result.supported {
                return Err(result.blocked_reasons.join("; "));
            }
            assert_eq!(result.value, Some(4.0));
        }
        _ => {
            let result = interpret_program_ad_effect_ir_value_and_gradient(source, inputs)?;
            if !result.supported {
                return Err(result.blocked_reasons.join("; "));
            }
            assert_eq!(result.value, Some(4.0));
            assert_eq!(result.gradient, [4.0]);
        }
    }
    Ok(())
}

#[test]
fn public_metadata_input_alias_and_branch_validation_cancels_and_recovers() {
    let source = branch_payload().to_string();
    for surface in 0..3 {
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        with_replay_checkpoint(
            move || {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || validate_surface(&source, surface, &[2.0]),
        )
        .unwrap();
        assert!(calls.get() > 1);
        for boundary in 1..=calls.get() {
            let observed = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&observed);
            let refused = with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get() + 1);
                    if recorded.get() >= boundary {
                        Err("metadata validation owner cancelled".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || validate_surface(&source, surface, &[2.0]),
            );
            assert!(refused
                .unwrap_err()
                .contains("metadata validation owner cancelled"));
            replay_checkpoint().unwrap();
            validate_surface(&source, surface, &[2.0]).unwrap();
        }
    }
}

#[test]
fn public_input_finite_validation_crosses_multiple_chunks_and_preserves_gradient() {
    let source = serde_json::json!({
        "format":"program_ad_effect_ir.v1",
        "ssa_values":[
            {"name":"%0","producer":0,"version":0,"shape":[513],"dtype":"float64","effect":0},
            {"name":"%1","producer":1,"version":0,"shape":[],"dtype":"float64","effect":1}
        ],
        "effects":[
            {"index":0,"kind":"parameter","target":"%0","inputs":["x"],"version":0,"ordering":0,"operation":"parameter"},
            {"index":1,"kind":"primitive","target":"%1","inputs":["%0"],"version":0,"ordering":1,"operation":"sum"}
        ],
        "alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]
    }).to_string();
    let inputs = vec![2.0; 513];
    let result = interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(1026.0));
    assert_eq!(result.gradient, vec![1.0; 513]);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for index in [0usize, 256, 512] {
            let mut invalid = inputs.clone();
            invalid[index] = bad;
            let result =
                interpret_program_ad_effect_ir_value_and_gradient(&source, &invalid).unwrap();
            assert!(!result.supported);
            assert_eq!(
                result.blocked_reasons,
                ["Rust Program AD value+gradient inputs must be finite"]
            );
            let result = interpret_program_ad_effect_ir_forward(&source, &invalid).unwrap();
            assert!(!result.supported);
            assert_eq!(
                result.blocked_reasons,
                ["Rust Program AD interpreter inputs must be finite"]
            );
            let retry =
                interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, Some(1026.0));
            assert_eq!(retry.gradient, vec![1.0; 513]);
        }
    }
}

#[test]
fn public_malformed_alias_branch_and_phi_contracts_refuse_without_poisoning_replay() {
    let healthy = branch_payload();
    for case in 0..8 {
        let mut invalid = healthy.clone();
        match case {
            0 => invalid["alias_edges"][256]["kind"] = serde_json::json!("mutation_alias"),
            1 => invalid["effects"][1]["kind"] = serde_json::json!("pure"),
            2 => invalid["effects"][1]["inputs"] = serde_json::json!(["%0"]),
            3 => invalid["control_regions"][0]["entered"] = serde_json::json!(false),
            4 => invalid["phi_nodes"][0]["selected"] = serde_json::json!(null),
            5 => invalid["phi_nodes"][0]["control_region"] = serde_json::json!(99),
            6 => invalid["phi_nodes"][0]["incoming"] = serde_json::json!(["left", "right"]),
            _ => {
                invalid["phi_nodes"][1]["control_region"] = serde_json::json!(0);
                invalid["phi_nodes"][1]["selected"] = serde_json::json!("executed_true");
            }
        }
        for surface in 1..3 {
            assert!(validate_surface(&invalid.to_string(), surface, &[2.0]).is_err());
            validate_surface(&healthy.to_string(), surface, &[2.0]).unwrap();
        }
    }
    let mut invalid = healthy.clone();
    invalid["phi_nodes"][0]["incoming"] = serde_json::json!(["executed_true"]);
    assert!(parse_program_ad_effect_ir(&invalid.to_string())
        .unwrap_err()
        .contains("incoming must contain at least two"));
    validate_surface(&healthy.to_string(), 0, &[2.0]).unwrap();
}

#[test]
fn public_branch_validation_metadata_refuses_before_numeric_replay_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, with_replay_metadata_admission,
    };
    use std::cell::RefCell;
    let source = branch_payload().to_string();
    let parser_calls = Rc::new(Cell::new(0usize));
    let recorded = Rc::clone(&parser_calls);
    with_replay_metadata_admission(
        move |_| {
            recorded.set(recorded.get() + 1);
            Ok(())
        },
        || parse_program_ad_effect_ir(&source),
    )
    .unwrap();
    for surface in [1usize, 2] {
        let numeric_started = Rc::new(Cell::new(false));
        let metadata_calls = Rc::new(RefCell::new(Vec::new()));
        let observed = Rc::clone(&metadata_calls);
        let started = Rc::clone(&numeric_started);
        let numeric = Rc::clone(&numeric_started);
        with_replay_metadata_admission(
            move |bytes| {
                if !started.get() {
                    observed.borrow_mut().push(bytes);
                }
                Ok(())
            },
            || {
                with_replay_memory_admission(
                    move |_| {
                        numeric.set(true);
                        Ok(())
                    },
                    || validate_surface(&source, surface, &[2.0]),
                )
            },
        )
        .unwrap();
        assert!(numeric_started.get());
        // Four separately owned branch/region/phi tables precede replay planning.
        assert!(metadata_calls.borrow().len() >= parser_calls.get() + 4);
        for boundary in parser_calls.get() + 1..=metadata_calls.borrow().len() {
            let calls = Cell::new(0usize);
            let numeric_calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&numeric_calls);
            let refused = with_replay_metadata_admission(
                move |_| {
                    calls.set(calls.get() + 1);
                    if calls.get() == boundary {
                        Err("validation metadata refused".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || {
                    with_replay_memory_admission(
                        move |_| {
                            recorded.set(recorded.get() + 1);
                            Ok(())
                        },
                        || validate_surface(&source, surface, &[2.0]),
                    )
                },
            );
            assert!(refused.unwrap_err().contains("validation metadata refused"));
            assert_eq!(numeric_calls.get(), 0);
            validate_surface(&source, surface, &[2.0]).unwrap();
        }
    }
}
