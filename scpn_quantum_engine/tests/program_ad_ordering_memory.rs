// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public effect-ordering lifecycle and recovery tests

use scpn_quantum_program_ad_replay::program_ad_ir::{
    interpret_program_ad_effect_ir_forward, interpret_program_ad_effect_ir_value_and_gradient,
};
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
    replay_checkpoint, with_replay_checkpoint,
};
use std::cell::Cell;
use std::rc::Rc;

fn ordered_chain(count: usize, tied: bool) -> String {
    let mut values = Vec::new();
    let mut effects = Vec::new();
    for index in 0..count {
        let target = format!("%{index}");
        let source = if index == 0 {
            "x".to_owned()
        } else {
            format!("%{}", index - 1)
        };
        let inputs = if index == 0 {
            vec![source]
        } else {
            vec![source.clone(), source]
        };
        values.push(serde_json::json!({"name":target,"producer":index,"version":0,"shape":[],"dtype":"float64","effect":index}));
        effects.push(serde_json::json!({"index":index,"kind":if index==0 {"parameter"} else {"primitive"},"target":target,"inputs":inputs,"version":0,"ordering":if tied {0} else {index},"operation":if index==0 {"parameter"} else {"add"}}));
    }
    if !tied {
        effects.reverse();
    }
    serde_json::json!({"format":"program_ad_effect_ir.v1","ssa_values":values,"effects":effects,"alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]}).to_string()
}

#[test]
fn public_ordering_preserves_original_ties_and_orders_shuffled_dependencies() {
    for count in [1usize, 2, 7, 18] {
        for tied in [false, true] {
            let source = ordered_chain(count, tied);
            // Repeated addition doubles the value and derivative at each effect.
            let derivative = (1usize << (count - 1)) as f64;
            let expected = 2.0 * derivative;
            let forward = interpret_program_ad_effect_ir_forward(&source, &[2.0]).unwrap();
            assert!(forward.supported, "{:?}", forward.blocked_reasons);
            assert_eq!(forward.value, Some(expected));
            let gradient =
                interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0]).unwrap();
            assert!(gradient.supported, "{:?}", gradient.blocked_reasons);
            assert_eq!(gradient.value, Some(expected));
            assert_eq!(gradient.gradient, [derivative]);
        }
    }
}

#[test]
fn public_forward_and_reverse_ordering_refuse_owned_cancellation_then_recover() {
    for tied in [false, true] {
        let source = ordered_chain(7, tied);
        for reverse in [false, true] {
            let observed = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&observed);
            with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get() + 1);
                    Ok(())
                },
                || {
                    if reverse {
                        let result =
                            interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0])
                                .unwrap();
                        assert!(result.supported, "{:?}", result.blocked_reasons);
                        assert_eq!(result.value, Some(128.0));
                        assert_eq!(result.gradient, [64.0]);
                    } else {
                        let result =
                            interpret_program_ad_effect_ir_forward(&source, &[2.0]).unwrap();
                        assert!(result.supported, "{:?}", result.blocked_reasons);
                        assert_eq!(result.value, Some(128.0));
                    }
                },
            );
            assert!(observed.get() > 1);
            for boundary in 1..=observed.get() {
                let calls = Rc::new(Cell::new(0usize));
                let recorded = Rc::clone(&calls);
                with_replay_checkpoint(
                    move || {
                        recorded.set(recorded.get() + 1);
                        if recorded.get() >= boundary {
                            Err("effect ordering owner cancelled".to_owned())
                        } else {
                            Ok(())
                        }
                    },
                    || {
                        let (supported, reasons) = if reverse {
                            match interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0])
                            {
                                Err(reason) => (false, vec![reason]),
                                Ok(result) => (result.supported, result.blocked_reasons),
                            }
                        } else {
                            match interpret_program_ad_effect_ir_forward(&source, &[2.0]) {
                                Err(reason) => (false, vec![reason]),
                                Ok(result) => (result.supported, result.blocked_reasons),
                            }
                        };
                        assert!(!supported);
                        assert!(
                            reasons
                                .iter()
                                .any(|reason| reason.contains("effect ordering owner cancelled")),
                            "{reasons:?}"
                        );
                    },
                );
                replay_checkpoint().unwrap();
                let forward = interpret_program_ad_effect_ir_forward(&source, &[2.0]).unwrap();
                assert!(forward.supported);
                assert_eq!(forward.value, Some(128.0));
                let retry =
                    interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0]).unwrap();
                assert!(retry.supported);
                assert_eq!(retry.value, Some(128.0));
                assert_eq!(retry.gradient, [64.0]);
            }
        }
    }
}
