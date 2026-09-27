// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public multi-dot admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
    replay_checkpoint, with_replay_checkpoint,
};
use std::cell::Cell;
use std::rc::Rc;

fn multi_dot_ir(operation: &str, count: usize) -> String {
    let mut values = Vec::new();
    let mut effects = Vec::new();
    let mut inputs = Vec::new();
    for index in 0..count {
        let name = format!("%{index}");
        inputs.push(name.clone());
        values.push(serde_json::json!({"name":name,"producer":index,"version":0,"shape":[],"dtype":"float64","effect":index}));
        effects.push(serde_json::json!({"index":index,"kind":"parameter","target":name,"inputs":[format!("x{index}")],"version":0,"ordering":index,"operation":"parameter"}));
    }
    let target = format!("%{count}");
    values.push(serde_json::json!({"name":target,"producer":count,"version":0,"shape":[],"dtype":"float64","effect":count}));
    effects.push(serde_json::json!({"index":count,"kind":"primitive","target":target,"inputs":inputs,"version":0,"ordering":count,"operation":operation}));
    serde_json::json!({"format":"program_ad_effect_ir.v1","ssa_values":values,"effects":effects,"alias_edges":[],"control_regions":[],"phi_nodes":[],"bytecode_offsets":[]}).to_string()
}

#[test]
fn public_multi_dot_matrix_vector_chains_values_gradients_and_owned_cancellation() {
    for (operation, inputs, expected, gradient) in [
        (
            "linalg:multi_dot:2x2__2x2:out:2x2:0",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            19.0,
            vec![5.0, 7.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0],
        ),
        (
            "linalg:multi_dot:2x2__2x2:out:2x2:3",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            50.0,
            vec![0.0, 0.0, 6.0, 8.0, 0.0, 3.0, 0.0, 4.0],
        ),
        (
            "linalg:multi_dot:2__2x2:out:2:1",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            16.0,
            vec![4.0, 6.0, 0.0, 1.0, 0.0, 2.0],
        ),
        (
            "linalg:multi_dot:2x2__2:out:2:1",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            39.0,
            vec![0.0, 0.0, 5.0, 6.0, 3.0, 4.0],
        ),
        (
            "linalg:multi_dot:2__2:out:scalar",
            vec![1.0, 2.0, 3.0, 4.0],
            11.0,
            vec![3.0, 4.0, 1.0, 2.0],
        ),
        (
            "linalg:multi_dot:2x2__2x2__2x1:out:2x1:0",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            391.0,
            vec![105.0, 143.0, 0.0, 0.0, 9.0, 10.0, 18.0, 20.0, 19.0, 22.0],
        ),
        (
            "linalg:multi_dot:2__2x2__2:out:scalar",
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            219.0,
            vec![53.0, 83.0, 7.0, 8.0, 14.0, 16.0, 13.0, 16.0],
        ),
    ] {
        let source = multi_dot_ir(operation, inputs.len());
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let baseline = with_replay_checkpoint(
            move || {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
        )
        .unwrap();
        assert!(baseline.supported, "{:?}", baseline.blocked_reasons);
        assert_eq!(baseline.value, Some(expected));
        assert_eq!(baseline.gradient, gradient);
        assert!(calls.get() > 1);
        for boundary in 1..=calls.get() {
            let observed = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&observed);
            let refused = with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get() + 1);
                    if recorded.get() >= boundary {
                        Err("multi-dot owner cancelled".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
            );
            match refused {
                Err(reason) => assert!(reason.contains("multi-dot owner cancelled"), "{reason}"),
                Ok(result) => {
                    assert!(!result.supported);
                    assert!(
                        result
                            .blocked_reasons
                            .iter()
                            .any(|r| r.contains("multi-dot owner cancelled")),
                        "{:?}",
                        result.blocked_reasons
                    );
                }
            }
            replay_checkpoint().unwrap();
            let retry =
                interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, baseline.value);
            assert_eq!(retry.gradient, baseline.gradient);
        }
    }
}

#[test]
fn public_multi_dot_absurd_dimensions_and_malformed_chains_refuse_then_retry() {
    for operation in [
        format!("linalg:multi_dot:{}x2__2x2:out:2x2:0", usize::MAX),
        "linalg:multi_dot:2x2:out:2x2:0".to_owned(),
        "linalg:multi_dot:0x2__2x2:out:2x2:0".to_owned(),
        "linalg:multi_dot:1x2x2__2x2:out:2x2:0".to_owned(),
        "linalg:multi_dot:2x2__3x2:out:2x2:0".to_owned(),
        "linalg:multi_dot:2x2__2x2:out:2x2:4".to_owned(),
        "linalg:multi_dot:2x2__2x2:out:2:0".to_owned(),
        "linalg:multi_dot:2x2__2x2:out:scalar".to_owned(),
        "linalg:multi_dot:2x2__2x2:out:2x2:0:extra".to_owned(),
    ] {
        let inputs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let refused = interpret_program_ad_effect_ir_value_and_gradient(
            &multi_dot_ir(&operation, 8),
            &inputs,
        )
        .unwrap();
        assert!(!refused.supported, "{operation}");
        let retry = interpret_program_ad_effect_ir_value_and_gradient(
            &multi_dot_ir("linalg:multi_dot:2x2__2x2:out:2x2:0", 8),
            &inputs,
        )
        .unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(19.0));
        assert_eq!(retry.gradient, vec![5.0, 7.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0]);
    }
}

#[test]
fn public_multi_dot_dot_product_signed_zero_and_retry_are_preserved() {
    let source = multi_dot_ir("linalg:multi_dot:2__2:out:scalar", 4);
    let result =
        interpret_program_ad_effect_ir_value_and_gradient(&source, &[-0.0, -0.0, 1.0, 1.0])
            .unwrap();
    assert!(result.supported);
    assert_eq!(result.value.unwrap().to_bits(), (-0.0_f64).to_bits());
    let retry =
        interpret_program_ad_effect_ir_value_and_gradient(&source, &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert!(retry.supported);
    assert_eq!(retry.value, Some(11.0));
    assert_eq!(retry.gradient, vec![3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn public_multi_dot_workspace_budget_is_inclusive_and_refusal_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, ReplayMemoryRequest,
    };
    let source = multi_dot_ir("linalg:multi_dot:2x1__1x2:out:2x2:0", 4);
    let inputs = [2.0, 3.0, 4.0, 5.0];
    // Four parameters plus one result; kernel reverse keeps 3*4 + 2*4 + 8 floats.
    let expected = ReplayMemoryRequest {
        forward_bytes: 40,
        adjoint_bytes: 72,
        intermediate_bytes: 224,
    };
    for budget in [335usize, 336, 337] {
        let result = with_replay_memory_admission(
            move |request| {
                assert_eq!(request, expected);
                if request.total_bytes()? > budget {
                    Err("multi-dot workspace budget refused".to_owned())
                } else {
                    Ok(())
                }
            },
            || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
        )
        .unwrap();
        assert_eq!(result.supported, budget >= 336);
        if result.supported {
            assert_eq!(result.value, Some(8.0));
            assert_eq!(result.gradient, [4.0, 0.0, 2.0, 0.0]);
        } else {
            assert!(result
                .blocked_reasons
                .iter()
                .any(|reason| reason.contains("multi-dot workspace budget refused")));
        }
        let retry = interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(8.0));
        assert_eq!(retry.gradient, [4.0, 0.0, 2.0, 0.0]);
    }
    for budget in [135usize, 136, 137] {
        let result = with_replay_memory_admission(
            move |request| {
                assert_eq!(
                    request,
                    ReplayMemoryRequest {
                        forward_bytes: 40,
                        adjoint_bytes: 0,
                        intermediate_bytes: 96
                    }
                );
                if request.total_bytes()? > budget {
                    Err("forward multi-dot budget refused".to_owned())
                } else {
                    Ok(())
                }
            },
            || interpret_program_ad_effect_ir_forward(&source, &inputs),
        );
        if budget < 136 {
            assert!(result
                .unwrap_err()
                .contains("forward multi-dot budget refused"));
        } else {
            let result = result.unwrap();
            assert!(result.supported);
            assert_eq!(result.value, Some(8.0));
        }
    }
}

#[test]
fn public_sequential_multi_dot_reuses_peak_workspace_declaration() {
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, ReplayMemoryRequest,
    };
    let mut ir: serde_json::Value =
        serde_json::from_str(&multi_dot_ir("linalg:multi_dot:2x1__1x2:out:2x2:0", 4)).unwrap();
    let mut second_value = ir["ssa_values"][4].clone();
    let mut second_effect = ir["effects"][4].clone();
    second_value["name"] = serde_json::json!("%5");
    second_value["producer"] = serde_json::json!(5);
    second_value["effect"] = serde_json::json!(5);
    second_effect["index"] = serde_json::json!(5);
    second_effect["target"] = serde_json::json!("%5");
    second_effect["ordering"] = serde_json::json!(5);
    ir["ssa_values"].as_array_mut().unwrap().push(second_value);
    ir["effects"].as_array_mut().unwrap().push(second_effect);
    ir["ssa_values"].as_array_mut().unwrap().push(serde_json::json!({"name":"%6","producer":6,"version":0,"shape":[],"dtype":"float64","effect":6}));
    ir["effects"].as_array_mut().unwrap().push(serde_json::json!({"index":6,"kind":"pure","target":"%6","inputs":["%4","%5"],"version":0,"ordering":6,"operation":"add"}));
    let result = with_replay_memory_admission(
        |request| {
            assert_eq!(
                request,
                ReplayMemoryRequest {
                    forward_bytes: 56,
                    adjoint_bytes: 88,
                    intermediate_bytes: 224
                }
            );
            Ok(())
        },
        || {
            interpret_program_ad_effect_ir_value_and_gradient(
                &ir.to_string(),
                &[2.0, 3.0, 4.0, 5.0],
            )
        },
    )
    .unwrap();
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(16.0));
    assert_eq!(result.gradient, [8.0, 0.0, 4.0, 0.0]);
}
