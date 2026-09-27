// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public stencil admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
    replay_checkpoint, with_replay_checkpoint,
};
use std::cell::Cell;
use std::rc::Rc;

fn stencil_ir(operation: &str, count: usize) -> String {
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

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "expected {expected}, got {actual}"
    );
}

struct StencilCase {
    shape: &'static str,
    axis: usize,
    edge: usize,
    spacing: &'static str,
    output: usize,
    inputs: Vec<f64>,
    expected: f64,
    gradient: Vec<f64>,
}

fn stencil_cases() -> Vec<StencilCase> {
    [
        (
            "3",
            0,
            1,
            "scalar=1",
            0,
            vec![0.0, 1.0, 4.0],
            1.0,
            vec![-1.0, 1.0, 0.0],
        ),
        (
            "3",
            0,
            1,
            "scalar=1",
            1,
            vec![0.0, 1.0, 4.0],
            2.0,
            vec![-0.5, 0.0, 0.5],
        ),
        (
            "3",
            0,
            1,
            "scalar=1",
            2,
            vec![0.0, 1.0, 4.0],
            3.0,
            vec![0.0, -1.0, 1.0],
        ),
        (
            "3",
            0,
            2,
            "scalar=1",
            0,
            vec![0.0, 1.0, 4.0],
            0.0,
            vec![-1.5, 2.0, -0.5],
        ),
        (
            "3",
            0,
            2,
            "scalar=1",
            2,
            vec![0.0, 1.0, 4.0],
            4.0,
            vec![0.5, -2.0, 1.5],
        ),
        (
            "3",
            0,
            2,
            "scalar=-1",
            2,
            vec![0.0, 1.0, 4.0],
            -4.0,
            vec![-0.5, 2.0, -1.5],
        ),
        (
            "3",
            0,
            2,
            "coordinates=0,1,3",
            0,
            vec![0.0, 1.0, 9.0],
            0.0,
            vec![-4.0 / 3.0, 1.5, -1.0 / 6.0],
        ),
        (
            "3",
            0,
            2,
            "coordinates=0,1,3",
            1,
            vec![0.0, 1.0, 9.0],
            2.0,
            vec![-2.0 / 3.0, 0.5, 1.0 / 6.0],
        ),
        (
            "3",
            0,
            2,
            "coordinates=0,1,3",
            2,
            vec![0.0, 1.0, 9.0],
            6.0,
            vec![2.0 / 3.0, -1.5, 5.0 / 6.0],
        ),
        (
            "3",
            0,
            2,
            "coordinates=3,1,0",
            2,
            vec![9.0, 1.0, 0.0],
            0.0,
            vec![-1.0 / 6.0, 1.5, -4.0 / 3.0],
        ),
        (
            "2",
            0,
            1,
            "coordinates=0,2",
            0,
            vec![2.0, 8.0],
            3.0,
            vec![-0.5, 0.5],
        ),
        (
            "2",
            0,
            1,
            "coordinates=0,2",
            1,
            vec![2.0, 8.0],
            3.0,
            vec![-0.5, 0.5],
        ),
        (
            "2",
            0,
            1,
            "coordinates=2,0",
            1,
            vec![2.0, 8.0],
            -3.0,
            vec![0.5, -0.5],
        ),
        (
            "2x3",
            1,
            1,
            "scalar=1",
            4,
            vec![0.0, 1.0, 4.0, 10.0, 11.0, 14.0],
            2.0,
            vec![0.0, 0.0, 0.0, -0.5, 0.0, 0.5],
        ),
        (
            "2x3",
            0,
            1,
            "scalar=1",
            4,
            vec![0.0, 1.0, 4.0, 10.0, 11.0, 14.0],
            10.0,
            vec![0.0, -1.0, 0.0, 0.0, 1.0, 0.0],
        ),
        (
            "3x2",
            0,
            2,
            "scalar=1",
            5,
            vec![0.0, 10.0, 1.0, 11.0, 4.0, 14.0],
            4.0,
            vec![0.0, 0.5, 0.0, -2.0, 0.0, 1.5],
        ),
        (
            "2x3",
            1,
            2,
            "coordinates=0,1,3",
            4,
            vec![0.0, 1.0, 9.0, 10.0, 11.0, 19.0],
            2.0,
            vec![0.0, 0.0, 0.0, -2.0 / 3.0, 0.5, 1.0 / 6.0],
        ),
        (
            "2x3",
            0,
            1,
            "coordinates=2,0",
            4,
            vec![0.0, 1.0, 4.0, 10.0, 11.0, 14.0],
            -5.0,
            vec![0.0, 0.5, 0.0, 0.0, -0.5, 0.0],
        ),
    ]
    .into_iter()
    .map(
        |(shape, axis, edge, spacing, output, inputs, expected, gradient)| StencilCase {
            shape,
            axis,
            edge,
            spacing,
            output,
            inputs,
            expected,
            gradient,
        },
    )
    .collect()
}

#[test]
fn public_stencil_scalar_coordinate_edges_and_axes_observe_owned_cancellation() {
    for StencilCase {
        shape,
        axis,
        edge,
        spacing,
        output,
        inputs,
        expected,
        gradient,
    } in stencil_cases()
    {
        let source=stencil_ir(&format!("stencil:gradient:shape:{shape}:axis:{axis}:edge:{edge}:spacing:{spacing}:out:{output}"),inputs.len());
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
        assert_close(baseline.value.unwrap(), expected);
        assert_eq!(baseline.gradient.len(), gradient.len());
        for (actual, expected) in baseline.gradient.iter().zip(&gradient) {
            assert_close(*actual, *expected);
        }
        assert!(calls.get() > 1);
        for boundary in 1..=calls.get() {
            let observed = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&observed);
            let refused = with_replay_checkpoint(
                move || {
                    recorded.set(recorded.get() + 1);
                    if recorded.get() >= boundary {
                        Err("stencil owner cancelled".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
            );
            match refused {
                Err(reason) => assert!(reason.contains("stencil owner cancelled"), "{reason}"),
                Ok(result) => {
                    assert!(!result.supported);
                    assert!(
                        result
                            .blocked_reasons
                            .iter()
                            .any(|r| r.contains("stencil owner cancelled")),
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
fn public_stencil_rejects_shape_spacing_and_coefficient_overflow_then_recovers() {
    let valid = "stencil:gradient:shape:3:axis:0:edge:2:spacing:scalar=1:out:2";
    let inputs = [0.0, 1.0, 4.0];
    for operation in [
        valid.replace("shape:3", &format!("shape:{}x2", usize::MAX)),
        valid.replace("shape:3", "shape:0"),
        valid.replace("shape:3", "shape:"),
        valid.replace("shape:3", "shape:2"),
        valid.replace("axis:0", "axis:1"),
        valid.replace("edge:2", "edge:3"),
        valid.replace("out:2", "out:3"),
        valid.replace("scalar=1", "scalar=0"),
        valid.replace("scalar=1", "scalar=NaN"),
        valid.replace("scalar=1", "scalar=1e-320"),
        valid.replace("scalar=1", "coordinates="),
        valid.replace("scalar=1", "coordinates=0,1"),
        valid.replace("scalar=1", "coordinates=0,0,2"),
        valid.replace("scalar=1", "coordinates=0,2,1"),
        valid.replace("scalar=1", "coordinates=0,NaN,2"),
        format!("{valid}:extra"),
        valid.replace(":out:2", ""),
    ] {
        let refused =
            interpret_program_ad_effect_ir_value_and_gradient(&stencil_ir(&operation, 3), &inputs)
                .unwrap();
        assert!(!refused.supported, "{operation}");
        let retry =
            interpret_program_ad_effect_ir_value_and_gradient(&stencil_ir(valid, 3), &inputs)
                .unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(4.0));
        assert_eq!(retry.gradient, vec![0.5, -2.0, 1.5]);
    }
}

#[test]
fn public_stencil_two_point_second_order_refusal_and_reverse_overflow_recover() {
    let insufficient = stencil_ir(
        "stencil:gradient:shape:2:axis:0:edge:2:spacing:coordinates=0,2:out:1",
        2,
    );
    let refused =
        interpret_program_ad_effect_ir_value_and_gradient(&insufficient, &[2.0, 8.0]).unwrap();
    assert!(!refused.supported);
    assert!(refused
        .blocked_reasons
        .iter()
        .any(|r| r.contains("requires at least 3 samples")));
    let valid = stencil_ir(
        "stencil:gradient:shape:2:axis:0:edge:1:spacing:coordinates=0,2:out:1",
        2,
    );
    let retry = interpret_program_ad_effect_ir_value_and_gradient(&valid, &[2.0, 8.0]).unwrap();
    assert!(retry.supported);
    assert_eq!(retry.value, Some(3.0));
    assert_eq!(retry.gradient, vec![-0.5, 0.5]);

    let mut ir: serde_json::Value = serde_json::from_str(&stencil_ir(
        "stencil:gradient:shape:2:axis:0:edge:1:spacing:scalar=1e-308:out:1",
        2,
    ))
    .unwrap();
    let values = ir["ssa_values"].as_array_mut().unwrap();
    for index in [3, 4] {
        values.push(serde_json::json!({"name":format!("%{index}"),"producer":index,"version":0,"shape":[],"dtype":"float64","effect":index}));
    }
    let effects = ir["effects"].as_array_mut().unwrap();
    effects.push(serde_json::json!({"index":3,"kind":"parameter","target":"%3","inputs":["weight"],"version":0,"ordering":3,"operation":"parameter"}));
    effects.push(serde_json::json!({"index":4,"kind":"pure","target":"%4","inputs":["%2","%3"],"version":0,"ordering":4,"operation":"mul"}));
    let source = ir.to_string();
    let refused =
        interpret_program_ad_effect_ir_value_and_gradient(&source, &[0.0, 0.0, 2.0]).unwrap();
    assert!(!refused.supported);
    assert!(
        refused
            .blocked_reasons
            .iter()
            .any(|r| r.contains("adjoint contribution must be finite")),
        "{:?}",
        refused.blocked_reasons
    );
    replay_checkpoint().unwrap();
    let healthy = source.replace("scalar=1e-308", "scalar=2");
    let retry =
        interpret_program_ad_effect_ir_value_and_gradient(&healthy, &[2.0, 8.0, 2.0]).unwrap();
    assert!(retry.supported);
    assert_eq!(retry.value, Some(6.0));
    assert_eq!(retry.gradient, vec![-1.0, 1.0, 3.0]);
}

#[test]
fn public_stencil_workspace_budgets_cover_shape_coordinates_and_reverse_storage() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, ReplayMemoryRequest,
    };
    for case in stencil_cases() {
        let operation = format!(
            "stencil:gradient:shape:{}:axis:{}:edge:{}:spacing:{}:out:{}",
            case.shape, case.axis, case.edge, case.spacing, case.output
        );
        let ir = stencil_ir(&operation, case.inputs.len());
        let n = case.inputs.len();
        let rank = case.shape.split('x').count();
        let coordinates = case
            .spacing
            .strip_prefix("coordinates=")
            .map_or(0, |label| label.split(',').count());
        for gradient_surface in [false, true] {
            let expected = ReplayMemoryRequest {
                forward_bytes: (n + 1) * 8,
                adjoint_bytes: if gradient_surface { (2 * n + 1) * 8 } else { 0 },
                intermediate_bytes: ((if gradient_surface { 2 * n } else { n }) + coordinates) * 8
                    + 2 * rank * std::mem::size_of::<usize>(),
            };
            let total = expected.total_bytes().unwrap();
            let replay = || {
                if gradient_surface {
                    interpret_program_ad_effect_ir_value_and_gradient(&ir, &case.inputs)
                        .map(|r| (r.supported, r.value, r.gradient, r.blocked_reasons))
                } else {
                    interpret_program_ad_effect_ir_forward(&ir, &case.inputs)
                        .map(|r| (r.supported, r.value, Vec::new(), r.blocked_reasons))
                }
            };
            for budget in [total - 1, total, total + 1] {
                let calls = Rc::new(Cell::new(0usize));
                let recorded = Rc::clone(&calls);
                let result = with_replay_memory_admission(
                    move |request| {
                        recorded.set(recorded.get() + 1);
                        assert_eq!(request, expected);
                        if request.total_bytes()? > budget {
                            Err("stencil workspace budget refused".to_owned())
                        } else {
                            Ok(())
                        }
                    },
                    replay,
                );
                assert_eq!(calls.get(), 1);
                match result {
                    Err(reason) => {
                        assert!(budget < total);
                        assert!(reason.contains("stencil workspace budget refused"));
                    }
                    Ok(result) => {
                        assert_eq!(result.0, budget >= total);
                        if result.0 {
                            assert_close(result.1.unwrap(), case.expected);
                            if gradient_surface {
                                assert_eq!(result.2.len(), case.gradient.len());
                                for (actual, expected) in result.2.iter().zip(&case.gradient) {
                                    assert_close(*actual, *expected);
                                }
                            }
                        } else {
                            assert!(result
                                .3
                                .iter()
                                .any(|r| r.contains("stencil workspace budget refused")));
                        }
                    }
                }
                let retry = replay().unwrap();
                assert!(retry.0, "{:?}", retry.3);
                assert_close(retry.1.unwrap(), case.expected);
                if gradient_surface {
                    assert_eq!(retry.2.len(), case.gradient.len());
                    for (actual, expected) in retry.2.iter().zip(&case.gradient) {
                        assert_close(*actual, *expected);
                    }
                }
            }
        }
    }
}

#[test]
fn public_stencil_metadata_refuses_before_numeric_admission_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    let valid = "stencil:gradient:shape:3:axis:0:edge:2:spacing:scalar=1:out:2";
    for operation in [
        valid.replace("shape:3", &format!("shape:{}x2", usize::MAX)),
        valid.replace("shape:3", "shape:0"),
        valid.replace("shape:3", "shape:"),
        valid.replace("shape:3", "shape:2"),
        valid.replace("axis:0", "axis:1"),
        valid.replace("edge:2", "edge:3"),
        valid.replace("out:2", "out:3"),
        valid.replace("scalar=1", "scalar=0"),
        valid.replace("scalar=1", "scalar=NaN"),
        valid.replace("scalar=1", "coordinates="),
        valid.replace("scalar=1", "coordinates=0,1"),
        valid.replace("scalar=1", "coordinates=0,0,2"),
        valid.replace("scalar=1", "coordinates=0,2,1"),
        valid.replace("scalar=1", "coordinates=0,NaN,2"),
        format!("{valid}:extra"),
        valid.replace(":out:2", ""),
        valid.replace("shape:3", "shape:bad"),
        valid.replace("axis:0", "axis:bad"),
        valid.replace("edge:2", "edge:bad"),
        valid.replace("out:2", "out:bad"),
        valid.replace("scalar=1", "coordinates=0,bad,2"),
        valid.replace("scalar=1", "unknown=1"),
    ] {
        for gradient_surface in [false, true] {
            let calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&calls);
            let ir = stencil_ir(&operation, 3);
            let supported = with_replay_memory_admission(
                move |_| {
                    recorded.set(recorded.get() + 1);
                    Ok(())
                },
                || {
                    if gradient_surface {
                        interpret_program_ad_effect_ir_value_and_gradient(&ir, &[0.0, 1.0, 4.0])
                            .map(|r| r.supported)
                    } else {
                        interpret_program_ad_effect_ir_forward(&ir, &[0.0, 1.0, 4.0])
                            .map(|r| r.supported)
                    }
                },
            )
            .unwrap();
            assert!(!supported, "{operation}");
            assert_eq!(calls.get(), 0);
            let retry = interpret_program_ad_effect_ir_value_and_gradient(
                &stencil_ir(valid, 3),
                &[0.0, 1.0, 4.0],
            )
            .unwrap();
            assert!(retry.supported);
            assert_eq!(retry.value, Some(4.0));
            assert_eq!(retry.gradient, [0.5, -2.0, 1.5]);
        }
    }
}

#[test]
fn public_stencil_coefficient_overflow_remains_numeric_after_admission() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    let ir = stencil_ir(
        "stencil:gradient:shape:3:axis:0:edge:2:spacing:scalar=1e-320:out:2",
        3,
    );
    for gradient_surface in [false, true] {
        let calls = Rc::new(Cell::new(0usize));
        let recorded = Rc::clone(&calls);
        let supported = with_replay_memory_admission(
            move |_| {
                recorded.set(recorded.get() + 1);
                Ok(())
            },
            || {
                if gradient_surface {
                    interpret_program_ad_effect_ir_value_and_gradient(&ir, &[0.0, 1.0, 4.0])
                        .map(|r| r.supported)
                } else {
                    interpret_program_ad_effect_ir_forward(&ir, &[0.0, 1.0, 4.0])
                        .map(|r| r.supported)
                }
            },
        )
        .unwrap();
        assert!(!supported);
        assert_eq!(calls.get(), 1);
        let healthy = ir.replace("scalar=1e-320", "scalar=1");
        let retry =
            interpret_program_ad_effect_ir_value_and_gradient(&healthy, &[0.0, 1.0, 4.0]).unwrap();
        assert!(retry.supported);
        assert_eq!(retry.value, Some(4.0));
        assert_eq!(retry.gradient, [0.5, -2.0, 1.5]);
    }
}
