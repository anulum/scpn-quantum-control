// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public spectral admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
    replay_checkpoint, with_replay_checkpoint,
};
use std::cell::Cell;
use std::rc::Rc;

fn spectral_ir(operation: &str, count: usize) -> String {
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

#[test]
fn public_spectral_values_vectors_and_gradients_observe_owned_cancellation() {
    let s = std::f64::consts::FRAC_1_SQRT_2;
    // Closed-form eigenpairs of [[2,1],[1,2]], with each API's frozen ordering/gauge.
    for (operation, expected, gradient) in [
        ("linalg:eigvalsh:0", 1.0, [0.5, -0.5, -0.5, 0.5]),
        ("linalg:eigvalsh:1", 3.0, [0.5, 0.5, 0.5, 0.5]),
        ("linalg:eigvals:2x2:1", 1.0, [0.5, -0.5, -0.5, 0.5]),
        ("linalg:eigvals:2x2:0", 3.0, [0.5, 0.5, 0.5, 0.5]),
        ("linalg:eig:eigenvalue:2x2:1", 1.0, [0.5, -0.5, -0.5, 0.5]),
        (
            "linalg:eig:eigenvector:2x2:1:0",
            -s,
            [s / 4.0, -s / 4.0, s / 4.0, -s / 4.0],
        ),
        (
            "linalg:eig:eigenvector:2x2:0:0",
            -s,
            [-s / 4.0, -s / 4.0, s / 4.0, s / 4.0],
        ),
        (
            "linalg:eigh:eigenvalue:2x2:L:0",
            1.0,
            [0.5, -0.5, -0.5, 0.5],
        ),
        ("linalg:eigh:eigenvalue:2x2:U:1", 3.0, [0.5, 0.5, 0.5, 0.5]),
        (
            "linalg:eigh:eigenvector:2x2:L:0:0",
            -s,
            [s / 4.0, 0.0, 0.0, -s / 4.0],
        ),
        (
            "linalg:eigh:eigenvector:2x2:U:1:0",
            s,
            [s / 4.0, 0.0, 0.0, -s / 4.0],
        ),
    ] {
        let source = spectral_ir(operation, 4);
        let inputs = [2.0, 1.0, 1.0, 2.0];
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
                        Err("spectral owner cancelled".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
            );
            match refused {
                Err(reason) => assert!(reason.contains("spectral owner cancelled"), "{reason}"),
                Ok(result) => {
                    assert!(!result.supported);
                    assert!(
                        result
                            .blocked_reasons
                            .iter()
                            .any(|r| r.contains("spectral owner cancelled")),
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
fn public_spectral_metadata_and_unsupported_spectra_refuse_before_retry() {
    for operation in [
        "linalg:eigvalsh:2",
        "linalg:eigvalsh:",
        "linalg:eigvalsh:0:extra",
        "linalg:eigvals:3x3:0",
        "linalg:eigvals:2x2:2",
        "linalg:eig:eigenvalue:2x2:2",
        "linalg:eig:eigenvector:2x2:0:2",
        "linalg:eigh:eigenvalue:2x2:X:0",
        "linalg:eigh:eigenvector:2x2:L:2:0",
        "linalg:eigh:eigenvector:2x2:L:0:0:extra:extra",
    ] {
        let refused = interpret_program_ad_effect_ir_value_and_gradient(
            &spectral_ir(operation, 4),
            &[2.0, 1.0, 1.0, 2.0],
        )
        .unwrap();
        assert!(!refused.supported, "{operation}");
        let retry = interpret_program_ad_effect_ir_value_and_gradient(
            &spectral_ir("linalg:eigvalsh:0", 4),
            &[2.0, 1.0, 1.0, 2.0],
        )
        .unwrap();
        assert!(retry.supported);
        assert_close(retry.value.unwrap(), 1.0);
    }
    for (operation, inputs) in [
        ("linalg:eigvalsh:0", [1.0, 0.0, 0.0, 1.0]),
        ("linalg:eigvals:2x2:0", [0.0, -1.0, 1.0, 0.0]),
        ("linalg:eig:eigenvalue:2x2:0", [0.0, -1.0, 1.0, 0.0]),
        ("linalg:eigh:eigenvector:2x2:L:0:0", [1.0, 0.0, 0.0, 2.0]),
    ] {
        let refused =
            interpret_program_ad_effect_ir_value_and_gradient(&spectral_ir(operation, 4), &inputs)
                .unwrap();
        assert!(!refused.supported);
        let retry = interpret_program_ad_effect_ir_value_and_gradient(
            &spectral_ir("linalg:eigvalsh:0", 4),
            &[2.0, 1.0, 1.0, 2.0],
        )
        .unwrap();
        assert!(retry.supported);
        assert_close(retry.value.unwrap(), 1.0);
    }
}

#[test]
fn public_spectral_finite_value_with_overflowing_reverse_contribution_recovers() {
    let mut ir: serde_json::Value =
        serde_json::from_str(&spectral_ir("linalg:eigh:eigenvector:2x2:L:0:0", 4)).unwrap();
    let values = ir["ssa_values"].as_array_mut().unwrap();
    for index in [5, 6] {
        values.push(serde_json::json!({"name":format!("%{index}"),"producer":index,"version":0,"shape":[],"dtype":"float64","effect":index}));
    }
    let effects = ir["effects"].as_array_mut().unwrap();
    effects.push(serde_json::json!({"index":5,"kind":"parameter","target":"%5","inputs":["weight"],"version":0,"ordering":5,"operation":"parameter"}));
    effects.push(serde_json::json!({"index":6,"kind":"pure","target":"%6","inputs":["%4","%5"],"version":0,"ordering":6,"operation":"mul"}));
    let source = ir.to_string();
    let refused = interpret_program_ad_effect_ir_value_and_gradient(
        &source,
        &[0.0, 5.0e-9, 5.0e-9, 0.0, 1.0e308],
    )
    .unwrap();
    assert!(!refused.supported);
    assert!(
        refused
            .blocked_reasons
            .iter()
            .any(|r| r.contains("spectral cotangent entries must be finite")),
        "{:?}",
        refused.blocked_reasons
    );
    replay_checkpoint().unwrap();
    let retry =
        interpret_program_ad_effect_ir_value_and_gradient(&source, &[2.0, 1.0, 1.0, 2.0, 2.0])
            .unwrap();
    assert!(retry.supported);
    let s = std::f64::consts::FRAC_1_SQRT_2;
    assert_close(retry.value.unwrap(), -2.0 * s);
    for (actual, expected) in retry.gradient.iter().zip([s / 2.0, 0.0, 0.0, -s / 2.0, -s]) {
        assert_close(*actual, expected);
    }
}

#[test]
fn public_spectral_workspace_budgets_preserve_eigenpairs_and_recover() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, ReplayMemoryRequest,
    };
    let s = std::f64::consts::FRAC_1_SQRT_2;
    for (operation, value, gradient) in [
        ("linalg:eigvalsh:0", 1.0, [0.5, -0.5, -0.5, 0.5]),
        ("linalg:eigvalsh:1", 3.0, [0.5, 0.5, 0.5, 0.5]),
        ("linalg:eigvals:2x2:0", 3.0, [0.5, 0.5, 0.5, 0.5]),
        ("linalg:eigvals:2x2:1", 1.0, [0.5, -0.5, -0.5, 0.5]),
        ("linalg:eig:eigenvalue:2x2:1", 1.0, [0.5, -0.5, -0.5, 0.5]),
        (
            "linalg:eig:eigenvector:2x2:1:0",
            -s,
            [s / 4.0, -s / 4.0, s / 4.0, -s / 4.0],
        ),
        (
            "linalg:eig:eigenvector:2x2:0:0",
            -s,
            [-s / 4.0, -s / 4.0, s / 4.0, s / 4.0],
        ),
        (
            "linalg:eigh:eigenvalue:2x2:L:0",
            1.0,
            [0.5, -0.5, -0.5, 0.5],
        ),
        ("linalg:eigh:eigenvalue:2x2:U:1", 3.0, [0.5, 0.5, 0.5, 0.5]),
        (
            "linalg:eigh:eigenvector:2x2:L:0:0",
            -s,
            [s / 4.0, 0.0, 0.0, -s / 4.0],
        ),
        (
            "linalg:eigh:eigenvector:2x2:U:1:0",
            s,
            [s / 4.0, 0.0, 0.0, -s / 4.0],
        ),
    ] {
        let ir = spectral_ir(operation, 4);
        let inputs = [2.0, 1.0, 1.0, 2.0];
        for gradient_surface in [false, true] {
            let expected = ReplayMemoryRequest {
                forward_bytes: 40,
                adjoint_bytes: if gradient_surface { 72 } else { 0 },
                intermediate_bytes: if gradient_surface { 64 } else { 32 },
            };
            let total = expected.total_bytes().unwrap();
            let replay = || {
                if gradient_surface {
                    interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs)
                        .map(|r| (r.supported, r.value, r.gradient, r.blocked_reasons))
                } else {
                    interpret_program_ad_effect_ir_forward(&ir, &inputs)
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
                            Err("spectral workspace budget refused".to_owned())
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
                        assert!(reason.contains("spectral workspace budget refused"));
                    }
                    Ok(result) => {
                        assert_eq!(result.0, budget >= total);
                        if result.0 {
                            assert_close(result.1.unwrap(), value);
                            if gradient_surface {
                                assert_eq!(result.2.len(), 4);
                                for (a, b) in result.2.iter().zip(&gradient) {
                                    assert_close(*a, *b);
                                }
                            }
                        } else {
                            assert!(result
                                .3
                                .iter()
                                .any(|r| r.contains("spectral workspace budget refused")));
                        }
                    }
                }
                let retry = replay().unwrap();
                assert!(retry.0, "{:?}", retry.3);
                assert_close(retry.1.unwrap(), value);
                if gradient_surface {
                    assert_eq!(retry.2.len(), 4);
                    for (a, b) in retry.2.iter().zip(&gradient) {
                        assert_close(*a, *b);
                    }
                }
            }
        }
    }
}

#[test]
fn public_spectral_bad_metadata_and_arity_refuse_before_numeric_admission() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    for (operation, count) in [
        ("linalg:eigvalsh:2", 4),
        ("linalg:eigvals:2x2:2", 4),
        ("linalg:eigvals:3x3:0", 4),
        ("linalg:eig:eigenvalue:3x3:0", 4),
        ("linalg:eig:eigenvector:2x2:2:0", 4),
        ("linalg:eigh:eigenvalue:2x2:X:0", 4),
        ("linalg:eigh:eigenvector:2x2:L:0:2", 4),
        ("linalg:eigvalsh:0:extra", 4),
        ("linalg:eigvalsh:0", 3),
        ("linalg:eigvals:2x2:0", 3),
        ("linalg:eig:eigenvalue:2x2:0", 3),
        ("linalg:eigh:eigenvalue:2x2:L:0", 3),
    ] {
        for gradient_surface in [false, true] {
            let calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&calls);
            let ir = spectral_ir(operation, count);
            let inputs = vec![2.0; count];
            let result = with_replay_memory_admission(
                move |_| {
                    recorded.set(recorded.get() + 1);
                    Ok(())
                },
                || {
                    if gradient_surface {
                        interpret_program_ad_effect_ir_value_and_gradient(&ir, &inputs)
                            .map(|r| r.supported)
                    } else {
                        interpret_program_ad_effect_ir_forward(&ir, &inputs).map(|r| r.supported)
                    }
                },
            )
            .unwrap();
            assert!(!result, "{operation}");
            assert_eq!(calls.get(), 0);
            let retry = interpret_program_ad_effect_ir_value_and_gradient(
                &spectral_ir("linalg:eigvalsh:0", 4),
                &[2.0, 1.0, 1.0, 2.0],
            )
            .unwrap();
            assert!(retry.supported);
            assert_close(retry.value.unwrap(), 1.0);
            assert_eq!(retry.gradient.len(), 4);
            for (a, b) in retry.gradient.iter().zip([0.5, -0.5, -0.5, 0.5]) {
                assert_close(*a, b);
            }
        }
    }
}
