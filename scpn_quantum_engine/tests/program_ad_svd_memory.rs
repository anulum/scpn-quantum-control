// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SCPN Quantum Control — public SVD admission and lifecycle tests

use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_value_and_gradient;
use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
    replay_checkpoint, with_replay_checkpoint,
};
use std::cell::Cell;
use std::rc::Rc;

fn svd_ir(operation: &str, count: usize) -> String {
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
fn public_svd_rectangular_and_square_values_gradients_and_outer_phase_cancellation() {
    // Singular values and derivatives of positive diagonal rectangular embeddings.
    for (shape, index, inputs, expected, gradient) in [
        (
            "2x3",
            0,
            vec![3.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            5.0,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
        (
            "2x3",
            1,
            vec![3.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            3.0,
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "3x2",
            0,
            vec![3.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            5.0,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ),
        (
            "3x2",
            1,
            vec![3.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            3.0,
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "3x3",
            0,
            vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0],
            5.0,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
        (
            "3x3",
            1,
            vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0],
            3.0,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ] {
        let source = svd_ir(&format!("linalg:svdvals:{shape}:{index}"), inputs.len());
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
                        Err("SVD owner cancelled".to_owned())
                    } else {
                        Ok(())
                    }
                },
                || interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs),
            );
            match refused {
                Err(reason) => assert!(reason.contains("SVD owner cancelled"), "{reason}"),
                Ok(result) => {
                    assert!(!result.supported);
                    assert!(
                        result
                            .blocked_reasons
                            .iter()
                            .any(|r| r.contains("SVD owner cancelled")),
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
fn public_svd_overflow_metadata_and_unsupported_spectra_refuse_before_retry() {
    let valid = "linalg:svdvals:2x3:0";
    let inputs = [3.0, 0.0, 0.0, 0.0, 5.0, 0.0];
    for operation in [
        format!("linalg:svdvals:{}x2:0", usize::MAX),
        format!("linalg:svdvals:2x{}:0", usize::MAX),
        "linalg:svdvals:0x3:0".to_owned(),
        "linalg:svdvals:2x0:0".to_owned(),
        "linalg:svdvals:2:0".to_owned(),
        "linalg:svdvals:2x3x4:0".to_owned(),
        "linalg:svdvals:2x3:2".to_owned(),
        "linalg:svdvals:2x3:-1".to_owned(),
        "linalg:svdvals:2x3:0:extra".to_owned(),
        "linalg:svdvals:2x3".to_owned(),
    ] {
        let refused =
            interpret_program_ad_effect_ir_value_and_gradient(&svd_ir(&operation, 6), &inputs)
                .unwrap();
        assert!(!refused.supported, "{operation}");
        let retry =
            interpret_program_ad_effect_ir_value_and_gradient(&svd_ir(valid, 6), &inputs).unwrap();
        assert!(retry.supported);
        assert_close(retry.value.unwrap(), 5.0);
    }
    for bad in [
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0, 3.0, 0.0],
    ] {
        let refused =
            interpret_program_ad_effect_ir_value_and_gradient(&svd_ir(valid, 6), &bad).unwrap();
        assert!(!refused.supported);
        let retry =
            interpret_program_ad_effect_ir_value_and_gradient(&svd_ir(valid, 6), &inputs).unwrap();
        assert!(retry.supported);
        assert_close(retry.value.unwrap(), 5.0);
    }
}

#[test]
fn public_svd_declared_matrix_budgets_preserve_values_gradients_and_retry() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_memory_admission, ReplayMemoryRequest,
    };
    for (shape, index, rows, cols, inputs, value, gradient) in [
        (
            "2x3",
            0,
            2,
            3,
            vec![3.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            5.0,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ),
        (
            "2x3",
            1,
            2,
            3,
            vec![3.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            3.0,
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "3x2",
            0,
            3,
            2,
            vec![3.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            5.0,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        ),
        (
            "3x2",
            1,
            3,
            2,
            vec![3.0, 0.0, 0.0, 5.0, 0.0, 0.0],
            3.0,
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "3x3",
            0,
            3,
            3,
            vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0],
            5.0,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ),
        (
            "3x3",
            1,
            3,
            3,
            vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0],
            3.0,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "1x3",
            0,
            1,
            3,
            vec![3.0, 4.0, 0.0],
            5.0,
            vec![0.6, 0.8, 0.0],
        ),
        (
            "3x1",
            0,
            3,
            1,
            vec![-3.0, 4.0, 0.0],
            5.0,
            vec![-0.6, 0.8, 0.0],
        ),
    ] {
        let ir = svd_ir(&format!("linalg:svdvals:{shape}:{index}"), inputs.len());
        let n = inputs.len();
        let k = rows.min(cols);
        for gradient_surface in [false, true] {
            // Source/matrix/vectors plus bidiagonal and copied solver scratch.
            // Sum all phase buffers conservatively, not a measured allocator peak.
            let expected = ReplayMemoryRequest {
                forward_bytes: (n + 1) * 8,
                adjoint_bytes: if gradient_surface { (2 * n + 1) * 8 } else { 0 },
                intermediate_bytes: ((if gradient_surface { 3 * n } else { 2 * n })
                    + rows * k
                    + k * cols
                    + k
                    + 4 * k
                    - 2
                    + rows
                    + 2 * cols)
                    * 8,
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
                            Err("SVD declared matrix budget refused".to_owned())
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
                        assert!(reason.contains("SVD declared matrix budget refused"));
                    }
                    Ok(result) => {
                        assert_eq!(result.0, budget >= total);
                        if result.0 {
                            assert_close(result.1.unwrap(), value);
                            if gradient_surface {
                                assert_eq!(result.2.len(), n);
                                for (a, b) in result.2.iter().zip(&gradient) {
                                    assert_close(*a, *b);
                                }
                            }
                        } else {
                            assert!(result
                                .3
                                .iter()
                                .any(|r| r.contains("SVD declared matrix budget refused")));
                        }
                    }
                }
                let retry = replay().unwrap();
                assert!(retry.0, "{:?}", retry.3);
                assert_close(retry.1.unwrap(), value);
                if gradient_surface {
                    assert_eq!(retry.2.len(), n);
                    for (a, b) in retry.2.iter().zip(&gradient) {
                        assert_close(*a, *b);
                    }
                }
            }
        }
    }
}

#[test]
fn public_svd_bad_layout_refuses_before_numeric_admission_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::with_replay_memory_admission;
    for (operation, count) in [
        (format!("linalg:svdvals:{}x2:0", usize::MAX), 6),
        (format!("linalg:svdvals:2x{}:0", usize::MAX), 6),
        ("linalg:svdvals:0x3:0".to_owned(), 6),
        ("linalg:svdvals:2x0:0".to_owned(), 6),
        ("linalg:svdvals:2:0".to_owned(), 6),
        ("linalg:svdvals:2x3x4:0".to_owned(), 6),
        ("linalg:svdvals:2x3:2".to_owned(), 6),
        ("linalg:svdvals:2x3:-1".to_owned(), 6),
        ("linalg:svdvals:2x3:0:extra".to_owned(), 6),
        ("linalg:svdvals:2x3".to_owned(), 6),
        ("linalg:svdvals:badx3:0".to_owned(), 6),
        ("linalg:svdvals:2xbad:0".to_owned(), 6),
        ("linalg:svdvals:2x3:bad".to_owned(), 6),
        ("linalg:svdvals:2x3:0".to_owned(), 5),
        ("linalg:svdvals:2x3:0".to_owned(), 7),
    ] {
        for gradient_surface in [false, true] {
            let ir = svd_ir(&operation, count);
            let inputs = vec![2.0; count];
            let calls = Rc::new(Cell::new(0usize));
            let recorded = Rc::clone(&calls);
            let supported = with_replay_memory_admission(
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
            assert!(!supported, "{operation}");
            assert_eq!(calls.get(), 0);
            let retry = interpret_program_ad_effect_ir_value_and_gradient(
                &svd_ir("linalg:svdvals:2x3:0", 6),
                &[3.0, 0.0, 0.0, 0.0, 5.0, 0.0],
            )
            .unwrap();
            assert!(retry.supported);
            assert_close(retry.value.unwrap(), 5.0);
            assert_eq!(retry.gradient.len(), 6);
            for (a, b) in retry.gradient.iter().zip([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]) {
                assert_close(*a, b);
            }
        }
    }
}

#[test]
fn public_svd_solver_ceiling_inherits_restores_and_recovers() {
    use scpn_quantum_program_ad_replay::program_ad_ir::interpret_program_ad_effect_ir_forward;
    use scpn_quantum_program_ad_replay::program_ad_lifecycle::{
        with_replay_solver_iterations, DEFAULT_REPLAY_SOLVER_ITERATIONS,
    };
    // Symmetric positive matrix: eigenvalues5,3,2 and top projector uu^T.
    let inputs = [4.0, 1.0, 0.0, 1.0, 4.0, 0.0, 0.0, 0.0, 2.0];
    let source = svd_ir("linalg:svdvals:3x3:0", 9);
    for adjoint in [false, true] {
        let replay = || {
            if adjoint {
                interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs)
                    .map(|r| (r.supported, r.value, r.gradient, r.blocked_reasons))
            } else {
                interpret_program_ad_effect_ir_forward(&source, &inputs)
                    .map(|r| (r.supported, r.value, Vec::new(), r.blocked_reasons))
            }
        };
        let assert_healthy = || {
            let result = replay().unwrap();
            assert!(result.0, "{:?}", result.3);
            assert_close(result.1.unwrap(), 5.0);
            if adjoint {
                assert_eq!(result.2.len(), 9);
                for (actual, expected) in result
                    .2
                    .iter()
                    .zip([0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
                {
                    assert_close(*actual, expected);
                }
            }
        };
        assert_healthy();
        let refused = with_replay_solver_iterations(1, || {
            with_replay_solver_iterations(DEFAULT_REPLAY_SOLVER_ITERATIONS, replay).unwrap()
        })
        .unwrap()
        .unwrap();
        assert!(!refused.0);
        assert!(
            refused
                .3
                .iter()
                .any(|r| r.contains("SVD solver iteration limit")),
            "{:?}",
            refused.3
        );
        assert_healthy();
        with_replay_solver_iterations(DEFAULT_REPLAY_SOLVER_ITERATIONS, || {
            let refused = with_replay_solver_iterations(1, replay).unwrap().unwrap();
            assert!(!refused.0);
            assert_healthy();
        })
        .unwrap();
        let entered = Cell::new(false);
        assert!(with_replay_solver_iterations(0, || entered.set(true)).is_err());
        assert!(!entered.get());
        assert_healthy();
        let panic = std::panic::catch_unwind(|| {
            with_replay_solver_iterations(1, || panic!("owned solver scope unwind")).unwrap();
        });
        assert!(panic.is_err());
        assert_healthy();
    }
}

#[test]
fn public_svd_nonfinite_spectrum_refuses_without_sort_panic_and_recovers() {
    let source = svd_ir("linalg:svdvals:2x2:0", 4);
    for inputs in [
        [f64::MAX, f64::MAX, f64::MAX, -f64::MAX],
        [f64::MAX, 0.0, 0.0, f64::MAX / 2.0],
    ] {
        let result = interpret_program_ad_effect_ir_value_and_gradient(&source, &inputs).unwrap();
        if inputs[1] != 0.0 {
            assert!(!result.supported);
            assert!(
                result
                    .blocked_reasons
                    .iter()
                    .any(|r| r.contains("must be finite")),
                "{:?}",
                result.blocked_reasons
            );
        } else {
            assert!(result.supported, "{:?}", result.blocked_reasons);
            assert_close(result.value.unwrap() / f64::MAX, 1.0);
            assert_eq!(result.gradient.len(), 4);
            for (actual, expected) in result.gradient.iter().zip([1.0, 0.0, 0.0, 0.0]) {
                assert_close(*actual, expected);
            }
        }
        let retry = interpret_program_ad_effect_ir_value_and_gradient(
            &svd_ir("linalg:svdvals:2x3:0", 6),
            &[3.0, 0.0, 0.0, 0.0, 5.0, 0.0],
        )
        .unwrap();
        assert!(retry.supported);
        assert_close(retry.value.unwrap(), 5.0);
    }
}
