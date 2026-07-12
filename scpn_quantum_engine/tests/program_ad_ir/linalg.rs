// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD IR linalg replay tests

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_trace() {
    // trace([[a,b],[c,d]]) = a + d; gradient is 1 on the diagonal, 0 off it.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_TRACE_2X2_PROGRAM_AD_IR,
        &[1.0, 2.0, 3.0, 4.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 5.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.gradient, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_det_2x2() {
    // det([[a,b],[c,d]]) = a*d - b*c; cofactor gradient [d, -c, -b, a].
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_DET_2X2_PROGRAM_AD_IR,
        &[2.0, 1.0, 1.0, 3.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 5.0_f64).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 4);
    assert!((result.gradient[0] - 3.0_f64).abs() <= 1.0e-12);
    assert!((result.gradient[1] - (-1.0_f64)).abs() <= 1.0e-12);
    assert!((result.gradient[2] - (-1.0_f64)).abs() <= 1.0e-12);
    assert!((result.gradient[3] - 2.0_f64).abs() <= 1.0e-12);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_inverse_element() {
    // inv([[a,b],[c,d]])[0,0] = d/det; reduced by *1.0 so it is the program value.
    // For [2,1,1,3]: det=5, M=[0.6,-0.2,-0.2,0.4]; d(M00)/dA = [-0.36, 0.12, 0.12, -0.04].
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_INV_2X2_ELEMENT_PROGRAM_AD_IR,
        &[2.0, 1.0, 1.0, 3.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 0.6_f64).abs() <= 1.0e-12);
    let expected = [-0.36_f64, 0.12, 0.12, -0.04];
    assert_eq!(result.gradient.len(), 4);
    for (got, want) in result.gradient.iter().zip(expected.iter()) {
        assert!((got - want).abs() <= 1.0e-12, "{got} vs {want}");
    }
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_fails_closed_on_indexed_multi_output_linalg() {
    // A bare solve element as the final effect: the IR does not record which component the
    // program returned, so the replay fails closed rather than replaying the wrong element.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_SOLVE_2X2_FINAL_PROGRAM_AD_IR,
        &[3.0, 1.0, 2.0, 4.0, 5.0, 6.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result.gradient.is_empty());
    assert!(result.blocked_reasons[0].contains("indexed multi-output linalg"));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_det_3x3() {
    // det of [[2,0,1],[1,3,2],[0,1,4]] = 21; gradient is the cofactor matrix.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_DET_3X3_PROGRAM_AD_IR,
        &[2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 21.0_f64).abs() <= 1.0e-12);
    let expected = [10.0_f64, -4.0, 1.0, 1.0, 8.0, -2.0, -3.0, -3.0, 6.0];
    assert_eq!(result.gradient.len(), 9);
    for (got, want) in result.gradient.iter().zip(expected.iter()) {
        assert!((got - want).abs() <= 1.0e-12, "{got} vs {want}");
    }
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_linalg_inverse_3x3_element() {
    // inv([[2,0,1],[1,3,2],[0,1,4]])[0,0] = 10/21; reduced by *1.0 so it is the program value.
    // d(M00)/dA00 = -M00^2 = -(10/21)^2.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_INV_3X3_ELEMENT_PROGRAM_AD_IR,
        &[2.0, 0.0, 1.0, 1.0, 3.0, 2.0, 0.0, 1.0, 4.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    let m00 = 10.0_f64 / 21.0;
    assert!((result.value.unwrap() - m00).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), 9);
    assert!((result.gradient[0] - (-m00 * m00)).abs() <= 1.0e-12);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_general_linalg_det_4x4() {
    // det(diag(2,3,4,5)) = 120 via the LU general path; gradient is the adjugate,
    // diag(60, 40, 30, 24), zero off the diagonal.
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        LINALG_DET_4X4_DIAGONAL_PROGRAM_AD_IR,
        &[
            2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0,
        ],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 120.0_f64).abs() <= 1.0e-9);
    let expected = [
        60.0_f64, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 30.0, 0.0, 0.0, 0.0, 0.0, 24.0,
    ];
    assert_eq!(result.gradient.len(), 16);
    for (got, want) in result.gradient.iter().zip(expected.iter()) {
        assert!((got - want).abs() <= 1.0e-9, "{got} vs {want}");
    }
}
