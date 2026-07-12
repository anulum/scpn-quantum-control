// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program-AD IR reduction replay tests

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_product_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_PRODUCT_REDUCTION_PROGRAM_AD_IR,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -1.0, 2.0, 3.0, -0.25, 0.1,
        ],
    )
    .unwrap();

    let expected_gradient = [
        92.0, 40.0, 42.0, 11.0, 6.4, 13.0, 4.0, 10.0, 18.0, 6.0, 120.0, 720.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(88.0));
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%0[4]", "%0[5]", "%1[0]", "%1[1]", "%1[2]",
            "%2[0]", "%2[1]", "%3"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 14);
    assert_eq!(result.supported_effect_count, 14);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_single_zero_product() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR,
        &[0.0, 2.0, 3.0, 4.0],
    )
    .unwrap();

    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert_eq!(result.value, Some(0.0));
    assert_eq!(result.gradient, vec![24.0, 0.0, 0.0, 0.0]);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_multi_zero_product() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR,
        &[0.0, 2.0, 0.0, 4.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("prod gradient supports at most one zero input")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_variance_std_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR,
        &[
            1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.25, 2.0, 1.5, -0.75, 0.25,
        ],
    )
    .unwrap();

    let sqrt_14 = 14.0_f64.sqrt();
    let sqrt_6 = 6.0_f64.sqrt();
    let expected_value = 455.0_f64 / 144.0 + 0.5 * (sqrt_14 - sqrt_6);
    let expected_gradient = [
        -0.5 - 2.0 / sqrt_14 - 2.0 / 9.0,
        1.875 - 0.5 / sqrt_14 - 5.0 / 36.0,
        -3.0 + 2.5 / sqrt_14 + 1.0 / 36.0,
        0.5 + 0.75 / sqrt_6 - 1.0 / 18.0,
        -1.875 + 1.0 / 9.0,
        3.0 - 0.75 / sqrt_6 + 5.0 / 18.0,
        1.0,
        2.25,
        2.25,
        sqrt_14 / 3.0,
        2.0 * sqrt_6 / 3.0,
        35.0 / 9.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - expected_value).abs() <= 1.0e-12);
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%0[4]", "%0[5]", "%1[0]", "%1[1]", "%1[2]",
            "%2[0]", "%2[1]", "%3"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 14);
    assert_eq!(result.supported_effect_count, 14);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_corrected_variance_std_reductions() {
    let ir = STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR
        .replace(
            "\"operation\": \"var:axis:0\"",
            "\"operation\": \"var:axis:0:ddof:1\"",
        )
        .replace(
            "\"operation\": \"std:axis:-1\"",
            "\"operation\": \"std:axis:-1:correction:1\"",
        )
        .replace("\"operation\": \"var\"", "\"operation\": \"var:ddof:2\"");
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &ir,
        &[
            1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.0, 2.0, 1.25, -0.75, 0.6,
        ],
    )
    .unwrap();

    let row_std = (7.0_f64 / 3.0).sqrt();
    let expected_value = 7.5 + 1.25 * row_std;
    let expected_gradient = [
        -1.8 - 5.0 / (6.0 * row_std),
        2.5 - 5.0 / (24.0 * row_std),
        -5.9 + 25.0 / (24.0 * row_std),
        1.175,
        -2.6,
        6.625,
        2.0,
        4.5,
        4.5,
        row_std,
        2.0,
        35.0 / 6.0,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - expected_value).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.supported_effect_count, 14);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_degenerate_moment_correction() {
    let ir = STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR.replace(
        "\"operation\": \"std:axis:-1\"",
        "\"operation\": \"std:axis:-1:ddof:3\"",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &ir,
        &[
            1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.25, 2.0, 1.5, -0.75, 0.25,
        ],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("correction must be less than reduction group size")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_zero_variance_std() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STD_ZERO_VARIANCE_PROGRAM_AD_IR,
        &[2.0, 2.0, 2.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("std gradient requires positive variance")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_order_statistic_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_ORDER_STATISTIC_REDUCTION_PROGRAM_AD_IR,
        &[
            3.0, -2.0, 0.5, 1.0, -1.5, 2.0, 0.7, -1.3, 0.25, 1.1, -0.4, 0.8, -0.6, 0.9, 1.2, -0.4,
            0.75, -1.1, 0.5,
        ],
    )
    .unwrap();

    let expected_gradient = [
        2.0625, 0.825, 1.175, 0.4375, -2.725, 0.625, 3.0, -1.5, 2.0, -2.0, -1.5, 3.0, -2.0, 0.75,
        -0.75, -0.25, 2.5, -1.625, 1.625,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 10.9_f64).abs() <= 1.0e-12);
    assert_eq!(
        result.parameter_targets,
        vec![
            "%0[0]", "%0[1]", "%0[2]", "%0[3]", "%0[4]", "%0[5]", "%1[0]", "%1[1]", "%1[2]",
            "%2[0]", "%2[1]", "%3", "%4", "%5", "%6[0]", "%6[1]", "%7[0]", "%7[1]", "%7[2]"
        ]
    );
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 32);
    assert_eq!(result.supported_effect_count, 32);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_order_statistic_ties() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        ORDER_STATISTIC_TIE_PROGRAM_AD_IR,
        &[2.0, 2.0, 1.0],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("strictly ordered values")));
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_static_trapezoid_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR,
        &[1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
    )
    .unwrap();

    let expected_gradient = [
        0.3125, 1.125, 0.875, -0.0625, -0.625, -0.5, 2.625, 0.4375, 1.75,
    ];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 5.46875).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 10);
    assert_eq!(result.supported_effect_count, 10);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_replays_full_grid_trapezoid_reductions() {
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        STATIC_TRAPEZOID_FULL_GRID_PROGRAM_AD_IR,
        &[1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
    )
    .unwrap();

    let expected_gradient = [0.3125, 1.125, 0.875, -0.25, -1.0, -0.6875, 2.625, 0.5, 1.75];
    assert!(result.supported, "{:?}", result.blocked_reasons);
    assert!((result.value.unwrap() - 5.375).abs() <= 1.0e-12);
    assert_eq!(result.gradient.len(), expected_gradient.len());
    for (actual, expected) in result.gradient.iter().zip(expected_gradient) {
        assert!((actual - expected).abs() <= 1.0e-12);
    }
    assert_eq!(result.effect_count, 10);
    assert_eq!(result.supported_effect_count, 10);
}

#[test]
fn program_ad_effect_ir_rust_value_and_gradient_rejects_invalid_trapezoid_metadata() {
    let invalid_ir = STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR.replace(
        "\"operation\": \"trapezoid:axis:1:x:0,0.25,1.0\"",
        "\"operation\": \"trapezoid:axis:1:x:0,1\"",
    );
    let result = interpret_program_ad_effect_ir_value_and_gradient(
        &invalid_ir,
        &[1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
    )
    .unwrap();

    assert!(!result.supported);
    assert!(result
        .blocked_reasons
        .iter()
        .any(|reason| reason.contains("x metadata length must match integration axis size")));
}
