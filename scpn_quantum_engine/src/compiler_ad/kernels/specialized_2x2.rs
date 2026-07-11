// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Specialized 2x2 compiler AD kernels

//! Fixed-size 2x2 and spectral numerical kernels for compiler-backed AD parity.

use super::checked_vector;

const DISCRIMINANT_EPS: f64 = 1.0e-24;
const UPPER_CHART_EPS: f64 = 1.0e-12;
const MATRIX_2X2_DETERMINANT_EPS: f64 = 1.0e-12;
const SYMMETRIC_2X2_SPD_EPS: f64 = 1.0e-12;
const SYMMETRIC_2X2_EIGENVALUE_GAP_EPS: f64 = 1.0e-12;

fn checked_matrix_2x2_eigenvalues_values(
    values: &[f64],
    primitive: &str,
) -> Result<[f64; 4], String> {
    if values.len() != 4 {
        return Err(format!("{primitive} requires row-major matrix values"));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("values[{index}] is not finite ({value})"));
        }
    }
    let checked = [values[0], values[1], values[2], values[3]];
    let delta = checked[0] - checked[3];
    let discriminant = delta * delta + 4.0 * checked[1] * checked[2];
    if !discriminant.is_finite() || discriminant <= DISCRIMINANT_EPS {
        return Err(format!("{primitive} requires real distinct eigenvalues"));
    }
    Ok(checked)
}

fn checked_matrix_2x2_values(values: &[f64]) -> Result<[f64; 4], String> {
    let checked =
        checked_matrix_2x2_eigenvalues_values(values, "native matrix 2x2 eigensystem Rust kernel")?;
    if checked[1].abs() <= UPPER_CHART_EPS {
        return Err(
            "native matrix 2x2 eigensystem Rust kernel requires a non-zero upper off-diagonal eigenvector chart"
                .into(),
        );
    }
    Ok(checked)
}

fn eigensystem_geometry(values: &[f64; 4]) -> ([f64; 2], [[f64; 2]; 2], [f64; 2], f64) {
    let [a, b, c, d] = *values;
    let trace = a + d;
    let delta = a - d;
    let discriminant = delta * delta + 4.0 * b * c;
    let root = discriminant.sqrt();
    let lower = 0.5 * (trace - root);
    let upper = 0.5 * (trace + root);
    let q_lower = 0.5 * (-delta - root);
    let q_upper = 0.5 * (-delta + root);
    let lower_norm = (b * b + q_lower * q_lower).sqrt();
    let upper_norm = (b * b + q_upper * q_upper).sqrt();
    (
        [lower, upper],
        [
            [b / lower_norm, q_lower / lower_norm],
            [b / upper_norm, q_upper / upper_norm],
        ],
        [q_lower, q_upper],
        root,
    )
}

fn eigenvalues_geometry(values: &[f64; 4]) -> ([f64; 2], f64, f64) {
    let [a, b, c, d] = *values;
    let trace = a + d;
    let delta = a - d;
    let discriminant = delta * delta + 4.0 * b * c;
    let root = discriminant.sqrt();
    ([0.5 * (trace - root), 0.5 * (trace + root)], delta, root)
}

/// Evaluate bounded real-simple nonsymmetric 2x2 eigenvalues.
pub fn matrix_2x2_eigenvalues_value_inner(values: &[f64]) -> Result<[f64; 2], String> {
    let checked = checked_matrix_2x2_eigenvalues_values(
        values,
        "native matrix 2x2 eigenvalue Rust value kernel",
    )?;
    let (eigenvalues, _, _) = eigenvalues_geometry(&checked);
    Ok(eigenvalues)
}

/// Apply the exact JVP for bounded real-simple nonsymmetric 2x2 eigenvalues.
pub fn matrix_2x2_eigenvalues_jvp_inner(
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 2], String> {
    let checked = checked_matrix_2x2_eigenvalues_values(
        values,
        "native matrix 2x2 eigenvalue Rust JVP kernel",
    )?;
    let tangent = checked_vector::<4>(
        tangent,
        "tangent",
        "native matrix 2x2 eigenvalue Rust JVP kernel",
    )?;
    let [a, b, c, d] = checked;
    let [ta, tb, tc, td] = tangent;
    let trace_tangent = ta + td;
    let delta = a - d;
    let delta_tangent = ta - td;
    let discriminant = delta * delta + 4.0 * b * c;
    let root = discriminant.sqrt();
    let discriminant_tangent = 2.0 * delta * delta_tangent + 4.0 * (tb * c + b * tc);
    let root_tangent = discriminant_tangent / (2.0 * root);
    Ok([
        0.5 * (trace_tangent - root_tangent),
        0.5 * (trace_tangent + root_tangent),
    ])
}

/// Apply the exact VJP for bounded real-simple nonsymmetric 2x2 eigenvalues.
pub fn matrix_2x2_eigenvalues_vjp_inner(
    values: &[f64],
    cotangent: &[f64],
) -> Result<[f64; 4], String> {
    let checked = checked_matrix_2x2_eigenvalues_values(
        values,
        "native matrix 2x2 eigenvalue Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<2>(
        cotangent,
        "cotangent",
        "native matrix 2x2 eigenvalue Rust VJP kernel",
    )?;
    let [_, b, c, _] = checked;
    let (_, delta, root) = eigenvalues_geometry(&checked);
    let alpha = 0.5 * (cotangent[0] + cotangent[1]);
    let beta = (cotangent[1] - cotangent[0]) / (4.0 * root);
    let delta_term = 2.0 * delta * beta;
    Ok([
        alpha + delta_term,
        4.0 * c * beta,
        4.0 * b * beta,
        alpha - delta_term,
    ])
}

/// Sum-output gradient provenance helper for the vector-output eigenvalue primitive.
pub fn matrix_2x2_eigenvalues_sum_gradient_inner(values: &[f64]) -> Result<[f64; 4], String> {
    matrix_2x2_eigenvalues_vjp_inner(values, &[1.0; 2])
}

/// Evaluate the bounded real-simple nonsymmetric 2x2 eigensystem chart.
pub fn matrix_2x2_eigensystem_value_inner(values: &[f64]) -> Result<[f64; 6], String> {
    let checked = checked_matrix_2x2_values(values)?;
    let (eigenvalues, eigenvectors, _, _) = eigensystem_geometry(&checked);
    Ok([
        eigenvalues[0],
        eigenvalues[1],
        eigenvectors[0][0],
        eigenvectors[1][0],
        eigenvectors[0][1],
        eigenvectors[1][1],
    ])
}

/// Apply the exact JVP for the bounded real-simple nonsymmetric 2x2 eigensystem chart.
pub fn matrix_2x2_eigensystem_jvp_inner(
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 6], String> {
    let checked = checked_matrix_2x2_values(values)?;
    let tangent = checked_vector::<4>(
        tangent,
        "tangent",
        "native matrix 2x2 eigensystem Rust JVP kernel",
    )?;
    let [a, b, c, d] = checked;
    let [ta, tb, tc, td] = tangent;
    let trace_tangent = ta + td;
    let delta = a - d;
    let delta_tangent = ta - td;
    let discriminant = delta * delta + 4.0 * b * c;
    let root = discriminant.sqrt();
    let discriminant_tangent = 2.0 * delta * delta_tangent + 4.0 * (tb * c + b * tc);
    let root_tangent = discriminant_tangent / (2.0 * root);
    let lower_tangent = 0.5 * (trace_tangent - root_tangent);
    let upper_tangent = 0.5 * (trace_tangent + root_tangent);
    let (_, eigenvectors, q_values, _) = eigensystem_geometry(&checked);
    let q_lower_tangent = lower_tangent - ta;
    let q_upper_tangent = upper_tangent - ta;

    fn vector_tangent(b: f64, q: f64, vector: [f64; 2], tb: f64, tq: f64) -> [f64; 2] {
        let norm = (b * b + q * q).sqrt();
        let dot = vector[0] * tb + vector[1] * tq;
        [(tb - vector[0] * dot) / norm, (tq - vector[1] * dot) / norm]
    }

    let lower_vector_tangent = vector_tangent(b, q_values[0], eigenvectors[0], tb, q_lower_tangent);
    let upper_vector_tangent = vector_tangent(b, q_values[1], eigenvectors[1], tb, q_upper_tangent);
    Ok([
        lower_tangent,
        upper_tangent,
        lower_vector_tangent[0],
        upper_vector_tangent[0],
        lower_vector_tangent[1],
        upper_vector_tangent[1],
    ])
}

/// Apply the exact VJP for the bounded real-simple nonsymmetric 2x2 eigensystem chart.
pub fn matrix_2x2_eigensystem_vjp_inner(
    values: &[f64],
    cotangent: &[f64],
) -> Result<[f64; 4], String> {
    let checked = checked_matrix_2x2_values(values)?;
    let cotangent = checked_vector::<6>(
        cotangent,
        "cotangent",
        "native matrix 2x2 eigensystem Rust VJP kernel",
    )?;
    let [a, b, c, d] = checked;
    let delta = a - d;
    let (_, eigenvectors, q_values, root) = eigensystem_geometry(&checked);

    fn raw_vector_adjoint(b: f64, q: f64, vector: [f64; 2], cotangent: [f64; 2]) -> [f64; 2] {
        let norm = (b * b + q * q).sqrt();
        let dot = vector[0] * cotangent[0] + vector[1] * cotangent[1];
        [
            (cotangent[0] - vector[0] * dot) / norm,
            (cotangent[1] - vector[1] * dot) / norm,
        ]
    }

    let lower_raw_adjoint = raw_vector_adjoint(
        b,
        q_values[0],
        eigenvectors[0],
        [cotangent[2], cotangent[4]],
    );
    let upper_raw_adjoint = raw_vector_adjoint(
        b,
        q_values[1],
        eigenvectors[1],
        [cotangent[3], cotangent[5]],
    );
    let lower_eigenvalue_adjoint = cotangent[0] + lower_raw_adjoint[1];
    let upper_eigenvalue_adjoint = cotangent[1] + upper_raw_adjoint[1];
    let alpha = 0.5 * (lower_eigenvalue_adjoint + upper_eigenvalue_adjoint);
    let beta = (upper_eigenvalue_adjoint - lower_eigenvalue_adjoint) / (4.0 * root);
    let adj_a_eigenvalues = alpha + 2.0 * delta * beta;
    let adj_d_eigenvalues = alpha - 2.0 * delta * beta;
    let adj_b_eigenvalues = 4.0 * c * beta;
    let adj_c_eigenvalues = 4.0 * b * beta;
    let q_adjoint_sum = lower_raw_adjoint[1] + upper_raw_adjoint[1];
    let b_chart_adjoint = lower_raw_adjoint[0] + upper_raw_adjoint[0];
    Ok([
        adj_a_eigenvalues - q_adjoint_sum,
        adj_b_eigenvalues + b_chart_adjoint,
        adj_c_eigenvalues,
        adj_d_eigenvalues,
    ])
}

/// Sum-output gradient provenance helper for the vector-output eigensystem primitive.
pub fn matrix_2x2_eigensystem_sum_gradient_inner(values: &[f64]) -> Result<[f64; 4], String> {
    matrix_2x2_eigensystem_vjp_inner(values, &[1.0; 6])
}

fn checked_matrix_2x2_determinant_values(
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<[f64; 4], String> {
    if values.len() != 4 {
        return Err(format!(
            "{primitive} requires row-major 2x2 matrix {label} values"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
    }
    Ok([values[0], values[1], values[2], values[3]])
}

/// Evaluate det(A) for a row-major finite real 2x2 matrix.
pub fn matrix_2x2_determinant_value_inner(values: &[f64]) -> Result<[f64; 1], String> {
    let [a00, a01, a10, a11] = checked_matrix_2x2_determinant_values(
        values,
        "values",
        "native matrix 2x2 determinant Rust value kernel",
    )?;
    Ok([a00 * a11 - a01 * a10])
}

/// Apply the exact JVP for det(A) over row-major 2x2 matrices.
pub fn matrix_2x2_determinant_jvp_inner(
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 1], String> {
    let [a00, a01, a10, a11] = checked_matrix_2x2_determinant_values(
        values,
        "values",
        "native matrix 2x2 determinant Rust JVP kernel",
    )?;
    let [t00, t01, t10, t11] = checked_matrix_2x2_determinant_values(
        tangent,
        "tangent",
        "native matrix 2x2 determinant Rust JVP kernel",
    )?;
    Ok([t00 * a11 + a00 * t11 - t01 * a10 - a01 * t10])
}

/// Apply the exact VJP for det(A) over row-major 2x2 matrices.
pub fn matrix_2x2_determinant_vjp_inner(
    values: &[f64],
    cotangent: &[f64],
) -> Result<[f64; 4], String> {
    let [a00, a01, a10, a11] = checked_matrix_2x2_determinant_values(
        values,
        "values",
        "native matrix 2x2 determinant Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<1>(
        cotangent,
        "cotangent",
        "native matrix 2x2 determinant Rust VJP kernel",
    )?;
    Ok([
        cotangent[0] * a11,
        -cotangent[0] * a10,
        -cotangent[0] * a01,
        cotangent[0] * a00,
    ])
}

/// Return the scalar-output adjugate gradient for det(A).
pub fn matrix_2x2_determinant_gradient_inner(values: &[f64]) -> Result<[f64; 4], String> {
    matrix_2x2_determinant_vjp_inner(values, &[1.0])
}

fn checked_matrix_2x2_inverse_values(
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<([f64; 4], f64), String> {
    let checked = checked_matrix_2x2_determinant_values(values, label, primitive)?;
    let determinant = checked[0] * checked[3] - checked[1] * checked[2];
    if !determinant.is_finite() || determinant.abs() <= MATRIX_2X2_DETERMINANT_EPS {
        return Err(format!(
            "{primitive} requires a nonsingular row-major 2x2 matrix"
        ));
    }
    Ok((checked, determinant))
}

fn inverse_2x2_from_checked(values: &[f64; 4], determinant: f64) -> [f64; 4] {
    [
        values[3] / determinant,
        -values[1] / determinant,
        -values[2] / determinant,
        values[0] / determinant,
    ]
}

fn matmul_2x2(left: &[f64; 4], right: &[f64; 4]) -> [f64; 4] {
    [
        left[0] * right[0] + left[1] * right[2],
        left[0] * right[1] + left[1] * right[3],
        left[2] * right[0] + left[3] * right[2],
        left[2] * right[1] + left[3] * right[3],
    ]
}

fn transpose_2x2(values: &[f64; 4]) -> [f64; 4] {
    [values[0], values[2], values[1], values[3]]
}

/// Evaluate inv(A) for a nonsingular row-major finite real 2x2 matrix.
pub fn matrix_2x2_inverse_value_inner(values: &[f64]) -> Result<[f64; 4], String> {
    let (checked, determinant) = checked_matrix_2x2_inverse_values(
        values,
        "values",
        "native matrix 2x2 inverse Rust value kernel",
    )?;
    Ok(inverse_2x2_from_checked(&checked, determinant))
}

/// Apply the exact JVP -A^-1 dA A^-1 for row-major 2x2 inverse.
pub fn matrix_2x2_inverse_jvp_inner(values: &[f64], tangent: &[f64]) -> Result<[f64; 4], String> {
    let (checked, determinant) = checked_matrix_2x2_inverse_values(
        values,
        "values",
        "native matrix 2x2 inverse Rust JVP kernel",
    )?;
    let tangent = checked_matrix_2x2_determinant_values(
        tangent,
        "tangent",
        "native matrix 2x2 inverse Rust JVP kernel",
    )?;
    let inverse = inverse_2x2_from_checked(&checked, determinant);
    let product = matmul_2x2(&matmul_2x2(&inverse, &tangent), &inverse);
    Ok([-product[0], -product[1], -product[2], -product[3]])
}

/// Apply the exact VJP -A^-T C A^-T for row-major 2x2 inverse.
pub fn matrix_2x2_inverse_vjp_inner(values: &[f64], cotangent: &[f64]) -> Result<[f64; 4], String> {
    let (checked, determinant) = checked_matrix_2x2_inverse_values(
        values,
        "values",
        "native matrix 2x2 inverse Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<4>(
        cotangent,
        "cotangent",
        "native matrix 2x2 inverse Rust VJP kernel",
    )?;
    let inverse = inverse_2x2_from_checked(&checked, determinant);
    let inverse_transpose = transpose_2x2(&inverse);
    let product = matmul_2x2(
        &matmul_2x2(&inverse_transpose, &cotangent),
        &inverse_transpose,
    );
    Ok([-product[0], -product[1], -product[2], -product[3]])
}

/// Return the sum-output gradient provenance for the vector-output 2x2 inverse.
pub fn matrix_2x2_inverse_sum_gradient_inner(values: &[f64]) -> Result<[f64; 4], String> {
    matrix_2x2_inverse_vjp_inner(values, &[1.0; 4])
}

fn checked_matrix_2x2_solve_values(
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<([f64; 4], [f64; 2], f64), String> {
    if values.len() != 6 {
        return Err(format!(
            "{primitive} requires row-major 2x2 matrix plus rhs {label} values"
        ));
    }
    let (matrix, determinant) = checked_matrix_2x2_inverse_values(&values[..4], label, primitive)?;
    for (offset, value) in values[4..6].iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "{primitive} {label} value {} is not finite",
                offset + 4
            ));
        }
    }
    Ok((matrix, [values[4], values[5]], determinant))
}

fn checked_matrix_2x2_solve_tangent_values(
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<([f64; 4], [f64; 2]), String> {
    if values.len() != 6 {
        return Err(format!(
            "{primitive} requires row-major 2x2 matrix plus rhs {label} values"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{primitive} {label} value {index} is not finite"));
        }
    }
    Ok((
        [values[0], values[1], values[2], values[3]],
        [values[4], values[5]],
    ))
}

fn solve_2x2_from_checked(matrix: &[f64; 4], determinant: f64, rhs: &[f64; 2]) -> [f64; 2] {
    [
        (matrix[3] * rhs[0] - matrix[1] * rhs[1]) / determinant,
        (-matrix[2] * rhs[0] + matrix[0] * rhs[1]) / determinant,
    ]
}

/// Evaluate A^-1 b for a nonsingular row-major finite real 2x2 system.
pub fn matrix_2x2_solve_value_inner(values: &[f64]) -> Result<[f64; 2], String> {
    let (matrix, rhs, determinant) = checked_matrix_2x2_solve_values(
        values,
        "values",
        "native matrix 2x2 solve Rust value kernel",
    )?;
    Ok(solve_2x2_from_checked(&matrix, determinant, &rhs))
}

/// Apply the exact JVP A^-1 (db - dA x) for row-major 2x2 linear solves.
pub fn matrix_2x2_solve_jvp_inner(values: &[f64], tangent: &[f64]) -> Result<[f64; 2], String> {
    let (matrix, rhs, determinant) = checked_matrix_2x2_solve_values(
        values,
        "values",
        "native matrix 2x2 solve Rust JVP kernel",
    )?;
    let (tangent_matrix, tangent_rhs) = checked_matrix_2x2_solve_tangent_values(
        tangent,
        "tangent",
        "native matrix 2x2 solve Rust JVP kernel",
    )?;
    let primal = solve_2x2_from_checked(&matrix, determinant, &rhs);
    let residual = [
        tangent_rhs[0] - tangent_matrix[0] * primal[0] - tangent_matrix[1] * primal[1],
        tangent_rhs[1] - tangent_matrix[2] * primal[0] - tangent_matrix[3] * primal[1],
    ];
    Ok(solve_2x2_from_checked(&matrix, determinant, &residual))
}

/// Apply the exact VJP for row-major 2x2 solves: dA=-lambda x^T, db=lambda.
pub fn matrix_2x2_solve_vjp_inner(values: &[f64], cotangent: &[f64]) -> Result<[f64; 6], String> {
    let (matrix, rhs, determinant) = checked_matrix_2x2_solve_values(
        values,
        "values",
        "native matrix 2x2 solve Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<2>(
        cotangent,
        "cotangent",
        "native matrix 2x2 solve Rust VJP kernel",
    )?;
    let primal = solve_2x2_from_checked(&matrix, determinant, &rhs);
    let inverse = inverse_2x2_from_checked(&matrix, determinant);
    let inverse_transpose = transpose_2x2(&inverse);
    let adjoint = [
        inverse_transpose[0] * cotangent[0] + inverse_transpose[1] * cotangent[1],
        inverse_transpose[2] * cotangent[0] + inverse_transpose[3] * cotangent[1],
    ];
    Ok([
        -adjoint[0] * primal[0],
        -adjoint[0] * primal[1],
        -adjoint[1] * primal[0],
        -adjoint[1] * primal[1],
        adjoint[0],
        adjoint[1],
    ])
}

/// Return the sum-output gradient provenance for the vector-output 2x2 solve.
pub fn matrix_2x2_solve_sum_gradient_inner(values: &[f64]) -> Result<[f64; 6], String> {
    matrix_2x2_solve_vjp_inner(values, &[1.0; 2])
}

fn checked_symmetric_2x2_values(
    values: &[f64],
    label: &str,
    primitive: &str,
) -> Result<[f64; 3], String> {
    if values.len() != 3 {
        return Err(format!(
            "{primitive} requires upper-triangle symmetric 2x2 {label} values"
        ));
    }
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{primitive} {label} value {index} is not finite"));
        }
    }
    Ok([values[0], values[1], values[2]])
}

fn cholesky_2x2_from_checked(values: &[f64; 3], primitive: &str) -> Result<[f64; 3], String> {
    if values[0] <= SYMMETRIC_2X2_SPD_EPS {
        return Err(format!(
            "{primitive} requires a positive definite symmetric 2x2 matrix"
        ));
    }
    let l00 = values[0].sqrt();
    let l10 = values[1] / l00;
    let schur = values[2] - l10 * l10;
    if !schur.is_finite() || schur <= SYMMETRIC_2X2_SPD_EPS {
        return Err(format!(
            "{primitive} requires a positive definite symmetric 2x2 matrix"
        ));
    }
    Ok([l00, l10, schur.sqrt()])
}

/// Evaluate lower-triangle Cholesky factors for an SPD symmetric 2x2 matrix.
pub fn symmetric_2x2_cholesky_value_inner(values: &[f64]) -> Result<[f64; 3], String> {
    let checked = checked_symmetric_2x2_values(
        values,
        "values",
        "native symmetric 2x2 Cholesky Rust value kernel",
    )?;
    cholesky_2x2_from_checked(&checked, "native symmetric 2x2 Cholesky Rust value kernel")
}

/// Apply the exact JVP for SPD symmetric 2x2 Cholesky factors.
pub fn symmetric_2x2_cholesky_jvp_inner(
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 3], String> {
    let checked = checked_symmetric_2x2_values(
        values,
        "values",
        "native symmetric 2x2 Cholesky Rust JVP kernel",
    )?;
    let tangent = checked_symmetric_2x2_values(
        tangent,
        "tangent",
        "native symmetric 2x2 Cholesky Rust JVP kernel",
    )?;
    let [l00, l10, l11] =
        cholesky_2x2_from_checked(&checked, "native symmetric 2x2 Cholesky Rust JVP kernel")?;
    let tangent_l00 = tangent[0] / (2.0 * l00);
    let tangent_l10 = tangent[1] / l00 - l10 * tangent_l00 / l00;
    let tangent_l11 = (tangent[2] - 2.0 * l10 * tangent_l10) / (2.0 * l11);
    Ok([tangent_l00, tangent_l10, tangent_l11])
}

/// Apply the exact VJP for SPD symmetric 2x2 Cholesky factors.
pub fn symmetric_2x2_cholesky_vjp_inner(
    values: &[f64],
    cotangent: &[f64],
) -> Result<[f64; 3], String> {
    let checked = checked_symmetric_2x2_values(
        values,
        "values",
        "native symmetric 2x2 Cholesky Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<3>(
        cotangent,
        "cotangent",
        "native symmetric 2x2 Cholesky Rust VJP kernel",
    )?;
    let [l00, l10, l11] =
        cholesky_2x2_from_checked(&checked, "native symmetric 2x2 Cholesky Rust VJP kernel")?;
    let adjoint_schur = cotangent[2] / (2.0 * l11);
    let adjoint_l10 = cotangent[1] - 2.0 * l10 * adjoint_schur;
    let adjoint_l00 = cotangent[0] - adjoint_l10 * checked[1] / (l00 * l00);
    Ok([adjoint_l00 / (2.0 * l00), adjoint_l10 / l00, adjoint_schur])
}

/// Return the sum-output gradient provenance for the vector-output 2x2 Cholesky.
pub fn symmetric_2x2_cholesky_sum_gradient_inner(values: &[f64]) -> Result<[f64; 3], String> {
    symmetric_2x2_cholesky_vjp_inner(values, &[1.0; 3])
}

fn symmetric_2x2_eigenvalue_geometry(
    values: &[f64; 3],
    primitive: &str,
) -> Result<(f64, f64, f64), String> {
    let centre = 0.5 * (values[0] + values[2]);
    let half_delta = 0.5 * (values[0] - values[2]);
    let radius_squared = half_delta * half_delta + values[1] * values[1];
    if !radius_squared.is_finite() || radius_squared <= SYMMETRIC_2X2_EIGENVALUE_GAP_EPS {
        return Err(format!(
            "{primitive} requires distinct symmetric 2x2 eigenvalues"
        ));
    }
    let radius = radius_squared.sqrt();
    if !radius.is_finite() || radius <= SYMMETRIC_2X2_EIGENVALUE_GAP_EPS {
        return Err(format!(
            "{primitive} requires distinct symmetric 2x2 eigenvalues"
        ));
    }
    Ok((centre, half_delta, radius))
}

/// Evaluate ordered eigenvalues for a distinct symmetric 2x2 matrix.
pub fn symmetric_2x2_eigenvalues_value_inner(values: &[f64]) -> Result<[f64; 2], String> {
    let checked = checked_symmetric_2x2_values(
        values,
        "values",
        "native symmetric 2x2 eigenvalue Rust value kernel",
    )?;
    let (centre, _, radius) = symmetric_2x2_eigenvalue_geometry(
        &checked,
        "native symmetric 2x2 eigenvalue Rust value kernel",
    )?;
    Ok([centre - radius, centre + radius])
}

/// Apply the exact JVP for distinct symmetric 2x2 eigenvalues.
pub fn symmetric_2x2_eigenvalues_jvp_inner(
    values: &[f64],
    tangent: &[f64],
) -> Result<[f64; 2], String> {
    let checked = checked_symmetric_2x2_values(
        values,
        "values",
        "native symmetric 2x2 eigenvalue Rust JVP kernel",
    )?;
    let tangent = checked_symmetric_2x2_values(
        tangent,
        "tangent",
        "native symmetric 2x2 eigenvalue Rust JVP kernel",
    )?;
    let (_, half_delta, radius) = symmetric_2x2_eigenvalue_geometry(
        &checked,
        "native symmetric 2x2 eigenvalue Rust JVP kernel",
    )?;
    let tangent_centre = 0.5 * (tangent[0] + tangent[2]);
    let tangent_half_delta = 0.5 * (tangent[0] - tangent[2]);
    let tangent_radius = (half_delta * tangent_half_delta + checked[1] * tangent[1]) / radius;
    Ok([
        tangent_centre - tangent_radius,
        tangent_centre + tangent_radius,
    ])
}

/// Apply the exact VJP for distinct symmetric 2x2 eigenvalues.
pub fn symmetric_2x2_eigenvalues_vjp_inner(
    values: &[f64],
    cotangent: &[f64],
) -> Result<[f64; 3], String> {
    let checked = checked_symmetric_2x2_values(
        values,
        "values",
        "native symmetric 2x2 eigenvalue Rust VJP kernel",
    )?;
    let cotangent = checked_vector::<2>(
        cotangent,
        "cotangent",
        "native symmetric 2x2 eigenvalue Rust VJP kernel",
    )?;
    let (_, half_delta, radius) = symmetric_2x2_eigenvalue_geometry(
        &checked,
        "native symmetric 2x2 eigenvalue Rust VJP kernel",
    )?;
    let half_term = half_delta / (2.0 * radius);
    let offdiag_term = checked[1] / radius;
    Ok([
        cotangent[0] * (0.5 - half_term) + cotangent[1] * (0.5 + half_term),
        (cotangent[1] - cotangent[0]) * offdiag_term,
        cotangent[0] * (0.5 + half_term) + cotangent[1] * (0.5 - half_term),
    ])
}

/// Return the sum-output gradient provenance for vector-output symmetric 2x2 eigenvalues.
pub fn symmetric_2x2_eigenvalues_sum_gradient_inner(values: &[f64]) -> Result<[f64; 3], String> {
    symmetric_2x2_eigenvalues_vjp_inner(values, &[1.0; 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (left, right)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*left - *right).abs() < 1e-12,
                "index {index}: left={left}, right={right}"
            );
        }
    }

    #[test]
    fn matrix_2x2_eigenvalues_value_jvp_vjp_and_sum_gradient_match_closed_form() {
        let values = [2.0, 0.25, 0.75, 1.0];
        let tangent = [0.1, -0.2, 0.4, -0.3];
        let cotangent = [1.25, -0.75];

        assert_close(
            &matrix_2x2_eigenvalues_value_inner(&values).unwrap(),
            &[0.838_562_172_233_852_3, 2.161_437_827_766_147_5],
        );
        assert_close(
            &matrix_2x2_eigenvalues_jvp_inner(&values, &tangent).unwrap(),
            &[-0.213_389_341_902_768_15, 0.013_389_341_902_768_165],
        );
        assert_close(
            &matrix_2x2_eigenvalues_vjp_inner(&values, &cotangent).unwrap(),
            &[
                -0.505_928_946_018_454_4,
                -1.133_893_419_027_681_7,
                -0.377_964_473_009_227_2,
                1.005_928_946_018_454_4,
            ],
        );
        assert_close(
            &matrix_2x2_eigenvalues_sum_gradient_inner(&values).unwrap(),
            &[1.0, 0.0, 0.0, 1.0],
        );
    }

    #[test]
    fn matrix_2x2_eigenvalues_boundaries_fail_closed() {
        let nonreal = matrix_2x2_eigenvalues_value_inner(&[0.0, -1.0, 1.0, 0.0]).unwrap_err();
        assert!(nonreal.contains("real distinct eigenvalues"));
        let repeated = matrix_2x2_eigenvalues_value_inner(&[1.0, 0.0, 0.0, 1.0]).unwrap_err();
        assert!(repeated.contains("real distinct eigenvalues"));
    }

    #[test]
    fn matrix_2x2_eigensystem_value_matches_closed_form() {
        let values = [2.0, 0.25, 0.75, 1.0];
        let result = matrix_2x2_eigensystem_value_inner(&values).unwrap();
        assert_close(
            &result,
            &[
                0.838_562_172_233_852_3,
                2.161_437_827_766_147_5,
                0.210_430_715_716_423_36,
                0.840_070_779_091_305_9,
                -0.977_608_773_427_833_9,
                0.542_476_806_985_263,
            ],
        );
    }

    #[test]
    fn matrix_2x2_eigensystem_jvp_matches_closed_form() {
        let values = [2.0, 0.25, 0.75, 1.0];
        let tangent = [0.1, -0.2, 0.4, -0.3];
        let result = matrix_2x2_eigensystem_jvp_inner(&values, &tangent).unwrap();
        assert_close(
            &result,
            &[
                -0.213_389_341_902_768_15,
                0.013_389_341_902_768_165,
                -0.215_156_061_857_423_42,
                -0.065_142_791_863_097_92,
                -0.046_312_436_342_641_75,
                0.100_879_070_234_798_1,
            ],
        );
    }

    #[test]
    fn matrix_2x2_eigensystem_vjp_matches_jvp_transpose_reference() {
        let values = [2.0, 0.25, 0.75, 1.0];
        let cotangent = [1.25, -0.75, 0.5, -0.25, 0.3, -0.6];
        let result = matrix_2x2_eigensystem_vjp_inner(&values, &cotangent).unwrap();
        assert_close(
            &result,
            &[
                -0.464_840_980_381_554,
                -0.653_188_087_262_615_7,
                -0.592_983_537_780_116_1,
                0.964_840_980_381_554_1,
            ],
        );
    }

    #[test]
    fn matrix_2x2_eigensystem_boundaries_fail_closed() {
        let nonreal = matrix_2x2_eigensystem_value_inner(&[0.0, -1.0, 1.0, 0.0]).unwrap_err();
        assert!(nonreal.contains("real distinct eigenvalues"));
        let repeated = matrix_2x2_eigensystem_value_inner(&[1.0, 0.0, 0.0, 1.0]).unwrap_err();
        assert!(repeated.contains("real distinct eigenvalues"));
        let zero_chart = matrix_2x2_eigensystem_value_inner(&[2.0, 0.0, 1.0, 1.0]).unwrap_err();
        assert!(zero_chart.contains("upper off-diagonal eigenvector chart"));
    }

    #[test]
    fn matrix_2x2_determinant_value_jvp_vjp_and_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0];
        let tangent = [0.1, -0.2, 0.3, 0.4];
        let cotangent = [1.25];

        assert_close(
            &matrix_2x2_determinant_value_inner(&values).unwrap(),
            &[6.5],
        );
        assert_close(
            &matrix_2x2_determinant_jvp_inner(&values, &tangent).unwrap(),
            &[1.5],
        );
        assert_close(
            &matrix_2x2_determinant_vjp_inner(&values, &cotangent).unwrap(),
            &[3.75, -0.625, 1.25, 2.5],
        );
        assert_close(
            &matrix_2x2_determinant_gradient_inner(&values).unwrap(),
            &[3.0, -0.5, 1.0, 2.0],
        );
    }

    #[test]
    fn matrix_2x2_determinant_boundaries_fail_closed() {
        let wrong_count = matrix_2x2_determinant_value_inner(&[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("row-major 2x2 matrix values"));
        let non_finite =
            matrix_2x2_determinant_gradient_inner(&[2.0, f64::NAN, 0.5, 3.0]).unwrap_err();
        assert!(non_finite.contains("not finite"));
        let tangent_count =
            matrix_2x2_determinant_jvp_inner(&[2.0, -1.0, 0.5, 3.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("row-major 2x2 matrix tangent values"));
        let cotangent_count =
            matrix_2x2_determinant_vjp_inner(&[2.0, -1.0, 0.5, 3.0], &[1.0, 2.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 1 cotangent value"));
    }

    #[test]
    fn matrix_2x2_inverse_value_jvp_vjp_and_sum_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0];
        let tangent = [0.1, -0.2, 0.3, 0.4];
        let cotangent = [0.75, -1.25, 0.5, 2.0];

        assert_close(
            &matrix_2x2_inverse_value_inner(&values).unwrap(),
            &[
                0.461_538_461_538_461_56,
                0.153_846_153_846_153_85,
                -0.076_923_076_923_076_93,
                0.307_692_307_692_307_7,
            ],
        );
        assert_close(
            &matrix_2x2_inverse_jvp_inner(&values, &tangent).unwrap(),
            &[
                -0.044_970_414_201_183_43,
                -0.004_733_727_810_650_887,
                -0.028_402_366_863_905_325,
                -0.055_621_301_775_147_93,
            ],
        );
        assert_close(
            &matrix_2x2_inverse_vjp_inner(&values, &cotangent).unwrap(),
            &[
                -0.029_585_798_816_568_046,
                0.248_520_710_059_171_6,
                -0.189_349_112_426_035_5,
                -0.109_467_455_621_301_78,
            ],
        );
        assert_close(
            &matrix_2x2_inverse_sum_gradient_inner(&values).unwrap(),
            &[
                -0.236_686_390_532_544_37,
                -0.088_757_396_449_704_14,
                -0.284_023_668_639_053_26,
                -0.106_508_875_739_644_97,
            ],
        );
    }

    #[test]
    fn matrix_2x2_inverse_boundaries_fail_closed() {
        let wrong_count = matrix_2x2_inverse_value_inner(&[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("row-major 2x2 matrix values"));
        let non_finite =
            matrix_2x2_inverse_sum_gradient_inner(&[2.0, f64::NAN, 0.5, 3.0]).unwrap_err();
        assert!(non_finite.contains("not finite"));
        let singular = matrix_2x2_inverse_value_inner(&[1.0, 2.0, 2.0, 4.0]).unwrap_err();
        assert!(singular.contains("nonsingular row-major 2x2 matrix"));
        let tangent_count =
            matrix_2x2_inverse_jvp_inner(&[2.0, -1.0, 0.5, 3.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("row-major 2x2 matrix tangent values"));
        let cotangent_count =
            matrix_2x2_inverse_vjp_inner(&[2.0, -1.0, 0.5, 3.0], &[1.0, 2.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 4 cotangent value"));
    }

    #[test]
    fn matrix_2x2_solve_value_jvp_vjp_and_sum_gradient_match_closed_form() {
        let values = [2.0, -1.0, 0.5, 3.0, 1.5, -2.0];
        let tangent = [0.1, -0.2, 0.3, 0.4, -0.5, 0.75];
        let cotangent = [1.25, -0.75];

        assert_close(
            &matrix_2x2_solve_value_inner(&values).unwrap(),
            &[0.384_615_384_615_384_64, -0.730_769_230_769_230_7],
        );
        assert_close(
            &matrix_2x2_solve_jvp_inner(&values, &tangent).unwrap(),
            &[-0.173_372_781_065_088_76, 0.337_869_822_485_207_1],
        );
        assert_close(
            &matrix_2x2_solve_vjp_inner(&values, &cotangent).unwrap(),
            &[
                -0.244_082_840_236_686_4,
                0.463_757_396_449_704_15,
                0.014_792_899_408_284_023,
                -0.028_106_508_875_739_646,
                0.634_615_384_615_384_6,
                -0.038_461_538_461_538_464,
            ],
        );
        assert_close(
            &matrix_2x2_solve_sum_gradient_inner(&values).unwrap(),
            &[
                -0.147_928_994_082_840_24,
                0.281_065_088_757_396_44,
                -0.177_514_792_899_408_27,
                0.337_278_106_508_875_74,
                0.384_615_384_615_384_64,
                0.461_538_461_538_461_56,
            ],
        );
    }

    #[test]
    fn matrix_2x2_solve_boundaries_fail_closed() {
        let wrong_count = matrix_2x2_solve_value_inner(&[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("matrix plus rhs values"));
        let non_finite =
            matrix_2x2_solve_sum_gradient_inner(&[2.0, -1.0, 0.5, 3.0, f64::NAN, -2.0])
                .unwrap_err();
        assert!(non_finite.contains("not finite"));
        let singular = matrix_2x2_solve_value_inner(&[1.0, 2.0, 2.0, 4.0, 1.0, -1.0]).unwrap_err();
        assert!(singular.contains("nonsingular row-major 2x2 matrix"));
        let tangent_count =
            matrix_2x2_solve_jvp_inner(&[2.0, -1.0, 0.5, 3.0, 1.5, -2.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("matrix plus rhs tangent values"));
        let cotangent_count =
            matrix_2x2_solve_vjp_inner(&[2.0, -1.0, 0.5, 3.0, 1.5, -2.0], &[1.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 2 cotangent value"));
    }

    #[test]
    fn symmetric_2x2_cholesky_value_jvp_vjp_and_sum_gradient_match_closed_form() {
        let values = [4.0, 1.0, 3.0];
        let tangent = [0.2, -0.3, 0.4];
        let cotangent = [1.25, -0.75, 0.5];

        assert_close(
            &symmetric_2x2_cholesky_value_inner(&values).unwrap(),
            &[2.0, 0.5, 1.658_312_395_177_7],
        );
        assert_close(
            &symmetric_2x2_cholesky_jvp_inner(&values, &tangent).unwrap(),
            &[0.05, -0.162_5, 0.169_600_131_324_992_05],
        );
        assert_close(
            &symmetric_2x2_cholesky_vjp_inner(&values, &cotangent).unwrap(),
            &[
                0.368_797_229_518_055_1,
                -0.450_377_836_144_440_94,
                0.150_755_672_288_881_81,
            ],
        );
        assert_close(
            &symmetric_2x2_cholesky_sum_gradient_inner(&values).unwrap(),
            &[
                0.206_344_459_036_110_23,
                0.349_244_327_711_118_2,
                0.301_511_344_577_763_63,
            ],
        );
    }

    #[test]
    fn symmetric_2x2_cholesky_boundaries_fail_closed() {
        let wrong_count = symmetric_2x2_cholesky_value_inner(&[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("upper-triangle symmetric 2x2 values"));
        let non_finite =
            symmetric_2x2_cholesky_sum_gradient_inner(&[4.0, f64::NAN, 3.0]).unwrap_err();
        assert!(non_finite.contains("not finite"));
        let non_spd = symmetric_2x2_cholesky_value_inner(&[1.0, 2.0, 1.0]).unwrap_err();
        assert!(non_spd.contains("positive definite symmetric 2x2 matrix"));
        let tangent_count = symmetric_2x2_cholesky_jvp_inner(&[4.0, 1.0, 3.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("upper-triangle symmetric 2x2 tangent values"));
        let cotangent_count =
            symmetric_2x2_cholesky_vjp_inner(&[4.0, 1.0, 3.0], &[1.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 3 cotangent value"));
    }

    #[test]
    fn symmetric_2x2_eigenvalues_value_jvp_vjp_and_sum_gradient_match_closed_form() {
        let values = [2.0, 0.5, 3.0];
        let tangent = [0.1, -0.2, 0.4];
        let cotangent = [1.25, -0.75];

        assert_close(
            &symmetric_2x2_eigenvalues_value_inner(&values).unwrap(),
            &[1.792_893_218_813_452_5, 3.207_106_781_186_547_5],
        );
        assert_close(
            &symmetric_2x2_eigenvalues_jvp_inner(&values, &tangent).unwrap(),
            &[0.285_355_339_059_327_36, 0.214_644_660_940_672_64],
        );
        assert_close(
            &symmetric_2x2_eigenvalues_vjp_inner(&values, &cotangent).unwrap(),
            &[
                0.957_106_781_186_547_5,
                -std::f64::consts::SQRT_2,
                -0.457_106_781_186_547_4,
            ],
        );
        assert_close(
            &symmetric_2x2_eigenvalues_sum_gradient_inner(&values).unwrap(),
            &[1.0, 0.0, 1.0],
        );
    }

    #[test]
    fn symmetric_2x2_eigenvalues_boundaries_fail_closed() {
        let wrong_count = symmetric_2x2_eigenvalues_value_inner(&[1.0, 2.0]).unwrap_err();
        assert!(wrong_count.contains("upper-triangle symmetric 2x2 values"));
        let non_finite =
            symmetric_2x2_eigenvalues_sum_gradient_inner(&[2.0, f64::NAN, 3.0]).unwrap_err();
        assert!(non_finite.contains("not finite"));
        let repeated = symmetric_2x2_eigenvalues_value_inner(&[1.0, 0.0, 1.0]).unwrap_err();
        assert!(repeated.contains("distinct symmetric 2x2 eigenvalues"));
        let tangent_count =
            symmetric_2x2_eigenvalues_jvp_inner(&[2.0, 0.5, 3.0], &[1.0]).unwrap_err();
        assert!(tangent_count.contains("upper-triangle symmetric 2x2 tangent values"));
        let cotangent_count =
            symmetric_2x2_eigenvalues_vjp_inner(&[2.0, 0.5, 3.0], &[1.0]).unwrap_err();
        assert!(cotangent_count.contains("requires 2 cotangent value"));
    }
}
