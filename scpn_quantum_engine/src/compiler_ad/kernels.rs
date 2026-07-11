// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Native compiler AD numerical kernels

//! Shared validation and stable re-exports for compiler-backed AD numerical kernels.

mod generic;
mod specialized_2x2;

pub use generic::{
    matrix_frobenius_norm_squared_gradient_inner, matrix_frobenius_norm_squared_jvp_inner,
    matrix_frobenius_norm_squared_value_inner, matrix_frobenius_norm_squared_vjp_inner,
    matrix_matrix_product_jvp_inner, matrix_matrix_product_sum_gradient_inner,
    matrix_matrix_product_value_inner, matrix_matrix_product_vjp_inner,
    matrix_quadratic_form_gradient_inner, matrix_quadratic_form_jvp_inner,
    matrix_quadratic_form_value_inner, matrix_quadratic_form_vjp_inner,
    matrix_trace_gradient_inner, matrix_trace_jvp_inner, matrix_trace_value_inner,
    matrix_trace_vjp_inner, matrix_vector_product_jvp_inner,
    matrix_vector_product_sum_gradient_inner, matrix_vector_product_value_inner,
    matrix_vector_product_vjp_inner, vector_dot_gradient_inner, vector_dot_jvp_inner,
    vector_dot_value_inner, vector_dot_vjp_inner, vector_squared_norm_gradient_inner,
    vector_squared_norm_jvp_inner, vector_squared_norm_value_inner, vector_squared_norm_vjp_inner,
};

pub use specialized_2x2::{
    matrix_2x2_determinant_gradient_inner, matrix_2x2_determinant_jvp_inner,
    matrix_2x2_determinant_value_inner, matrix_2x2_determinant_vjp_inner,
    matrix_2x2_eigensystem_jvp_inner, matrix_2x2_eigensystem_sum_gradient_inner,
    matrix_2x2_eigensystem_value_inner, matrix_2x2_eigensystem_vjp_inner,
    matrix_2x2_eigenvalues_jvp_inner, matrix_2x2_eigenvalues_sum_gradient_inner,
    matrix_2x2_eigenvalues_value_inner, matrix_2x2_eigenvalues_vjp_inner,
    matrix_2x2_inverse_jvp_inner, matrix_2x2_inverse_sum_gradient_inner,
    matrix_2x2_inverse_value_inner, matrix_2x2_inverse_vjp_inner, matrix_2x2_solve_jvp_inner,
    matrix_2x2_solve_sum_gradient_inner, matrix_2x2_solve_value_inner, matrix_2x2_solve_vjp_inner,
    symmetric_2x2_cholesky_jvp_inner, symmetric_2x2_cholesky_sum_gradient_inner,
    symmetric_2x2_cholesky_value_inner, symmetric_2x2_cholesky_vjp_inner,
    symmetric_2x2_eigenvalues_jvp_inner, symmetric_2x2_eigenvalues_sum_gradient_inner,
    symmetric_2x2_eigenvalues_value_inner, symmetric_2x2_eigenvalues_vjp_inner,
};

fn checked_vector<const N: usize>(
    vector: &[f64],
    label: &str,
    primitive: &str,
) -> Result<[f64; N], String> {
    if vector.len() != N {
        return Err(format!("{primitive} requires {N} {label} value(s)"));
    }
    let mut checked = [0.0; N];
    for (index, value) in vector.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is not finite ({value})"));
        }
        checked[index] = *value;
    }
    Ok(checked)
}
