// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD spectral linalg replay helpers

//! Bounded spectral linear-algebra replay helpers for Program AD effect IR.
//!
//! Python Program AD emits one `linalg:eigvalsh:<index>` SSA node per
//! eigenvalue. This module owns the Rust-side 2x2 symmetric, distinct-spectrum
//! contract so the main Program AD IR evaluator stays a dispatcher. The helper
//! deliberately fails closed for non-2x2 matrices, non-symmetric inputs,
//! repeated eigenvalues, and malformed output metadata because those cases need
//! broader spectral policy before they can be promoted.

const DISTINCT_EIGENVALUE_TOLERANCE: f64 = 1.0e-10;
const SYMMETRY_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
struct Eigvalsh2x2 {
    output_index: usize,
    diagonal: [f64; 2],
    off_diagonal: f64,
    eigenvalues: [f64; 2],
}

/// Return whether an operation label belongs to bounded `np.linalg.eigvalsh` replay.
pub(crate) fn is_eigvalsh_operation(operation: &str) -> bool {
    operation.starts_with("linalg:eigvalsh:")
}

/// Evaluate one scalar eigenvalue from a row-major 2x2 symmetric Program AD node.
pub(crate) fn eigvalsh_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_eigvalsh_2x2(effect_index, operation, input_values)?;
    Ok(metadata.eigenvalues[metadata.output_index])
}

/// Return local reverse contributions for one scalar 2x2 `eigvalsh` output node.
pub(crate) fn eigvalsh_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} eigvalsh cotangent must be finite"
        ));
    }
    let metadata = parse_eigvalsh_2x2(effect_index, operation, input_values)?;
    let outer = eigenvector_outer(&metadata)?;
    Ok(outer
        .iter()
        .map(|component| output_cotangent * component)
        .collect())
}

fn parse_eigvalsh_2x2(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<Eigvalsh2x2, String> {
    if input_values.len() != 4 {
        return Err(format!(
            "effect {effect_index} eigvalsh Rust replay supports only 2x2 matrices"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} eigvalsh inputs must be finite"
        ));
    }
    let output_index = parse_eigvalsh_index(effect_index, operation)?;
    if output_index >= 2 {
        return Err(format!(
            "effect {effect_index} eigvalsh 2x2 output index must be 0 or 1"
        ));
    }
    let [a, b, c, d] = input_values else {
        return Err(format!(
            "effect {effect_index} eigvalsh Rust replay supports only 2x2 matrices"
        ));
    };
    let scale = input_values
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    if (b - c).abs() > SYMMETRY_TOLERANCE * scale {
        return Err(format!(
            "effect {effect_index} eigvalsh requires a symmetric 2x2 matrix"
        ));
    }
    let off_diagonal = 0.5 * (b + c);
    let diagonal_delta = a - d;
    let gap = (diagonal_delta * diagonal_delta + 4.0 * off_diagonal * off_diagonal).sqrt();
    let center = 0.5 * (a + d);
    let radius = 0.5 * gap;
    let eigenvalues = [center - radius, center + radius];
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} eigvalsh output must be finite"
        ));
    }
    let eigen_scale = eigenvalues
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    if gap <= DISTINCT_EIGENVALUE_TOLERANCE * eigen_scale {
        return Err(format!(
            "effect {effect_index} eigvalsh gradient requires distinct eigenvalues"
        ));
    }
    Ok(Eigvalsh2x2 {
        output_index,
        diagonal: [*a, *d],
        off_diagonal,
        eigenvalues,
    })
}

fn parse_eigvalsh_index(effect_index: usize, operation: &str) -> Result<usize, String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() != 3 || parts[0] != "linalg" || parts[1] != "eigvalsh" {
        return Err(format!(
            "effect {effect_index} eigvalsh operation metadata is malformed"
        ));
    }
    parts[2]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} eigvalsh output index metadata is malformed"))
}

fn eigenvector_outer(metadata: &Eigvalsh2x2) -> Result<[f64; 4], String> {
    let [a, d] = metadata.diagonal;
    if metadata.off_diagonal.abs() <= SYMMETRY_TOLERANCE {
        return Ok(diagonal_eigenvector_outer(a, d, metadata.output_index));
    }
    let lambda = metadata.eigenvalues[metadata.output_index];
    let primary = [metadata.off_diagonal, lambda - a];
    let secondary = [lambda - d, metadata.off_diagonal];
    let raw = if squared_norm(primary) >= squared_norm(secondary) {
        primary
    } else {
        secondary
    };
    let norm = squared_norm(raw).sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return Err("eigvalsh eigenvector normalization must be finite".to_owned());
    }
    let x = raw[0] / norm;
    let y = raw[1] / norm;
    Ok([x * x, x * y, y * x, y * y])
}

fn diagonal_eigenvector_outer(a: f64, d: f64, output_index: usize) -> [f64; 4] {
    let lower_is_first_axis = a <= d;
    if (output_index == 0 && lower_is_first_axis) || (output_index == 1 && !lower_is_first_axis) {
        [1.0, 0.0, 0.0, 0.0]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

fn squared_norm(vector: [f64; 2]) -> f64 {
    vector[0] * vector[0] + vector[1] * vector[1]
}
