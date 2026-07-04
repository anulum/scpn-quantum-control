// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Program AD spectral linalg replay helpers

//! Bounded spectral linear-algebra replay helpers for Program AD effect IR.
//!
//! Python Program AD emits one `linalg:eigvalsh:<index>`,
//! `linalg:eigvals:2x2:<index>`, `linalg:eigh:eigenvalue:2x2:<UPLO>:<index>`,
//! or `linalg:eigh:eigenvector:2x2:<UPLO>:<column>:<row>` SSA node per scalar
//! spectral output. This module owns the Rust-side 2x2 spectral contracts so
//! the main Program AD IR evaluator stays a dispatcher. The helper
//! deliberately fails closed for non-2x2 matrices, non-symmetric Hermitian
//! inputs, zero-offdiagonal `eigh` eigenvector outputs, complex or repeated
//! spectra, and malformed output metadata because those cases need broader
//! spectral policy before they can be promoted.

const DISTINCT_EIGENVALUE_TOLERANCE: f64 = 1.0e-10;
const REAL_SPECTRUM_TOLERANCE: f64 = 1.0e-12;
const SYMMETRY_TOLERANCE: f64 = 1.0e-12;

#[derive(Debug, Clone, PartialEq)]
struct Eigvalsh2x2 {
    output_index: usize,
    diagonal: [f64; 2],
    off_diagonal: f64,
    eigenvalues: [f64; 2],
}

#[derive(Debug, Clone, PartialEq)]
struct Eigvals2x2 {
    output_index: usize,
    sign: f64,
    values: [f64; 4],
    eigenvalues: [f64; 2],
    gap: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum EighOutput {
    Eigenvalue { index: usize },
    Eigenvector { column: usize, row: usize },
}

#[derive(Debug, Clone, PartialEq)]
struct Eigh2x2 {
    output: EighOutput,
    eigenvalues: [f64; 2],
    eigenvectors: [[f64; 2]; 2],
}

/// Return whether an operation label belongs to bounded `np.linalg.eigvalsh` replay.
pub(crate) fn is_eigvalsh_operation(operation: &str) -> bool {
    operation.starts_with("linalg:eigvalsh:")
}

/// Return whether an operation label belongs to bounded `np.linalg.eigvals` replay.
pub(crate) fn is_eigvals_operation(operation: &str) -> bool {
    operation.starts_with("linalg:eigvals:")
}

/// Return whether an operation label belongs to bounded `np.linalg.eigh` replay.
pub(crate) fn is_eigh_operation(operation: &str) -> bool {
    operation.starts_with("linalg:eigh:")
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

/// Evaluate one scalar eigenvalue from a row-major 2x2 real-simple Program AD node.
pub(crate) fn eigvals_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_eigvals_2x2(effect_index, operation, input_values)?;
    Ok(metadata.eigenvalues[metadata.output_index])
}

/// Evaluate one scalar output from a row-major 2x2 symmetric `eigh` Program AD node.
pub(crate) fn eigh_output_value(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<f64, String> {
    let metadata = parse_eigh_2x2(effect_index, operation, input_values)?;
    match metadata.output {
        EighOutput::Eigenvalue { index } => Ok(metadata.eigenvalues[index]),
        EighOutput::Eigenvector { column, row } => Ok(metadata.eigenvectors[row][column]),
    }
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

/// Return local reverse contributions for one scalar 2x2 `eigvals` output node.
pub(crate) fn eigvals_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} eigvals cotangent must be finite"
        ));
    }
    let metadata = parse_eigvals_2x2(effect_index, operation, input_values)?;
    let [a, b, c, d] = metadata.values;
    let diagonal_delta = a - d;
    let sign = metadata.sign;
    Ok(vec![
        output_cotangent * (0.5 + sign * diagonal_delta / (2.0 * metadata.gap)),
        output_cotangent * sign * c / metadata.gap,
        output_cotangent * sign * b / metadata.gap,
        output_cotangent * (0.5 - sign * diagonal_delta / (2.0 * metadata.gap)),
    ])
}

/// Return local reverse contributions for one scalar 2x2 `eigh` output node.
pub(crate) fn eigh_output_cotangent(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
    output_cotangent: f64,
) -> Result<Vec<f64>, String> {
    if !output_cotangent.is_finite() {
        return Err(format!(
            "effect {effect_index} eigh cotangent must be finite"
        ));
    }
    let metadata = parse_eigh_2x2(effect_index, operation, input_values)?;
    match metadata.output {
        EighOutput::Eigenvalue { index } => {
            let vector = [
                metadata.eigenvectors[0][index],
                metadata.eigenvectors[1][index],
            ];
            Ok(vector_outer(vector)
                .iter()
                .map(|component| output_cotangent * component)
                .collect())
        }
        EighOutput::Eigenvector { column, row } => {
            let other = 1 - column;
            let lambda_delta = metadata.eigenvalues[column] - metadata.eigenvalues[other];
            let other_vector = [
                metadata.eigenvectors[0][other],
                metadata.eigenvectors[1][other],
            ];
            let column_vector = [
                metadata.eigenvectors[0][column],
                metadata.eigenvectors[1][column],
            ];
            let scale = output_cotangent * other_vector[row] / lambda_delta;
            let raw = [
                scale * other_vector[0] * column_vector[0],
                scale * other_vector[0] * column_vector[1],
                scale * other_vector[1] * column_vector[0],
                scale * other_vector[1] * column_vector[1],
            ];
            Ok(vec![
                raw[0],
                0.5 * (raw[1] + raw[2]),
                0.5 * (raw[2] + raw[1]),
                raw[3],
            ])
        }
    }
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

fn parse_eigvals_2x2(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<Eigvals2x2, String> {
    if input_values.len() != 4 {
        return Err(format!(
            "effect {effect_index} eigvals Rust replay supports only 2x2 matrices"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} eigvals inputs must be finite"
        ));
    }
    let output_index = parse_eigvals_index(effect_index, operation)?;
    if output_index >= 2 {
        return Err(format!(
            "effect {effect_index} eigvals 2x2 output index must be 0 or 1"
        ));
    }
    let [a, b, c, d] = input_values else {
        return Err(format!(
            "effect {effect_index} eigvals Rust replay supports only 2x2 matrices"
        ));
    };
    let values = [*a, *b, *c, *d];
    let scale = values
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    let diagonal_delta = a - d;
    let discriminant = diagonal_delta * diagonal_delta + 4.0 * b * c;
    let discriminant_tolerance = REAL_SPECTRUM_TOLERANCE * scale * scale;
    if discriminant < -discriminant_tolerance {
        return Err(format!(
            "effect {effect_index} eigvals requires real distinct eigenvalues"
        ));
    }
    let gap = discriminant.max(0.0).sqrt();
    let center = 0.5 * (a + d);
    let lower = center - 0.5 * gap;
    let upper = center + 0.5 * gap;
    let eigen_scale = 1.0_f64.max(lower.abs()).max(upper.abs());
    if gap <= DISTINCT_EIGENVALUE_TOLERANCE * eigen_scale {
        return Err(format!(
            "effect {effect_index} eigvals requires real distinct eigenvalues"
        ));
    }
    let ordered_signs = eigvals_2x2_order_signs(*a, *b, *c, *d, scale);
    let sign = ordered_signs[output_index];
    let eigenvalues = ordered_signs.map(
        |ordered_sign| {
            if ordered_sign < 0.0 {
                lower
            } else {
                upper
            }
        },
    );
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "effect {effect_index} eigvals output must be finite"
        ));
    }
    Ok(Eigvals2x2 {
        output_index,
        sign,
        values,
        eigenvalues,
        gap,
    })
}

fn parse_eigh_2x2(
    effect_index: usize,
    operation: &str,
    input_values: &[f64],
) -> Result<Eigh2x2, String> {
    if input_values.len() != 4 {
        return Err(format!(
            "effect {effect_index} eigh Rust replay supports only 2x2 matrices"
        ));
    }
    if input_values.iter().any(|value| !value.is_finite()) {
        return Err(format!("effect {effect_index} eigh inputs must be finite"));
    }
    let output = parse_eigh_output(effect_index, operation)?;
    let [a, b, c, d] = input_values else {
        return Err(format!(
            "effect {effect_index} eigh Rust replay supports only 2x2 matrices"
        ));
    };
    let values = [*a, *b, *c, *d];
    let scale = values
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    if (b - c).abs() > SYMMETRY_TOLERANCE * scale {
        return Err(format!(
            "effect {effect_index} eigh requires a symmetric 2x2 matrix"
        ));
    }
    let off_diagonal = 0.5 * (b + c);
    if matches!(output, EighOutput::Eigenvector { .. })
        && off_diagonal.abs() <= SYMMETRY_TOLERANCE * scale
    {
        return Err(format!(
            "effect {effect_index} eigh eigenvector gradient requires nonzero off-diagonal entries"
        ));
    }
    let diagonal_delta = a - d;
    let gap = (diagonal_delta * diagonal_delta + 4.0 * off_diagonal * off_diagonal).sqrt();
    let center = 0.5 * (a + d);
    let radius = 0.5 * gap;
    let eigenvalues = [center - radius, center + radius];
    if eigenvalues.iter().any(|value| !value.is_finite()) {
        return Err(format!("effect {effect_index} eigh output must be finite"));
    }
    let eigen_scale = eigenvalues
        .iter()
        .fold(1.0_f64, |current, value| current.max(value.abs()));
    if gap <= DISTINCT_EIGENVALUE_TOLERANCE * eigen_scale {
        return Err(format!(
            "effect {effect_index} eigh gradient requires distinct eigenvalues"
        ));
    }
    let eigenvectors = eigh_eigenvectors_2x2(*a, off_diagonal, *d, eigenvalues)?;
    Ok(Eigh2x2 {
        output,
        eigenvalues,
        eigenvectors,
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

fn parse_eigvals_index(effect_index: usize, operation: &str) -> Result<usize, String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() != 4 || parts[0] != "linalg" || parts[1] != "eigvals" || parts[2] != "2x2" {
        return Err(format!(
            "effect {effect_index} eigvals operation metadata is malformed"
        ));
    }
    parts[3]
        .parse::<usize>()
        .map_err(|_| format!("effect {effect_index} eigvals output index metadata is malformed"))
}

fn parse_eigh_output(effect_index: usize, operation: &str) -> Result<EighOutput, String> {
    let parts = operation.split(':').collect::<Vec<&str>>();
    if parts.len() < 6 || parts[0] != "linalg" || parts[1] != "eigh" {
        return Err(format!(
            "effect {effect_index} eigh operation metadata is malformed"
        ));
    }
    if parts[3] != "2x2" {
        return Err(format!(
            "effect {effect_index} eigh Rust replay supports only 2x2 matrices"
        ));
    }
    if parts[4] != "L" && parts[4] != "U" {
        return Err(format!(
            "effect {effect_index} eigh UPLO metadata must be L or U"
        ));
    }
    match parts[2] {
        "eigenvalue" if parts.len() == 6 => {
            let index = parts[5].parse::<usize>().map_err(|_| {
                format!("effect {effect_index} eigh eigenvalue index metadata is malformed")
            })?;
            if index >= 2 {
                return Err(format!(
                    "effect {effect_index} eigh eigenvalue index must be 0 or 1"
                ));
            }
            Ok(EighOutput::Eigenvalue { index })
        }
        "eigenvector" if parts.len() == 7 => {
            let column = parts[5].parse::<usize>().map_err(|_| {
                format!("effect {effect_index} eigh eigenvector column metadata is malformed")
            })?;
            let row = parts[6].parse::<usize>().map_err(|_| {
                format!("effect {effect_index} eigh eigenvector row metadata is malformed")
            })?;
            if column >= 2 || row >= 2 {
                return Err(format!(
                    "effect {effect_index} eigh eigenvector column and row must be 0 or 1"
                ));
            }
            Ok(EighOutput::Eigenvector { column, row })
        }
        _ => Err(format!(
            "effect {effect_index} eigh operation metadata is malformed"
        )),
    }
}

fn eigvals_2x2_order_signs(a: f64, b: f64, c: f64, d: f64, scale: f64) -> [f64; 2] {
    let off_diagonal_tolerance = SYMMETRY_TOLERANCE * scale;
    let b_is_zero = b.abs() <= off_diagonal_tolerance;
    let c_is_zero = c.abs() <= off_diagonal_tolerance;
    if b_is_zero && !c_is_zero {
        if d <= a {
            [-1.0, 1.0]
        } else {
            [1.0, -1.0]
        }
    } else if a < d {
        [-1.0, 1.0]
    } else {
        [1.0, -1.0]
    }
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

fn eigh_eigenvectors_2x2(
    a: f64,
    b: f64,
    d: f64,
    eigenvalues: [f64; 2],
) -> Result<[[f64; 2]; 2], String> {
    let scale = 1.0_f64.max(a.abs()).max(b.abs()).max(d.abs());
    if b.abs() <= SYMMETRY_TOLERANCE * scale {
        return Ok(diagonal_eigenvectors(a, d));
    }
    let raw0 = if b > 0.0 && a <= d {
        [-b, a - eigenvalues[0]]
    } else {
        [b, eigenvalues[0] - a]
    };
    let raw1 = if b > 0.0 && a > d {
        [-b, a - eigenvalues[1]]
    } else {
        [b, eigenvalues[1] - a]
    };
    let column0 = normalise_eigh_vector(raw0)?;
    let column1 = normalise_eigh_vector(raw1)?;
    Ok([[column0[0], column1[0]], [column0[1], column1[1]]])
}

fn diagonal_eigenvectors(a: f64, d: f64) -> [[f64; 2]; 2] {
    if a <= d {
        [[1.0, 0.0], [0.0, 1.0]]
    } else {
        [[0.0, 1.0], [1.0, 0.0]]
    }
}

fn normalise_eigh_vector(raw: [f64; 2]) -> Result<[f64; 2], String> {
    let norm = squared_norm(raw).sqrt();
    if norm <= 0.0 || !norm.is_finite() {
        return Err("eigh eigenvector normalization must be finite".to_owned());
    }
    Ok([raw[0] / norm, raw[1] / norm])
}

fn vector_outer(vector: [f64; 2]) -> [f64; 4] {
    [
        vector[0] * vector[0],
        vector[0] * vector[1],
        vector[1] * vector[0],
        vector[1] * vector[1],
    ]
}

fn squared_norm(vector: [f64; 2]) -> f64 {
    vector[0] * vector[0] + vector[1] * vector[1]
}
