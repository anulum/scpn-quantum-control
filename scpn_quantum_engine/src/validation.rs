// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — FFI Boundary Validation Utilities

//! Validation helpers for PyO3 FFI boundary.
//!
//! Every exported function should validate inputs at the boundary
//! before passing to pure Rust internals. These utilities provide
//! consistent error messages and prevent silent clamping.
//!
//! Core functions return `Result<(), String>` for testability.
//! PyO3 wrappers convert to `PyResult<()>` at call site via `map_err`.

use ndarray::{ArrayView, Dimension};
use numpy::{Element, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Convert a validation result to PyResult.
#[inline]
pub fn to_pyresult(r: Result<(), String>) -> PyResult<()> {
    r.map_err(PyValueError::new_err)
}

/// Validate that a float slice contains no NaN or Inf values.
pub fn check_finite(arr: &[f64], name: &str) -> Result<(), String> {
    for (i, &v) in arr.iter().enumerate() {
        if !v.is_finite() {
            return Err(format!("{name}[{i}] is not finite ({v})"));
        }
    }
    Ok(())
}

/// Validate that a value is strictly positive (and finite).
///
/// Written as a positive predicate so NaN fails CLOSED: the old
/// `val <= 0.0` rejection let NaN through (every comparison with NaN is
/// false), and a NaN `dt`/`k_base`/`alpha` would poison whole kernels.
pub fn check_positive(val: f64, name: &str) -> Result<(), String> {
    if !(val.is_finite() && val > 0.0) {
        return Err(format!("{name} must be positive and finite, got {val}"));
    }
    Ok(())
}

/// Validate that a value is in the range [lo, hi].
///
/// Written as a positive predicate so a NaN value (or NaN bounds) fails
/// CLOSED instead of slipping through the negated comparison.
pub fn check_range(val: f64, lo: f64, hi: f64, name: &str) -> Result<(), String> {
    if !(val >= lo && val <= hi) {
        return Err(format!("{name} must be in [{lo}, {hi}], got {val}"));
    }
    Ok(())
}

/// Validate that n > 0 for matrix/qubit count.
pub fn check_n(n: usize, name: &str) -> Result<(), String> {
    if n == 0 {
        return Err(format!("{name} must be > 0"));
    }
    Ok(())
}

/// Validate flat array length matches expected n*n.
///
/// Fail-closed on `n * n` overflow: an `n` too large to square can never
/// describe a real matrix, so it is a mismatch, not a panic.
pub fn check_flat_square(arr: &[f64], n: usize, name: &str) -> Result<(), String> {
    match n.checked_mul(n) {
        Some(expected) if arr.len() == expected => Ok(()),
        Some(expected) => Err(format!("{name} length {} != {n}² = {expected}", arr.len())),
        None => Err(format!("{name}: n = {n} overflows n² on this platform")),
    }
}

/// Validate statevector length is 2^n.
///
/// Fail-closed on `2^n` overflow: for `n` at or beyond the pointer width the
/// shift would wrap (release) or panic (debug), so it is a mismatch instead —
/// no admissible statevector has that many amplitudes.
pub fn check_statevec_len(len: usize, n: usize, name: &str) -> Result<(), String> {
    if n >= usize::BITS as usize {
        return Err(format!("{name}: 2^{n} overflows usize on this platform"));
    }
    let expected = 1usize << n;
    if len != expected {
        return Err(format!("{name} length {len} != 2^{n} = {expected}"));
    }
    Ok(())
}

/// Validate that an array's length matches the dimension established elsewhere.
///
/// The FEP exports take their dimension from `mu` and then index every other
/// argument with it. Without this check an undersized argument reaches an
/// ndarray bounds panic, which crosses PyO3 as `PanicException` — a
/// `BaseException` that ordinary Python error handling does not catch.
pub fn check_vector_len(len: usize, expected: usize, name: &str) -> Result<(), String> {
    if len != expected {
        return Err(format!("{name} length {len} != {expected}"));
    }
    Ok(())
}

/// Validate that a matrix is exactly `n` by `n`.
///
/// Both dimensions are checked. A matrix that is merely large enough to index
/// — a 2x3 read as 2x2 — silently contributes a sub-block to the result, which
/// is worse than a panic because nothing reports it.
pub fn check_square_matrix(rows: usize, cols: usize, n: usize, name: &str) -> Result<(), String> {
    if rows != n || cols != n {
        return Err(format!("{name} shape {rows}x{cols} != {n}x{n}"));
    }
    Ok(())
}

/// Validate that every element of an array view is finite.
///
/// Works on any dimensionality and on non-contiguous views, so a strided
/// NumPy slice is checked rather than rejected. The reported index is the
/// position in logical iteration order.
pub fn check_finite_array<D: Dimension>(
    array: &ArrayView<'_, f64, D>,
    name: &str,
) -> Result<(), String> {
    for (index, &value) in array.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{name}[{index}] is not finite ({value})"));
        }
    }
    Ok(())
}

/// Validate that a scalar parameter is finite.
///
/// Separate from [`check_positive`] because some parameters, such as a ridge,
/// admit zero and negative values but never NaN or infinity.
pub fn check_finite_scalar(value: f64, name: &str) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{name} must be finite, got {value}"));
    }
    Ok(())
}

/// Validate finite square shape and symmetry within absolute tolerance `atol`.
///
/// A Cholesky factorisation reads only one triangle, so an asymmetric matrix
/// silently yields the determinant of its symmetrised lower triangle instead of
/// an error. Callers using relative symmetry must scale `atol` themselves.
/// Empty square matrices are admitted; dimension requirements belong to callers.
///
/// # Errors
/// Returns an error for rectangular shape, non-finite entries, non-finite or
/// negative tolerance, or an off-diagonal difference exceeding the tolerance.
pub fn check_symmetric(
    matrix: &ArrayView<'_, f64, ndarray::Ix2>,
    atol: f64,
    name: &str,
) -> Result<(), String> {
    let n = matrix.nrows();
    check_square_matrix(n, matrix.ncols(), n, name)?;
    if !(atol.is_finite() && atol >= 0.0) {
        return Err(format!(
            "{name} symmetry tolerance must be finite and non-negative, got {atol}"
        ));
    }
    check_finite_array(matrix, name)?;
    for i in 0..n {
        for j in 0..i {
            let difference = (matrix[[i, j]] - matrix[[j, i]]).abs();
            // Positive predicate, as in `check_positive`: a NaN difference is
            // not finite, so a matrix containing NaN fails CLOSED here rather
            // than slipping through a negated comparison.
            if !(difference.is_finite() && difference <= atol) {
                return Err(format!(
                    "{name} must be symmetric within {atol}: [{i},{j}] and [{j},{i}] differ by {difference}"
                ));
            }
        }
    }
    Ok(())
}

/// Validate domain range indices.
pub fn check_domain_range(start: usize, end: usize, n: usize, name: &str) -> Result<(), String> {
    if start >= n {
        return Err(format!("{name} start ({start}) >= matrix size ({n})"));
    }
    if end >= n {
        return Err(format!("{name} end ({end}) >= matrix size ({n})"));
    }
    if start > end {
        return Err(format!("{name} start ({start}) > end ({end})"));
    }
    Ok(())
}

// Convenience wrappers that return PyResult directly. Each defers to the
// `check_` function of the same name, which carries the rule itself; these
// exist so a PyO3 entry point can use `?` without converting at every call.
/// [`check_finite`] as a `PyResult`.
pub fn validate_finite(arr: &[f64], name: &str) -> PyResult<()> {
    to_pyresult(check_finite(arr, name))
}
/// [`check_positive`] as a `PyResult`.
pub fn validate_positive(val: f64, name: &str) -> PyResult<()> {
    to_pyresult(check_positive(val, name))
}
/// [`check_range`] as a `PyResult`.
pub fn validate_range(val: f64, lo: f64, hi: f64, name: &str) -> PyResult<()> {
    to_pyresult(check_range(val, lo, hi, name))
}
/// [`check_n`] as a `PyResult`.
pub fn validate_n(n: usize, name: &str) -> PyResult<()> {
    to_pyresult(check_n(n, name))
}
/// [`check_flat_square`] as a `PyResult`.
pub fn validate_flat_square(arr: &[f64], n: usize, name: &str) -> PyResult<()> {
    to_pyresult(check_flat_square(arr, n, name))
}
/// [`check_vector_len`] as a `PyResult`.
pub fn validate_vector_len(len: usize, expected: usize, name: &str) -> PyResult<()> {
    to_pyresult(check_vector_len(len, expected, name))
}
/// [`check_square_matrix`] as a `PyResult`.
pub fn validate_square_matrix(rows: usize, cols: usize, n: usize, name: &str) -> PyResult<()> {
    to_pyresult(check_square_matrix(rows, cols, n, name))
}
/// [`check_finite_array`] as a `PyResult`.
pub fn validate_finite_array<D: Dimension>(
    array: &ArrayView<'_, f64, D>,
    name: &str,
) -> PyResult<()> {
    to_pyresult(check_finite_array(array, name))
}
/// [`check_finite_scalar`] as a `PyResult`.
pub fn validate_finite_scalar(value: f64, name: &str) -> PyResult<()> {
    to_pyresult(check_finite_scalar(value, name))
}
/// [`check_symmetric`] as a `PyResult`.
pub fn validate_symmetric(
    matrix: &ArrayView<'_, f64, ndarray::Ix2>,
    atol: f64,
    name: &str,
) -> PyResult<()> {
    to_pyresult(check_symmetric(matrix, atol, name))
}
/// [`check_domain_range`] as a `PyResult`.
pub fn validate_domain_range(start: usize, end: usize, n: usize, name: &str) -> PyResult<()> {
    to_pyresult(check_domain_range(start, end, n, name))
}
/// Borrow a NumPy array as a slice, requiring C-contiguous storage.
///
/// Unlike the other wrappers this one has no `check_` counterpart: it is the
/// point where a Python buffer becomes a Rust slice, and a non-contiguous
/// array is rejected rather than copied or strided over.
pub fn validate_contiguous_slice<'a, T: Element>(
    arr: &'a PyReadonlyArray1<'_, T>,
    name: &str,
) -> PyResult<&'a [T]> {
    arr.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a C-contiguous NumPy array")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_finite_ok() {
        assert!(check_finite(&[1.0, 2.0, 3.0], "test").is_ok());
    }

    #[test]
    fn test_check_finite_nan() {
        let err = check_finite(&[1.0, f64::NAN], "arr").unwrap_err();
        assert!(err.contains("not finite"));
    }

    #[test]
    fn test_check_finite_inf() {
        let err = check_finite(&[f64::INFINITY], "arr").unwrap_err();
        assert!(err.contains("not finite"));
    }

    #[test]
    fn test_check_positive_ok() {
        assert!(check_positive(1.0, "x").is_ok());
    }

    #[test]
    fn test_check_positive_zero() {
        assert!(check_positive(0.0, "x").is_err());
    }

    #[test]
    fn test_check_positive_negative() {
        assert!(check_positive(-1.0, "x").is_err());
    }

    #[test]
    fn test_check_positive_rejects_nan_and_inf() {
        // Fuzz-found (knm_validators): NaN passed the old `val <= 0.0`
        // rejection because every NaN comparison is false — fail-open.
        assert!(check_positive(f64::NAN, "x").is_err());
        assert!(check_positive(f64::INFINITY, "x").is_err());
        assert!(check_positive(f64::NEG_INFINITY, "x").is_err());
    }

    #[test]
    fn test_check_range_rejects_nan_value_and_bounds() {
        assert!(check_range(f64::NAN, 0.0, 1.0, "p").is_err());
        assert!(check_range(0.5, f64::NAN, 1.0, "p").is_err());
        assert!(check_range(0.5, 0.0, f64::NAN, "p").is_err());
    }

    #[test]
    fn test_check_range_ok() {
        assert!(check_range(0.5, 0.0, 1.0, "p").is_ok());
    }

    #[test]
    fn test_check_range_below() {
        assert!(check_range(-0.1, 0.0, 1.0, "p").is_err());
    }

    #[test]
    fn test_check_n_zero() {
        assert!(check_n(0, "n").is_err());
    }

    #[test]
    fn test_check_flat_square_ok() {
        assert!(check_flat_square(&[0.0; 9], 3, "K").is_ok());
    }

    #[test]
    fn test_check_flat_square_wrong() {
        assert!(check_flat_square(&[0.0; 8], 3, "K").is_err());
    }

    #[test]
    fn test_check_flat_square_overflowing_n_fails_closed() {
        // n² overflows usize: must be a mismatch error, never a panic.
        let err = check_flat_square(&[0.0; 4], usize::MAX, "K").unwrap_err();
        assert!(err.contains("overflows"));
    }

    #[test]
    fn test_check_statevec_ok() {
        assert!(check_statevec_len(4, 2, "psi").is_ok());
    }

    #[test]
    fn test_check_statevec_wrong() {
        assert!(check_statevec_len(5, 2, "psi").is_err());
    }

    #[test]
    fn test_check_statevec_overflowing_n_fails_closed() {
        // 2^n beyond the pointer width: must be an error, never a shift panic
        // (debug) or a wrapped, silently wrong expectation (release).
        let err = check_statevec_len(1, usize::BITS as usize, "psi").unwrap_err();
        assert!(err.contains("overflows"));
        assert!(check_statevec_len(usize::MAX, 10_000, "psi").is_err());
    }

    #[test]
    fn test_check_vector_len_matches_and_mismatches() {
        assert!(check_vector_len(4, 4, "x_observed").is_ok());
        let err = check_vector_len(3, 4, "x_observed").unwrap_err();
        assert_eq!(err, "x_observed length 3 != 4");
    }

    #[test]
    fn test_check_square_matrix_requires_both_dimensions() {
        assert!(check_square_matrix(2, 2, 2, "k").is_ok());
        // A 2x3 read as 2x2 would index without panicking and quietly use a
        // sub-block, so the column count must be checked too.
        assert_eq!(
            check_square_matrix(2, 3, 2, "k").unwrap_err(),
            "k shape 2x3 != 2x2"
        );
        assert_eq!(
            check_square_matrix(3, 2, 2, "k").unwrap_err(),
            "k shape 3x2 != 2x2"
        );
    }

    #[test]
    fn test_check_finite_array_reports_the_offending_index() {
        let ok = ndarray::Array2::<f64>::zeros((2, 2));
        assert!(check_finite_array(&ok.view(), "k").is_ok());

        let mut bad = ndarray::Array2::<f64>::zeros((2, 2));
        bad[[1, 0]] = f64::NAN;
        let err = check_finite_array(&bad.view(), "k").unwrap_err();
        assert!(err.starts_with("k[2] is not finite"), "got {err}");

        let mut infinite = ndarray::Array1::<f64>::zeros(3);
        infinite[2] = f64::NEG_INFINITY;
        assert!(check_finite_array(&infinite.view(), "mu").is_err());
    }

    #[test]
    fn test_check_finite_array_reads_non_contiguous_views() {
        // A strided NumPy slice arrives as a non-contiguous view. It must be
        // checked through its strides, not rejected and not skipped.
        let dense = ndarray::Array1::from_vec(vec![1.0, f64::NAN, 2.0, 3.0]);
        let strided = dense.slice(ndarray::s![..;2]);
        assert!(check_finite_array(&strided, "mu").is_ok());
        let other = dense.slice(ndarray::s![1..;2]);
        assert!(check_finite_array(&other, "mu").is_err());
    }

    #[test]
    fn test_check_finite_scalar() {
        assert!(check_finite_scalar(0.0, "ridge").is_ok());
        assert!(check_finite_scalar(-1.0, "ridge").is_ok());
        assert_eq!(
            check_finite_scalar(f64::NAN, "ridge").unwrap_err(),
            "ridge must be finite, got NaN"
        );
        assert!(check_finite_scalar(f64::INFINITY, "ridge").is_err());
    }

    #[test]
    fn test_check_symmetric_accepts_rounding_and_rejects_real_asymmetry() {
        let symmetric = ndarray::arr2(&[[2.0, 1.0], [1.0, 3.0]]);
        assert!(check_symmetric(&symmetric.view(), 1e-10, "k").is_ok());

        let rounded = ndarray::arr2(&[[2.0, 1.0], [1.0 + 1e-12, 3.0]]);
        assert!(check_symmetric(&rounded.view(), 1e-10, "k").is_ok());

        let asymmetric = ndarray::arr2(&[[2.0, 1.0], [0.0, 3.0]]);
        let err = check_symmetric(&asymmetric.view(), 1e-10, "k").unwrap_err();
        assert!(err.contains("must be symmetric within"), "got {err}");
        assert!(err.contains("[1,0] and [0,1]"), "got {err}");
    }

    #[test]
    fn test_check_symmetric_fails_closed_on_nan() {
        // Written as a positive predicate: a NaN difference must reject, not
        // slip through a negated comparison.
        let with_nan = ndarray::arr2(&[[2.0, f64::NAN], [1.0, 3.0]]);
        assert!(check_symmetric(&with_nan.view(), 1e-10, "k").is_err());
    }

    /// Rectangular input returns an error before any transposed indexing.
    #[test]
    fn test_check_symmetric_rejects_rectangular_shape_without_panicking() {
        for shape in [(2, 1), (1, 2), (0, 1), (1, 0)] {
            let matrix = ndarray::Array2::<f64>::zeros(shape);
            assert!(check_symmetric(&matrix.view(), 1e-10, "k").is_err());
        }
        let empty = ndarray::Array2::<f64>::zeros((0, 0));
        assert!(check_symmetric(&empty.view(), 0.0, "k").is_ok());
    }

    /// Diagonal entries must be finite even when no off-diagonal pairs exist.
    #[test]
    fn test_check_symmetric_rejects_nonfinite_diagonal() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let matrix = ndarray::arr2(&[[value]]);
            assert!(check_symmetric(&matrix.view(), 1e-10, "k").is_err());
        }
    }

    /// Invalid tolerances fail independently of matrix size; exact symmetry is valid.
    #[test]
    fn test_check_symmetric_rejects_invalid_tolerance() {
        let matrix = ndarray::arr2(&[[1.0]]);
        for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(check_symmetric(&matrix.view(), tolerance, "k").is_err());
        }
        assert!(check_symmetric(&matrix.view(), 0.0, "k").is_ok());
    }

    #[test]
    fn test_check_domain_range_ok() {
        assert!(check_domain_range(0, 3, 16, "domain_a").is_ok());
    }

    #[test]
    fn test_check_domain_range_oob() {
        assert!(check_domain_range(0, 16, 16, "domain_a").is_err());
    }

    #[test]
    fn test_check_domain_range_inverted() {
        assert!(check_domain_range(5, 3, 16, "domain_a").is_err());
    }
}
