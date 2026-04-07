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

use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

/// Convert a validation result to PyResult.
#[inline]
pub fn to_pyresult(r: Result<(), String>) -> PyResult<()> {
    r.map_err(|e| PyValueError::new_err(e))
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

/// Validate that a value is strictly positive.
pub fn check_positive(val: f64, name: &str) -> Result<(), String> {
    if val <= 0.0 {
        return Err(format!("{name} must be positive, got {val}"));
    }
    Ok(())
}

/// Validate that a value is in the range [lo, hi].
pub fn check_range(val: f64, lo: f64, hi: f64, name: &str) -> Result<(), String> {
    if val < lo || val > hi {
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
pub fn check_flat_square(arr: &[f64], n: usize, name: &str) -> Result<(), String> {
    if arr.len() != n * n {
        return Err(format!(
            "{name} length {} != {n}² = {}",
            arr.len(),
            n * n
        ));
    }
    Ok(())
}

/// Validate statevector length is 2^n.
pub fn check_statevec_len(len: usize, n: usize, name: &str) -> Result<(), String> {
    let expected = 1usize << n;
    if len != expected {
        return Err(format!("{name} length {len} != 2^{n} = {expected}"));
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

// Convenience wrappers that return PyResult directly
pub fn validate_finite(arr: &[f64], name: &str) -> PyResult<()> {
    to_pyresult(check_finite(arr, name))
}
pub fn validate_positive(val: f64, name: &str) -> PyResult<()> {
    to_pyresult(check_positive(val, name))
}
pub fn validate_range(val: f64, lo: f64, hi: f64, name: &str) -> PyResult<()> {
    to_pyresult(check_range(val, lo, hi, name))
}
pub fn validate_n(n: usize, name: &str) -> PyResult<()> {
    to_pyresult(check_n(n, name))
}
pub fn validate_flat_square(arr: &[f64], n: usize, name: &str) -> PyResult<()> {
    to_pyresult(check_flat_square(arr, n, name))
}
pub fn validate_domain_range(start: usize, end: usize, n: usize, name: &str) -> PyResult<()> {
    to_pyresult(check_domain_range(start, end, n, name))
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
    fn test_check_statevec_ok() {
        assert!(check_statevec_len(4, 2, "psi").is_ok());
    }

    #[test]
    fn test_check_statevec_wrong() {
        assert!(check_statevec_len(5, 2, "psi").is_err());
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
