// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Kuramoto shared input validators
//! Shared phase-vector validators for the Kuramoto compute kernels.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::validation::validate_contiguous_slice;

pub(crate) fn validate_finite_slice(values: &[f64], name: &str) -> PyResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(())
}

pub(crate) fn validate_phase_vector<'a>(
    values: &'a PyReadonlyArray1<'_, f64>,
    name: &str,
) -> PyResult<&'a [f64]> {
    let slice = validate_contiguous_slice(values, name)?;
    validate_finite_slice(slice, name)?;
    Ok(slice)
}
