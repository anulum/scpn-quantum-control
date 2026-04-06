// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Krylov Complexity (Operator Lanczos)

//! Operator Lanczos algorithm for Krylov complexity b-coefficients.
//!
//! Computes the Lanczos b-coefficients for the Liouvillian L=[H,·] acting on
//! operator space. The growth rate of b_n characterises operator spreading and
//! quantum chaos. For the SCPN XY Hamiltonian, linear growth indicates maximal
//! chaos (Maldacena bound), while bounded growth indicates integrability.
//!
//! Ref: Parker et al., Phys. Rev. X 9, 041017 (2019)

use ndarray::Array2;
use pyo3::prelude::*;

use crate::complex_utils::{c64, cmat_from_flat, hs_inner_real, C64};

/// Operator Lanczos: b-coefficients for Liouvillian L=[H,·] on d×d matrices.
///
/// Avoids Python per-step overhead for the commutator loop (2 matrix multiplies
/// per step). For dim ≤ 256 (8 qubits), ~5–10× faster than numpy.
#[pyfunction]
pub fn lanczos_b_coefficients(
    h_re: numpy::PyReadonlyArray1<'_, f64>,
    h_im: numpy::PyReadonlyArray1<'_, f64>,
    o_re: numpy::PyReadonlyArray1<'_, f64>,
    o_im: numpy::PyReadonlyArray1<'_, f64>,
    dim: usize,
    max_steps: usize,
    tol: f64,
) -> Vec<f64> {
    let h = cmat_from_flat(h_re.as_slice().unwrap(), h_im.as_slice().unwrap(), dim);
    let o_init = cmat_from_flat(o_re.as_slice().unwrap(), o_im.as_slice().unwrap(), dim);

    lanczos_b_inner(&h, &o_init, max_steps, tol)
}

/// Pure Rust Lanczos (no PyO3) for testing and internal use.
pub fn lanczos_b_inner(
    h: &Array2<C64>,
    o_init: &Array2<C64>,
    max_steps: usize,
    tol: f64,
) -> Vec<f64> {
    let dim = h.nrows();
    let norm_0 = hs_inner_real(o_init, o_init).max(0.0).sqrt();
    if norm_0 < tol {
        return vec![0.0];
    }

    let mut o_prev = Array2::<C64>::zeros((dim, dim));
    let mut o_curr = o_init / c64(norm_0, 0.0);
    let mut b_list: Vec<f64> = Vec::with_capacity(max_steps);

    for _ in 0..max_steps {
        // A = [H, O_curr] = H·O − O·H
        let mut a_next = h.dot(&o_curr);
        {
            let oh = o_curr.dot(h);
            a_next -= &oh;
        }

        if let Some(&b_last) = b_list.last() {
            a_next.scaled_add(c64(-b_last, 0.0), &o_prev);
        }

        let a_n = hs_inner_real(&o_curr, &a_next);
        a_next.scaled_add(c64(-a_n, 0.0), &o_curr);

        let b_next = hs_inner_real(&a_next, &a_next).max(0.0).sqrt();
        if b_next < tol {
            break;
        }

        b_list.push(b_next);
        o_prev = o_curr;
        o_curr = a_next / c64(b_next, 0.0);
    }

    b_list
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_lanczos_identity_operator() {
        // [H, I] = 0 for any H → b_list should be empty or [0]
        let dim = 2;
        let h = Array2::from_shape_vec(
            (dim, dim),
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(-1.0, 0.0)],
        )
        .unwrap();
        let id = Array2::from_shape_fn((dim, dim), |(i, j)| {
            if i == j {
                c64(1.0, 0.0)
            } else {
                c64(0.0, 0.0)
            }
        });
        let b = lanczos_b_inner(&h, &id, 10, 1e-10);
        assert!(b.is_empty(), "[H, I] = 0 → no b-coefficients");
    }

    #[test]
    fn test_lanczos_pauli_x_with_z() {
        // H = Z, O = X → [Z,X] = 2iY, [Z,Y] = -2iX → periodic, b_n should be non-zero
        let dim = 2;
        let h = Array2::from_shape_vec(
            (dim, dim),
            vec![c64(1.0, 0.0), c64(0.0, 0.0), c64(0.0, 0.0), c64(-1.0, 0.0)],
        )
        .unwrap();
        let x = Array2::from_shape_vec(
            (dim, dim),
            vec![c64(0.0, 0.0), c64(1.0, 0.0), c64(1.0, 0.0), c64(0.0, 0.0)],
        )
        .unwrap();
        let b = lanczos_b_inner(&h, &x, 10, 1e-10);
        assert!(!b.is_empty(), "non-commuting operators produce b > 0");
        assert!(b[0] > 0.0);
    }

    #[test]
    fn test_lanczos_zero_operator() {
        let dim = 2;
        let h = Array2::from_shape_fn((dim, dim), |(i, j)| {
            if i == j {
                c64(1.0, 0.0)
            } else {
                c64(0.0, 0.0)
            }
        });
        let zero = Array2::<C64>::zeros((dim, dim));
        let b = lanczos_b_inner(&h, &zero, 10, 1e-10);
        assert_eq!(b, vec![0.0], "zero operator → [0.0]");
    }
}
