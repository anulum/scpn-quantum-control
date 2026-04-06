// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Complex-Valued Linear Algebra Utilities

//! Shared complex-valued linear algebra helpers for Krylov, OTOC, and
//! future analysis modules. All operate on ndarray complex matrices.

use ndarray::{Array1, Array2};
use num_complex::Complex;

pub type C64 = Complex<f64>;

/// Construct a complex number from real and imaginary parts.
#[inline]
pub fn c64(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// Build a complex matrix from flat real and imaginary arrays (row-major).
pub fn cmat_from_flat(re: &[f64], im: &[f64], dim: usize) -> Array2<C64> {
    Array2::from_shape_fn((dim, dim), |(i, j)| c64(re[i * dim + j], im[i * dim + j]))
}

/// Build a complex vector from real and imaginary arrays.
pub fn cvec_from_parts(re: &[f64], im: &[f64]) -> Array1<C64> {
    Array1::from_shape_fn(re.len(), |i| c64(re[i], im[i]))
}

/// Conjugate transpose A†.
pub fn conj_transpose(a: &Array2<C64>) -> Array2<C64> {
    let (m, n) = a.dim();
    Array2::from_shape_fn((n, m), |(i, j)| a[[j, i]].conj())
}

/// Re(Tr(A†B) / d) — Hilbert-Schmidt inner product normalised by dimension.
pub fn hs_inner_real(a: &Array2<C64>, b: &Array2<C64>) -> f64 {
    let d = a.nrows() as f64;
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| (av.conj() * bv).re)
        .sum::<f64>()
        / d
}

/// A† × x without materialising A†.
pub fn ct_matvec(a: &Array2<C64>, x: &Array1<C64>) -> Array1<C64> {
    let (m, n) = a.dim();
    Array1::from_shape_fn(n, |col| {
        (0..m)
            .map(|row| a[[row, col]].conj() * x[row])
            .sum::<C64>()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_c64_constructor() {
        let z = c64(3.0, 4.0);
        assert!((z.re - 3.0).abs() < 1e-12);
        assert!((z.im - 4.0).abs() < 1e-12);
        assert!((z.norm() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_conj_transpose_hermitian() {
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                c64(1.0, 0.0),
                c64(2.0, 3.0),
                c64(2.0, -3.0),
                c64(4.0, 0.0),
            ],
        )
        .unwrap();
        let ah = conj_transpose(&a);
        for i in 0..2 {
            for j in 0..2 {
                assert!((a[[i, j]] - ah[[i, j]]).norm() < 1e-12);
            }
        }
    }

    #[test]
    fn test_hs_inner_real_identity() {
        let d = 2;
        let id = Array2::from_shape_fn((d, d), |(i, j)| {
            if i == j {
                c64(1.0, 0.0)
            } else {
                c64(0.0, 0.0)
            }
        });
        let ip = hs_inner_real(&id, &id);
        assert!((ip - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_hs_inner_real_orthogonal_paulis() {
        let x = Array2::from_shape_vec(
            (2, 2),
            vec![
                c64(0.0, 0.0),
                c64(1.0, 0.0),
                c64(1.0, 0.0),
                c64(0.0, 0.0),
            ],
        )
        .unwrap();
        let z = Array2::from_shape_vec(
            (2, 2),
            vec![
                c64(1.0, 0.0),
                c64(0.0, 0.0),
                c64(0.0, 0.0),
                c64(-1.0, 0.0),
            ],
        )
        .unwrap();
        let ip = hs_inner_real(&x, &z);
        assert!(ip.abs() < 1e-12, "Tr(X†Z)/d = 0");
    }

    #[test]
    fn test_ct_matvec_identity() {
        let id = Array2::from_shape_fn((2, 2), |(i, j)| {
            if i == j {
                c64(1.0, 0.0)
            } else {
                c64(0.0, 0.0)
            }
        });
        let x = Array1::from_vec(vec![c64(1.0, 2.0), c64(3.0, 4.0)]);
        let result = ct_matvec(&id, &x);
        // I† = I, so result should equal x
        for i in 0..2 {
            assert!((result[i] - x[i]).norm() < 1e-12);
        }
    }

    #[test]
    fn test_cmat_from_flat_roundtrip() {
        let re = vec![1.0, 2.0, 3.0, 4.0];
        let im = vec![0.1, 0.2, 0.3, 0.4];
        let m = cmat_from_flat(&re, &im, 2);
        assert!((m[[0, 0]] - c64(1.0, 0.1)).norm() < 1e-12);
        assert!((m[[1, 1]] - c64(4.0, 0.4)).norm() < 1e-12);
    }
}
