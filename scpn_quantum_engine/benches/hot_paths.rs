// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// scpn-quantum-engine — Criterion benchmarks for hot paths
//
// Runs native Rust benchmarks for the compute-hot inner helpers
// (not the `#[pyfunction]` wrappers — those need a Python
// interpreter and are covered by
// `tests/test_rust_path_benchmarks.py` on the Python side).
//
// Usage:
//     cargo bench --bench hot_paths
//     cargo bench --bench hot_paths -- --save-baseline main
//     cargo bench --bench hot_paths -- --baseline main
//
// CI saves the baseline on push to `main` and compares PR runs
// against it; critcmp produces a human-readable diff.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

use scpn_quantum_engine::biological_qec::biological_decode_inner;
use scpn_quantum_engine::compiler_ad::{
    matrix_2x2_determinant_jvp_inner, matrix_2x2_determinant_value_inner,
    matrix_2x2_determinant_vjp_inner, matrix_2x2_eigensystem_jvp_inner,
    matrix_2x2_eigensystem_value_inner, matrix_2x2_eigensystem_vjp_inner,
    matrix_2x2_inverse_jvp_inner, matrix_2x2_inverse_value_inner, matrix_2x2_inverse_vjp_inner,
    matrix_frobenius_norm_squared_jvp_inner, matrix_frobenius_norm_squared_value_inner,
    matrix_frobenius_norm_squared_vjp_inner, matrix_quadratic_form_jvp_inner,
    matrix_quadratic_form_value_inner, matrix_quadratic_form_vjp_inner, matrix_trace_jvp_inner,
    matrix_trace_value_inner, matrix_trace_vjp_inner, symmetric_2x2_cholesky_jvp_inner,
    symmetric_2x2_cholesky_value_inner, symmetric_2x2_cholesky_vjp_inner, vector_dot_jvp_inner,
    vector_dot_value_inner, vector_dot_vjp_inner, vector_squared_norm_jvp_inner,
    vector_squared_norm_value_inner, vector_squared_norm_vjp_inner,
};
use scpn_quantum_engine::dla::{commutator_dense, is_independent_fast};
use scpn_quantum_engine::knm::build_knm_inner;
use scpn_quantum_engine::kuramoto::order_parameter_inner;

fn bench_build_knm(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_knm_inner");
    for &n in &[4usize, 8, 16, 32] {
        group.bench_function(format!("n={}", n), |b| {
            b.iter(|| build_knm_inner(black_box(n), black_box(0.45), black_box(0.3)));
        });
    }
    group.finish();
}

fn bench_order_parameter(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_parameter_inner");
    for &n in &[8usize, 16, 32, 64] {
        let theta: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
        group.bench_function(format!("n={}", n), |b| {
            b.iter(|| order_parameter_inner(black_box(&theta)));
        });
    }
    group.finish();
}

fn bench_commutator_dense(c: &mut Criterion) {
    let mut group = c.benchmark_group("commutator_dense");
    for &dim in &[4usize, 8, 16, 32] {
        // Two random symmetric-ish real matrices; content is arbitrary,
        // we only care about the f64 arithmetic cost.
        let a: Vec<f64> = (0..dim * dim).map(|i| ((i as f64) * 0.37).sin()).collect();
        let b: Vec<f64> = (0..dim * dim).map(|i| ((i as f64) * 0.91).cos()).collect();
        group.bench_function(format!("dim={}", dim), |bench| {
            bench.iter(|| commutator_dense(black_box(&a), black_box(&b), black_box(dim)));
        });
    }
    group.finish();
}

fn bench_is_independent_fast(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_independent_fast");
    for &dim in &[16usize, 32, 64] {
        // Build a `basis` of 16 vectors of dimension dim*dim plus a
        // candidate `new_op`; the cost is dominated by the dot-product
        // sweep against basis.
        let basis: Vec<Vec<f64>> = (0..16)
            .map(|k| {
                (0..dim * dim)
                    .map(|i| ((i * (k + 1)) as f64 * 0.17).sin())
                    .collect()
            })
            .collect();
        let new_op: Vec<f64> = (0..dim * dim).map(|i| ((i as f64) * 0.53).cos()).collect();
        group.bench_function(format!("dim={} basis=16", dim), |bench| {
            bench.iter(|| {
                is_independent_fast(black_box(&new_op), black_box(&basis), black_box(1e-9))
            });
        });
    }
    group.finish();
}

fn bench_biological_decode_inner(c: &mut Criterion) {
    let mut group = c.benchmark_group("biological_decode_inner");
    for &n in &[8usize, 12, 16] {
        let mut edge_u = Vec::new();
        let mut edge_v = Vec::new();
        let mut edge_w = Vec::new();

        // Ring
        for i in 0..n {
            let j = (i + 1) % n;
            edge_u.push(i as i64);
            edge_v.push(j as i64);
            edge_w.push(1.0);
        }
        // Chords for alternative shortest paths
        for i in 0..(n / 2) {
            let j = (i + 2) % n;
            edge_u.push(i as i64);
            edge_v.push(j as i64);
            edge_w.push(1.5);
        }

        let mut syndrome = vec![0i8; n];
        syndrome[1] = 1;
        syndrome[n / 2] = 1;
        syndrome[(n / 2) + 1] = 1;
        syndrome[n - 1] = 1;

        group.bench_function(format!("n={}", n), |bench| {
            bench.iter(|| {
                biological_decode_inner(
                    black_box(&edge_u),
                    black_box(&edge_v),
                    black_box(&edge_w),
                    black_box(n),
                    black_box(&syndrome),
                )
            });
        });
    }
    group.finish();
}

fn bench_matrix_2x2_eigensystem_ad(c: &mut Criterion) {
    let values = [2.0, 0.25, 0.75, 1.0];
    let tangent = [0.1, -0.2, 0.4, -0.3];
    let cotangent = [1.25, -0.75, 0.5, -0.25, 0.3, -0.6];
    let mut group = c.benchmark_group("matrix_2x2_eigensystem_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| matrix_2x2_eigensystem_value_inner(black_box(&values)).unwrap());
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| {
            matrix_2x2_eigensystem_jvp_inner(black_box(&values), black_box(&tangent)).unwrap()
        });
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            matrix_2x2_eigensystem_vjp_inner(black_box(&values), black_box(&cotangent)).unwrap()
        });
    });
    group.finish();
}

fn bench_matrix_quadratic_form_ad(c: &mut Criterion) {
    let values = [2.0, -1.0, 0.5, 3.0, 1.5, -2.0];
    let tangent = [0.1, -0.2, 0.3, 0.4, -0.5, 0.25];
    let cotangent = [1.25];
    let mut group = c.benchmark_group("matrix_quadratic_form_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| matrix_quadratic_form_value_inner(black_box(2), black_box(&values)).unwrap());
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| {
            matrix_quadratic_form_jvp_inner(black_box(2), black_box(&values), black_box(&tangent))
                .unwrap()
        });
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            matrix_quadratic_form_vjp_inner(black_box(2), black_box(&values), black_box(&cotangent))
                .unwrap()
        });
    });
    group.finish();
}

fn bench_vector_squared_norm_ad(c: &mut Criterion) {
    let values = [1.5, -2.0, 0.25];
    let tangent = [-0.5, 0.75, 2.0];
    let cotangent = [1.25];
    let mut group = c.benchmark_group("vector_squared_norm_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| vector_squared_norm_value_inner(black_box(3), black_box(&values)).unwrap());
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| {
            vector_squared_norm_jvp_inner(black_box(3), black_box(&values), black_box(&tangent))
                .unwrap()
        });
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            vector_squared_norm_vjp_inner(black_box(3), black_box(&values), black_box(&cotangent))
                .unwrap()
        });
    });
    group.finish();
}

fn bench_vector_dot_ad(c: &mut Criterion) {
    let values = [1.0, 2.0, -3.0, 4.0];
    let tangent = [0.5, -1.0, 2.0, -0.25];
    let cotangent = [1.25];
    let mut group = c.benchmark_group("vector_dot_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| vector_dot_value_inner(black_box(2), black_box(&values)).unwrap());
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| {
            vector_dot_jvp_inner(black_box(2), black_box(&values), black_box(&tangent)).unwrap()
        });
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            vector_dot_vjp_inner(black_box(2), black_box(&values), black_box(&cotangent)).unwrap()
        });
    });
    group.finish();
}

fn bench_matrix_trace_ad(c: &mut Criterion) {
    let values = [2.0, -1.0, 0.5, 3.0];
    let tangent = [0.1, -0.2, 0.3, 0.4];
    let cotangent = [1.25];
    let mut group = c.benchmark_group("matrix_trace_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| matrix_trace_value_inner(black_box(2), black_box(&values)).unwrap());
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| {
            matrix_trace_jvp_inner(black_box(2), black_box(&values), black_box(&tangent)).unwrap()
        });
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            matrix_trace_vjp_inner(black_box(2), black_box(&values), black_box(&cotangent)).unwrap()
        });
    });
    group.finish();
}

fn bench_matrix_frobenius_norm_squared_ad(c: &mut Criterion) {
    let values = [2.0, -1.0, 0.5, 3.0];
    let tangent = [0.1, -0.2, 0.3, 0.4];
    let cotangent = [1.25];
    let mut group = c.benchmark_group("matrix_frobenius_norm_squared_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| {
            matrix_frobenius_norm_squared_value_inner(black_box(2), black_box(&values)).unwrap()
        });
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| {
            matrix_frobenius_norm_squared_jvp_inner(
                black_box(2),
                black_box(&values),
                black_box(&tangent),
            )
            .unwrap()
        });
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            matrix_frobenius_norm_squared_vjp_inner(
                black_box(2),
                black_box(&values),
                black_box(&cotangent),
            )
            .unwrap()
        });
    });
    group.finish();
}

fn bench_matrix_2x2_determinant_ad(c: &mut Criterion) {
    let values = [2.0, -1.0, 0.5, 3.0];
    let tangent = [0.1, -0.2, 0.3, 0.4];
    let cotangent = [1.25];
    let mut group = c.benchmark_group("matrix_2x2_determinant_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| matrix_2x2_determinant_value_inner(black_box(&values)).unwrap());
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| {
            matrix_2x2_determinant_jvp_inner(black_box(&values), black_box(&tangent)).unwrap()
        });
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            matrix_2x2_determinant_vjp_inner(black_box(&values), black_box(&cotangent)).unwrap()
        });
    });
    group.finish();
}

fn bench_matrix_2x2_inverse_ad(c: &mut Criterion) {
    let values = [2.0, -1.0, 0.5, 3.0];
    let tangent = [0.1, -0.2, 0.3, 0.4];
    let cotangent = [0.75, -1.25, 0.5, 2.0];
    let mut group = c.benchmark_group("matrix_2x2_inverse_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| matrix_2x2_inverse_value_inner(black_box(&values)).unwrap());
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| {
            matrix_2x2_inverse_jvp_inner(black_box(&values), black_box(&tangent)).unwrap()
        });
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            matrix_2x2_inverse_vjp_inner(black_box(&values), black_box(&cotangent)).unwrap()
        });
    });
    group.finish();
}

fn bench_matrix_2x2_solve_ad(c: &mut Criterion) {
    use scpn_quantum_engine::compiler_ad::{
        matrix_2x2_solve_jvp_inner, matrix_2x2_solve_value_inner, matrix_2x2_solve_vjp_inner,
    };

    let values = [2.0, -1.0, 0.5, 3.0, 1.5, -2.0];
    let tangent = [0.1, -0.2, 0.3, 0.4, -0.5, 0.75];
    let cotangent = [1.25, -0.75];
    let mut group = c.benchmark_group("matrix_2x2_solve_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| matrix_2x2_solve_value_inner(black_box(&values)).unwrap());
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| matrix_2x2_solve_jvp_inner(black_box(&values), black_box(&tangent)).unwrap());
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            matrix_2x2_solve_vjp_inner(black_box(&values), black_box(&cotangent)).unwrap()
        });
    });
    group.finish();
}

fn bench_symmetric_2x2_cholesky_ad(c: &mut Criterion) {
    let values = [4.0, 1.0, 3.0];
    let tangent = [0.2, -0.3, 0.4];
    let cotangent = [1.25, -0.75, 0.5];
    let mut group = c.benchmark_group("symmetric_2x2_cholesky_ad");
    group.bench_function("value", |bench| {
        bench.iter(|| symmetric_2x2_cholesky_value_inner(black_box(&values)).unwrap());
    });
    group.bench_function("jvp", |bench| {
        bench.iter(|| {
            symmetric_2x2_cholesky_jvp_inner(black_box(&values), black_box(&tangent)).unwrap()
        });
    });
    group.bench_function("vjp", |bench| {
        bench.iter(|| {
            symmetric_2x2_cholesky_vjp_inner(black_box(&values), black_box(&cotangent)).unwrap()
        });
    });
    group.finish();
}

criterion_group!(
    hot_paths,
    bench_build_knm,
    bench_order_parameter,
    bench_commutator_dense,
    bench_is_independent_fast,
    bench_biological_decode_inner,
    bench_matrix_2x2_eigensystem_ad,
    bench_matrix_2x2_determinant_ad,
    bench_matrix_2x2_inverse_ad,
    bench_matrix_2x2_solve_ad,
    bench_symmetric_2x2_cholesky_ad,
    bench_matrix_quadratic_form_ad,
    bench_matrix_frobenius_norm_squared_ad,
    bench_matrix_trace_ad,
    bench_vector_dot_ad,
    bench_vector_squared_norm_ad
);
criterion_main!(hot_paths);
