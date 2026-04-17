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
        let new_op: Vec<f64> = (0..dim * dim)
            .map(|i| ((i as f64) * 0.53).cos())
            .collect();
        group.bench_function(format!("dim={} basis=16", dim), |bench| {
            bench.iter(|| {
                is_independent_fast(
                    black_box(&new_op),
                    black_box(&basis),
                    black_box(1e-9),
                )
            });
        });
    }
    group.finish();
}

criterion_group!(
    hot_paths,
    bench_build_knm,
    bench_order_parameter,
    bench_commutator_dense,
    bench_is_independent_fast
);
criterion_main!(hot_paths);
