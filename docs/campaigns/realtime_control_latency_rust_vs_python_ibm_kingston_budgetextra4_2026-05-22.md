# Realtime Control Latency Rust vs Python

- Rust mean: `7.213458` s (n=6)
- Python mean: `10.428570` s (n=6)
- Delta (Python - Rust): `3.215112` s

## Per-lane Means

| Lane | Rust mean (s) | Python mean (s) |
|---|---:|---:|
| `capacity_sweep` | 6.982567 | 7.017786 |
| `rt_adaptive_dense_dynamic` | 7.720416 | 14.143182 |
| `rt_adaptive_dense_open_loop` | 7.355900 | 13.825718 |
| `rt_adaptive_sparse_dynamic` | 7.006032 | 6.990277 |
| `rt_adaptive_sparse_open_loop` | 7.233269 | 13.576672 |
