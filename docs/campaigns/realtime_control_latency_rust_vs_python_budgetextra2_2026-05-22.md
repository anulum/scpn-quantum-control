# Realtime Control Latency Rust vs Python

- Rust mean: `6.517766` s (n=6)
- Python mean: `6.409961` s (n=6)
- Delta (Python - Rust): `-0.107805` s

## Per-lane Means

| Lane | Rust mean (s) | Python mean (s) |
|---|---:|---:|
| `capacity_sweep` | 7.058155 | 6.985230 |
| `rt_adaptive_dense_dynamic` | 5.619616 | 5.237468 |
| `rt_adaptive_dense_open_loop` | 7.393288 | 7.195799 |
| `rt_adaptive_sparse_dynamic` | 4.968204 | 5.064707 |
| `rt_adaptive_sparse_open_loop` | 7.009176 | 6.991331 |
