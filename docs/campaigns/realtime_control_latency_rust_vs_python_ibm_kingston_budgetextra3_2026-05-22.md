# Realtime Control Latency Rust vs Python

- Rust mean: `8.305232` s (n=6)
- Python mean: `7.180488` s (n=6)
- Delta (Python - Rust): `-1.124744` s

## Per-lane Means

| Lane | Rust mean (s) | Python mean (s) |
|---|---:|---:|
| `capacity_sweep` | 11.425383 | 6.977111 |
| `rt_adaptive_dense_dynamic` | 5.475620 | 7.711961 |
| `rt_adaptive_dense_open_loop` | 7.215859 | 7.361422 |
| `rt_adaptive_sparse_dynamic` | 6.957157 | 7.049236 |
| `rt_adaptive_sparse_open_loop` | 7.331991 | 7.006088 |
