# Realtime Control Latency Rust vs Python

- Rust mean: `7.264898` s (n=6)
- Python mean: `7.091577` s (n=6)
- Delta (Python - Rust): `-0.173320` s

## Per-lane Means

| Lane | Rust mean (s) | Python mean (s) |
|---|---:|---:|
| `capacity_sweep` | 7.400608 | 6.973887 |
| `rt_adaptive_dense_dynamic` | 7.480990 | 7.253924 |
| `rt_adaptive_dense_open_loop` | 7.352379 | 7.383076 |
| `rt_adaptive_sparse_dynamic` | 6.962690 | 6.992270 |
| `rt_adaptive_sparse_open_loop` | 6.992113 | 6.972421 |
