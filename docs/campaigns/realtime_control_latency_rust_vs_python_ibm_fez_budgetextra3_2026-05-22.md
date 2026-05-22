# Realtime Control Latency Rust vs Python

- Rust mean: `6.928691` s (n=6)
- Python mean: `7.126052` s (n=6)
- Delta (Python - Rust): `0.197361` s

## Per-lane Means

| Lane | Rust mean (s) | Python mean (s) |
|---|---:|---:|
| `capacity_sweep` | 6.968364 | 6.961222 |
| `rt_adaptive_dense_dynamic` | 6.279036 | 7.739433 |
| `rt_adaptive_dense_open_loop` | 7.150257 | 6.944611 |
| `rt_adaptive_sparse_dynamic` | 7.221342 | 6.960247 |
| `rt_adaptive_sparse_open_loop` | 6.984782 | 7.189576 |
