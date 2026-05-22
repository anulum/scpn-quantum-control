# Realtime Control Latency Rust vs Python

- Rust mean: `6.801363` s (n=6)
- Python mean: `6.737755` s (n=6)
- Delta (Python - Rust): `-0.063607` s

## Per-lane Means

| Lane | Rust mean (s) | Python mean (s) |
|---|---:|---:|
| `capacity_sweep` | 7.071447 | 6.996617 |
| `rt_adaptive_dense_dynamic` | 5.558445 | 7.143310 |
| `rt_adaptive_dense_open_loop` | 7.146573 | 5.184031 |
| `rt_adaptive_sparse_dynamic` | 7.006965 | 6.941932 |
| `rt_adaptive_sparse_open_loop` | 6.953297 | 7.164026 |
