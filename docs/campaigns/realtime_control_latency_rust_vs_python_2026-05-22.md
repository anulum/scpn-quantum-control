# Realtime Control Latency Rust vs Python

- Rust mean: `7.144280` s (n=12)
- Python mean: `6.789447` s (n=12)
- Delta (Python - Rust): `-0.354833` s

## Per-lane Means

| Lane | Rust mean (s) | Python mean (s) |
|---|---:|---:|
| `capacity_sweep` | 7.075722 | 6.818273 |
| `rt_adaptive_dense_dynamic` | 7.817875 | 7.611475 |
| `rt_adaptive_dense_open_loop` | 6.991504 | 5.245447 |
| `rt_adaptive_sparse_dynamic` | 7.356091 | 7.094629 |
| `rt_adaptive_sparse_open_loop` | 6.960109 | 6.975633 |
