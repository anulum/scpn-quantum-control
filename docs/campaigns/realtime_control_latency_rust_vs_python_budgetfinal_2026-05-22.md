# Realtime Control Latency Rust vs Python

- Rust mean: `9.019362` s (n=6)
- Python mean: `11.578785` s (n=6)
- Delta (Python - Rust): `2.559423` s

## Per-lane Means

| Lane | Rust mean (s) | Python mean (s) |
|---|---:|---:|
| `capacity_sweep` | 7.084427 | 10.221194 |
| `rt_adaptive_dense_dynamic` | 7.558378 | 21.290231 |
| `rt_adaptive_dense_open_loop` | 18.386289 | 7.024094 |
| `rt_adaptive_sparse_dynamic` | 6.990436 | 13.510960 |
| `rt_adaptive_sparse_open_loop` | 7.012216 | 7.205036 |
