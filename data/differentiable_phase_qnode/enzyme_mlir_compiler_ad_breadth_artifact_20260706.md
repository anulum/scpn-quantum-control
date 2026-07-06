# Enzyme/MLIR Compiler-AD Breadth Artifact

- artifact_id: `enzyme-mlir-compiler-ad-breadth-artifact-20260706`
- promotion_ready: `False`
- case_count: `11`
- max_abs_error: `0.000e+00`
- runtime_seconds: `1.424147`
- isolated_benchmark_artifact_id: `phase-qnode-affinity:e5e0a86508e365e8`
- isolated_benchmark_promotion_ready: `False`

Passed cases:

- `llvm_ir_generation`
- `loop_activity`
- `matrix_vjp`
- `native_enzyme_execution`
- `scalar_reverse_mode`
- `vector_vjp`

Failed cases:

- `alias_activity`
- `matrix_jvp`
- `mlir_lowering`
- `scalar_forward_mode`
- `vector_jvp`

| Case | Status | Transform modes | Frontend | Failure class |
|---|---|---|---|---|
| `alias_activity` | `hard_gap` | `forward, jvp, reverse, vjp` | `program_ad_alias` | program_ad_alias_not_enzyme_mlir_raw_case |
| `llvm_ir_generation` | `success` | `reverse, vjp` | `llvm_ir` | none |
| `loop_activity` | `success` | `reverse, vjp` | `llvm_ir_c_loop` | none |
| `matrix_jvp` | `hard_gap` | `jvp` | `llvm_ir` | matrix_jvp_raw_enzyme_case_missing |
| `matrix_vjp` | `success` | `reverse, vjp` | `llvm_ir` | none |
| `mlir_lowering` | `hard_gap` | `forward, jvp, reverse, vjp` | `mlir` | mlir_lowering_runtime_row_missing |
| `native_enzyme_execution` | `success` | `reverse, vjp` | `native_llvm_enzyme` | none |
| `scalar_forward_mode` | `hard_gap` | `forward` | `llvm_ir` | scalar_forward_raw_enzyme_case_missing |
| `scalar_reverse_mode` | `success` | `reverse, vjp` | `llvm_ir` | none |
| `vector_jvp` | `hard_gap` | `jvp` | `llvm_ir` | vector_jvp_raw_enzyme_case_missing |
| `vector_vjp` | `success` | `reverse, vjp` | `llvm_ir` | none |

Claim boundary: Raw Enzyme/MLIR compiler-AD breadth artifact assembled from committed bounded execution evidence. Passing rows carry derivative-correctness evidence only; hard-gap rows remain explicit. This artifact does not promote provider, hardware, GPU, QPU, isolated benchmark, arbitrary compiler-AD, or performance claims.
