# Compiler Evidence Boundary

- artifact_id: `compiler-evidence-boundary-20260705`
- source_commit: `b75113fb`
- classification: `functional_non_isolated`
- promotion_ready: `False`
- native LLVM/JIT focused selector: `37 passed, 39 deselected`
- Enzyme/LLVM execution: scalar, vector, and matrix cases executed with max gradient error `0.0`

| Requirement | Status | Evidence |
|---|---|---|
| `scalar_forward_mode` | `bounded_evidence_attached` | `llvm-jit-claim-gate-20260704` |
| `scalar_reverse_mode` | `evidence_attached` | `enzyme-toolchain-ad-execution-20260705`, `llvm-jit-claim-gate-20260704` |
| `vector_jvp` | `bounded_evidence_attached` | `llvm-jit-claim-gate-20260704` |
| `vector_vjp` | `evidence_attached` | `enzyme-toolchain-ad-execution-20260705`, `llvm-jit-claim-gate-20260704` |
| `matrix_jvp` | `bounded_evidence_attached` | `llvm-jit-claim-gate-20260704` |
| `matrix_vjp` | `evidence_attached` | `enzyme-toolchain-ad-execution-20260705`, `llvm-jit-claim-gate-20260704` |
| `loop_activity` | `bounded_evidence_attached` | `enzyme-toolchain-ad-execution-20260705` |
| `alias_activity` | `blocked` | No dedicated alias-activity compiler evidence artifact is attached. |
| `mlir_lowering` | `bounded_evidence_attached` | `enzyme_mlir_maturity_audit_20260616`, `llvm-jit-claim-gate-20260704` |
| `llvm_ir_generation` | `evidence_attached` | `native-whole-program-ad-execution-20260622`, `enzyme-toolchain-ad-execution-20260705` |
| `native_enzyme_execution` | `evidence_attached` | `enzyme-toolchain-ad-execution-20260705` |

Promotion blockers:

- prescribed native_llvm_jit selector selected zero tests
- isolated compiler benchmark artifact IDs missing
- native LLVM/JIT crash-safety tests missing
- alias-activity compiler evidence missing
- compiler promotion batch not assembled

Claim boundary: Compiler evidence boundary only. Bounded native LLVM/JIT tests and current Enzyme/LLVM execution evidence are attached for reviewer triage. This artifact does not promote general compiler AD, isolated benchmarks, provider, hardware, GPU, or performance claim.
