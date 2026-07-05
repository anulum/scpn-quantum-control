# Compiler Evidence Boundary

- artifact_id: `compiler-evidence-boundary-20260705`
- source_commit: `534010da`
- classification: `functional_non_isolated`
- promotion_ready: `False`
- native LLVM/JIT focused selector: `37 passed, 39 deselected`
- native LLVM/JIT prescribed selector: `37 passed, 39 deselected`
- native LLVM/JIT crash-safety selector: `3 passed`
- Enzyme/LLVM execution: scalar, vector, and matrix cases executed with max gradient error `0.0`
- alias activity: `5 complete lattice cases, 3 fail-closed lattice cases`

| Requirement | Status | Evidence |
|---|---|---|
| `scalar_forward_mode` | `bounded_evidence_attached` | `llvm-jit-claim-gate-20260704` |
| `scalar_reverse_mode` | `evidence_attached` | `enzyme-toolchain-ad-execution-20260705`, `llvm-jit-claim-gate-20260704` |
| `vector_jvp` | `bounded_evidence_attached` | `llvm-jit-claim-gate-20260704` |
| `vector_vjp` | `evidence_attached` | `enzyme-toolchain-ad-execution-20260705`, `llvm-jit-claim-gate-20260704` |
| `matrix_jvp` | `bounded_evidence_attached` | `llvm-jit-claim-gate-20260704` |
| `matrix_vjp` | `evidence_attached` | `enzyme-toolchain-ad-execution-20260705`, `llvm-jit-claim-gate-20260704` |
| `loop_activity` | `bounded_evidence_attached` | `enzyme-toolchain-ad-execution-20260705` |
| `alias_activity` | `bounded_evidence_attached` | `compiler-alias-activity-evidence-20260706` |
| `mlir_lowering` | `bounded_evidence_attached` | `enzyme_mlir_maturity_audit_20260616`, `llvm-jit-claim-gate-20260704` |
| `llvm_ir_generation` | `evidence_attached` | `native-whole-program-ad-execution-20260622`, `enzyme-toolchain-ad-execution-20260705` |
| `native_enzyme_execution` | `evidence_attached` | `enzyme-toolchain-ad-execution-20260705` |

Promotion blockers:

- isolated compiler benchmark artifact IDs missing
- compiler promotion batch not assembled

Selector policy: the prescribed native LLVM/JIT selector is the path-targeted positive pytest selector `PYTHONPATH=src:oscillatools/src python3 -m pytest tests/test_mlir_realtime_cloud.py -q -k "native_llvm_jit"`. It avoids path-name negation and selected `37` tests with `39` deselected in the 2026-07-06 local evidence run.

Crash-safety evidence: `tests/test_llvm_jit_crash_safety.py::test_native_llvm_jit_reports_unsupported_wide_determinant_before_compile`, `tests/test_llvm_jit_crash_safety.py::test_native_llvm_jit_rejects_nondifferentiable_selection_boundary`, and `tests/test_llvm_jit_crash_safety.py::test_native_llvm_jit_support_metadata_declares_fail_closed_boundaries`.

Alias-activity evidence: `compiler-alias-activity-evidence-20260706` records bounded Program AD static alias-lattice activity for `5` complete cases and `3` fail-closed cases. Observed alias kinds: `alias_analysis`, `control_path_alias`, `expression_rebinding_alias`, `list_alias`, `local_rebinding_alias`, `loop_carried_state`, `mutation_version`, `object_attribute_alias`, `view_alias`.

Claim boundary: Compiler evidence boundary only. Bounded native LLVM/JIT tests, crash-safety tests, current Enzyme/LLVM execution evidence, and Program AD alias-activity lattice evidence are attached for reviewer triage. This artifact does not promote general compiler AD, isolated benchmarks, provider, hardware, GPU, or performance claim.
