# LLVM/JIT claim gate

- artifact_id: `llvm-jit-claim-gate-20260704`
- executable_lowering_evidence_id: `native-whole-program-ad-execution-20260622`
- executable_lowering_verified: True
- promotion_ready: False
- benchmark_artifact_ids: none

| requirement | status | evidence |
|-------------|--------|----------|
| executable_lowering | ready | `native-whole-program-ad-execution-20260622` |
| correctness_tests | ready | `tests/test_native_whole_program_ad_execution_evidence.py::test_runner_executes_beyond_scalar_with_reference_parity`; `tests/test_native_whole_program_ad_execution_evidence.py::test_committed_evidence_artifact_is_valid` |
| crash_safety_tests | ready | `tests/test_llvm_jit_crash_safety.py::test_native_llvm_jit_reports_unsupported_wide_determinant_before_compile`; `tests/test_llvm_jit_crash_safety.py::test_native_llvm_jit_rejects_nondifferentiable_selection_boundary`; `tests/test_llvm_jit_crash_safety.py::test_native_llvm_jit_support_metadata_declares_fail_closed_boundaries` |
| benchmark_artifact_ids | blocked | missing |
| rollback_policy | ready | Disable native LLVM/JIT promotion, withdraw any generated public claim row, and route release notes through the claim ledger if crash-safety tests, isolated benchmark artifacts, or dependency checks regress. |
| fallback_policy | ready | Use interpreted Program AD or the existing Python/Rust replay path for value-and-gradient execution whenever the native LLVM/JIT gate is not promotion-ready. |

Claim boundary: Bounded native LLVM/JIT claim gate: executable lowering evidence alone is not a promoted LLVM/JIT claim. Promotion requires verified beyond-scalar native execution, correctness test identifiers, crash-safety test identifiers, isolated benchmark artifact identifiers, rollback policy, and fallback policy; until every requirement is attached there is no LLVM/JIT promotion, provider, hardware, GPU, or performance claim.
