# Compiler Promotion Batch

- artifact_id: `compiler-promotion-batch-20260706`
- source_commit: `8df68be7`
- classification: `functional_non_isolated`
- status: `blocked_missing_isolated_compiler_benchmark_ids`
- promotion_ready: `False`
- assembled evidence files: `10`

Missing requirements:

- `isolated compiler benchmark artifact IDs`

Promotion blockers:

- isolated compiler benchmark artifact IDs missing

| Evidence file | Role | Artifact ID | SHA-256 |
|---|---|---|---|
| `data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.json` | Program AD compiler alias-activity evidence | `compiler-alias-activity-evidence-20260706` | `87f194287513dc254e07195858fe13a9b479c00d4cc73aba2653baaba5c4bc35` |
| `data/differentiable_phase_qnode/compiler_alias_activity_evidence_20260706.md` | Program AD compiler alias-activity reviewer summary | `compiler-alias-activity-evidence-20260706` | `93e71c273495508e86ca1ed0a116c243f58c2ccdf3ed81e18287bf8d89fd67ce` |
| `data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.json` | Compiler evidence boundary and remaining promotion gates | `compiler-evidence-boundary-20260705` | `4b818e07e1de839ffd185cf0694c67a36de9a4532ecaf85626cb005a40ac823b` |
| `data/differentiable_phase_qnode/compiler_evidence_boundary_20260705.md` | Compiler evidence boundary reviewer summary | `compiler-evidence-boundary-20260705` | `d89870f45d1ff2faa6a72cf007e72d897c302869b32a6385179d49efa7ff50be` |
| `data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.json` | Enzyme/MLIR maturity and hard-gap evidence | `enzyme-mlir-maturity-audit-20260616` | `4c22d858a28b1ff94b06b4a135039eccc49890607c584a38b413e0ead46fb707` |
| `data/differentiable_phase_qnode/enzyme_mlir_maturity_audit_20260616.md` | Enzyme/MLIR maturity reviewer summary | `enzyme-mlir-maturity-audit-20260616` | `ec3b9caedfc2b20863b37309027a8cc49162de6400e0959a93cb75af4ca24656` |
| `data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.json` | Native LLVM/JIT promotion claim gate | `llvm-jit-claim-gate-20260704` | `e040b1d07b0e5c8195edfe18702f428510e2b05292ab934d9926b54ad7e73a26` |
| `data/differentiable_phase_qnode/llvm_jit_claim_gate_20260704.md` | Native LLVM/JIT promotion claim-gate reviewer summary | `llvm-jit-claim-gate-20260704` | `05b1fa03eb5dde4796e0dba999cf8e03fae3bffae028c3a0ef7122f9419febdd` |
| `data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.json` | Native LLVM/JIT whole-program AD execution evidence | `native-whole-program-ad-execution-20260622` | `c73447bb64f21e7412eb479ee67a773fd1687ebce5357dc7a07c6f9256faf6b0` |
| `data/differentiable_phase_qnode/native_whole_program_ad_execution_evidence_20260622.md` | Native LLVM/JIT whole-program AD execution reviewer summary | `native-whole-program-ad-execution-20260622` | `6e6d58e4e99595c271164fe0ace562e2e1a9fc3c099a34bdb8b312b7010c93c0` |

Claim boundary: Compiler promotion batch assembly only: committed compiler-boundary, alias-activity, native LLVM/JIT, native whole-program AD, and Enzyme/MLIR maturity evidence are checksummed for reviewer triage; this does not promote general compiler AD, isolated benchmarks, provider, hardware, GPU, or performance claim.
