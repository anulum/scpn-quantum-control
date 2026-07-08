# TN/MPS Crossover Stage-1 Gate

This QWC-5.1 artifact admits the larger-than-16-node N=30-40
tensor-network crossover row format before any owner-gated compute run.

## Boundary

Stage-1 schema and admission evidence for N=30-40 tensor-network crossover rows only; no N=30-40 solver row, hardware advantage, tensor-network hardness, or broad quantum-advantage claim is established.

## Row Schema

- protocol: `qwc_5_1_tn_mps_crossover_stage1`
- target sizes: `30, 32, 36, 40`
- required baselines: `classical_ode, mps_tensor_network, aer_statevector_or_skip`

| required field |
| --- |
| `protocol_id` |
| `n_qubits` |
| `baseline` |
| `status` |
| `wall_time_ms` |
| `memory_bytes` |
| `max_bond` |
| `discarded_weight` |
| `entropy_proxy` |
| `truncation_policy` |
| `omitted_coupling_mass` |
| `command` |
| `machine` |
| `dependencies` |
| `git_commit` |
| `host_load` |
| `claim_boundary` |
| `notes` |

## Stage-1 Gates

| gate | passed | evidence |
| --- | --- | --- |
| `target_sizes_exceed_sixteen` | `True` | target_sizes=(30, 32, 36, 40) |
| `row_schema_pinned` | `True` | 18 required row fields pinned |
| `stage2_compute_owner_gated` | `True` | stage-2 N=30-40 execution remains owner-gated |
| `claim_boundary_closed` | `True` | design keeps advantage_claim_allowed=false and benchmark_execution_performed=false |

## Blocked Claims

- broad quantum advantage
- tensor-network hardness
- hardware scaling win
- GPU tensor-network comparison
- Julia/ITensor parity

## Owner-Gated Follow-ups

- execute N=30-40 CPU-first quimb rows with isolated host metadata
- promote or skip GPU tensor-network rows only through the owner-gated GPU lane
- add maintained Julia/ITensor parity only after Julia toolchain ownership exists

## Regeneration

```bash
scpn-bench s2-tn-crossover-stage1
```
