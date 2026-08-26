# TN/MPS Baseline Design

This is the CPU-first design artifact for the N=30-40 tensor-network
baseline path. It is planning and preregistration evidence only.

## Boundary

Design and preregistration evidence only; no N>=30 tensor-network row, hardware advantage, or broad quantum-advantage claim is established.

## Decision

Use the Python/quimb CPU MPS adapter as the first execution path, with the bounded native Schmidt/resource model as a deterministic fallback and validation scaffold. Keep ITensor/Julia and GPU TN as explicitly blocked follow-ups until owner-gated toolchain work.

## Adapters

| adapter | language | dependency | status | role | max N |
| --- | --- | --- | --- | --- | ---: |
| quimb_mps_cpu | Python | quimb[tensor] optional extra | optional_dependency | CPU DMRG/TEBD execution adapter for nearest-neighbour or explicitly truncated K_nm | 40 |
| bounded_native_schmidt | Python/NumPy | none | ready | Deterministic resource and discarded-weight scaffold for row validation | 40 |
| itensor_julia | Julia | ITensor.jl | blocked | Future independent MPS parity adapter after Julia ownership is assigned | 40 |
| gpu_tn | CUDA/Python | cuTensorNet or equivalent | owner_gated | Future GPU tensor-network comparison lane | 40 |

## Size Plan

| N | class | CPU-first adapter | CPU execution enabled | GPU follow-up |
| ---: | --- | --- | --- | --- |
| 30 | pilot | quimb_mps_cpu | True | defer to the owner-gated GPU lane |
| 32 | pilot | quimb_mps_cpu | True | defer to the owner-gated GPU lane |
| 36 | extension | quimb_mps_cpu | True | defer to the owner-gated GPU lane |
| 40 | extension | quimb_mps_cpu | True | defer to the owner-gated GPU lane |

## Acceptance Gates

- Every N=30-40 row must record status, wall_time_ms, memory_bytes, max_bond, discarded_weight, entropy proxy, command, machine, dependencies, and git_commit.
- Skipped rows must carry explicit size, dependency, or resource-gate reasons.
- TN/MPS rows must be compared against the registered scaling-protocol matrix before any execution evidence is promoted.

## Blocked Claims

- No broad quantum advantage claim from this design artifact.
- No tensor-network hardness claim until measured N=30-40 TN rows exist.
- No GPU TN comparison until the owner-gated GPU lane promotes it.
- No Julia/ITensor parity claim until a maintained Julia adapter is measured.

## CPU Execution Prerequisites

The CPU-first TN/MPS rows can execute once the quimb dependency, resource caps, and scaling-row schema are pinned by this manifest.

## Regeneration

```bash
scpn-bench s2-tn-mps-baseline-design
```
